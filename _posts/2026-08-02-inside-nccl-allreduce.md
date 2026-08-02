---
layout: post
title: "Inside NCCL's all-reduce: ring, double binary tree, or neither?"
date: 2026-08-02
tags: [nccl, distributed-training, gpu, collectives]
---

In the [last post](/2026/07/26/fsdp-collectives-101.html) I said the standard ring
all-reduce is literally a reduce-scatter and an all-gather run back to back. That's
true, and it's also where most explanations stop, mine included. It left me with a
picture of one algorithm, the ring, faithfully executed every time someone calls
`ncclAllReduce`. So I cloned NCCL (pronounced "nickel", version 2.30, current master)
and read the implementation, and the picture underneath is much better than the one I
was carrying. NCCL doesn't have an all-reduce algorithm. It has six, plus three
different wire protocols to run them over, and it prices every valid combination with
a little cost model on every call and takes the cheapest. The ring you learned from
the classic blog posts is just one row of that menu, and on a modern H100 machine it
often loses to an algorithm where no GPU sends data to any other GPU at all, because
the NVLink switch does the arithmetic.

This post is a guided tour of that machinery, with file and line references into the
source so you can check everything I claim. Everything below is from NCCL 2.30
(master as of August 2026); constants do drift between releases.

If you only take three lines from this post:

1. The ring is bandwidth optimal but its latency grows linearly with the number of
   GPUs. The tree is the opposite: latency grows with the log of the node count, but
   it gives up some bandwidth.
2. There is no threshold constant that picks between them. NCCL models every
   algorithm and protocol pair as `time = latency + bytes/bandwidth` and takes the
   argmin, per call, at enqueue time.
3. On NVSwitch systems the winner is often neither: the switch reduces the data
   itself (NVLink SHARP), and across nodes the InfiniBand switches can too.

## The ring, exactly as the kernel runs it

Start with the algorithm the last post promised. Every rank splits the buffer into
`k` chunks, one per rank in the ring. The reduce-scatter half takes `k-1` steps: each
step, every rank sends one chunk to its ring neighbor and receives a different chunk,
adding what arrives into its own copy. After `k-1` hops a chunk has visited everyone
and the rank holding it has the full sum. The all-gather half is another `k-1` steps
of the same motion, except now the finished chunks circulate unchanged. Total:
`2(k-1)` steps, and every link carries a different chunk on every step, which is why
nothing idles and why the ring hits the bandwidth lower bound of `2(k-1)/k` of the
buffer size sent per GPU.

Here is one chunk's journey on four GPUs. All four chunks make this same trip
simultaneously, one position apart, so this diagram is happening four times at once,
rotated:

<div style="text-align:center">
<svg viewBox="0 0 680 300" width="100%" style="max-width:680px;height:auto" role="img" aria-label="Diagram of one chunk traveling a 4-GPU ring: reduce-scatter phase accumulates the sum, all-gather phase distributes it">
<rect x="14" y="6" width="316" height="266" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<rect x="354" y="6" width="316" height="266" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<text x="172" y="28" text-anchor="middle" font-size="14" font-weight="bold" fill="#a05c1a">reduce-scatter half</text>
<text x="512" y="28" text-anchor="middle" font-size="14" font-weight="bold" fill="#3f7a3f">all-gather half</text>
<!-- left ring: GPU0 TL, GPU1 TR, GPU2 BR, GPU3 BL -->
<rect x="60" y="52" width="58" height="30" rx="5" fill="#fff" stroke="#999"/><text x="89" y="72" text-anchor="middle" font-size="12" fill="#333">GPU 0</text>
<rect x="226" y="52" width="58" height="30" rx="5" fill="#fff" stroke="#999"/><text x="255" y="72" text-anchor="middle" font-size="12" fill="#333">GPU 1</text>
<rect x="226" y="196" width="58" height="30" rx="5" fill="#fff" stroke="#999"/><text x="255" y="216" text-anchor="middle" font-size="12" fill="#333">GPU 2</text>
<rect x="60" y="196" width="58" height="30" rx="5" fill="#f6dfc4" stroke="#b06f2a" stroke-width="1.6"/><text x="89" y="216" text-anchor="middle" font-size="12" fill="#333">GPU 3</text>
<line x1="122" y1="67" x2="220" y2="67" stroke="#b06f2a" stroke-width="1.6"/><polygon points="220,63 228,67 220,71" fill="#b06f2a"/>
<text x="172" y="60" text-anchor="middle" font-size="10.5" fill="#a05c1a">g0</text>
<line x1="255" y1="86" x2="255" y2="190" stroke="#b06f2a" stroke-width="1.6"/><polygon points="251,190 255,198 259,190" fill="#b06f2a"/>
<text x="264" y="140" font-size="10.5" fill="#a05c1a">g0+g1</text>
<line x1="222" y1="211" x2="124" y2="211" stroke="#b06f2a" stroke-width="1.6"/><polygon points="124,207 116,211 124,215" fill="#b06f2a"/>
<text x="172" y="204" text-anchor="middle" font-size="10.5" fill="#a05c1a">g0+g1+g2</text>
<text x="89" y="244" text-anchor="middle" font-size="10.5" fill="#a05c1a">+g3 = full sum</text>
<text x="172" y="262" text-anchor="middle" font-size="10.5" fill="#888">3 hops, adding at every stop</text>
<!-- right ring -->
<rect x="400" y="52" width="58" height="30" rx="5" fill="#d8ecd8" stroke="#9cc49c"/><text x="429" y="72" text-anchor="middle" font-size="12" fill="#333">GPU 0</text>
<rect x="566" y="52" width="58" height="30" rx="5" fill="#d8ecd8" stroke="#9cc49c"/><text x="595" y="72" text-anchor="middle" font-size="12" fill="#333">GPU 1</text>
<rect x="566" y="196" width="58" height="30" rx="5" fill="#d8ecd8" stroke="#9cc49c"/><text x="595" y="216" text-anchor="middle" font-size="12" fill="#333">GPU 2</text>
<rect x="400" y="196" width="58" height="30" rx="5" fill="#7fb97f" stroke="#4e8a4e" stroke-width="1.6"/><text x="429" y="216" text-anchor="middle" font-size="12" fill="#333">GPU 3</text>
<line x1="429" y1="190" x2="429" y2="86" stroke="#4e8a4e" stroke-width="1.6"/><polygon points="425,86 429,78 433,86" fill="#4e8a4e"/>
<text x="404" y="140" font-size="10.5" fill="#3f7a3f">sum</text>
<line x1="462" y1="67" x2="560" y2="67" stroke="#4e8a4e" stroke-width="1.6"/><polygon points="560,63 568,67 560,71" fill="#4e8a4e"/>
<text x="512" y="60" text-anchor="middle" font-size="10.5" fill="#3f7a3f">sum</text>
<line x1="595" y1="86" x2="595" y2="190" stroke="#4e8a4e" stroke-width="1.6"/><polygon points="591,190 595,198 599,190" fill="#4e8a4e"/>
<text x="604" y="140" font-size="10.5" fill="#3f7a3f">sum</text>
<text x="512" y="244" text-anchor="middle" font-size="10.5" fill="#3f7a3f">3 more hops, copying only</text>
<text x="512" y="262" text-anchor="middle" font-size="10.5" fill="#888">2(k-1) = 6 steps total for k = 4</text>
</svg>
</div>

The orange and green are deliberate: they're the same colors the last post used for
reduce-scatter and all-gather, because the ring all-reduce literally is those two
collectives fused.

And it really is fused, not two calls. The whole thing is one loop in the device
kernel, `runRing` in `src/device/all_reduce.h:14`. Trimmed to its skeleton:

```c
// step 0: push my chunk to the next GPU
prims.directSend(offset, offset, nelem);

// k-2 steps: receive a chunk, add mine, forward the partial sum
for (int j = 2; j < nranks; ++j)
  prims.directRecvReduceDirectSend(offset, offset, nelem);

// step k-1: the arriving chunk completes here; keep it and forward it
prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

// k-2 steps: receive a finished chunk, keep it, forward it
for (int j = 1; j < nranks - 1; ++j)
  prims.directRecvCopyDirectSend(offset, offset, nelem);

// last step: receive the final chunk, nothing left to forward
prims.directRecv(offset, nelem);
```

Those primitive names are the vocabulary the whole library is written in.
`recvReduceSend` means "receive from my ring predecessor, add my contribution,
send the result to my successor", and it happens as one fused operation: data
streams from the receive buffer through the adds and out the send buffer without a
round trip to memory in between. The `postOp=true` on the middle step is where an
average gets its divide, folded into the step where each chunk's sum completes.

Two details the textbook picture leaves out. First, a rank doesn't wait for a whole
chunk before forwarding. Chunks are cut into slices and pushed through an 8-slot
FIFO per peer (`NCCL_STEPS` in `src/include/device.h:26`), so step `j+1` of the
pipeline starts while step `j` is still arriving. Second, none of this runs once.
NCCL carves the buffer across many independent rings.

## Channels: the ring is plural

A "channel" is NCCL's unit of parallelism: one CUDA thread block, on one SM, with
its own ring order, its own FIFO buffers, and its own slice of the input
(`grid.x` is exactly the channel count, `src/enqueue.cc:1753`). A 350 GB/s NVLink
mesh can't be saturated by one block doing loads and stores, so NCCL runs up to 64
channels (`MAXCHANNELS`, `src/include/device.h`) and splits every collective across
them. The ring orderings themselves come out of a topology search
(`src/graph/search.cc`) that walks the PCIe/NVLink/NIC graph at init time looking
for orderings that maximize per-channel bandwidth, which is why the ring order
rarely matches rank order.

This matters for reading the rest of the post: when the cost model says "tree gets
half the bandwidth", the mechanism is channels. Odd work goes to one structure,
even work to another, and the two run concurrently on different SMs.

## Where the ring hurts

Count the steps again: `2(k-1)`, and they're sequential. Each chunk's sum isn't done
until it has physically visited every rank. On 8 GPUs that's 14 hops. On 1024 GPUs
it's 2046 hops, and that cost is paid even by a 4-byte all-reduce, because hops are
hops regardless of size. Bandwidth optimal, latency linear. For big gradient buckets
the pipeline hides it; for the small, frequent all-reduces that show up everywhere
in real systems (loss scalars, norms, router statistics, anything at high world
size) the alpha term is the whole cost.

The fix is old: reduce up a tree, broadcast back down. Latency becomes logarithmic
in the number of nodes. The problem that kept trees out of NCCL for years is
bandwidth: in a binary tree, roughly half the ranks are leaves. A leaf only pushes
its own data up once, while an interior rank moves three times that much (two
children's data coming in, the merged stream going up, and the broadcast coming
back down), so the collective runs at the speed of its busiest ranks while the
leaves' links sit mostly idle. That imbalance is the problem the double binary
tree solves.

## The double binary tree

NCCL builds the tree in `src/graph/trees.cc:32` with a bit trick: for a power-of-two
world, take each rank's lowest set bit; flipping it gives your parent, halving it
gives your children. The comment in the source draws it better than I can, so here
it is, lifted directly (14 ranks):

```
0---------------8
         ______/ \______
        4               12
      /   \            /  \
    2       6       10     \
   / \     / \     /  \     \
  1   3   5   7   9   11    13
```

Notice who the leaves are: the odd ranks. Every interior rank is even. So build a
second tree that's the mirror image of the first (`ncclGetDtree`,
`src/graph/trees.cc:90`), and the roles swap exactly: every rank that idles as a
leaf in tree one works as an interior node in tree two. NCCL then assigns half its
channels to each tree, so half of every buffer flows up one tree while the other
half flows up the other. Both trees together use every rank's send bandwidth every
step. This is the construction from Sanders, Speck and Träff's two-tree paper, and
it's what NCCL 2.4 shipped as "double binary trees": tree latency at roughly ring
bandwidth.

<div style="text-align:center">
<svg viewBox="0 0 680 250" width="100%" style="max-width:680px;height:auto" role="img" aria-label="Two mirrored binary trees over 12 ranks; leaves of one tree are interior nodes of the other">
<rect x="14" y="6" width="316" height="226" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<rect x="354" y="6" width="316" height="226" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<text x="172" y="26" text-anchor="middle" font-size="13" font-weight="bold" fill="#a05c1a">tree one</text>
<text x="512" y="26" text-anchor="middle" font-size="13" font-weight="bold" fill="#3f7a3f">tree two (mirror)</text>
<!-- tree1 edges: 0-8, 8-4, 8-10, 4-2, 4-6, 10-9, 10-11, 2-1, 2-3, 6-5, 6-7 -->
<g stroke="#d9ae7a" stroke-width="1.4">
<line x1="52" y1="52" x2="216" y2="52"/><line x1="216" y1="52" x2="132" y2="100"/><line x1="216" y1="52" x2="262" y2="148"/>
<line x1="132" y1="100" x2="86" y2="148"/><line x1="132" y1="100" x2="178" y2="148"/>
<line x1="262" y1="148" x2="240" y2="196"/><line x1="262" y1="148" x2="288" y2="196"/>
<line x1="86" y1="148" x2="64" y2="196"/><line x1="86" y1="148" x2="108" y2="196"/>
<line x1="178" y1="148" x2="156" y2="196"/><line x1="178" y1="148" x2="200" y2="196"/>
</g>
<!-- tree1 nodes: interior even=solid orange, leaves odd=pale -->
<g font-size="11">
<circle cx="52" cy="52" r="13" fill="#e6a15c" stroke="#b06f2a"/><text x="52" y="56" text-anchor="middle" fill="#333">0</text>
<circle cx="216" cy="52" r="13" fill="#e6a15c" stroke="#b06f2a"/><text x="216" y="56" text-anchor="middle" fill="#333">8</text>
<circle cx="132" cy="100" r="13" fill="#e6a15c" stroke="#b06f2a"/><text x="132" y="104" text-anchor="middle" fill="#333">4</text>
<circle cx="262" cy="148" r="13" fill="#e6a15c" stroke="#b06f2a"/><text x="262" y="152" text-anchor="middle" fill="#333">10</text>
<circle cx="86" cy="148" r="13" fill="#e6a15c" stroke="#b06f2a"/><text x="86" y="152" text-anchor="middle" fill="#333">2</text>
<circle cx="178" cy="148" r="13" fill="#e6a15c" stroke="#b06f2a"/><text x="178" y="152" text-anchor="middle" fill="#333">6</text>
<circle cx="64" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="64" y="200" text-anchor="middle" fill="#777">1</text>
<circle cx="108" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="108" y="200" text-anchor="middle" fill="#777">3</text>
<circle cx="156" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="156" y="200" text-anchor="middle" fill="#777">5</text>
<circle cx="200" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="200" y="200" text-anchor="middle" fill="#777">7</text>
<circle cx="240" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="240" y="200" text-anchor="middle" fill="#777">9</text>
<circle cx="288" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="288" y="200" text-anchor="middle" fill="#777">11</text>
</g>
<!-- tree2: mirror, node v = 11 - v of tree1. edges: 11-3, 3-7, 3-1, 7-9, 7-5, 1-2, 1-0, 9-10, 9-8, 5-6, 5-4 -->
<g stroke="#9cc49c" stroke-width="1.4">
<line x1="628" y1="52" x2="464" y2="52"/><line x1="464" y1="52" x2="548" y2="100"/><line x1="464" y1="52" x2="418" y2="148"/>
<line x1="548" y1="100" x2="594" y2="148"/><line x1="548" y1="100" x2="502" y2="148"/>
<line x1="418" y1="148" x2="440" y2="196"/><line x1="418" y1="148" x2="392" y2="196"/>
<line x1="594" y1="148" x2="616" y2="196"/><line x1="594" y1="148" x2="572" y2="196"/>
<line x1="502" y1="148" x2="524" y2="196"/><line x1="502" y1="148" x2="480" y2="196"/>
</g>
<g font-size="11">
<circle cx="628" cy="52" r="13" fill="#7fb97f" stroke="#4e8a4e"/><text x="628" y="56" text-anchor="middle" fill="#333">11</text>
<circle cx="464" cy="52" r="13" fill="#7fb97f" stroke="#4e8a4e"/><text x="464" y="56" text-anchor="middle" fill="#333">3</text>
<circle cx="548" cy="100" r="13" fill="#7fb97f" stroke="#4e8a4e"/><text x="548" y="104" text-anchor="middle" fill="#333">7</text>
<circle cx="418" cy="148" r="13" fill="#7fb97f" stroke="#4e8a4e"/><text x="418" y="152" text-anchor="middle" fill="#333">1</text>
<circle cx="594" cy="148" r="13" fill="#7fb97f" stroke="#4e8a4e"/><text x="594" y="152" text-anchor="middle" fill="#333">9</text>
<circle cx="502" cy="148" r="13" fill="#7fb97f" stroke="#4e8a4e"/><text x="502" y="152" text-anchor="middle" fill="#333">5</text>
<circle cx="392" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="392" y="200" text-anchor="middle" fill="#777">0</text>
<circle cx="440" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="440" y="200" text-anchor="middle" fill="#777">2</text>
<circle cx="480" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="480" y="200" text-anchor="middle" fill="#777">4</text>
<circle cx="524" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="524" y="200" text-anchor="middle" fill="#777">6</text>
<circle cx="572" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="572" y="200" text-anchor="middle" fill="#777">8</text>
<circle cx="616" cy="196" r="13" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,2"/><text x="616" y="200" text-anchor="middle" fill="#777">10</text>
</g>
<text x="340" y="242" text-anchor="middle" font-size="10.5" fill="#888">solid = interior (reduces and forwards), dashed = leaf. Every rank is solid in exactly one tree.</text>
</svg>
</div>

Three implementation details that surprised me:

**The tree is between nodes, not GPUs.** The double binary tree is built over
*nodes* (`connectTrees`, `src/graph/connect.cc:138`). Inside a node, the local GPUs
form a simple chain hanging off the node's position in the tree
(`src/graph/connect.cc:61`). So a 128-node, 1024-GPU job has a 128-node double tree
with 8-GPU chains inside NVLink domains, where hops are cheap. On a single node the
"tree" degenerates to just the chain, which buys nothing over the ring; the tree's
win is a multi-node story.

**Reduce and broadcast run at the same time.** I pictured tree all-reduce as two
phases: everything reduces to the root, then everything broadcasts down. The kernel
doesn't work that way. `runTreeSplit` (`src/device/all_reduce.h:146`) splits each
rank's thread block into two teams: one runs `recvReduceSend` up the tree while the
other simultaneously runs `recvCopySend` down it, chunk by chunk. A chunk bounces
off the root and heads back down while later chunks are still climbing. The split
is 70/30 in favor of the reduce side for the low-latency protocols, because
reducing three children's data costs more than forwarding to three children (the
comment at `all_reduce.h:160` says exactly this).

**There's no whole-message wait anywhere.** Same slicing and 8-slot FIFOs as the
ring, so tree latency really is proportional to depth, not depth times message
size.

## How NCCL picks: a cost model, not a threshold

Old NCCL had `NCCL_TREE_THRESHOLD`. It was removed in 2.5, and what replaced it is
nicer. At init, `ncclTopoTuneModel` (`src/graph/tuning.cc:243`) fills two tables,
`latencies[collective][algorithm][protocol]` and
`bandwidths[collective][algorithm][protocol]`, from measured constants: base launch
overheads, per-hop latencies for NVLink vs PCIe vs network, per-architecture
bandwidth ceilings. Then every call (really every aggregated batch of calls) runs
the argmin in `topoGetAlgoInfo` (`src/enqueue.cc:2028`) over all pairs, where the
cost of a pair is one line (`src/graph/tuning.cc:646`):

```c
*time = lat * latCount + nBytes / (1000 * bw);
```

Latency plus bytes over bandwidth. For ring all-reduce the latency entry works out
to the hop count you'd derive on paper, split by link type:

```
ring:  2(nRanks-1) hops:  (2(nRanks-1) - 2(nNodes-1)) intra-node
                          + 2(nNodes-1) network hops
tree:  2((ranksPerNode-1) intra-node + log2(nNodes) network hops)
```

That last line is the whole tree story in one expression: the network term, the
expensive one, went from linear in nodes to logarithmic. At 16 nodes, ring pays 30
network-latency units, tree pays 8. At 128 nodes it's 254 versus 14. Meanwhile the
bandwidth table charges the tree for its structural overheads (a factor around 0.9,
plus per-architecture ceilings), so the model naturally produces the classic
picture: tree wins small, ring wins large, and the crossover slides upward with
node count. No threshold anywhere; it falls out of two lines crossing.

<div style="text-align:center">
<svg viewBox="0 0 680 270" width="100%" style="max-width:680px;height:auto" role="img" aria-label="Sketch of the cost model: time versus message size for tree and ring, with tree cheaper at small sizes and ring cheaper at large sizes">
<rect x="14" y="6" width="652" height="246" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<line x1="70" y1="210" x2="630" y2="210" stroke="#bbb" stroke-width="1"/>
<line x1="70" y1="210" x2="70" y2="30" stroke="#bbb" stroke-width="1"/>
<text x="350" y="234" text-anchor="middle" font-size="11.5" fill="#666">message size (log scale)</text>
<text x="36" y="120" text-anchor="middle" font-size="11.5" fill="#666" transform="rotate(-90 36 120)">time per call</text>
<!-- ring: high intercept, shallow late rise -->
<path d="M 70 150 C 220 149, 340 146, 430 132 C 510 119, 580 96, 630 72" fill="none" stroke="#4e8a4e" stroke-width="2.2"/>
<!-- tree: low intercept, steeper rise -->
<path d="M 70 190 C 200 188, 300 178, 390 148 C 470 121, 550 74, 615 32" fill="none" stroke="#5b6ee1" stroke-width="2.2"/>
<line x1="452" y1="210" x2="452" y2="46" stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>
<text x="446" y="58" text-anchor="end" font-size="10.5" fill="#888">crossover</text>
<text x="150" y="176" font-size="11.5" fill="#5b6ee1">tree: low latency floor</text>
<text x="132" y="136" font-size="11.5" fill="#3f7a3f">ring: pays 2(n-1) hops up front</text>
<text x="622" y="145" text-anchor="end" font-size="11.5" fill="#3f7a3f">ring: near-optimal bandwidth</text>
<text x="608" y="56" text-anchor="end" font-size="11.5" fill="#5b6ee1" stroke="#fafafa" stroke-width="3" paint-order="stroke">tree: bandwidth derated</text>
<text x="255" y="200" text-anchor="middle" font-size="10.5" fill="#888">tree wins here</text>
<text x="555" y="200" text-anchor="middle" font-size="10.5" fill="#888">ring wins here</text>
<text x="350" y="248" text-anchor="middle" font-size="10" fill="#999">shape of the model, not a measurement; the crossover moves right as node count grows</text>
</svg>
</div>

My favorite artifact in this file is `treeCorrectionFactor`
(`src/graph/tuning.cc:623`), a hand-tuned table of 24 numbers, one per power-of-two
size from 64 B up, that derates tree bandwidth by up to 60 percent in the awkward
middle sizes around 128 KB to 1 MB:

```c
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4, ... },
  ...
```

The comment above it is disarmingly honest: "Trees are not perfectly sticking to
the model for medium sizes. Applying a static correction factor is not ideal but
works quite well." A reminder that under the clean alpha-beta model there's an
engineer with a benchmark harness making the numbers match reality.

You can overrule all of it with `NCCL_ALGO=Tree` or `NCCL_ALGO=Ring` (and
`NCCL_PROTO=...`), which is also the best way to feel the difference on your own
cluster.

## Three ways to move a byte

The same argmin also picks the wire protocol, and this layer was completely new to
me. The algorithm says who talks to whom; the protocol says what a message
physically looks like, and it exists because of a synchronization problem: how does
the receiver know the data in the FIFO slot is ready?

**Simple** is the obvious design. Write the payload, execute a memory fence, then
bump a tail counter the receiver is polling (`src/device/prims_simple.h:164`). Full
bandwidth, but the fence is expensive and sits on the critical path of every hop,
so it shows up as latency. NCCL even dedicates a warp per block just to overlap the
fence with the copies (the "extra warp for sync" at `src/enqueue.cc:2102`).

**LL (low latency)** makes the fence disappear with a trick. Data travels in
16-byte lines: 4 bytes of data, 4 bytes of flag, 4 of data, 4 of flag
(`ncclLLFifoLine`, `src/include/device.h:75`). The GPU writes each line with a
single 16-byte atomic store, so the flag and the data arrive together, and a
receiver spinning on the flags can consume the data the moment it sees them. No
fence, no tail pointer, no waiting for a whole slot:

```
one LL line, 16 bytes on the wire:
+--------+--------+--------+--------+
| data   | flag   | data   | flag   |    8 payload bytes per 16 wire bytes
+--------+--------+--------+--------+
```

The price is brutal and paid knowingly: half the wire bytes are flags, so LL tops
out at 50 percent of link bandwidth. For a 4 KB all-reduce, nobody cares; latency
is everything.

**LL128** is the same flag trick with better arithmetic, for links that can
deliver a 128-byte write atomically (NVLink). The unit becomes a 128-byte line:
15 words of data, 1 word of flag, so 120 of 128 bytes are payload, 93.75 percent
of bandwidth at latency close to LL (`src/device/prims_ll128.h`). On NVLink paths
LL128 is such a good default that it covers a huge range of sizes, which is why
the model bothers pricing all three (`src/graph/tuning.cc:328`).

| | Simple | LL | LL128 |
|---|---|---|---|
| readiness signal | fence + tail counter | flag inside each 16 B line | flag inside each 128 B line |
| wire efficiency | ~100% | 50% | 93.75% |
| relative latency | high | lowest | low |
| typical home | large messages | tiny messages | NVLink, small to medium |

So "which all-reduce am I running" is really a pair like Ring+LL128 or
Tree+Simple, and both coordinates come out of the same cost table. Six algorithms
times three protocols, minus invalid combinations, priced per call.

## When the switch does the math

Here's the part that retired my mental model. Everything above assumes GPUs do the
reducing and links do the moving. On Hopper and newer machines with NVSwitch, the
switch itself can reduce, and NCCL's fastest single-node algorithm is built on
that. NVIDIA calls it NVLink SHARP; in the code it's `NCCL_ALGO_NVLS`.

The mechanism sits on CUDA multicast memory. At init, NCCL creates a multicast
object (`cuMulticastCreate`, `src/transport/nvls.cc`) that every local GPU binds
its buffer into, giving each GPU a second address for the same logical memory.
Loads and stores through that address are special:

{% raw %}
```c
// src/device/reduce_kernel.h: the load returns the SUM across all GPUs
multimem.ld_reduce.relaxed.sys.global.add.f32  %0, [%1];

// src/device/op128.h: the store lands on EVERY GPU
multimem.st.global.v4.f32  [%0], {%1,%2,%3,%4};
```
{% endraw %}

Read the whole all-reduce kernel loop for the registered-buffer case and it's
almost nothing: each GPU walks its slice issuing `multimem.ld_reduce`, which asks
the switch to fetch that address from all peers and add the values in transit,
then `multimem.st`, which asks the switch to replicate the sum back to everyone
(`src/device/all_reduce.h:441`). One read, one write, per byte. No ring position,
no steps, no per-peer anything. The reduction happens in the switch fabric.

<div style="text-align:center">
<svg viewBox="0 0 680 240" width="100%" style="max-width:680px;height:auto" role="img" aria-label="NVLS all-reduce: GPUs issue multimem loads that the NVSwitch reduces, and multimem stores that it replicates">
<rect x="14" y="6" width="652" height="216" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<rect x="90" y="26" width="500" height="40" rx="6" fill="#fff" stroke="#999" stroke-width="1.4"/>
<text x="340" y="46" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">NVSwitch</text>
<text x="340" y="59" text-anchor="middle" font-size="10" fill="#888">adds in transit, replicates in transit</text>
<g>
<rect x="80" y="160" width="80" height="34" rx="6" fill="#fff" stroke="#999"/><text x="120" y="181" text-anchor="middle" font-size="12" fill="#333">GPU 0</text>
<rect x="220" y="160" width="80" height="34" rx="6" fill="#fff" stroke="#999"/><text x="260" y="181" text-anchor="middle" font-size="12" fill="#333">GPU 1</text>
<rect x="360" y="160" width="80" height="34" rx="6" fill="#fff" stroke="#999"/><text x="400" y="181" text-anchor="middle" font-size="12" fill="#333">GPU 2</text>
<rect x="500" y="160" width="80" height="34" rx="6" fill="#fff" stroke="#999"/><text x="540" y="181" text-anchor="middle" font-size="12" fill="#333">GPU 3</text>
</g>
<g stroke="#b06f2a" stroke-width="1.8">
<line x1="112" y1="158" x2="112" y2="72"/><line x1="252" y1="158" x2="252" y2="72"/>
<line x1="392" y1="158" x2="392" y2="72"/><line x1="532" y1="158" x2="532" y2="72"/>
</g>
<polygon points="108,74 112,66 116,74" fill="#b06f2a"/><polygon points="248,74 252,66 256,74" fill="#b06f2a"/>
<polygon points="388,74 392,66 396,74" fill="#b06f2a"/><polygon points="528,74 532,66 536,74" fill="#b06f2a"/>
<g stroke="#4e8a4e" stroke-width="1.8">
<line x1="128" y1="70" x2="128" y2="156"/><line x1="268" y1="70" x2="268" y2="156"/>
<line x1="408" y1="70" x2="408" y2="156"/><line x1="548" y1="70" x2="548" y2="156"/>
</g>
<polygon points="124,154 128,162 132,154" fill="#4e8a4e"/><polygon points="264,154 268,162 272,154" fill="#4e8a4e"/>
<polygon points="404,154 408,162 412,154" fill="#4e8a4e"/><polygon points="544,154 548,162 552,154" fill="#4e8a4e"/>
<text x="190" y="105" text-anchor="middle" font-size="10.5" fill="#a05c1a" stroke="#fafafa" stroke-width="3" paint-order="stroke">multimem.ld_reduce</text>
<text x="190" y="118" text-anchor="middle" font-size="10.5" fill="#a05c1a" stroke="#fafafa" stroke-width="3" paint-order="stroke">one load returns the sum</text>
<text x="470" y="105" text-anchor="middle" font-size="10.5" fill="#3f7a3f" stroke="#fafafa" stroke-width="3" paint-order="stroke">multimem.st</text>
<text x="470" y="118" text-anchor="middle" font-size="10.5" fill="#3f7a3f" stroke="#fafafa" stroke-width="3" paint-order="stroke">one store lands everywhere</text>
<text x="340" y="214" text-anchor="middle" font-size="10.5" fill="#888">no GPU-to-GPU sends, no ring steps: the reduction happens in the switch fabric</text>
</svg>
</div>

In the cost tables NVLS carries a high fixed latency (25 microseconds in
`src/graph/tuning.cc`, versus 3.4 for a ring hop over NVLink) and a bandwidth
entry that gets doubled for all-reduce because the reduce-in and broadcast-out
directions pipeline through the switch simultaneously. So tiny all-reduces still
go to LL rings or trees, and big single-node ones go to the switch.

The same idea exists between nodes. InfiniBand switches with SHARP can reduce in
the network too, and NCCL reaches them through its CollNet plugin: the proxy
literally calls `iallreduce` on the network (`src/transport/coll_net.cc:815`) and
gets back fully reduced data, no inter-node ring or tree traffic at all. And the
hybrids compose exactly like you'd hope: multi-node NVLS uses the NVSwitch for
the intra-node reduction and IB SHARP or an inter-node double binary tree
(`NVLS_TREE`) for the cross-node part. The full all-reduce menu in 2.30 is Ring,
Tree, CollNetDirect, CollNetChain, NVLS, and NVLSTree
(`src/device/generate.py:87`), and every entry is just a different answer to "who
does the adds, and who moves the bytes".

## Watch it decide

Don't take the cost model's word for it; it will happily show you its choices.
Two env vars make NCCL's tuning layer chatty:

```
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING ./build/all_reduce_perf -b 256 -e 1G -f 2 -g 8
```

(`all_reduce_perf` is from [nccl-tests](https://github.com/NVIDIA/nccl-tests);
any PyTorch job with those env vars works the same.) At init, rank 0 dumps the
entire latency and bandwidth table it computed for your exact topology. Then, for
every collective, you get one line from `src/enqueue.cc:809` naming the winner:

```
AllReduce: 4096 Bytes -> Algo Tree proto LL channel{Lo..Hi}={0..1}
AllReduce: 8388608 Bytes -> Algo Ring proto LL128 channel{Lo..Hi}={0..15}
AllReduce: 268435456 Bytes -> Algo NVLS proto Simple channel{Lo..Hi}={0..15}
```

(Those three lines are the shape to expect on a single Hopper node; your sizes and
winners will differ, which is rather the point.) Sweep sizes with `-b`/`-e`/`-f`
and you can watch the argmin walk the menu: LL to LL128 to Simple, tree to ring to
switch. Then pin `NCCL_ALGO=Ring NCCL_PROTO=Simple`, rerun the sweep, and compare
the small-message latencies against the free choice. That gap is the cost model
earning its keep.

## The mental model that replaced mine

What I had before reading the source: "NCCL does ring all-reduce."

What I have now:

- All-reduce is a menu, not an algorithm: six who-does-what structures times three
  wire protocols, priced per call by `latency + bytes/bandwidth`, cheapest wins.
- Ring and tree are both built from the same five primitives (`send`,
  `recvReduceSend`, `recvReduceCopySend`, `recvCopySend`, `recv`); the
  reduce-scatter plus all-gather structure from the last post is visible as the
  two halves of the ring loop, and as the up and down teams of the tree kernel.
- The tree is a double binary tree over nodes, with every rank interior in
  exactly one tree so no send bandwidth idles, and chains inside each node.
- Latency work rides flags packed inside the data (LL, LL128); bandwidth work
  pays for fences (Simple).
- On modern fabric, the best all-reduce is often no all-reduce: one load that
  returns the sum, one store that lands everywhere, and the switch does the math.

The FSDP series will pick this thread right back up: FSDP's actual traffic is
all-gather and reduce-scatter, and those have their own menu (including PAT,
parallel aggregated trees, an algorithm that never applies to all-reduce, and
NVLS variants of their own). Plus the overlap post I already owe you, where these
channel counts and SM budgets stop being trivia and start being the thing that
eats your compute.

## References

Claims about NCCL internals are checked against the NCCL master source at commit
`5067397` (v2.30, August 2026); file and line references throughout point there.

- [NCCL source on GitHub](https://github.com/NVIDIA/nccl), specifically
  `src/device/all_reduce.h` (kernels), `src/graph/trees.cc` and
  `src/graph/rings.cc` (structure construction), `src/graph/tuning.cc` (the cost
  model), and `src/enqueue.cc` (selection and launch).
- [Massively Scale Your Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/),
  Jeaugey. The double binary tree announcement, with measurements to 24,576 GPUs.
- [Two-tree algorithms for full bandwidth broadcast, reduction and scan](https://doi.org/10.1016/j.parco.2009.09.001),
  Sanders, Speck, Träff. The construction NCCL's double tree implements.
- [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/),
  Gibiansky. The 2017 post that made ring all-reduce common knowledge in deep
  learning.
- [Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521),
  Thakur, Rabenseifner, Gropp. The classic treatment of allreduce algorithm
  selection by message size, twenty years before this cost model.
- [Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/)
  and [NVLink SHARP documentation](https://docs.nvidia.com/networking/display/sharpv300)
  for the in-network reduction hardware.
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html),
  including `NCCL_ALGO`, `NCCL_PROTO`, and the debug switches used above.
