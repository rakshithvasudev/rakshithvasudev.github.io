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
`ncclAllReduce`. So I cloned NCCL (version 2.30, current master)
and read the implementation (with the help of my preferred agent of the day :D),
and the picture underneath is much better than the one I was carrying. NCCL does not have one all-reduce algorithm. It has six, and three wire
protocols to carry them. Each time you call it, it estimates the cost of each
valid pairing for your message and hardware, then selects the lowest-cost
candidate. The ring you learned from
the classic blog posts is just one row of that menu, and on the 8x H100 machine I
measured, it stops being the pick once messages get large: the estimate starts
favoring an algorithm in which no GPU addresses any other GPU, because the switch
hardware does the arithmetic.

Said as three plain claims, since the whole post is really me testing them. One:
there is no best all-reduce, only a best all-reduce for this message on this
machine. Two: NCCL behaves like a database query planner.
The hardware decides which algorithm and protocol pairings are legal, and a cost
model estimates each pairing's runtime and picks a winner, per call. Three: what
the winner mostly trades is fixed per-operation latency against sustained data
movement, right up until hardware appears that shifts the trade itself by doing
the reduction inside the switch.

This post is a guided tour of that machinery, with file and line references into the
source so you can check everything I claim. Everything below is from NCCL 2.30
(master as of August 2026); constants do drift between releases.

If you only take three lines from this post:

1. Every all-reduce pays two costs: latency, the fixed overhead of each
   communication step, and bandwidth, the time spent moving bytes. Among
   conventional point-to-point all-reduce algorithms, a ring minimizes the
   total data each GPU sends, so it usually delivers the best sustained
   bandwidth for large tensors. But it needs `2(n-1)` sequential
   steps, so its latency grows linearly with the number of GPUs. A tree needs
   only a logarithmic number of steps, so it wins for small tensors, where
   latency dominates. In practice, trees sustain less bandwidth than rings, so
   large tensors usually go back to the ring.
2. There is no threshold constant that picks between them. NCCL models every
   algorithm and protocol pair as `time = latency + bytes/bandwidth` and takes the
   argmin, per call, at enqueue time.
3. On NVSwitch systems the winner is often neither: the switch reduces the data
   itself (NVLink SHARP), and across nodes the InfiniBand switches can too.

## What all-reduce promises

First, the contract, for anyone landing here without the last post. All-reduce
takes one same-shaped tensor per rank, combines them element wise, and leaves
every rank holding the identical combined result. The numbers from last time work
just as well here:

```
before:  A: [8, 0, 4, 2]      B: [0, 4, 8, 6]
all-reduce (sum) ------------------------------
after:   A: [8, 4, 12, 8]     B: [8, 4, 12, 8]
```

(NCCL reduces with sum, prod, min, max, or avg. The average is a sum with the
division by n folded into the collective itself: for floating point types each
rank's contribution is pre-scaled by 1/n as it's read, and for integer types
the finished sum is divided once at the last step; both variants live in
`hostToDevRedOp`, [`src/enqueue.cc:2526`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/enqueue.cc#L2526).) One fact from the last post carries most of
this one: all-reduce = reduce-scatter + all-gather. First
every rank ends up owning the finished sum of one slice, then the finished
slices circulate until everyone has all of them. Keep that split in mind: the
ring is exactly those two steps, run over real wires.

## The ring, exactly as the kernel runs it

Start with the algorithm the last post promised. Every rank splits the buffer into
`n` chunks, one per rank in the ring. The reduce-scatter half takes `n-1` steps: each
step, every rank receives a chunk from one neighbor, adds its own contribution to
it, and passes the running total to the other neighbor. After `n-1` hops a chunk
has collected every rank's contribution, and the rank where it lands holds the
full sum. The all-gather half is another `n-1` steps
of the same motion, except now the finished chunks circulate unchanged. Total:
`2(n-1)` steps, and every link carries a different chunk on every step, so nothing
idles.

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
<text x="512" y="262" text-anchor="middle" font-size="10.5" fill="#888">2(n-1) = 6 steps total for n = 4</text>
</svg>
</div>

The orange and green are deliberate: they're the same colors the last post used for
reduce-scatter and all-gather, because the ring all-reduce literally is those two
collectives fused.

The one-chunk view shows the journey; it hides the schedule. To see where `2(n-1)`
actually comes from, track who holds which running total after every step. Three GPUs keep it
readable: call GPU i's contributions to the three chunks `ai`, `bi`, `ci`, and
watch four steps do the whole job:

<div style="text-align:center">
<svg viewBox="0 0 680 302" width="100%" style="max-width:680px;height:auto" role="img" aria-label="Full state evolution of a 3-GPU ring all-reduce over 4 steps: two reduce-scatter steps complete each chunk's sum, two all-gather steps distribute them">
<rect x="14" y="6" width="652" height="290" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<text x="171" y="32" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#555">GPU 0</text>
<text x="367" y="32" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#555">GPU 1</text>
<text x="563" y="32" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#555">GPU 2</text>
<!-- t0 -->
<text x="76" y="59" text-anchor="end" font-size="11" fill="#666">start</text>
<g font-size="11">
<rect x="84" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="112" y="59" text-anchor="middle" fill="#333">a0</text>
<rect x="143" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="171" y="59" text-anchor="middle" fill="#333">b0</text>
<rect x="202" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="230" y="59" text-anchor="middle" fill="#333">c0</text>
<rect x="280" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="308" y="59" text-anchor="middle" fill="#333">a1</text>
<rect x="339" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="367" y="59" text-anchor="middle" fill="#333">b1</text>
<rect x="398" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="426" y="59" text-anchor="middle" fill="#333">c1</text>
<rect x="476" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="504" y="59" text-anchor="middle" fill="#333">a2</text>
<rect x="535" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="563" y="59" text-anchor="middle" fill="#333">b2</text>
<rect x="594" y="42" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="622" y="59" text-anchor="middle" fill="#333">c2</text>
</g>
<text x="340" y="86" text-anchor="middle" font-size="10.5" font-style="italic" fill="#a05c1a">reduce-scatter: each GPU sends one chunk right, adds what arrives</text>
<!-- t1 -->
<text x="76" y="111" text-anchor="end" font-size="11" fill="#666">t1</text>
<g font-size="10.5">
<rect x="84" y="94" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="112" y="111" text-anchor="middle" fill="#333">a0</text>
<rect x="143" y="94" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="171" y="111" text-anchor="middle" fill="#333">b0+b2</text>
<rect x="202" y="94" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="230" y="111" text-anchor="middle" fill="#333">c0</text>
<rect x="280" y="94" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="308" y="111" text-anchor="middle" fill="#333">a1</text>
<rect x="339" y="94" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="367" y="111" text-anchor="middle" fill="#333">b1</text>
<rect x="398" y="94" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="426" y="111" text-anchor="middle" fill="#333">c0+c1</text>
<rect x="476" y="94" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="504" y="111" text-anchor="middle" fill="#333">a1+a2</text>
<rect x="535" y="94" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="563" y="111" text-anchor="middle" fill="#333">b2</text>
<rect x="594" y="94" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="622" y="111" text-anchor="middle" fill="#333">c2</text>
</g>
<!-- t2 -->
<text x="76" y="141" text-anchor="end" font-size="11" fill="#666">t2</text>
<g font-size="10.5">
<rect x="84" y="124" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="112" y="141" text-anchor="middle" fill="#333">Σa</text>
<rect x="143" y="124" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="171" y="141" text-anchor="middle" fill="#333">b0+b2</text>
<rect x="202" y="124" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="230" y="141" text-anchor="middle" fill="#333">c0</text>
<rect x="280" y="124" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="308" y="141" text-anchor="middle" fill="#333">a1</text>
<rect x="339" y="124" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="367" y="141" text-anchor="middle" fill="#333">Σb</text>
<rect x="398" y="124" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="426" y="141" text-anchor="middle" fill="#333">c0+c1</text>
<rect x="476" y="124" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="504" y="141" text-anchor="middle" fill="#333">a1+a2</text>
<rect x="535" y="124" width="56" height="26" fill="#fff" stroke="#c4c4c4"/><text x="563" y="141" text-anchor="middle" fill="#333">b2</text>
<rect x="594" y="124" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="622" y="141" text-anchor="middle" fill="#333">Σc</text>
</g>
<text x="340" y="168" text-anchor="middle" font-size="10.5" font-style="italic" fill="#3f7a3f">all-gather: forward the finished chunks around the same ring</text>
<!-- t3 -->
<text x="76" y="193" text-anchor="end" font-size="11" fill="#666">t3</text>
<g font-size="10.5">
<rect x="84" y="176" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="112" y="193" text-anchor="middle" fill="#333">Σa</text>
<rect x="143" y="176" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="171" y="193" text-anchor="middle" fill="#333">b0+b2</text>
<rect x="202" y="176" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="230" y="193" text-anchor="middle" fill="#333">Σc</text>
<rect x="280" y="176" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="308" y="193" text-anchor="middle" fill="#333">Σa</text>
<rect x="339" y="176" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="367" y="193" text-anchor="middle" fill="#333">Σb</text>
<rect x="398" y="176" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="426" y="193" text-anchor="middle" fill="#333">c0+c1</text>
<rect x="476" y="176" width="56" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="504" y="193" text-anchor="middle" fill="#333">a1+a2</text>
<rect x="535" y="176" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="563" y="193" text-anchor="middle" fill="#333">Σb</text>
<rect x="594" y="176" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="622" y="193" text-anchor="middle" fill="#333">Σc</text>
</g>
<!-- t4 -->
<text x="76" y="223" text-anchor="end" font-size="11" fill="#666">t4</text>
<g font-size="10.5">
<rect x="84" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="112" y="223" text-anchor="middle" fill="#333">Σa</text>
<rect x="143" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="171" y="223" text-anchor="middle" fill="#333">Σb</text>
<rect x="202" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="230" y="223" text-anchor="middle" fill="#333">Σc</text>
<rect x="280" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="308" y="223" text-anchor="middle" fill="#333">Σa</text>
<rect x="339" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="367" y="223" text-anchor="middle" fill="#333">Σb</text>
<rect x="398" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="426" y="223" text-anchor="middle" fill="#333">Σc</text>
<rect x="476" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="504" y="223" text-anchor="middle" fill="#333">Σa</text>
<rect x="535" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="563" y="223" text-anchor="middle" fill="#333">Σb</text>
<rect x="594" y="206" width="56" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="622" y="223" text-anchor="middle" fill="#333">Σc</text>
</g>
<g font-size="10.5">
<rect x="120" y="246" width="14" height="14" fill="#fff" stroke="#c4c4c4"/><text x="141" y="257" fill="#666">one rank's piece</text>
<rect x="280" y="246" width="14" height="14" fill="#f6dfc4" stroke="#d9ae7a"/><text x="301" y="257" fill="#666">partial sum</text>
<rect x="420" y="246" width="14" height="14" fill="#7fb97f" stroke="#4e8a4e"/><text x="441" y="257" fill="#666">finished sum Σ</text>
</g>
<text x="340" y="284" text-anchor="middle" font-size="10.5" fill="#888">each chunk takes n-1 = 2 hops to finish and n-1 = 2 more to reach everyone: 2(n-1) = 4 steps</text>
</svg>
</div>

Now the number falls out of two counts. A chunk's full sum has `n` contributions
sitting on `n` different GPUs, and under the ring's discipline (only talk to your
neighbor) each hop merges exactly one more GPU into the running total, so a chunk
takes `n-1` hops to finish. The moment it finishes it exists on exactly one GPU,
and the other `n-1` GPUs still need it, so it takes `n-1` more forwards to
deliver. That's `2(n-1)`, and the ring's real trick is visible in the diagram: all
`n` chunks run through that pipeline simultaneously, one position out of phase, so
every link is busy every step and no step is wasted on data anyone already has.

To be clear about what's optimal here: the step count isn't. A tree can finish a
sum in logarithmic depth, and that's where this post goes next. The bytes
are what's optimal: every hop carries fresh, never-repeated data, so each GPU
sends `(2(n-1)/n) * S` total for an `S`-byte buffer, a hair under `2S`, which is
the proven floor for any all-reduce built out of point-to-point sends between
endpoints, however clever. Hold onto that qualifier about endpoints; hardware
shows up later in this post that changes the assumption it rests on. Latency
linear, bandwidth optimal. Keep that trade in your head; the rest of this post is
NCCL renegotiating it from every direction.

And it really is fused, not two calls. The whole thing is one loop in the device
kernel, `runRing` in [`src/device/all_reduce.h:14`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/all_reduce.h#L14). Trimmed to its skeleton:

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

Those primitive names are the vocabulary the rest of NCCL is written in.
`recvReduceSend` means "receive from my ring predecessor, add my contribution,
send the result to my successor", and it happens as one fused operation: data
streams from the receive buffer through the adds and out the send buffer without a
round trip to memory in between. The `postOp=true` on the middle step marks where
each chunk's sum completes, and any final fixup runs exactly there, like the
divide of an integer average (floating point averages need no fixup, since every
contribution was pre-scaled by 1/n on the way in).

Two details the textbook picture leaves out. First, a rank doesn't wait for a whole
chunk before forwarding. Chunks are cut into slices and pushed through an 8-slot
FIFO per peer (`NCCL_STEPS` in [`src/include/device.h:26`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/include/device.h#L26)), so step `j+1` of the
pipeline starts while step `j` is still arriving. Second, none of this runs once.
NCCL carves the buffer across many independent rings.

## Channels: the ring is plural

A "channel" is one independent copy of the whole communication pipeline: its own
ring order, its own FIFO buffers, its own slice of the input, driven by its own
CUDA thread block on its own SM. Channels exist because one thread block doing
loads and stores cannot come close to saturating a fabric that moves hundreds of
gigabytes per second, so NCCL runs up to 64 of these pipelines side by side
(`MAXCHANNELS`,
[`src/include/device.h`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/include/device.h)) and splits every collective across them. The channel
count is literally the kernel's launch geometry: `grid.x` is the number of
channels ([`src/enqueue.cc:1758`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/enqueue.cc#L1758)). The ring orderings themselves come out of a topology search
([`src/graph/search.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/search.cc)) that walks the PCIe/NVLink/NIC graph at init time looking
for orderings that maximize per-channel bandwidth, which is why the ring order
rarely matches rank order.

This matters for reading the rest of the post: channels are how one collective can
run over two structures at once. When the double binary tree later splits every
buffer across two complementary trees, the split is by channel: half the channels
climb one tree, the other half the other, concurrently on different SMs. The
tuning model's derating of tree bandwidth is a separate and purely empirical
story, which we'll get to.

## Why a 10 GB all-reduce doesn't OOM

The last post spent a section on why FSDP's photocopies don't blow up memory. The
same worry transfers here, sharpened. Eight ranks all-reduce a 10 GB gradient, so
over the course of the collective each GPU receives many gigabytes of other GPUs'
partial sums. Where does all of that land? If your instinct says "some staging
buffer proportional to the message", all-reduce should be scary. It isn't, and the
short answer is that the gigabytes never live anywhere as a whole: data streams
through a handful of fixed-size reusable transfer slots, and the tensor's size
only determines how long the stream runs, never how wide the staging window gets.
The precise version is worth having, because it's the same streaming discipline
every algorithm in this post shares.

First, nothing proportional to the message is ever allocated, because arriving
data is consumed the moment it lands. Look at the ring loop again: the workhorse
step is `recvReduceSend`. A slice arrives in a FIFO slot, gets added to the local
values in registers on its way through the SM, and the result leaves out the send
side. The partial sum is never stored anywhere except in flight; the only
long-lived bytes are the finished chunks, and those land in your own output
tensor, which you already allocated. (For the usual PyTorch gradient all-reduce,
`sendbuff == recvbuff`: the whole operation is in place, and NCCL supports that
explicitly.)

Second, the staging that does exist is fixed size and allocated exactly once. The
FIFO between two ring neighbors is the per-connection buffer from earlier: 4 MiB
for Simple, 512 KiB for LL, 4.6875 MiB for LL128 ([`src/init.cc:810`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/init.cc#L810)), each carved
into `NCCL_STEPS = 8` slots. These are allocated when the communicator is created
(that memory bump you see at `init_process_group` time is exactly this, plus
peers and channels), and then reused for every collective for the life of the
communicator. A 4 KB all-reduce and a 10 GB all-reduce flow through the same
slots.

Third, backpressure. The sender is allowed to run at most 8 slots ahead of the
receiver: `waitPeer` ([`src/device/prims_simple.h:100`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/prims_simple.h#L100)) spins until the receiver's
head counter says a slot has been drained before writing another. So the bytes in
flight per connection are capped at the buffer size no matter how mismatched the
two GPUs' progress is. The tensor streams through a fixed window, like a river
through a lock:

<div style="text-align:center">
<svg viewBox="0 0 680 240" width="100%" style="max-width:680px;height:auto" role="img" aria-label="A large tensor streaming through a fixed 8-slot FIFO between two GPUs, with head and tail pointers providing backpressure">
<rect x="14" y="6" width="652" height="222" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<text x="85" y="32" text-anchor="middle" font-size="11.5" fill="#555">your tensor</text>
<g>
<rect x="50" y="40" width="70" height="17" fill="#eee" stroke="#ccc"/>
<rect x="50" y="57" width="70" height="17" fill="#eee" stroke="#ccc"/>
<rect x="50" y="74" width="70" height="17" fill="#eee" stroke="#ccc"/>
<rect x="50" y="91" width="70" height="17" fill="#f6dfc4" stroke="#d9ae7a"/>
<rect x="50" y="108" width="70" height="17" fill="#fff" stroke="#ccc"/>
<rect x="50" y="125" width="70" height="17" fill="#fff" stroke="#ccc"/>
<rect x="50" y="142" width="70" height="17" fill="#fff" stroke="#ccc"/>
<rect x="50" y="159" width="70" height="17" fill="#fff" stroke="#ccc"/>
</g>
<text x="85" y="192" text-anchor="middle" font-size="10.5" fill="#888">any size at all</text>
<line x1="126" y1="100" x2="200" y2="110" stroke="#b06f2a" stroke-width="1.6"/><polygon points="200,106 208,111 199,114" fill="#b06f2a"/>
<text x="368" y="78" text-anchor="middle" font-size="11" fill="#555">8 fixed slots of 512 KiB, allocated once at init</text>
<g>
<rect x="210" y="92" width="36" height="30" fill="#fff" stroke="#c4c4c4"/>
<rect x="250" y="92" width="36" height="30" fill="#f6dfc4" stroke="#d9ae7a"/>
<rect x="290" y="92" width="36" height="30" fill="#f6dfc4" stroke="#d9ae7a"/>
<rect x="330" y="92" width="36" height="30" fill="#f6dfc4" stroke="#d9ae7a"/>
<rect x="370" y="92" width="36" height="30" fill="#fff" stroke="#c4c4c4"/>
<rect x="410" y="92" width="36" height="30" fill="#fff" stroke="#c4c4c4"/>
<rect x="450" y="92" width="36" height="30" fill="#fff" stroke="#c4c4c4"/>
<rect x="490" y="92" width="36" height="30" fill="#fff" stroke="#c4c4c4"/>
</g>
<polygon points="264,132 272,132 268,125" fill="#4e8a4e"/>
<text x="268" y="147" text-anchor="middle" font-size="10" fill="#3f7a3f">head: receiver drains</text>
<polygon points="344,132 352,132 348,125" fill="#b06f2a"/>
<text x="358" y="161" text-anchor="middle" font-size="10" fill="#a05c1a">tail: sender fills, blocks when 8 ahead</text>
<line x1="530" y1="110" x2="566" y2="102" stroke="#4e8a4e" stroke-width="1.6"/><polygon points="565,98 574,101 566,106" fill="#4e8a4e"/>
<text x="615" y="32" text-anchor="middle" font-size="11.5" fill="#555">peer's tensor</text>
<g>
<rect x="580" y="40" width="70" height="17" fill="#7fb97f" stroke="#4e8a4e"/>
<rect x="580" y="57" width="70" height="17" fill="#7fb97f" stroke="#4e8a4e"/>
<rect x="580" y="74" width="70" height="17" fill="#7fb97f" stroke="#4e8a4e"/>
<rect x="580" y="91" width="70" height="17" fill="#fff" stroke="#ccc"/>
<rect x="580" y="108" width="70" height="17" fill="#fff" stroke="#ccc"/>
<rect x="580" y="125" width="70" height="17" fill="#fff" stroke="#ccc"/>
<rect x="580" y="142" width="70" height="17" fill="#fff" stroke="#ccc"/>
<rect x="580" y="159" width="70" height="17" fill="#fff" stroke="#ccc"/>
</g>
<text x="615" y="192" text-anchor="middle" font-size="10.5" fill="#888">reduced in place</text>
<text x="340" y="214" text-anchor="middle" font-size="10.5" fill="#888">in-flight staging per connection stays constant no matter how big the tensor is</text>
</svg>
</div>

Add it up and the total staging per rank is channels times connections times
roughly 9 MiB (the three protocol buffers together, carved out per connection in
[`src/transport/p2p.cc:488`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/transport/p2p.cc#L488)). Order of 100 to 300 MiB for a typical communicator,
fixed at init, flat forever after. That's why NCCL's memory footprint shows up
when you create the communicator and then never moves during training, and why
"how big is the tensor" never appears in the memory story at all. The
registered-buffer paths later in this post (NVLS user-buffer registration and the
network's direct modes) push this to its logical end: even the fixed staging copy
disappears, and the hardware reads your tensors where they sit.

## Where the ring hurts

If the ring moves the fewest bytes any conventional point-to-point all-reduce
can, why would NCCL ever run anything else? Because bytes are only half of the
bill.

Count the steps again: `2(n-1)`, and they're sequential. Each chunk's sum isn't done
until it has physically visited every rank. On 8 GPUs that's 14 hops. On 1024 GPUs
it's 2046 hops, and that cost is paid even by a 4-byte all-reduce, because hops are
hops regardless of size. Bandwidth optimal, latency linear. For big gradient buckets
the pipeline hides it; for the small, frequent all-reduces that show up everywhere
in real systems (loss scalars, norms, router statistics, anything at high world
size) the fixed per-hop cost, the alpha term, dominates everything else.

The fix is old: reduce up a tree, broadcast back down. Latency becomes logarithmic
in the number of nodes. The problem that kept trees out of NCCL for years is
bandwidth: in a binary tree, roughly half the ranks are leaves. A leaf sends once per chunk,
its own contribution going up. An interior rank sends three times: the merged
stream up, plus the broadcast copy out to each of its two children on the way
back down. The collective runs at the speed of its busiest ranks, so the tree
pays that 3x while the leaves' send links sit mostly idle. That imbalance is the problem the double binary
tree solves.

## The double binary tree

First, a thirty-second recap of the data structure itself, since most of us last
drew one in a classroom. A binary tree is a set of nodes in which each node has
at most two children, and every node except one, the root, has exactly one
parent. Nodes with no children are leaves; everything between the leaves and the
root is an interior node. The property that makes trees worth the bother: a
balanced binary tree over n nodes is only about log2(n) levels deep, so a
message can climb from any node to the root in log2(n) hops. Around a ring, the
same trip can take n-1. That gap is what the tree buys: 1024
nodes is ten hops up a tree and 1023 around a ring.

Map communication onto the structure and the two motions you get are exactly the
halves of an all-reduce. Send data from the leaves toward the root, each parent
adding what its children deliver before passing the total on, and by the time it
arrives at the root you have reduced. Push the result from the root back down,
each node handing copies to its children, and you have broadcast. One difference
from the trees you may remember from algorithms class: here every node holds
data, not just the leaves. Every GPU is a node somewhere in the tree, adds its
own values to whatever flows up through it, and keeps a copy of whatever flows
down.

NCCL builds the tree in [`src/graph/trees.cc:32`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/trees.cc#L32) with a bit trick: a rank's lowest
set bit fixes its depth, and a couple of integer operations on that bit produce
its parent (clear it, set the next bit up, with a fallback at the edge of the
rank range) and its children (the same bit halved, subtracted and added). It
works for any rank count. The comment in the source draws the result better than
I can, so here it is, lifted directly (14 ranks):

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
second tree with the roles swapped: the mirror image of the first when the count
is even, or the same tree shifted by one rank when it's odd (`ncclGetDtree`,
[`src/graph/trees.cc:90`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/trees.cc#L90)). Either way, a rank that idles as a leaf in tree one
works as an interior node in tree two, give or take one boundary rank pulling
interior duty in both when the count is odd. NCCL then assigns half its
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
*nodes* (`connectTrees`, [`src/graph/connect.cc:138`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/connect.cc#L138)). Inside a node, the local GPUs
form a simple chain hanging off the node's position in the tree
([`src/graph/connect.cc:61`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/connect.cc#L61)). So a 128-node, 1024-GPU job has a 128-node double tree
with 8-GPU chains inside NVLink domains, where hops are cheap. On a single node the
"tree" degenerates to just the chain, which buys nothing over the ring; the tree's
win is a multi-node story.

**Reduce and broadcast run at the same time.** I pictured tree all-reduce as two
phases: everything reduces to the root, then everything broadcasts down. The kernel
doesn't work that way. `runTreeSplit` ([`src/device/all_reduce.h:146`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/all_reduce.h#L146)) splits each
non-root rank's thread block into two teams: one runs `recvReduceSend` up the tree
while the other simultaneously runs `recvCopySend` down it, chunk by chunk (the
root, with no up direction, keeps all its threads on one team that turns sums
around). A chunk bounces
off the root and heads back down while later chunks are still climbing. The split
is 70/30 in favor of the reduce side for the low-latency protocols, because
reducing three children's data costs more than forwarding to three children (the
comment at [`src/device/all_reduce.h:161`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/all_reduce.h#L161) says exactly this).

**There's no whole-message wait anywhere.** Same slicing and 8-slot FIFOs as the
ring, so tree latency really is proportional to depth, not depth times message
size. The memory story from earlier survives the change too. A tree
rank keeps connections to at most four neighbors per channel, up to three
children and a parent, against the ring's two, and each connection carries the
same fixed slots. A parent reduces its children's incoming slices against its
own contribution in registers as they stream through
(`recvReduceSend` with a fan-in of three,
[`src/device/all_reduce.h:200`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/all_reduce.h#L200)),
so nothing accumulates anywhere. More neighbors costs a few more megabytes of
staging, never anything proportional to the tensor.

The clean way to carry the ring-versus-tree comparison out of these two sections:
the ring is extremely efficient at steady-state data movement, but its dependency
chain grows linearly with participants; the tree gives up some practical
sustained throughput (the tuning model derates it, as you're about to see) to
make that chain logarithmic. Small and frequent leans latency, huge and rare
leans bandwidth. Those are tendencies, not rules, and that's precisely the
problem: NCCL now holds two legitimate algorithms for the same collective. How
does it decide, call by call?

## How NCCL picks: a cost model, not a threshold

Old NCCL had `NCCL_TREE_THRESHOLD`. It was removed in 2.5, and what replaced it is
nicer. At init, `ncclTopoTuneModel` ([`src/graph/tuning.cc:243`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L243)) fills two tables,
`latencies[collective][algorithm][protocol]` and
`bandwidths[collective][algorithm][protocol]`, from measured constants: base launch
overheads, per-hop latencies for NVLink vs PCIe vs network, per-architecture
bandwidth ceilings. Then every call (really every aggregated batch of calls) runs
the argmin in `topoGetAlgoInfo` ([`src/enqueue.cc:2028`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/enqueue.cc#L2028)) over all pairs, where the
cost of a pair is one line ([`src/graph/tuning.cc:653`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L653)):

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

Read the last line closely: the network term, the
expensive one, went from linear in nodes to logarithmic. At 16 nodes, ring pays 30
network-latency units, tree pays 8. At 128 nodes it's 254 versus 14. Meanwhile the
bandwidth table charges the tree for its structural overheads (a factor around 0.9,
plus per-architecture ceilings), so the model naturally produces the classic
picture: tree wins small, ring wins large, and the crossover slides upward with
node count. No threshold anywhere; it falls out of two lines crossing.

Notice that two independent dials moved in that story, and it pays to keep them
separate. Message size moves the bytes-over-bandwidth term: more bytes, more
reason to care about sustained throughput. Node count moves the latency term:
more nodes, and the gap between the ring's roughly linear network path and the
tree's logarithmic one widens. Growing the cluster makes the tree competitive
across a wider range of sizes; it does not make the tree win. A large enough
message still goes to whichever plan moves bytes fastest, which might be the
ring, or, later in this post, the switch. The model computes each crossover from
both dials; it never assumes one.

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
<text x="350" y="248" text-anchor="middle" font-size="10" fill="#999">drawn from the cost formulas, not measured; the crossover moves right as node count grows</text>
</svg>
</div>

My favorite artifact in this file is `treeCorrectionFactor`
([`src/graph/tuning.cc:623`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L623)), a hand-tuned table with 24 entries per protocol, one per power-of-two
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
physically looks like on the wire and how the receiver learns it has arrived. So a
full plan is a pair, Ring plus LL, Ring plus Simple, Tree plus LL128, and the
protocols are not three more algorithms: any algorithm can ride any protocol the
path supports, and the argmin prices the pairs. The protocol layer exists because
of a synchronization problem: how does the receiver know the data in the FIFO
slot is ready?

**Simple** is the obvious design. Write the payload, execute a memory fence, then
bump a tail counter the receiver is polling ([`src/device/prims_simple.h:164`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/prims_simple.h#L164)). Full
bandwidth, but the fence is expensive and sits on the critical path of every hop,
so it shows up as latency. NCCL even carves a warp off the workers so the fence
and pointer updates overlap with the copies ("we need an extra warp to overlap
the threadfence and the copy", [`src/device/prims_simple.h:585`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/prims_simple.h#L585)), and budgets one
extra warp for exactly this when launching Simple ring kernels
([`src/enqueue.cc:2107`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/enqueue.cc#L2107)).

**LL (low latency)** makes the fence disappear with a trick. Data travels in
16-byte lines: 4 bytes of data, 4 bytes of flag, 4 of data, 4 of flag
(`ncclLLFifoLine`, [`src/include/device.h:75`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/include/device.h#L75)). The layout is the trick: each
8-byte half of the line carries its own flag right beside its own data, so as
long as the transport delivers 8 bytes atomically (NVLink does, RDMA writes do),
a flag can never show up ahead of the data it vouches for. The comment above the
struct spells out exactly this. A receiver spinning on the flags can therefore
consume the data the moment it sees them. No fence, no tail pointer, no waiting
for a whole slot:

```
one LL line, 16 bytes on the wire:
+--------+--------+--------+--------+
| data   | flag   | data   | flag   |    8 payload bytes per 16 wire bytes
+--------+--------+--------+--------+
```

The price is brutal and paid knowingly: half the wire bytes are flags, so LL tops
out at 50 percent of link bandwidth. For a 4 KB all-reduce, nobody cares; latency
is everything.

**LL128** is the same flag trick with better arithmetic, for paths that can
guarantee a 128-byte write lands whole and in order: NVLink inside the node, and
network routes that preserve the guarantee end to end. The unit becomes a
128-byte line: 15 words of data, 1 word of flag, so 120 of 128 bytes are
payload, 93.75 percent of bandwidth at roughly half of Simple's per-hop latency
([`src/device/prims_ll128.h`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/prims_ll128.h)). On NVLink paths LL128 is such a good default that
it covers a huge range of sizes, which is why the model bothers pricing all
three ([`src/graph/tuning.cc:328`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L328)).

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

The mechanism sits on CUDA multicast memory: one virtual address range that names
a group of physical memories, one per GPU, so that a single load or store can
address all of them at once. At init, every local GPU binds
NCCL's staging buffers into a shared multicast object (`cuMulticastCreate`,
[`src/transport/nvls.cc:60`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/transport/nvls.cc#L60); registering your own tensors later binds them into a
second object of their own). Each GPU then holds two kinds of pointer: a unicast
pointer naming its own pages, and a multicast pointer naming the whole group at
once. Loads and stores through the multicast pointer are special:

{% raw %}
```c
// src/device/reduce_kernel.h: the load returns the SUM across all GPUs
multimem.ld_reduce.relaxed.sys.global.add.f32  %0, [%1];

// src/device/op128.h: the store lands on EVERY GPU
multimem.st.global.v4.f32  [%0], {%1,%2,%3,%4};
```
{% endraw %}

Read the reduce team's loop in the NVLS kernel and it's almost nothing: each GPU
walks its slice of the buffer with a `directRecvDirectSend` whose template
arguments mark both source and destination as multimem
([`src/device/all_reduce.h:447`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/all_reduce.h#L447)). That compiles down to the pair above: one
`multimem.ld_reduce`, asking the switch to fetch that address from all peers and
add the values in transit, and one `multimem.st`, asking it to replicate the sum
back to everyone. One read, one write, per byte. No ring position, no steps, no
per-peer anything. The reduction happens in the switch fabric.

That last sentence sounds implausible, so here is exactly where the work goes. The switch really does execute the adds:
starting with the third generation, the NVSwitch ASIC carries dedicated SHARP
reduction hardware, and a `multimem.ld_reduce` is a load whose
responses from all subscribed memories get combined at the switch ports before
one result returns to the GPU that asked. But the GPUs are not idle and the
wires are not free. Every GPU still runs this kernel over its `1/n` share of the
buffer, issuing every load and store; the diagram's arrows are real NVLink
traffic, one pass up and one pass down per byte on each GPU's link. That's the
actual win over the ring, where each link carries every byte roughly twice in
each direction: NVLS halves per-link traffic, which is why the cost model
credits all-reduce with doubled NVLS bandwidth (`intraBw *= 2.0f` in
[`src/graph/tuning.cc:315`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L315)). It's also why the 2S floor from the ring section
doesn't constrain this: that bound was derived for endpoints exchanging
point-to-point messages, and a switch that sums in transit isn't beating the
bound, it's playing outside the assumptions the bound was built on. Measured on my
nodes the end-to-end advantage over the best ring is about 30 percent at 4 GiB,
not 2x; the numbers are in the last section. And in the common unregistered path the scatter and
gather warp teams still stage your data into the multicast buffers with plain
copies. Reductions the switch can't express never leave the GPU at all: the
multimem path covers sums and min/max only (`ncclNvlsSupported`,
[`src/include/device.h:587`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/include/device.h#L587)), so a floating point average, which NCCL implements
as a pre-scaled sum, is routed to ring or tree even on this hardware. So
"the switch does the math" is precise about the bulk sums, and only the sums.
The choreography, the staging, and the fixups stay on the GPU; what disappears
is GPU ALUs touching the reduction and any software notion of a peer.

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

Which raises the obvious question: if the switch can do the math, why wouldn't
NCCL always use it? Because capability only puts the row on the menu; it doesn't
win the argmin. In the cost tables NVLS carries a high fixed latency (25
microseconds in
[`src/graph/tuning.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc), versus 3.4 for a ring hop over NVLink) and a bandwidth
entry that gets doubled for all-reduce because the reduce-in and broadcast-out
directions pipeline through the switch simultaneously. Add the operator
restriction from above (a floating point average can't ride it) and the fact
that the bytes still have to move, and the switch is just another candidate with
its own constants, priced against everything else per call. So tiny all-reduces
still go to LL rings or trees, and big single-node ones go to the switch.

The same idea exists between nodes. InfiniBand switches with SHARP can reduce in
the network too, and NCCL reaches them through CollNet, its generic interface for
a network that can run collectives itself rather than merely deliver bytes: the proxy
literally calls `iallreduce` on the network ([`src/transport/coll_net.cc:815`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/transport/coll_net.cc#L815)) and
gets back fully reduced data, no inter-node ring or tree traffic at all. And the
hybrids compose exactly like you'd hope: multi-node NVLS uses the NVSwitch for
the intra-node reduction and IB SHARP or an inter-node double binary tree
(`NVLS_TREE`) for the cross-node part. The full all-reduce menu in 2.30 is Ring,
Tree, CollNetDirect, CollNetChain, NVLS, and NVLSTree
([`src/device/generate.py:87`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/generate.py#L87)), and every entry is just a different answer to "who
does the adds, and who moves the bytes".

## What your hardware takes off the menu

Everything above described the full menu, and if you're on older or plainer
hardware you may reasonably ask which parts still apply to you. Almost all of
it. Hardware affects the planner in two ways: it sets the constants in
the cost tables, and it decides which plans are legal at all. Every algorithm
row and protocol column is really a bet on one specific hardware capability, and
the machinery for a missing capability is the one you've already seen: the row's
bandwidth entry reads zero, and the argmin simply never considers it. There is no "cloud mode" or "legacy mode"
anywhere in NCCL; there are only capabilities present or absent.

The bets, one per row:

| menu entry | the capability it bets on | without it |
|---|---|---|
| Ring, Tree | any link that moves bytes | always available |
| LL | nothing extra (flags ride inside the data) | always available |
| LL128 | 128-byte writes land whole and in order | row zeroed |
| NVLS, NVLS_TREE | a switch with reduction hardware inside the node | rows zeroed |
| CollNetDirect/Chain | a network whose switches reduce, plus its plugin | rows zeroed |

Now walk the generations with that table in hand. A PCIe-only server, no
NVLink, is the floor: ring and tree over PCIe with LL and Simple, and that's the
whole menu, because LL128 demands NVLink-grade write atomicity even inside the
node (the gate is `graphs->typeIntra <= PATH_NVB`, [`src/graph/tuning.cc:531`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L531)).
A V100 or A100 box with NVLink and an earlier NVSwitch gets LL128 back and full
ring/tree bandwidth, but no switch arithmetic: those switch generations forward
bytes and do no math, and the code encodes that bluntly, an efficiency table
with literal zeros for Volta and Ampere (`nvlsEfficiency`,
[`src/graph/tuning.cc:139`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L139)). Everything in this post up through the cost model
applies to these machines unchanged; the offload sections just aren't about
them. Reduction-capable switches inside the node arrived with Hopper, and only
then does the NVLS row light up.

Between nodes the same logic repeats one level out. Plain Ethernet, RoCE, or
InfiniBand without SHARP configured moves bytes and does no math, so the
CollNet rows and multi-node NVLS are zeroed ([`src/graph/tuning.cc:504`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L504)), and
inter-node all-reduce is rings and double binary trees, exactly the two
algorithms this post spent most of its length on. That is the common case in
most datacenters, not the exception. If the nodes themselves have
reduction-capable switches, NVLS_TREE survives as the hybrid: switch math
inside the node, ordinary tree traffic between nodes, no cooperation needed
from the network at all.

Cloud fabrics slot into the same table rather than getting special treatment.
AWS's EFA, to take the biggest one, has no in-network reduction and no CollNet
plugin, so it's the "moves bytes, does no math" row above. Its one extra wrinkle
is the LL128 bet, and it's a clean example of capability framing, because the
answer changed over the years without NCCL changing at all. LL128 is legal only
where the transport can promise that a 128-byte write lands whole and in order.
EFA's base transport makes no such promise, so the
[plugin](https://github.com/aws/aws-ofi-nccl) historically exported
`NCCL_PROTO=simple` to protect you, zeroing the fast-protocol rows through the
same mask as everything else; on instance generations where the plugin can make
the guarantee, it stopped doing so. And because such fabrics
typically carry a higher per-message latency than InfiniBand, which enters the
model through the NIC latency added to every inter-node hop
(`graphs->latencyInter`, [`src/graph/tuning.cc:389`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc#L389)), the ring's
`2(nNodes-1)` inter-node hops hurt more against the tree's `2·log2(nNodes)`,
and the tree stays the right answer out to larger sizes than it would on a
lower-latency fabric.

The sentence to keep from this section: the cost model has no idea what brand
anything is. Hardware zeroes some rows and sets some constants, and the same
argmin over whatever remains explains everything the log shows you, on a
PCIe box from 2018 or on whatever ships next year.

## How this survives ten thousand GPUs

Everything above was worked out on a whiteboard, three GPUs here, a node
there. Does any of it still matter when the job has thousands of GPUs, or does
scale wash the tuning machinery out? It matters more, and the pattern across
the frontier reports fits in one sentence: at extreme scale, collective
communication stops being a library implementation detail and becomes an input
to model and system architecture. NCCL running the biggest jobs in the world
is not news; that's its day job. What I actually wanted to know is which of
these mechanisms does the heavy lifting up there, and what gives out first.
Three papers answer that, and reading them after tracing the source is a
different experience than reading them before.

Do one division before anything else, because it rearranged how I read all
three. The ring cuts the buffer into n chunks. Across 16,384 ranks, a 1 GiB
gradient bucket, the kind that feels enormous, is a 64 KiB chunk per rank. To
be precise about what that does and does not mean: NCCL still prices the
operation as a 1 GiB collective; the per-rank chunk is not what goes into the
cost tables. What the division changes is the physical texture of the work:
tiny slices, where per-hop overheads and pipelining granularity count for more
and more. And the dominant scaling pressure sits in the latency column
regardless: the ring's inter-node stage count grows roughly linearly with node
count while the tree's grows logarithmically, so every added node argues a
little harder for the low-depth plans. That latency pressure helps explain why
the mechanisms that looked like small-message footnotes earlier, the trees,
the flag protocols, the switch offload, become increasingly important
ingredients at frontier scale, each one where its transport and hardware
allow, while the ring keeps the traffic that stays genuinely huge.

The first thing scale changed was NCCL itself: the ring's linear latency growth
became unacceptable, so the library grew a new algorithm. NVIDIA shipped it
with measurements
([the NCCL 2.4 announcement](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/),
latency plot in figure 3): on Summit, at up to 24,576 GPUs, small-message
all-reduce latency beat rings by up to 180x. You can sanity-check that number
with nothing but the hop counts from the cost model section: 24k GPUs is
about 4,096 nodes, a ring serializes about 8,000 network hops, a tree needs
about 24. The same announcement admits what gave out: full bandwidth held
until traffic crossed the InfiniBand fabric's top switch layer. Even the
algorithm built for scale pays the topology tax we keep running into.

The next thing to give is the tuning.
[Llama 3](https://arxiv.org/abs/2407.21783)'s 405B model trained on up to
16,000 H100s, in a 24,000-GPU cluster wired as a three-layer Ethernet Clos
fabric with RoCE. Check that against the capability ladder: switches that
move bytes and do no math, so ring and tree territory. The stock constants
stopped fitting at that size, and the fixes Meta lists for NCCLX, their NCCL
fork, are knobs you now know by name. They "tuned chunking and data transfer
to fit network latencies": that's the chunk and FIFO sizing from the memory
section. They gave small control messages priority so they don't queue behind
bulk data in deep-buffer switches: that's the latency alpha, defended at the
fabric level. I reread that paragraph after finishing this post and it had
turned from color into a checklist.

Past that point, teams stop tuning around collective costs and start
designing for them.
[Kimi K3's report](https://arxiv.org/abs/2607.24653) (2.8T parameters,
mixture-of-experts, July 2026) does it twice. Their load balancer needs a
quantile over millions of values scattered across every rank. Gathering them
at training time is a non-starter, so each rank builds a histogram and "a
single all-reduce sums the per-rank bin counts," priced in the report at
under one percent of shipping the raw values. Notice what happened there: the
computation got reshaped until its communication collapsed into one small
all-reduce, dropped deliberately into the cheap regime where the trees and
flag protocols live. Their serving stack pulls the reverse trick: the
tensor-parallel all-reduce is "decomposed into a reduce-scatter and an
all-gather," a compute kernel slots in between the halves, and a norm gets
fused into the collective. That's the identity this post opened with, pried
apart so work can hide inside it.

One caveat before you go read these reports yourself: in mixture-of-experts
training the bulkiest traffic has moved to all-to-all expert dispatch (K3's
MoonEP, the pipeline co-design in
[DeepSeek-V3's report](https://arxiv.org/abs/2412.19437), and
[GLM-5](https://arxiv.org/abs/2602.15763)'s hierarchical all-to-all that
splits the intra-node and inter-node halves, the same fabric-level split as
NCCL's chain-inside-node, tree-across-nodes construction). That's a different
collective with different math, and it deserves its own post. Gradient sync
and tensor parallelism still run on the reduce-scatter, all-gather, and
all-reduce described here.

The pattern across all four systems is not that one collective dominates, or
that one trick keeps winning. It's that at this scale, communication structure
becomes something designers co-design with the computation, instead of
something a library quietly handles underneath it.

And when you want measured curves instead of my sketches,
["Demystifying NCCL"](https://arxiv.org/abs/2507.04786) (2025, revised 2026) benchmarks all
three protocols and both algorithms across message sizes and cluster sizes
and maps where each one wins: LL and LL128 small, Simple large, trees pulling
ahead as node counts grow. It's against NCCL 2.19, but it's the same
machinery, measured by people with no stake in the cost model being right.

## Watch it decide

Don't take the cost model's word for it; it will happily show you its choices.
Before looking, write down what everything so far predicts for a single 8-GPU
Hopper node: the tree has no expensive network depth to dodge, so it should
never win here; tiny messages should ride the ring on a low-latency protocol;
and once sizes are large enough to amortize the switch's 25 microsecond entry
fee, NVLS should take the rest. Two env vars make NCCL's tuning layer chatty:

```
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING ./build/all_reduce_perf -b 256 -e 1G -f 2 -g 8
```

(`all_reduce_perf` is from [nccl-tests](https://github.com/NVIDIA/nccl-tests);
any PyTorch job with those env vars works the same.) At init, rank 0 dumps the
entire latency and bandwidth table it computed for your exact topology. Then, for
every collective, you get one line from [`src/enqueue.cc:822`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/enqueue.cc#L822) naming the winner:

```
AllReduce: 4096 Bytes -> Algo RING proto LL channel{Lo..Hi}={0..0}
AllReduce: 1048576 Bytes -> Algo RING proto LL channel{Lo..Hi}={0..23}
AllReduce: 2097152 Bytes -> Algo NVLS proto SIMPLE channel{Lo..Hi}={0..15}
```

Those lines are measured, not invented: a single 8x H100 node, NCCL 2.30.7 built
from the same commit every file reference in this post points at, swept from 256 B
to 4 GiB. The walk on this machine is simpler than the full menu: ring with LL up
to 1 MiB, then straight to the switch from 2 MiB on, everything on Simple after
that. No tree at any size, which the chain-inside-the-node section predicted. No
LL128 window either; NVLS arrives before LL stops winning. The same sweep on an
8x H200 node decides identically, which is its own small lesson: the crossovers
follow the interconnect, and these two machines share their NVSwitch generation.
Your fabric will draw its own map, which is rather the point. Against the
predictions written down above: three for three, with the missing LL128 window
as the one detail the hand-waved version didn't see coming and the argmin did.

Then pin things and rerun, and each layer's contribution becomes a number. Pin the
algorithm to ring both times and flip only the protocol: with the algorithm held
constant, whatever difference appears is purely the synchronization scheme, the
fence against the flags, nothing else. That is the price of the fence (all numbers
here are the in-place halves, the PyTorch gradient case, on the H100 node):

| size | ring, protocol free (picks LL) | ring, Simple forced |
|---|---|---|
| 64 KiB | 50 us | 76 us |
| 1 MiB | 61 us | 112 us |
| 8 MiB | 79 us, 186 GB/s | 140 us, 105 GB/s |

The flag protocol is worth 45 percent at 1 MiB, right in the awkward middle band.
At 256 B both sit on the same ~50 microsecond launch floor; single node, so the
latency drama the tree section promised needs node counts to appear. It appears
one section down.

Now let the argmin run free again, so the algorithm itself may change. This
experiment asks a different question from the last one: does moving the
reduction into the switch pay, and does it pay more as the payload grows? The
large sizes leave the ring for the switch, which is worth this much bus
bandwidth:

| size | free choice (NVLS) | best ring |
|---|---|---|
| 32 MiB | 289 GB/s | 279 GB/s |
| 128 MiB | 399 GB/s | 333 GB/s |
| 4 GiB | 475 GB/s | 366 GB/s |

A near tie where NVLS first takes over, growing to 30 percent at full size,
against the doubled bandwidth the cost table promises. The growth pattern is the
bandwidth term of the model earning its keep: as bytes grow, the switch's fixed
entry fee stops mattering and only its halved per-link traffic remains. The H200
node lands within a few percent of every number here.

And one more sweep pays off the post's opening claim in wall clock. Run the two
halves separately and compare against running them fused:

```
4 GiB, in place:   reduce-scatter  10,506 us  +  all-gather  10,462 us  =  20,968 us
                   ring all-reduce                                         20,537 us
                   NVLS all-reduce                                         15,808 us
```

The identity holds to within two percent on real wires. And NVLS beats it by a
quarter, which is the measured value of not being made of those two halves.

## The same sweep, off the node

A single node hid the tree, so I took the sweep across nodes: two and four 8x H200
nodes over EFA (NCCL reports the network as Libfabric, the aws-ofi plugin from the
capability section). Same build, same commit, same command plus MPI. Write the
forecast down first, like before: inter-node hops now carry real latency, so
the tree should finally earn its keep at the small end; the ring's steady-state
bandwidth should keep it in the running for the biggest payloads; and the
hybrid rows are legal now, free to take whatever slice their constants favor.
The argmin draws a different map:

| size band | 2 nodes | 4 nodes |
|---|---|---|
| 256 B to 128 KiB | Tree + LL | Tree + LL |
| 256 KiB to 8 MiB | Tree + LL128 | Tree + LL128 |
| 16 MiB | Ring + LL128 | Tree + LL128 |
| 32 MiB and up | NVLS_TREE + Simple | Ring + LL128, Simple from 512 MiB |

Everything the single node deleted from the menu is back. The tree owns the small
and medium sizes. The protocols climb LL to LL128 to Simple inside each algorithm's
reign. And at two nodes the bulk sizes go to NVLS_TREE, the hybrid from the switch
section: NVSwitch arithmetic inside each node, tree traffic between nodes, and no
cooperation needed from a network that does no math. At 4 GiB the hybrid moves 464
GB/s against the pinned ring's 366. Between one, two, and four nodes, the sweeps
have now surfaced every algorithm this hardware admits: ring, tree, NVLS, NVLS_TREE,
and all three protocols. The only rows never seen are the CollNet ones, which is the
capability table working as written, because EFA's switches move bytes and do no
math.

Pin tree and ring, and the crossover the sketch promised becomes numbers:

| size | tree, 2 nodes | ring, 2 nodes | tree, 4 nodes | ring, 4 nodes |
|---|---|---|---|---|
| 256 B | 44 us | 100 us | 71 us | 229 us |
| 64 KiB | 48 us | 111 us | 87 us | 237 us |
| 1 MiB | 130 us | 158 us | 120 us | 401 us |
| 16 MiB | 233 us | 226 us | 327 us | 393 us |
| 64 MiB | 651 us | 522 us | 970 us | 634 us |

Read the columns against the cost model. The ring's small-message floor is its
linear hop count made visible: roughly 100 microseconds at two nodes, 229 at four.
The tree's floor barely moves, 44 to 71, logarithmic depth doing exactly what it
was hired to do. So the tree's advantage at 256 B grows from 2.3x to 3.2x with the
node count, and the size where the ring catches up slides from 16 MiB at two nodes
to 32 MiB at four. That is the two lines of the sketch crossing on real wires, and
the crossover moving in the direction the model predicts as nodes are added.

One number from my first sweep deserves a confession. The free choice at
16 MiB on four nodes initially measured 525 microseconds while a pinned tree
ran 327, and I nearly published that as the argmin mispicking a protocol in
the awkward middle band. A controlled rerun says otherwise: same four nodes
for every configuration, five repetitions each, selections read from the
tuning log rather than assumed. (The crossover table above keeps the original
single-sweep values; the numbers here are medians over five runs, so the two
sets differ by a few percent, which is expected.) The free choice picks Tree with LL128, and
Tree with LL128 is the fastest of every algorithm this topology admits,
each one forced and measured: 340 microseconds median against 968 for Tree
with LL, 1117 for Tree with Simple, 407 for the best ring, and 357 for the
forced NVLS_TREE hybrid. My first number was one uncontrolled
sample on a shared fabric (I also suspected the debug logging enabled on that
run; reproducing that exact environment measured 334, so it wasn't that
either), and the rerun's own spread shows how easy such a sample is to
collect: one of five ring repetitions spiked to nearly double its median. The
argmin was right and my first measurement wasn't. The model still isn't an
oracle, its own correction tables say as much, but the one time I thought I'd
caught it red-handed, the thing that needed correcting was my benchmark.

And the identity survives leaving the node: at 4 GiB on four nodes,
reduce-scatter plus all-gather sum to 23.4 milliseconds against the pinned ring
all-reduce's 22.8, within three percent over EFA.

I later reran the two-node sweep on a pair of H100 nodes, and the result is the
right closing note for all of this. Same map, same sequence of regimes, but the
border posts sit one power of two off: LL hands over to LL128 at 128 KiB instead
of 256, and NVLS_TREE takes the bulk sizes from 64 MiB instead of 32 (468 GB/s
against the ring's 345 at 4 GiB). Same fabric generation, same switches, slightly
different constants, slightly different borders. Nobody moved a threshold, because
there is no threshold. Two machines computed the same argmin over their own
numbers and drew their own maps, which is the whole post in one sentence.

Since the two dials keep doing the work in this post, I gave them one final sweep
of their own: nine sizes from 16 MiB to 4 GiB, on one, two, and four H200 nodes,
free choice plus every legal algorithm forced at every size, five repetitions on
the big sizes and on every crossover cell. All three topologies use nodes from
the same four-node pool, so the columns differ in participating node count and
in nothing else I could control. The 25 MiB point doubles as a training-scale
anchor: it is PyTorch DDP's default gradient bucket (25 MiB, checked against
the 2.11 source); the rest of the range is representative bucket-and-shard
territory, and the GiB points deliberately push into the bandwidth-dominated
regime. The fastest measured plan in each cell, with its bus bandwidth:

| size | 1 node | 2 nodes | 4 nodes |
|---|---|---|---|
| 16 MiB | Ring (NVLS ties), 255 GB/s | NVLS_TREE, 146 GB/s | Tree, 98 GB/s |
| 25 MiB | NVLS (ring ties), 273 GB/s | NVLS_TREE, 178 GB/s | Ring, 119 GB/s |
| 50 MiB | NVLS, 325 GB/s | NVLS_TREE, 249 GB/s | Ring, 184 GB/s |
| 100 MiB | NVLS, 398 GB/s | NVLS_TREE, 300 GB/s | Ring, 233 GB/s |
| 256 MiB | NVLS, 441 GB/s | NVLS_TREE, 392 GB/s | Ring, 280 GB/s |
| 512 MiB | NVLS, 452 GB/s | NVLS_TREE, 408 GB/s | Ring, 291 GB/s |
| 1 GiB | NVLS, 461 GB/s | NVLS_TREE, 441 GB/s | Ring, 337 GB/s |
| 4 GiB | NVLS, 471 GB/s | NVLS_TREE, 459 GB/s | Ring, 357 GB/s |

Read the rows and the columns separately, because they are the two dials, finally
turned independently. Down any column runs the message-size dial: every topology
climbs to its own bandwidth plateau, about 470 GB/s for the switch on one node,
about 460 for the hybrid on two, about 360 for the ring on four. Across any row
runs the node-count dial, and the 16 MiB row is the whole argument in one line:
the same collective that runs fastest as a ring inside one node (with the switch
in a dead heat) hands to the hybrid at two nodes and to the tree at four. Within
one pool of machines, the only thing that changes across that row is how many
nodes participate, and with them the inter-node depth every candidate plan has
to price. The single-node column is the control: with no network depth to avoid,
the tree never wins a cell there, and it never wins a bulk cell anywhere. Its
one win sits exactly where the model says it should, the smallest payload on
the deepest topology. At the bulk end the bandwidth term takes over on every
topology, and the four-node column hands the biggest payloads to the plain
ring, which suggests the hybrid's inter-node half gives back enough of its
switch half's gains to lose at those sizes.

Two honest footnotes from the finer grid. First, with the 25 MiB points added,
the pinned tree-to-ring handover lands between 16 and 25 MiB at both two and
four nodes, so the crossover shift with node count that the coarser sweep put
at 16 versus 32 MiB is real in direction but smaller than power-of-two sampling
made it look. Second, away from the crossover boundaries the free choice
selects the same plan as the measured winner in every cell; right at the
borders it sometimes holds the neighboring plan instead, and five-run reruns
put that toll between three and twelve percent, the price a model pays for
drawing clean lines through noisy territory.

## The mental model that replaced mine

What I had before reading the source: "NCCL does ring all-reduce."

What I have now:

- `ncclAllReduce` is a request, not an algorithm. A planner prices six
  who-does-what structures times three wire protocols against your message,
  your topology, and your transport's capabilities, `latency + bytes/bandwidth`,
  cheapest wins, and launches that physical plan. Sometimes the plan is the
  ring. Sometimes it's a tree. Sometimes the switch does the math.
- The ring is built from five primitives (`send`, `recvReduceSend`,
  `recvReduceCopySend`, `recvCopySend`, `recv`), and the tree reuses the same
  set plus two more for its root; the reduce-scatter plus all-gather structure
  from the last post is visible as the two halves of the ring loop, and as the
  up and down teams of the tree kernel.
- The tree is a double binary tree over nodes, the two trees complementing
  leaf and interior roles so send bandwidth stays busy, and chains inside
  each node.
- Latency work rides flags packed inside the data (LL, LL128); bandwidth work
  pays for fences (Simple).
- On modern fabric, the fastest all-reduce is sometimes offloaded to the
  switch fabric: one load that returns the sum, one store that lands everywhere,
  and the switch does the math.

The FSDP series will pick this thread right back up: FSDP's actual traffic is
all-gather and reduce-scatter, and those have their own menu (including PAT,
parallel aggregated trees, an algorithm that never applies to all-reduce, and
NVLS variants of their own). Plus the overlap post I already owe you. The channel
counts in this one will matter there, because every SM a collective occupies is
an SM your matmuls don't get.

## References

Claims about NCCL internals are checked against the NCCL master source at commit
`5067397` (v2.30, August 2026); file and line references throughout point there.
The measured numbers are [nccl-tests](https://github.com/NVIDIA/nccl-tests) sweeps
against NCCL built from that same commit (reports as 2.30.7), on one 8x H100 node,
one 8x H200 node, and two- and four-node H200 clusters over EFA, in-place columns
throughout.

- [NCCL source on GitHub](https://github.com/NVIDIA/nccl), specifically
  [`src/device/all_reduce.h`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/device/all_reduce.h) (kernels), [`src/graph/trees.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/trees.cc) and
  [`src/graph/rings.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/rings.cc) (structure construction), [`src/graph/tuning.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/graph/tuning.cc) (the cost
  model), and [`src/enqueue.cc`](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/src/enqueue.cc) (selection and launch).
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
- [NCCL 2.17.1 release notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-17-1.html),
  where in-switch reduction landed in NCCL ("Add support for NVLink SHARP
  Reduction / Broadcast to accelerate intra-node allreduce operations"), and
  [NVIDIA SHARP documentation](https://docs.nvidia.com/networking/display/sharpv300)
  for the InfiniBand side of in-network reduction.
- [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl), the plugin NCCL uses on
  AWS EFA, whose [release notes](https://github.com/aws/aws-ofi-nccl/releases)
  track when LL and LL128 stopped being disabled on p5-class instances.
- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and
  Algorithms](https://arxiv.org/abs/2507.04786), Hu et al. Independent
  microbenchmarks of the protocols and algorithms this post describes, against
  NCCL 2.19.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), the source
  for the 16K-GPU RoCE training setup and the NCCLX collective-communication
  changes discussed above.
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), the
  source for the histogram all-reduce and the decomposed tensor-parallel
  all-reduce quoted above, and
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437) and
  [GLM-5](https://arxiv.org/abs/2602.15763) for the all-to-all-centric side of
  mixture-of-experts communication.
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html),
  including `NCCL_ALGO`, `NCCL_PROTO`, and the debug switches used above.
