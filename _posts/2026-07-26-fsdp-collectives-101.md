---
layout: post
title: "FSDP collectives 101: why reduce-scatter, and why not broadcast?"
date: 2026-07-26
tags: [fsdp, distributed-training, pytorch]
---

If you learned distributed training through DDP, you probably carry two instincts: after
the backward pass, all-reduce the gradients; and if only one place has the freshest
weights, broadcast them out. I carried both into FSDP and they cost me real confusion,
because both are wrong there. Not slightly wrong, wrong in a way that means the mental
model underneath is wrong. Working out why fixed my understanding of FSDP more than
anything else, so this post is that explanation: what all-gather and reduce-scatter
actually do, why reduce-scatter specifically is the right collective after backward, and
why broadcast and all-reduce are answers to questions FSDP never asks.

Before the story starts, the two operations themselves, in plain terms, no FSDP
attached. **All-gather**: every rank contributes its piece of a tensor, and afterwards
every rank holds the complete tensor. **Reduce-scatter**: every rank contributes a full
size tensor, the tensors get combined element wise (averaged, for our purposes), and
each rank keeps only its own slice of the result. One assembles pieces, the other
merges disagreeing copies and deals out the shares. That's the entire vocabulary of
this post (the formal definitions live in [NCCL's collective operations
docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)).
Everything that follows is about why FSDP uses exactly these two, at the moments it
does, and not the collectives you might reach for instead.

Here's the same vocabulary as a picture. Two GPUs, four numbers, A responsible for the
first half and B for the second. Notice the mirror: one op goes small in, big out; the
other goes big in, small out.

<div style="text-align:center">
<svg viewBox="0 0 680 290" width="100%" style="max-width:680px;height:auto" role="img" aria-label="Diagram of all-gather and reduce-scatter with two GPUs">
<rect x="14" y="6" width="316" height="276" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<rect x="354" y="6" width="316" height="276" rx="8" fill="#fafafa" stroke="#e6e6e6"/>
<text x="122" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#3f7a3f">all-gather</text>
<text x="462" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#a05c1a">reduce-scatter</text>
<text x="56" y="42" font-size="11" fill="#888">before</text>
<text x="396" y="42" font-size="11" fill="#888">before</text>
<text x="48" y="64" text-anchor="end" font-size="11.5" fill="#666">A</text>
<rect x="56" y="46" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="71" y="64" text-anchor="middle" font-size="12.5" fill="#333">1</text>
<rect x="89" y="46" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="104" y="64" text-anchor="middle" font-size="12.5" fill="#333">2</text>
<rect x="122" y="46" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<rect x="155" y="46" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<text x="48" y="96" text-anchor="end" font-size="11.5" fill="#666">B</text>
<rect x="56" y="78" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<rect x="89" y="78" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<rect x="122" y="78" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="137" y="96" text-anchor="middle" font-size="12.5" fill="#333">3</text>
<rect x="155" y="78" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="170" y="96" text-anchor="middle" font-size="12.5" fill="#333">4</text>
<line x1="122" y1="114" x2="122" y2="152" stroke="#4e8a4e" stroke-width="1.5"/>
<polygon points="117,152 127,152 122,162" fill="#4e8a4e"/>
<text x="132" y="140" font-size="12" fill="#3f7a3f" font-style="italic">all-gather</text>
<text x="56" y="166" font-size="11" fill="#888">after</text>
<text x="48" y="188" text-anchor="end" font-size="11.5" fill="#666">A</text>
<rect x="56" y="170" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="71" y="188" text-anchor="middle" font-size="12.5" fill="#333">1</text>
<rect x="89" y="170" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="104" y="188" text-anchor="middle" font-size="12.5" fill="#333">2</text>
<rect x="122" y="170" width="30" height="26" fill="#d8ecd8" stroke="#9cc49c"/><text x="137" y="188" text-anchor="middle" font-size="12.5" fill="#333">3</text>
<rect x="155" y="170" width="30" height="26" fill="#d8ecd8" stroke="#9cc49c"/><text x="170" y="188" text-anchor="middle" font-size="12.5" fill="#333">4</text>
<text x="48" y="220" text-anchor="end" font-size="11.5" fill="#666">B</text>
<rect x="56" y="202" width="30" height="26" fill="#d8ecd8" stroke="#9cc49c"/><text x="71" y="220" text-anchor="middle" font-size="12.5" fill="#333">1</text>
<rect x="89" y="202" width="30" height="26" fill="#d8ecd8" stroke="#9cc49c"/><text x="104" y="220" text-anchor="middle" font-size="12.5" fill="#333">2</text>
<rect x="122" y="202" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="137" y="220" text-anchor="middle" font-size="12.5" fill="#333">3</text>
<rect x="155" y="202" width="30" height="26" fill="#7fb97f" stroke="#4e8a4e"/><text x="170" y="220" text-anchor="middle" font-size="12.5" fill="#333">4</text>
<text x="122" y="250" text-anchor="middle" font-size="12" fill="#555">small in, big out</text>
<text x="122" y="266" text-anchor="middle" font-size="10.5" fill="#888">no math, pure assembly</text>
<text x="388" y="64" text-anchor="end" font-size="11.5" fill="#666">A</text>
<rect x="396" y="46" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="411" y="64" text-anchor="middle" font-size="12.5" fill="#333">8</text>
<rect x="429" y="46" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="444" y="64" text-anchor="middle" font-size="12.5" fill="#333">0</text>
<rect x="462" y="46" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="477" y="64" text-anchor="middle" font-size="12.5" fill="#333">4</text>
<rect x="495" y="46" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="510" y="64" text-anchor="middle" font-size="12.5" fill="#333">2</text>
<text x="388" y="96" text-anchor="end" font-size="11.5" fill="#666">B</text>
<rect x="396" y="78" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="411" y="96" text-anchor="middle" font-size="12.5" fill="#333">0</text>
<rect x="429" y="78" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="444" y="96" text-anchor="middle" font-size="12.5" fill="#333">4</text>
<rect x="462" y="78" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="477" y="96" text-anchor="middle" font-size="12.5" fill="#333">8</text>
<rect x="495" y="78" width="30" height="26" fill="#f6dfc4" stroke="#d9ae7a"/><text x="510" y="96" text-anchor="middle" font-size="12.5" fill="#333">6</text>
<line x1="462" y1="114" x2="462" y2="152" stroke="#b06f2a" stroke-width="1.5"/>
<polygon points="457,152 467,152 462,162" fill="#b06f2a"/>
<text x="472" y="134" font-size="12" fill="#a05c1a" font-style="italic">reduce-scatter</text>
<text x="472" y="150" font-size="10.5" fill="#888">avg = [4,2,6,4], in flight only</text>
<text x="396" y="166" font-size="11" fill="#888">after</text>
<text x="388" y="188" text-anchor="end" font-size="11.5" fill="#666">A</text>
<rect x="396" y="170" width="30" height="26" fill="#e6a15c" stroke="#b06f2a"/><text x="411" y="188" text-anchor="middle" font-size="12.5" fill="#333">4</text>
<rect x="429" y="170" width="30" height="26" fill="#e6a15c" stroke="#b06f2a"/><text x="444" y="188" text-anchor="middle" font-size="12.5" fill="#333">2</text>
<rect x="462" y="170" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<rect x="495" y="170" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<text x="388" y="220" text-anchor="end" font-size="11.5" fill="#666">B</text>
<rect x="396" y="202" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<rect x="429" y="202" width="30" height="26" fill="#fff" stroke="#c4c4c4" stroke-dasharray="3,3"/>
<rect x="462" y="202" width="30" height="26" fill="#e6a15c" stroke="#b06f2a"/><text x="477" y="220" text-anchor="middle" font-size="12.5" fill="#333">6</text>
<rect x="495" y="202" width="30" height="26" fill="#e6a15c" stroke="#b06f2a"/><text x="510" y="220" text-anchor="middle" font-size="12.5" fill="#333">4</text>
<text x="462" y="250" text-anchor="middle" font-size="12" fill="#555">big in, small out</text>
<text x="462" y="266" text-anchor="middle" font-size="10.5" fill="#888">merges disagreeing copies, deals out slices</text>
</svg>
</div>

Solid cells are what a rank contributed, pale cells are what arrived over the wire, and
dashed cells hold nothing. The same numbers show up again below, when these two ops go
to work inside FSDP.

One scope note as well: everything below describes plain one dimensional full sharding,
FSDP2's default. Hybrid sharding adds a replica dimension on top, and with it extra
communication (including, yes, an all-reduce). That's a different post.

## Two different worlds

In DDP, every GPU permanently stores the entire model. 8 GPUs means 8 full copies that
have to stay bit identical forever. All of DDP's communication exists to keep those
copies in sync.

In FSDP, the model exists exactly once, chopped into W pieces. GPU k permanently owns
piece k of every weight tensor, and of its gradient and optimizer state too. No full
copy of anything exists anywhere at rest. Full size tensors only show up as short lived
photocopies during compute, and then they get shredded. (If you're already asking "why
doesn't gathering full tensors blow up memory?", good question, held for one section.)

Once this picture is in your head, every "which collective goes here?" question answers
itself. You just ask: in this world, who is allowed to permanently hold what? Both of
the DDP instincts above are symptoms of the same bug: imagining full copies that need to
be kept in sync, in a world that deliberately has none.

## All-gather: everyone shows their piece

Each of W ranks contributes its piece, and afterwards everyone holds the concatenation
of all the pieces.

Say we have two GPUs and one 4 element weight. A owns `[w1,w2] = [1, 2]` and B owns
`[w3,w4] = [3, 4]`:

```
before:   A: [1, 2, ., .]        B: [., ., 3, 4]
all-gather ------------------------------------
after:    A: [1, 2, 3, 4]        B: [1, 2, 3, 4]
```

FSDP needs this because a matmul touches every entry of the weight. So right before a
layer runs, its custodians pool their slices into a temporary full copy. Compute,
shred, move on to the next layer. The whole forward pass is just gather, use, shred,
repeated per layer.

Notice the direction: each rank starts with 1/W of the data and ends with all of it.
Small in, big out. And there's no arithmetic anywhere, it's pure assembly.

I'll keep saying "layer" because it reads better, but strictly the unit is the FSDP
communication group: whatever you wrapped in one
[`fully_shard()`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
call. Wrap per transformer block, the common setup, and "group" and "layer" mean the
same thing.

This is also where the OOM question from earlier gets its answer. The photocopies don't
blow up memory because they never all exist at once: only the layer currently computing
is unsharded (plus the next one, fetched early to hide latency). For a 5B model split
into 24 blocks, that's roughly 0.4 GB of full bf16 weights alive at any moment, against
about 10 GB if every block stayed gathered. So the memory spike scales with your
largest block, not with the model. And "short lived" is literal: a block's photocopy
exists for the few milliseconds its compute takes, then the buffer is recycled for the
next block. The fine print is that this guarantee comes from how you wrap. Call
`fully_shard()` only on the root and there's one group, the whole model becomes one
giant photocopy, and that can absolutely OOM.

## Why backward needs a different collective

Here's the question that unlocked this for me. FSDP really has two communication jobs,
attached to two different things. Parameters get all-gathered whenever compute needs
them in full: before a layer's forward, and, because the photocopy gets shredded right
after forward, usually again before that layer's backward. Gradients get
reduce-scattered once backward has produced them. So the real split isn't "forward vs
backward", it's parameters vs gradients: why do parameters gather while gradients
reduce? Where does the "reduce" suddenly come from?

Look at what the ranks are holding in each case.

When parameters move, the shards are complementary pieces of one true weight. A's
`[1, 2]` and B's `[3, 4]` don't disagree about anything; they're different chapters of
the same book. Assembling them takes concatenation and nothing else. No arithmetic, so
no reduce. All-gather.

When gradients move, the situation is completely different. Each GPU ran the same
weights on different data, so each holds a full size gradient and the copies disagree:

```
A's grad:  [8, 0, 4, 2]
B's grad:  [0, 4, 8, 6]
average:   [4, 2, 6, 4]   <- computed in flight, never assembled on any GPU
```

Disagreeing copies can't be concatenated, they have to be combined. That combining step
is the "reduce". So the rule that generalizes: **reduce shows up exactly when the
per-rank copies disagree and must be merged.** Parameters never disagree, there's one
true weight living in pieces. Gradients disagree in general, because each rank saw
different data. That's the whole reason the two use different collectives.

And the "scatter" half? After averaging, each rank only needs its own slice. A will only
ever update w1 and w2, so shipping it the averaged gradient for w3 and w4 would be
spending network bandwidth on numbers it throws away. Reduce-scatter does both at once:
averages everyone's full gradients and delivers each custodian just its slice. A gets
`[4, 2]`, B gets `[6, 4]`, and the full averaged gradient never exists on any single
GPU.

Direction-wise this is the exact mirror of all-gather: big in, small out. One detail
worth knowing: the reduction you want is an average, not a sum. For bf16 and fp32,
NCCL's `AVG` op folds the divide by W into the collective itself, no separate division
kernel. Other dtypes take slightly different routes (fp16 splits the divisor across pre
and post scaling to avoid overflow), but every route ends the same place: each rank
holds its shard of the averaged gradient. All of this is visible in [PyTorch's FSDP2
collectives
source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py)
if you want to see the machinery.

A terminology note, since "scatter" and "sharding" sound interchangeable: sharding is a
state, scatter is an action. Think of a card game. Dealing a card to each player is a
scatter. Each player holding their own hand is being sharded. FSDP's weights *are*
sharded (the standing layout); reduce-scatter is the verb that re-establishes that
layout for gradients, with an average folded in.

## The identity that ties it together

There's a neat identity here: all-reduce = reduce-scatter + all-gather. "Everyone ends
up with the full averaged tensor" breaks into "everyone gets their averaged slice"
followed by "everyone shows their slice".

DDP needs the full all-reduce on gradients, since every GPU stores full weights and
therefore needs the full averaged gradient. FSDP runs only the first half after
backward. Each rank updates only its slice, so reduce-scatter is enough. The second
half isn't skipped though. It moves to the next forward pass, where an all-gather was
needed anyway to build the photocopy.

So FSDP is DDP's all-reduce sawed in half, with each half moved to where the data is
actually needed. Nothing new gets invented. The pieces just run at different times.

One precision for the careful reader: the second half doesn't carry the same tensor.
The optimizer steps in between, so the later all-gather moves updated parameter shards,
not the gradient shards that came out of reduce-scatter. The saw cuts the communication
pattern in half, not one particular tensor.

## Why not broadcast?

Broadcast means "one rank has the truth, copy it to everyone". The DDP instinct says:
the optimizer just updated the weights, other ranks need them, broadcast. But broadcast
exists to fix stale copies, so count the copies. After the optimizer updates w3, how
many permanent copies of w3 exist? Exactly one, on its custodian, freshly updated.
There is nothing to be stale. The other ranks don't hold an outdated w3. They hold
nothing at all, because their photocopy was shredded after backward. They'll get the
fresh w3 automatically at the next forward's all-gather, straight from the one rank
that owns it.

Broadcast is DDP thinking. It assumes replicas that can drift apart. When there's one
original per weight, staleness isn't a thing you have to prevent. It just can't happen.

## Why not the other collectives?

Each collective answers a specific question. "Which one goes here?" really means "which
question is FSDP asking right now?"

| Collective | The question it answers | Does FSDP ask it? |
|---|---|---|
| broadcast | one rank knows, everyone needs a copy | no, there are no replicas to sync |
| scatter | one rank holds everything, deal out the pieces | no, pieces never start centralized |
| gather | collect all pieces onto one rank | no, every rank needs the full layer, not just one |
| all-gather | everyone has a piece, everyone needs the whole | yes, before every layer's compute |
| all-reduce | everyone has a full version, everyone needs the full average | no, that's DDP |
| reduce-scatter | everyone has a full version, each rank needs its slice of the average | yes, after gradients |

Look at the "no" rows. They all either assume a central rank (scatter, gather,
broadcast) or assume full replicas (broadcast, all-reduce). FSDP's world has neither.
Every rank is a custodian of equal standing, and there's exactly one original of
everything.

## Run it yourself

Don't take my word for the numbers. This reproduces every value in this post with the
[torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html) API directly,
no FSDP involved:

```python
import torch
import torch.distributed as dist

use_cuda = torch.cuda.is_available()
dist.init_process_group("nccl" if use_cuda else "gloo")
rank = dist.get_rank()
if use_cuda:
    torch.cuda.set_device(rank)
dev = torch.device("cuda", rank) if use_cuda else torch.device("cpu")

# all-gather: A contributes [1,2], B contributes [3,4]
shard = torch.tensor([1.0, 2.0] if rank == 0 else [3.0, 4.0], device=dev)
full = torch.empty(4, device=dev)
dist.all_gather_into_tensor(full, shard)
print(f"rank {rank} after all-gather:     {full.tolist()}")

# reduce-scatter: A contributes [8,0,4,2], B contributes [0,4,8,6]
grad = torch.tensor([8.0, 0.0, 4.0, 2.0] if rank == 0 else [0.0, 4.0, 8.0, 6.0], device=dev)
mine = torch.empty(2, device=dev)
if use_cuda:
    dist.reduce_scatter_tensor(mine, grad, op=dist.ReduceOp.AVG)
else:
    dist.reduce_scatter_tensor(mine, grad, op=dist.ReduceOp.SUM)
    mine /= dist.get_world_size()
print(f"rank {rank} after reduce-scatter: {mine.tolist()}")

dist.destroy_process_group()
```

Save it as `collectives_demo.py` and run:

```
torchrun --nproc_per_node=2 collectives_demo.py
```

Output:

```
rank 0 after all-gather:     [1.0, 2.0, 3.0, 4.0]
rank 1 after all-gather:     [1.0, 2.0, 3.0, 4.0]
rank 0 after reduce-scatter: [4.0, 2.0]
rank 1 after reduce-scatter: [6.0, 4.0]
```

It runs on two GPUs over NCCL, or on plain CPU over gloo, and the fallback branch is a
small lesson in itself: `ReduceOp.AVG` is NCCL only, so on CPU you sum and divide
yourself. Verified on PyTorch 2.11.0.

## The check that fixed my mental model

When I'm not sure which collective belongs somewhere, I stop and count who permanently
holds what. If my answer requires a full copy of anything sitting on a GPU at rest,
I've slipped back into DDP world. In FSDP world:

- permanent state is shards only: weights, grads, optimizer moments, all 1/W
- full tensors are photocopies that live for one layer's compute
- there's one original of every number, so "keeping copies in sync" isn't a concept

And if the copies flowing through a collective disagree with each other, expect a
reduce in its name. If they're complementary pieces of one thing, expect a gather.

Everything else in FSDP, the CUDA streams, prefetching, `reshard_after_forward`,
overlap, is engineering on top of one follow up question: the gathers cost time, can we
hide them behind compute? That's the next post, with real profiler traces.

## References

The mechanism claims in this post are checked against the PyTorch 2.11.0 source, and
the memory numbers come from my own 8 GPU H100 runs (scripts will ship with the next
post).

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054),
  Rajbhandari et al. The paper that introduced sharding params, grads, and optimizer
  state across data parallel ranks. FSDP is PyTorch's native take on this idea.
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277),
  Zhao et al. The FSDP design paper.
- [`fully_shard` documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html),
  the FSDP2 API this post describes.
- [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html),
  the formal definitions of all-gather, reduce-scatter, and friends.
- [`_fsdp_collectives.py` at v2.11.0](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py),
  where the all-gather and reduce-scatter described here are actually implemented.
