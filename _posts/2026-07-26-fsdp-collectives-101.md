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

One scope note before we start: everything below describes plain one dimensional full
sharding, FSDP2's default. Hybrid sharding adds a replica dimension on top, and with it
extra communication (including, yes, an all-reduce). That's a different post.

## Two different worlds

In DDP, every GPU permanently stores the entire model. 8 GPUs means 8 full copies that
have to stay bit identical forever. All of DDP's communication exists to keep those
copies in sync.

In FSDP, the model exists exactly once, chopped into W pieces. GPU k permanently owns
piece k of every weight tensor, and of its gradient and optimizer state too. No full
copy of anything exists anywhere at rest. Full size tensors only show up as short lived
photocopies during compute, and then they get shredded.

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
communication group: whatever you wrapped in one `fully_shard()` call. Wrap per
transformer block, the common setup, and "group" and "layer" mean the same thing.

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
holds its shard of the averaged gradient.

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
