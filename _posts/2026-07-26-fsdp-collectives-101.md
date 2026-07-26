---
layout: post
title: "FSDP collectives 101: why reduce-scatter, and why not broadcast?"
date: 2026-07-26
tags: [fsdp, distributed-training, pytorch]
---

I failed the same quiz question twice while learning FSDP. Both times I made the same
two mistakes: after the backward pass I said "all-reduce", and after the optimizer step
I was sure the updated weights needed a broadcast. Both answers are correct for DDP.
Both are wrong for FSDP, and figuring out why turned out to be the most useful thing
I've picked up about distributed training so far. This post is the explanation I wish
I'd read before failing that quiz.

## Two different worlds

In DDP, every GPU permanently stores the entire model. 8 GPUs means 8 full copies that
have to stay bit identical forever. All of DDP's communication exists to keep those
copies in sync.

In FSDP, the model exists exactly once, chopped into W pieces. GPU k permanently owns
piece k of every weight tensor, and of its gradient and optimizer state too. No full
copy of anything exists anywhere at rest. Full size tensors only show up as short lived
photocopies during compute, and then they get shredded.

Once this picture is in your head, every "which collective goes here?" question answers
itself. You just ask: in this world, who is allowed to permanently hold what? My two
wrong quiz answers were both symptoms of the same bug: I was mentally living in DDP
world, imagining full copies that need to be kept in sync, while writing FSDP code.

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

## Reduce-scatter: average everything, take home only your slice

Here each rank contributes a full size tensor. The tensors get element wise averaged,
and each rank receives only its own slice of the result.

After backward, A and B each hold a full size gradient. Full size because the
photocopied weights were full size and autograd doesn't know anything about sharding.
The two gradients are different because each GPU trained on different data:

```
A's grad:  [8, 0, 4, 2]
B's grad:  [0, 4, 8, 6]
average:   [4, 2, 6, 4]   <- computed in flight, never assembled on any GPU
```

Reduce-scatter delivers `[4, 2]` to A and `[6, 4]` to B. Why only a slice? Because A
will only ever update w1 and w2. Shipping it the averaged gradient for w3 and w4 would
be spending network bandwidth on numbers it would immediately throw away.

This is the exact mirror of all-gather: big in, small out. It's also the only one of
these collectives that does any math. A detail I got wrong once myself: the reduction
is an average, not a sum. NCCL's `AVG` op folds the divide by W into the collective, so
there's no separate division kernel.

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

## Why not broadcast? (my failed quiz answer)

Broadcast means "one rank has the truth, copy it to everyone". It exists to fix stale
copies. So count the copies. After the optimizer updates w3, how many permanent copies
of w3 exist? Exactly one, on its custodian, freshly updated. There is nothing to be
stale. The other ranks don't hold an outdated w3. They hold nothing at all, because
their photocopy was shredded after backward. They'll get the fresh w3 automatically at
the next forward's all-gather, straight from the one rank that owns it.

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

Everything else in FSDP, the CUDA streams, prefetching, `reshard_after_forward`,
overlap, is engineering on top of one follow up question: the gathers cost time, can we
hide them behind compute? That's the next post, with real profiler traces.
