---
layout: post
title: "FSDP Collectives 101: why reduce-scatter, and why not broadcast?"
date: 2026-07-26
tags: [fsdp, distributed-training, pytorch]
---

I failed the same quiz question twice while learning FSDP. Both times, the same two
mistakes: after the backward pass I said "all-reduce," and after the optimizer step I
insisted the updated weights needed a broadcast. Both answers are *correct* — for DDP.
Both are wrong for FSDP, and the reason they're wrong turned out to be the single most
useful thing I've learned about distributed training. This post is the explanation I
wish I'd read first: what all-gather and reduce-scatter actually are, and — just as
important — why the other collectives you might reach for are answers to questions FSDP
never asks.

## The root fact: two different worlds

**DDP world:** every GPU permanently stores the entire model. 8 GPUs = 8 full copies
that must be kept bit-identical forever. All communication in DDP exists to preserve
that invariant.

**FSDP world:** the model exists exactly **once**, chopped into W pieces. GPU k
permanently owns piece k of every weight tensor — and of its gradient and optimizer
state. No full copy of anything exists anywhere, at rest, ever. Full-size tensors
appear only as short-lived *photocopies* during compute, then get shredded.

Every "which collective goes here?" question answers itself once you ask: **in this
world, who is allowed to permanently hold what?** My two wrong quiz answers were both
symptoms of mentally living in DDP world — imagining full copies that need to be kept
in sync — while writing FSDP code.

## All-gather: everyone shows their piece

Each of W ranks contributes its piece; afterward everyone holds the concatenation.

Two GPUs, one 4-element weight. A owns `[w1,w2] = [1, 2]`, B owns `[w3,w4] = [3, 4]`:

```
before:   A: [1, 2, ·, ·]        B: [·, ·, 3, 4]
all-gather ────────────────────────────────────
after:    A: [1, 2, 3, 4]        B: [1, 2, 3, 4]
```

Why FSDP needs it: a matmul touches every entry of the weight, so right before a layer
runs, its custodians pool their slices into a temporary full copy. Compute, shred, move
to the next layer. The whole forward pass is gather → use → shred, layer by layer.

Data flow direction: **small in, big out**. No arithmetic — pure assembly.

## Reduce-scatter: average everything, take home only your slice

Each rank contributes a **full-size** tensor; the tensors are element-wise reduced
(averaged), and each rank receives only its own slice of the result.

After backward, A and B each hold a full-size gradient — full-size because the
photocopied weights were full-size and autograd doesn't know about sharding. The
gradients *differ* because each GPU saw different data:

```
A's grad:  [8, 0, 4, 2]
B's grad:  [0, 4, 8, 6]
average:   [4, 2, 6, 4]   ← computed in flight, never assembled on any GPU
```

Reduce-scatter delivers `[4, 2]` to A and `[6, 4]` to B. Why only a slice? A will only
ever update w1 and w2 — shipping it the averaged gradient for w3 and w4 would spend
network bandwidth delivering numbers it would immediately throw away.

Data flow direction: **big in, small out** — the exact mirror of all-gather, and the
only one of these collectives that does math. (Detail I got wrong once: the reduction
is an *average*, not a sum — NCCL's `AVG` op folds the divide-by-W into the collective
itself, no separate division kernel.)

## The identity that ties it together

> **all-reduce = reduce-scatter + all-gather**

"Everyone ends with the full averaged tensor" decomposes into "everyone gets their
averaged slice" then "everyone shows their slice."

DDP needs the full all-reduce on gradients because every GPU stores full weights, so
every GPU needs the full averaged gradient. FSDP runs only the **first half**
(reduce-scatter) after backward — each rank updates only its slice. The second half
isn't skipped; it's **moved to the next forward pass**, where an all-gather was needed
anyway to build the photocopy.

FSDP is DDP's all-reduce sawed in half, with each half relocated to where the data is
actually needed. Nothing new is invented — the pieces are just rescheduled.

## Why not broadcast? (my failed quiz answer #2)

Broadcast is "one rank has the truth; copy it to everyone." It exists to fix **stale
copies**. So count the copies: after the optimizer updates w3, how many permanent
copies of w3 exist? Exactly one — on its custodian, freshly updated. There is nothing
to be stale. The other ranks don't hold an outdated w3; they hold *nothing* — their
photocopy was shredded after backward. They receive fresh w3 automatically at the next
forward's all-gather, straight from the single source of truth.

Broadcast is DDP-world thinking: it presumes replicas that can drift. With one original
per weight, staleness is structurally impossible.

## Why not the other collectives?

Each collective is the answer to a specific question. Asking "which one goes here?"
really means asking "which question is FSDP asking at this moment?"

| Collective | The question it answers | Does FSDP ask it? |
|---|---|---|
| broadcast | "One rank knows; everyone needs a copy" | No — no replicas exist to sync |
| scatter | "One rank holds everything; deal out the pieces" | No — pieces never start centralized |
| gather | "Collect all pieces *onto one rank*" | No — every rank needs the full layer, not one |
| all-gather | "Everyone has a piece; everyone needs the whole" | **Yes** — before every layer's compute |
| all-reduce | "Everyone has a full version; everyone needs the full average" | No — that's DDP; each rank needs only its slice |
| reduce-scatter | "Everyone has a full version; each rank needs its slice of the average" | **Yes** — after gradients |

The pattern in the "no" rows: they either presume a central rank (scatter, gather,
broadcast) or presume full replicas (broadcast, all-reduce). FSDP's world has neither —
it is symmetric (every rank is a custodian of equal standing) and deduplicated (one
original of everything).

## The diagnostic that fixed my mental model

When unsure which collective belongs somewhere, stop and **count who permanently holds
what**. If your answer requires a full copy of anything sitting on a GPU at rest,
you've slipped back into DDP world. In FSDP world:

- permanent state = shards only (weights, grads, optimizer moments — all 1/W)
- full tensors = photocopies with a lifetime of one layer's compute
- one original per number → synchronization is a non-concept

Everything else about FSDP — the CUDA streams, prefetching, `reshard_after_forward`,
overlap — is engineering on top of one follow-up question: *the gathers cost time; can
we hide them behind compute?* That's the next post, with real profiler traces.
