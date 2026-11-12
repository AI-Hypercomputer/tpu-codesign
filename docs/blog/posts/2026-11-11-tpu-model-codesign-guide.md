---
date: 2026-11-11
categories:
  - Architecture Co-Design
  - Ironwood (TPU v7x)
---

# Introducing the TPU Model Design Considerations and Optimization Guide

**Authors:** Gagik Amirkhanyan and Vaibhav Singh (Google Cloud)

Google designed TPUs from the ground up for machine learning efficiency across both training and inference, delivering top-tier performance per total cost of ownership (TCO) by trading general-purpose hardware scheduling for wide systolic arrays, dedicated SparseCore engines, and custom interconnect fabrics. Architectural decisions made before the first training step set permanent ceilings on Model FLOP Utilization (MFU), training cost, and serving latency. We are publishing the **TPU Model Design Considerations and Optimization Guide** as an open, living reference distilling practices co-evolved over a decade of TPU performance engineering for teams building new foundation models or adapting open-weight checkpoints on Ironwood (TPU v7x) and future TPU generations.

<!-- more -->

## What the guide covers

The guide organizes 13 concrete, hardware-grounded design and systems rules into three groups based on engineering stage and model-quality risk:

1. **[Shape rules](../../shape-rules.md) (Principles 1–3, near-zero quality risk):** Aligning per-device reduction and output dimensions to the 256×256 MXU systolic array (especially `head_dim = 256`, worth **+21% to +46% end-to-end MFU** across 8K–32K context), planning one parallelism strategy from 4 chips to multi-pod, and separating compile-time buffer shape from runtime data extent via `SMEM` scalar prefetch (`PrefetchScalarGridSpec`) for dropless MoE (`Megablox`, Tokamax GMM) and paged attention.
2. **[Architecture choices](../../architecture-choices.md) (Principles 4–9, validate with ablations):** Compressing routed token dispatch with **LatentMoE** (**+14% tokens/s** and **2.8×** faster permutes on Ironwood), raising compute intensity with one large shared expert in every LatentMoE layer and wider `d_ff` (**+57% tokens/s** and lower loss across 120.8B tokens on 512 TPU v7x chips, plus **+28% tokens/s** from wider `d_ff` alone), designing dynamic-range bounding (`QK-norm`, `z-loss`) for low precision (FP8 block size ≥ 512 on v7x), hybrid long-context attention, overcoming the decode bandwidth wall via low-diameter topologies (Boardfly) and Multi-Token Prediction (MTP) speculative decoding, and sizing fused VPU work inside the MXU tile window ($B \ge 1.5 \times r_{\text{MXU}} / r_{\text{VPU}} \approx 360$ at v7x peak).
3. **[Expressing parallelism & systems mapping](../../systems-mapping.md) (Principles 10–13, no model architecture changes):** Mapping parallel axes onto physical network links (`D2D → 3D ICI torus → DCN`) with SparseCore async collective offload (**15–22%** step-time reduction), Pallas VMEM pipelining and `tune-jax` block search, prefill/decode disaggregation with `block-size = 256` FP8 KV caches, and multi-pass BF16 for high-precision-bound compute.

## Where to dive in

* Start with [Overview & Hardware Facts](../../index.md) for the summary matrix and Ironwood foundational performance parameters.
* Jump straight to [Blueprint & Diagnostics](../../blueprint-and-diagnostics.md) for the 6-signal XProf triage checklist and a complete **~100B-parameter Ironwood LatentMoE reference config**.
* Read everything (including all figures and Appendices A & B) on one page in the [Complete Guide (Single Page)](../../guide.md).
