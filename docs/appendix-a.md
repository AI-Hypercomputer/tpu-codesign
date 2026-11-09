# Appendix A: Supporting Design and Evaluation Topics

---

## A.1. Stay in the conventional aspect-ratio band; explore wider-not-deeper within it via ladder ablations

At a fixed parameter budget, wider-shallower models execute faster on TPUs: larger matrix dimensions keep the 256×256 MXU saturated, while fewer sequential layers reduce kernel launch overhead, pipeline bubbles, and autoregressive decode latency.

Model quality must still be validated empirically per family. Pretraining loss is only weakly sensitive to width-to-depth ratio **within the conventional aspect-ratio band** (loss remains nearly flat across a ~40× aspect-ratio span at fixed parameters in [Kaplan et al.](https://arxiv.org/abs/2001.08361), and depth efficiency saturates logarithmically at large width in [Levine et al.](https://arxiv.org/abs/2006.12467)), whereas downstream reasoning and compositional tasks are more shape-sensitive than pretraining perplexity and can favor depth ([Tay et al.](https://arxiv.org/abs/2109.10686)). Treat width as an optimization lever to test via **parameter-matched ladder ablations** gated on downstream evaluations while staying inside established aspect ratios. As a reference point, [MAI-Base-1](https://microsoft.ai/pdf/mai-thinking-1.pdf) pins $D = L \times 256 / 3$ across its scaling ladder specifically because it lands cleanly on hardware-aligned widths. Choose layer counts divisible by candidate pipeline degrees (`48, 64, 72`).

**Measured (directional):** On Ironwood, a wider-shallower 30B variant achieved **31.4% vs. 29.3% MFU**; confirm downstream task parity with a parameter-matched ladder ablation before adopting.

---

## A.2. One knob (layer count) scales the family; ratios stay TPU-friendly

Parameterize an entire model family with a single scaling knob, the layer count $L$, deriving widths and head counts from fixed ratios that land on TPU-aligned values across every point on a constant tokens-per-parameter scaling ladder:

* **Hidden and intermediate widths (`d_model`, `d_ff`, `ℓ`):** Multiples of 256 per TensorCore after sharding.
* **Head counts (`num_q_heads`, `num_kv_heads`):** Divisible by planned TP degrees (`2` for training chiplet D2D, `8–16` for serving; MAI rounds query heads to multiples of 16) with `head_dim = 256`.
* **Routed expert count (`N`):** Divisible by candidate EP degrees (`8, 16, 32`).
* **Layer count (`L`):** Divisible by candidate PP degrees and hybrid local:global attention cycles (`48, 64, 72`).

Every small-scale ablation point then shares the exact hardware alignment properties of the flagship production candidate.

---

## A.3. Remat-friendly blocks

Activation rematerialization (checkpointing) trades ~25–30% additional compute FLOPs to prevent activation out-of-memory errors at long sequence lengths or large per-device microbatches. Design transformer blocks so that expensive-to-recompute tensors (attention outputs and MoE expert outputs) form clean, compact checkpoint boundaries with cheap elementwise work between them. That structure makes selective rematerialization policies (`save_qkv_proj`, `save_out_proj`, or [MaxText](https://github.com/AI-Hypercomputer/maxtext)'s `custom` remat policy) effective instead of forcing `full` layer rematerialization.

**Measured (keep activations in HBM by default):** Offloading activations to host CPU memory crosses a link far slower than HBM; in Ironwood benchmarks, keeping attention outputs resident in HBM outperformed every host-offload variant by **+8–19%**. Selective host offload becomes worthwhile only when model weights and optimizer state crowd HBM on very large dense checkpoints (for example, the Ironwood Llama-3.1 405B recipe in [tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) offloads decoder-layer inputs, while older smaller-HBM chip recipes additionally offload QKV projections). Always validate offload choices per model and chip generation with a short profile.

**LatentMoE remat bonus:** Anchoring activation checkpoints on the compressed latent tensor (`ℓ`) rather than full `d_model` freed **~25 GB/device** of HBM batch headroom on our 30B MoE runs.

---

## A.4. Size the model to your real serving concurrency

Low-concurrency autoregressive decode is HBM-bandwidth-bound. In weight-only FP8, a dense weight shard requires ~300 concurrent tokens sharing each weight read to cross from memory-bound into compute-bound execution (compare the token batch seen *per weight shard*, not total replica concurrency divided by TP chip count). Because a single deployed checkpoint serves varying traffic regimes, size the architecture for the demanding end of production traffic (latency-sensitive requests at long context) using two pre-commit equations:

1. **Per-token latency floor $= \text{per-chip streamed weight bytes} \div \text{HBM bandwidth}$.** Each decode step streams the weights resident on each chip: for a dense model, its sharded slice under TP; for an MoE, **the union of routed expert weights selected across the batch**, plus shared-expert and dense weights. As batch size increases, touched experts rapidly approach 100% of the expert pool. *Example:* When the batch touches all experts and placement is balanced, Qwen3.5-397B in FP8 across 4 Ironwood chips streams ~100 GB/chip, setting a **~13.5 ms/token** weight-read latency floor before counting KV cache reads; scaling out across a larger TP/EP domain lowers per-chip weight bytes at the cost of interconnect collectives ([Principle 8.1](architecture-choices.md#81-scale-out-over-high-connectivity-low-diameter-topologies)).
2. **Memory-limited concurrency $\approx \text{HBM remaining after weights, runtime buffers, and workspaces} \div \text{allocated KV/state per sequence}$.** Either this HBM capacity ceiling or your TPOT latency SLA caps the batch sharing each weight read, setting per-token bandwidth cost at `streamed bytes ÷ batch`. Use observed expert batch occupancy and physical per-rank KV allocation in this budget. Your byte-reduction levers are FP8 weights and KV (4-bit PTQ halves weight bytes again, [Appendix A.7](#a7-quantize-further-for-serving)), TP/EP scale-out, hybrid or compressed KV ([Principle 7](architecture-choices.md#7-hybrid-attention-for-long-context-pick-the-tpu-mature-variant)), and MTP speculative decoding ([Principle 8.2](architecture-choices.md#82-speculative-decoding-as-a-design-input-mtp-heads-or-drafter)). **If real production concurrency will be small, design fewer active bytes (fewer active parameters or a smaller model) rather than planning around a batch size you will not have.**

---

## A.5. Train-big-serve-small as a planned path

Production deployments typically serve a family ladder rather than a single checkpoint: the flagship tier is usually the pretrained MoE served in FP8 or 4-bit ([DeepSeek-V3](https://arxiv.org/abs/2412.19437), [Kimi K3](https://arxiv.org/abs/2607.24653), Qwen3-235B), while high-volume latency tiers are distilled or pruned student models, often MoEs themselves (Qwen3-30B-A3B, Llama 3.2, Minitron). Two design rules follow:

* **Design the flagship directly for serving** ([Principle 12](systems-mapping.md#12-co-design-for-prefilldecode-disaggregation-and-align-with-the-tpu-serving-stack), [Appendix A.4](#a4-size-the-model-to-your-real-serving-concurrency), [Appendix A.7](#a7-quantize-further-for-serving)).
* **Enforce the 256-alignment shape rules on every distilled or pruned student tier** ([Principle 1](shape-rules.md#1-align-every-per-device-reduction-and-output-dimension-to-256), [Appendices A.1–A.2](#a1-stay-in-the-conventional-aspect-ratio-band-explore-wider-not-deeper-within-it-via-ladder-ablations)). Systolic padding waste hurts *more* at smaller student dimensions, where a remainder tile represents a much larger fraction of total GEMM work.

---

## A.6. Measure ablations in wall-clock, not FLOPs

Paper FLOP counts and real accelerator wall-clock time frequently rank candidate architectures in opposite orders because hardware efficiency differs sharply across operator classes (dense MXU matmuls versus VPU elementwise ops, routing sorts, collectives, and recurrent scans). [MAI-Thinking-1](https://microsoft.ai/pdf/mai-thinking-1.pdf) explicitly tracks equal-FLOP and equal-time scaling curves side by side because rankings flip when communication and kernel execution costs enter the picture.

Always run architectural ablations on the production software stack (SplashAttention, Megablox or Tokamax scalar-prefetch ragged GMMs, SparseCore collective offload, and `tune-jax` block sizes) and compare **wall-clock training/serving cost at matched quality**. Equally important, **tune the systems baseline first**: in our Ironwood benchmarks, pure systems tuning (SparseCore offload, MXU-aligned GMM tiles, and remat tuning) lifted an open-weight 30B MoE from **12% to 23% MFU** before any architectural change. Running ablations against an untuned 12% MFU baseline severely mis-ranks architectural trade-offs.

---

## A.7. Quantize further for serving

Training precision is not the serving floor. Because autoregressive decode streams weights once per step and reuses them across active batch tokens ([Appendix A.4](#a4-size-the-model-to-your-real-serving-concurrency)), halving weight precision roughly halves weight-read bytes before scale metadata and dequantization overhead. Serving FP8 weights is the zero-friction train-consistent default ([Principle 6](architecture-choices.md#6-low-precision-as-architecture-not-afterthought)), and weight-only INT4, W4A8, or FP4 post-training quantization (PTQ) halves stored weight bytes again without retraining:

* **Match quantization mode to the serving regime:** Small-batch, latency-bound decode is dominated by weight reads, where weight-only 4-bit pays most. Large-batch, throughput-bound serving is compute-bound, where W4A8 or FP8 fits best.
* **Reuse the training sensitive-layer map ([Principle 6](architecture-choices.md#6-low-precision-as-architecture-not-afterthought)):** Keep the exact same operations in high precision (attention/router/output pre-softmax logits, router gates, norms, embeddings, final layers). Range-bounding choices baked in during pretraining (`QK-norm`, `QK-clip`, `z-loss`) allow 4-bit PTQ to hold quality with minimal degradation.
* **Quantize the KV cache:** At long context, KV reads rival weight reads; FP8 KV cache is already standard in Ironwood serving recipes ([Principle 12](systems-mapping.md#12-co-design-for-prefilldecode-disaggregation-and-align-with-the-tpu-serving-stack)).
* **Calibrate on production-matched traffic:** Static activation scales calibrated on mismatched domains, languages, or sequence lengths clip outliers or waste dynamic range in production.
* **Gate on downstream task evaluations, not perplexity alone:** Quantization degradation surfaces first on multi-step reasoning, coding, and long-context retrieval tasks; validate directly on your target serving benchmarks.

---

## A.8. Reference performance targets

Use published Ironwood [MaxText](https://github.com/AI-Hypercomputer/maxtext) and [tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) throughput numbers (`TFLOP/s/chip`; divide by 2,307 BF16 or 4,614 FP8 chip peak for MFU) as sanity-check targets for your own training runs:

| Reference model (sequence length) | BF16 (TFLOP/s/chip) | FP8 (TFLOP/s/chip) |
| :--- | ---: | ---: |
| Llama-3.1 70B (8K) | 1,207 | 1,854 |
| Llama-3.1 405B (8K) | 1,261 | 1,927 |
| Llama-3.1 70B (128K) | 898 | 1,014 |
| DeepSeek-V3 671B (4K) | 608–613 | 731–743 |
| Qwen3-235B-A22B (4K) | 630 | 703 |
| GPT-OSS-120B (8K) | 330 | n/a |

Two structural lessons stand out in this table:

1. **Dense vs. MoE MFU gap:** Sparse MoE models buy substantial quality per FLOP, but pay a real MFU tax in token routing, cross-chip communication, and ragged grouped GEMMs; budget cluster sizing around MoE MFU rather than dense MFU.
2. **Architecture within MoEs:** DeepSeek-V3 and Qwen3-235B (608–630 BF16 TFLOP/s/chip) align hidden and expert intermediate widths to multiples of 256 with `head_dim = 128`, whereas GPT-OSS-120B (330 BF16 TFLOP/s/chip, with unaligned 2,880 = $11.25 \times 256$ hidden/expert widths and `head_dim = 64`) lands at roughly half their per-chip throughput, directly illustrating [Principle 1](shape-rules.md#1-align-every-per-device-reduction-and-output-dimension-to-256).
