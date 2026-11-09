# TPU Model Design Considerations and Optimization Guide

**Authors:** Gagik Amirkhanyan and Vaibhav Singh (Google Cloud)

## Purpose

Google designed TPUs from the ground up for machine learning efficiency across both training and inference. Getting the most out of TPUs, or any other system, comes down to a set of architectural best practices that have co-evolved over a decade of performance engineering by TPU users inside and outside Google. We are making these practices openly available to the TPU community through this living guide. Large language models (LLMs) and Mixture-of-Experts (MoE) architectures receive the deepest coverage ([Principles 3–5](shape-rules.md#3-static-shapes-everywhere-including-dropless-moe-routing), [Principle 9](architecture-choices.md#9-minimize-exposed-vpu-time)) because sparse routing and cross-chip dispatch benefit most from TPU-aware design, while the underlying principles apply to any computation you want to accelerate on TPUs (including multimodal and world models) and largely carry over to other accelerators as well.

Prefer reading on one page? Open the [Complete Guide (Single Page)](guide.md).

## Co-designing models for Ironwood (TPU v7x) and future TPU generations: training and inference

Performance tuning starts with model definition. Architectural choices made before the first training step set permanent ceilings on Model FLOP Utilization (MFU), training cost, and serving latency. This guide provides concrete, hardware-grounded design rules for teams building or adapting models for TPUs, organized into three groups by engineering stage and quality risk:

1. **[Shape rules](shape-rules.md) (Principles 1–3):** Hardware-aligned dimensions, single-plan parallelism, and static shapes with dynamic runtime extent; near-zero model-quality risk. *Adopt always.*
2. **[Architecture choices](architecture-choices.md) (Principles 4–9):** Structural and numerical design choices; most involve quality trade-offs. *Validate with ablations.*
3. **[Expressing parallelism & systems mapping](systems-mapping.md) (Principles 10–13):** Parallelism placement across physical links, memory pipelining, serving disaggregation, and compiler tuning; no model checkpoint changes required.

**How to use this guide.** The shape rules apply to anyone designing a new model; use the summary table below to filter the rest, with the *Targets* column routing training versus inference owners and the *Scope* column marking the MoE-specific rules. If you are adapting an existing checkpoint rather than designing a new architecture, jump directly to [Systems mapping](systems-mapping.md) and [Blueprint & Diagnostics](blueprint-and-diagnostics.md). The general rules hold for any matmul-dominated architecture, and this repository grows model- and workload-specific studies over time.

## Principles at a glance

| # | Group | Guideline summary | Scope | Targets | Quality risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [1](shape-rules.md#1-align-every-per-device-reduction-and-output-dimension-to-256) | Shape | Make every per-device model reduction and output dimension a multiple of 256 (most frequently missed: `head_dim = 256`); counts such as `num_heads`, `layers`, and `batch` are exempt. | General | Both | ~None |
| [2](shape-rules.md#2-plan-one-parallelism-strategy-from-4-chips-to-multi-pod) | Shape | Plan dimensions for a large TPU scale-up domain (FSDP/TP/EP over ICI) paired with multi-slice scaling (DP/FSDP/PP over DCN, never EP or TP). | General | Training | ~None |
| [3](shape-rules.md#3-static-shapes-everywhere-including-dropless-moe-routing) | Shape | Separate compile-time buffer shape from runtime data extent via bucketing, masking, and `SMEM` scalar prefetch (`PrefetchScalarGridSpec`) so dropless MoE (`Megablox`, Tokamax GMM) and paged attention run without recompilation or padding waste. | General + MoE | Both | ~None |
| [4](architecture-choices.md#4-compress-routed-token-traffic-with-latentmoe) | Arch | Compress routed token communication volumes with **LatentMoE** (reinvesting saved bandwidth into expert capacity) and low-precision (FP8) all-to-all dispatch. | MoE | Both | Ablate (LatentMoE); FP8 A2A is quality-neutral |
| [5](architecture-choices.md#5-raise-compute-intensity-with-a-large-shared-expert-and-wider-experts) | Arch | Prefer higher compute intensity when quality is neutral: pair every LatentMoE layer with one large shared expert (overlapping EP dispatch) and trade excessive `(N, K)` fragmentation for wider expert `d_ff`. | MoE | Both | Ablate |
| [6](architecture-choices.md#6-low-precision-as-architecture-not-afterthought) | Arch | Bound dynamic ranges (`QK-norm`, `z-loss`) for low precision. Sub-channel block scaling runs on the VPU on TPU v7x and earlier (~240:1 MXU:VPU ratio; use FP8 block size ≥ 512). | General | Both | Ablate |
| [7](architecture-choices.md#7-hybrid-attention-for-long-context-pick-the-tpu-mature-variant) | Arch | Adopt hybrid sub-quadratic attention for long context: 5:1 local/global sliding-window interleave as the low-risk default, or linear-attention / compressed-KV hybrids to cut prefill FLOPs and decode KV footprint. | General | Both | Ablate |
| [8.1](architecture-choices.md#81-scale-out-over-high-connectivity-low-diameter-topologies) | Arch | Overcome memory-bandwidth-bound decode at low latency by scaling out across high-connectivity, low-network-diameter topologies (such as Boardfly on TPU 8i). | General | Inference | None |
| [8.2](architecture-choices.md#82-speculative-decoding-as-a-design-input-mtp-heads-or-drafter) | Arch | Design in Multi-Token Prediction (MTP) speculative decoding to verify multiple drafted tokens per HBM weight read; size active bytes to real serving concurrency. | General | Inference | Low |
| [9](architecture-choices.md#9-minimize-exposed-vpu-time) | Arch | Minimize exposed VPU time by dropping non-quality layout churn, fusing RMSNorm/SwiGLU into adjacent GEMMs, offloading MoE sorts/gathers to SparseCore, and sizing fused VPU work inside the MXU tile window ($B \ge 1.5 \times r_{\text{MXU}} / r_{\text{VPU}} \approx 340$ at v7x peak). | General + MoE | Training | ~None |
| [10](systems-mapping.md#10-map-model-axes-onto-the-physical-mesh) | Systems | Map every parallelism axis onto physical network links from fastest to slowest (D2D → 3D ICI torus → DCN) and enable SparseCore async collective offload. | General + MoE | Training | None |
| [11](systems-mapping.md#11-tune-the-memory-pipeline) | Systems | Use explicit Pallas HBM↔VMEM pipelining, tune shape-dependent kernel block sizes (`tune-jax`), and prefer selective rematerialization in HBM over host offload. | General | Both | None |
| [12](systems-mapping.md#12-co-design-for-prefilldecode-disaggregation-and-align-with-the-tpu-serving-stack) | Systems | Disaggregate compute-bound prefill (TTFT) and bandwidth-bound decode (TPOT) onto tailored slice shapes and shardings, using FP8 KV cache with `block-size = 256`. | General | Inference | None |
| [13](systems-mapping.md#13-use-multi-pass-bf16-for-high-precision-bound-compute) | Systems | For BF16-dominated compute workloads (such as long-context quadratic attention where direct FP8 degrades quality), use the multi-pass BF16 algorithm whenever the target TPU's low-precision-to-BF16 MXU throughput ratio exceeds the pass count. | General | Both | Low |

!!! note "Measured evidence"
    Alongside published sources, notes marked **Measured** report benchmarks from our Ironwood (TPU v7x) profiling program across open 20–30B MoEs (a Qwen3-30B-derived LatentMoE variant and GPT-OSS-20B) plus dense controls on a single slice in BF16 at training sequence lengths up to 32K (together with the two-slice DCN scaling check in Principle 2, and the 512-chip TPU v7x 120.8B-token and 1,024-chip TPU v4 50B-token compute-intensity ablations in Principle 5), all evaluated against tuned systems baselines. Treat the directional trends and rankings as transferable across configurations, and treat exact percentages as workload-specific.

## Foundational performance parameters (Ironwood / TPU v7x)

TPUs are custom ML ASICs engineered for performance per TCO, trading general-purpose hardware scheduling for wide 256×256 systolic arrays, software-managed VMEM, dedicated SparseCore engines, and direct multi-dimensional ICI fabrics. Reaching peak efficiency means aligning model shapes, routing, and parallelism with those hardware choices:

| Property | Value | Design consequence |
| :--- | :--- | :--- |
| **MXU (matrix multiply unit)** | 256×256 systolic array | Make every per-device contracting and output dimension a multiple of 256: matmuls execute in 256-wide tiles, and unaligned remainders are zero-padded, wasting cycles. |
| **Peak FP8** | 4,614 TFLOPS/chip (2× BF16) | Default to FP8 for training and serving: 8-bit operands double MXU math throughput and halve bytes moved. |
| **HBM** | 192 GB @ 7.38 TB/s | An op needs ≈ 625 FP8 FLOPs per byte read from HBM (≈ 312 BF16) to stay compute-bound rather than HBM-bound. Elementwise ops sit far below that ridge (minimize and fuse them). Weight-only FP8 decode re-reads weights every step (2 FLOPs per parameter per token) and therefore needs ≈ 300 concurrent tokens per weight shard to reach compute roofline, or fewer bytes per token (e.g., 4-bit weights or MTP). |
| **VMEM** | 64 MB per TensorCore | Kernels execute tile by tile: DMA a tile from HBM into VMEM, compute on it, and write results back while prefetching the next tile. Larger tiles reduce HBM↔VMEM traffic, but all live tiles and scratch buffers must fit inside 64 MB, bounding kernel block shapes (including SplashAttention blocks). |
| **1D arithmetic intensity over ICI** | ~11,500 FLOPs/byte (BF16; ~23,000 FP8 @ 200 GB/s ICI) | A collective stays hidden only if compute per byte transferred over ICI clears this threshold; TP moves activations every layer and rarely clears it, so avoid inter-chip TP in training. |
| **Dual-chiplet layout** | Two chiplets per chip, each with a TensorCore and 96 GB HBM; die-to-die (D2D) link ~6× faster than 1D ICI | Map TP = 2 onto the two chiplets so tensor-parallel collectives stay on the fast intra-package D2D link; wider TP crosses slower ICI. Memory is per chiplet: each TensorCore addresses its own 96 GB, not a shared 192 GB pool. |
| **Topology** | 3D torus (ICI) + DCN (data-center network) across pods | EP and CP require torus bandwidth every layer; DP and PP tolerate DCN best (one fixed-size sync per step or stage). FSDP over DCN re-gathers sharded weights on every accumulation microbatch, so reserve it for models exceeding one slice's HBM. |
| **SparseCore** | Dedicated offload engines for irregular memory access and collectives | Embedding lookups, MoE dispatch (`sort`, `gather`, `scatter`), and async collectives can execute off the TensorCore, overlapped with MXU compute. |

*Parallelism shorthand used throughout:* DP = data, TP = tensor, EP = expert, CP = context, PP = pipeline parallelism; FSDP = fully sharded data parallel.
