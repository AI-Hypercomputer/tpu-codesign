# Appendix B: Inference Hardware-Software Co-Design and References

---

## Inference hardware-software co-design checklist

Evaluate model architecture and serving configuration jointly at your target prompt lengths, output lengths, batch concurrency, and latency SLAs. Compare candidate designs on quality and **goodput** (completed requests per second meeting both TTFT and TPOT targets on an identical chip allocation). The table below summarizes supporting co-design checks across the serving stack:

| Co-design area | Guidance and mechanism | What to measure |
| :--- | :--- | :--- |
| **Attention, KV format, and kernel layout** | Pair `head_dim = 256` with proportionally fewer Q and KV heads, preserving total projection widths, logical KV bytes, and iso-quality. Inspect *physical* KV memory allocation after sharding, paging, and padding. Tune KV page sizes (`block-size = 256`) and attention compute blocks separately; prefer fused KV updates and phase-matched kernels. Ensure quantized KV saves physical HBM bytes while holding long-context quality ([TPU Ragged Paged Attention](https://arxiv.org/abs/2604.15464)). | Physical allocated KV bytes/token, attention and KV-update kernel time, tile padding waste, layout conversions, long-context task quality. |
| **Attention parallelism vs. expert parallelism** | Select attention TP/DP independently from expert EP. A large routed expert pool does not require equally wide attention TP; account for KV head replication when `TP > num_kv_heads`. Compare wider sharding against running more independent replicas at a fixed chip budget, since replica count governs the per-shard batch available to amortize weight reads. The [Qwen3.5-397B on Ironwood playbook](https://developers.googleblog.com/systems-engineering-playbook-optimizing-qwen-35-397b-moe-on-ironwood-tpu7x/) pairs attention DP with expert EP to avoid duplicating KV heads. | Per-replica concurrency, KV replication factor, collective latency, net goodput/chip, P99 TPOT. |
| **MoE batch occupancy and load balance** | Size expert width, count, and active top-$K$ for real decode batch sizes. Per-token active parameters do not capture the **union of expert weights** streamed across a batch. Profile expert routing skew alongside mean tokens/expert, since the busiest rank sets step latency. Rebalance expert placement or replicate hot experts in HBM without altering router math ([DeepSeek-V3](https://arxiv.org/pdf/2412.19437)). | Distinct experts touched per step, tokens/expert skew, busiest-rank latency, dispatch/combine and weight-read bytes. |
| **Batching and prefill/decode placement** | Prewarm a compact set of compiled token and batch-slot buckets. Tune continuous batching and prefill chunk size jointly so long prefills do not stall ongoing decode while avoiding tiny compute-starved chunks. Benchmark colocated chunked prefill ([Sarathi-Serve](https://arxiv.org/abs/2403.02310)) against disaggregated prefill/decode pools ([DistServe](https://arxiv.org/abs/2401.09670)) with separately tuned pool sizes and shardings, including KV transfer, layout conversion, and queueing delays. | TTFT and TPOT percentiles (P50/P99), queue wait time, KV handoff bytes and latency, prefill/decode pool utilization, net goodput/chip. |
| **Prefix reuse and locality-aware routing** | Reuse exact matching prefix KV blocks where supported ([vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)) and route related requests using both KV cache locality and queue load ([llm-d routing](https://llm-d.ai/docs/architecture)) so queueing hotspots do not erase saved prefill compute. Note that prefix caching skips repeated prefill FLOPs, not decode KV reads over that cached context. | Reused prefix token fraction, end-to-end TTFT including queueing, cache eviction/recompute rate, active KV occupancy, rank load skew. |
| **MTP / speculative decoding** | Tune draft length across the complete draft, verify, and KV commit/rollback cycle at target concurrency ([Principle 8.2](architecture-choices.md#82-speculative-decoding-as-a-design-input-mtp-heads-or-drafter)). Acceptance rate alone is insufficient once drafter latency, verification FLOPs, MoE touched-expert expansion, and rollback overhead are included. Validate rejection-sampling correctness so the output distribution matches the target model ([speculative decoding](https://arxiv.org/abs/2211.17192)). | Mean accepted prefix length, net committed tokens/s, full cycle latency, sampler distribution parity, net goodput/chip. |

---

## Two useful serving budget checks

1. **Logical KV cache footprint:** For standard multi-head / grouped-query attention with equal K and V head dimensions and element byte size $s_{\text{KV}}$, logical KV memory across active requests $r$ and attention layers $l$ is:

    $$M_{\text{KV}} = 2 \cdot s_{\text{KV}} \sum_{r, l} S_{r,l} \cdot n_{\text{KV},l} \cdot d_{\text{head},l}$$

    where $S_{r,l}$ is the cached token length for request $r$ at layer $l$. Halving $n_{\text{KV},l}$ when doubling $d_{\text{head},l}: 128 \to 256$ preserves logical $M_{\text{KV}}$ identically. Physical HBM allocation additionally includes page fragmentation, padding, quantization scale metadata, and any TP replication; latent-compressed KV (MLA) and recurrent linear-attention states require their own state accounting.

2. **Expected MoE batch expert occupancy and LatentMoE dispatch parity:** Under uniform independent routing across $E$ experts with $K$ distinct experts per token and $B$ concurrent decode tokens, the expected number of distinct experts touched in a single MoE layer is:

    $$E_{\text{touched}}(B) = E \cdot \left[1 - \left(1 - \frac{K}{E}\right)^B\right]$$

    Use this expression only as a back-of-the-envelope sanity check, and rely on profiled expert occupancy and HBM weight reads on real traffic. Separately, LatentMoE's save-and-reinvest condition:

    $$K_{\text{new}} \cdot \ell \le K_{\text{old}} \cdot d_{\text{model}}$$

    guarantees that routed activation dispatch volume does not increase at equal precision; expert weight-read bytes and MXU compute FLOPs remain distinct budgets to check alongside it. Note that the iso-quality `head_dim = 256` results in [Principle 1](shape-rules.md#1-align-every-per-device-reduction-and-output-dimension-to-256) hold total projection widths constant via proportional head-count reduction; the fixed-head-count `head_dim` sweep in the [TPU Ragged Paged Attention](https://arxiv.org/abs/2604.15464) paper evaluates a different scaling experiment.

---

## References

### Cloud TPU, JAX, and reference stacks

* Google Cloud, [*TPU7x (Ironwood) performance optimizations*](https://cloud.google.com/tpu/docs/ironwood-performance)
* Google Cloud Blog, [*TPU 8t and TPU 8i technical deep dive*](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)
* Google Developers Blog, [*Systems Engineering Playbook: Optimizing Qwen3.5-397B MoE on Ironwood (TPU v7x)*](https://developers.googleblog.com/systems-engineering-playbook-optimizing-qwen-35-397b-moe-on-ironwood-tpu7x/)
* JAX ML, [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/) (`jax-ml/scaling-book`)
* AI-Hypercomputer, [*tpu-recipes*](https://github.com/AI-Hypercomputer/tpu-recipes) (Ironwood training and inference recipes)
* AI-Hypercomputer, [*MaxText*](https://github.com/AI-Hypercomputer/maxtext) (open-source reference JAX/TPU LLM and MoE implementation)
* Google DeepMind, [*Tokamax*](https://github.com/openxla/tokamax) (open-source JAX/Pallas custom kernel library for TPUs and GPUs)
* [*TPU Ragged Paged Attention*](https://arxiv.org/abs/2604.15464) (arXiv:2604.15464)

### Frontier model architecture reports

* NVIDIA, [*Nemotron 3: Efficient and Open Intelligence*](https://arxiv.org/abs/2512.20856) (arXiv:2512.20856)
* Microsoft AI, [*MAI-Thinking-1: Building a Hill-Climbing Machine*](https://microsoft.ai/pdf/mai-thinking-1.pdf)
* DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) (arXiv:2412.19437)
* Moonshot AI, [*Kimi K3 Technical Report*](https://arxiv.org/abs/2607.24653) (arXiv:2607.24653)
* Gemma Team, [*Gemma 3 Technical Report*](https://arxiv.org/abs/2503.19786) (arXiv:2503.19786)
* Yang et al., [*Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464) (arXiv:2412.06464)
* Qwen Team, [*Qwen3-Next-80B-A3B*](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)
* Zhipu AI, [*GLM-4.5 Technical Report*](https://arxiv.org/abs/2508.06471) (arXiv:2508.06471)

### Scaling laws, serving systems, and profiling tools

* Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (arXiv:2001.08361)
* Levine et al., [*The Depth-to-Width Interplay in Self-Attention*](https://arxiv.org/abs/2006.12467) (arXiv:2006.12467)
* Tay et al., [*Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers*](https://arxiv.org/abs/2109.10686) (arXiv:2109.10686)
* Agrawal et al., [*Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Chunked-Prefills*](https://arxiv.org/abs/2403.02310) (arXiv:2403.02310)
* Zhong et al., [*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*](https://arxiv.org/abs/2401.09670) (arXiv:2401.09670)
* Leviathan et al., [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192) (arXiv:2211.17192)
* [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) and [llm-d Architecture](https://llm-d.ai/docs/architecture)
* Profiling & kernel tools: [XProf / TensorBoard Profiler](https://openxla.org/xprof), [Pallas TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html), [`tune-jax`](https://github.com/rdyro/tune-jax), and [`jax.experimental.roofline`](https://github.com/jax-ml/jax/tree/main/jax/experimental/roofline)
