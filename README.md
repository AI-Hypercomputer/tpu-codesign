# TPU Model Design Considerations and Optimization Guide

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Google designed TPUs from the ground up for machine learning efficiency and performance per total cost of ownership (TCO), trading general-purpose hardware scheduling for wide systolic arrays, dedicated SparseCore engines, and custom interconnect fabrics. This living guide provides practical, hardware-grounded architectural design principles, Mixture-of-Experts (MoE) co-design patterns, low-precision numerics rules, and distributed systems mappings for training and serving LLMs, MoEs, and other hardware-intensive workloads on Google Cloud TPUs (Ironwood / TPU v7x and future TPU generations).

* **Online Book & Blog:** <https://ai-hypercomputer.github.io/tpu-codesign/>
* **Single-Page Guide:** [`docs/guide.md`](docs/guide.md)

## Chapters

| Section | Contents | Quality Risk |
| :--- | :--- | :--- |
| **[Overview & Hardware Facts](docs/index.md)** | 13-principle summary matrix, Measured evidence methodology, and Ironwood (TPU v7x) foundational performance parameters | n/a |
| **[Part I: Shape Rules (Principles 1–3)](docs/shape-rules.md)** | 256-aligned per-device dimensions (`head_dim = 256`), single-plan parallelism from 4 chips to multi-pod, and static buffer shapes with `SMEM` scalar prefetch (`PrefetchScalarGridSpec`) for dropless MoE and paged attention | Near-zero (*adopt always*) |
| **[Part II: Architecture Choices (Principles 4–9)](docs/architecture-choices.md)** | LatentMoE token compression, higher compute intensity via one large shared expert + wider `d_ff` (+57% tok/s across 120.8B tokens on 512 TPU v7x chips), native low-precision design (FP8 block size ≥ 512 on v7x), hybrid long-context attention, decode scale-out (Boardfly) & MTP speculative decoding, and minimizing exposed VPU time | *Validate with ablations* |
| **[Part III: Expressing Parallelism & Systems Mapping (Principles 10–13)](docs/systems-mapping.md)** | Physical link hierarchy (`D2D → 3D ICI torus → DCN`) + SparseCore async offload, Pallas VMEM pipelining & `tune-jax`, prefill/decode disaggregation, and multi-pass BF16 for high-precision-bound compute | None–Low (*no model architecture changes*) |
| **[Part IV: Blueprint & Diagnostics](docs/blueprint-and-diagnostics.md)** | Regime prioritization, 6-signal XProf triage checklist, kernel roofline & HBM budgeting rules, and a complete **~100B Ironwood LatentMoE reference config** | n/a |
| **[Appendix A: Design & Evaluation (`A.1–A.8`)](docs/appendix-a.md)** | Aspect-ratio ladders, single-knob (`L`) family scaling, HBM remat vs. offload, serving concurrency sizing, distilled tiers, wall-clock ablations, serving PTQ, and published Ironwood reference targets | n/a |
| **[Appendix B: Inference Co-Design & References](docs/appendix-b.md)** | End-to-end serving co-design checklist, KV & MoE batch-occupancy budget equations, and primary references | n/a |
| **[Complete Guide (Single Page)](docs/guide.md)** | All chapters, figures, and appendices combined in one self-contained Markdown document | n/a |

## Previewing the Site Locally

The book and blog are built from the Markdown files in [`docs/`](docs/) using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/):

```bash
pip install "mkdocs<2.0" "mkdocs-material>=9.5,<10.0"
mkdocs serve
```

## Citation

```bibtex
@misc{tpucodesign2026,
  title  = {TPU Model Design Considerations and Optimization Guide},
  author = {{TPU Co-Design Authors}},
  year   = {2026},
  howpublished = {\url{https://ai-hypercomputer.github.io/tpu-codesign/}},
  note   = {Google Cloud AI Hypercomputer}
}
```

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## License & Terms

This repository is licensed under the [Apache License 2.0](LICENSE). Use of Google Cloud TPUs is subject to the [Google Cloud Terms of Service](https://cloud.google.com/terms).

## Security & Vulnerability Disclosure

Eligibility for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security) is determined by the [Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).
