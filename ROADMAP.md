# Swizzle Runtime (`swizzle-rt`) Technical Roadmap

This document tracks the open engineering milestones for `Swizzle-AI/swizzle-runtime` and its companion repositories, aligned with our foundation and grant deliverables (**LF AI & Data**, **LFX Mentorship**, **NLnet NGI Zero**, and **FLOSS/fund**).

---

## Milestone 1: Bit-Exact Cross-Vendor Kernel Parity (Q2–Q3 2026)
- [x] Zero-padding 16-byte XOR shared-memory CUDA GEMM tiling (`GemmTile`, `sm_75`–`sm_120`).
- [x] Compile-time balanced binary tree reductions (`TreeSum<K>`) for batch-size invariant dot products.
- [x] Zero-copy CPU & GPU-resident `Shortlist` exact vocabulary screening (`Theorem 9.1`).
- [ ] Extend `TreeSum<K>` and `Shortlist` warp-ballot screening to **Vulkan subgroup operations** (`VK_KHR_shader_subgroup_ballot`) for bit-exact parity across AMD RDNA3/4, Intel Arc, and Apple Silicon (MoltenVK).
- [ ] Expand quantized format coverage with bit-exact reference verification suites for `IQ2_XXS`, `IQ3_S`, and `Q6_K`.

## Milestone 2: Speculative Decoding & Stateful Prefix Caching (Q3 2026)
- [x] `RAD-1` online score-weighted ensemble (`SessionIndex` 262k suffix automaton + prompt `Lookup` + neural `MtpDrafter`).
- [x] `Warm Start` two-tier RAM/NVMe prefix cache for KV and recurrent states (`DeltaNetState`, `Mamba2State`).
- [ ] Distributed `Warm Start` cache synchronization over zero-copy RDMA / gRPC between `swizzle-gateway` and local `swizzle-rt` workers.
- [ ] Tree-attention verification in a single fused Vulkan/CUDA pass for branching factor $B > 4$.

## Milestone 3: Cryptographic Inference Receipts & Auditable Local AI (Q3–Q4 2026)
- [x] `swizzle-rt verify-receipt` deterministic seed, hardware backend, and token hash verification (`X-Swizzle-Receipt-Hash`).
- [ ] Self-contained JSON-LD / CBOR verifiable receipt format binding GGUF tensor SHA-256 digests, quantization kernel version, and prompt/output token streams.
- [ ] Open benchmark reproducibility harness comparing `swizzle-rt`, `llama.cpp`, and `vLLM` across latency (`TTFT`, `ITL p50/p95/p99`) and exact output divergence.

## Milestone 4: Real-Time Edge & Robotics VLA Inference (Q4 2026 – Q1 2027)
- [ ] Integration between `swizzle-runtime` and `swizzle-robotics` for deterministic `80 Hz` Vision-Language-Action (VLA) control loops on edge GPUs (Jetson Orin / x86 IPCs).
- [ ] Hard real-time memory locking (`mlock`), pre-allocated `Torque` graph pools, and bounded worst-case execution time (WCET) telemetry.
