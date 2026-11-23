# Swizzle Runtime (`swizzle-rt`) Release Methodology

Swizzle AI follows **Semantic Versioning 2.0.0 (`MAJOR.MINOR.PATCH`)** across the 10-crate `swizzle-runtime` Cargo workspace (`swizzle-quant`, `swizzle-cuda`, `swizzle-vulkan`, `swizzle-engine`, `swizzle-server`, `swizzle-cli`, and companion crates) as well as `swizzle-gateway`, `swizzle-router`, and `swizzle-robotics`.

## 1. Versioning Policy

- **MAJOR (`X.0.0`)**: Breaking changes to the CLI flags, HTTP wire protocol (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`), `.ktc` / GGUF quantization tensor layout, or deterministic receipt verification format (`swizzle-rt verify-receipt`).
- **MINOR (`0.X.0`)**: Backward-compatible architectural features, new model family support (e.g., Gated DeltaNet, Mamba2, MoE), new hardware acceleration targets (CUDA `sm_75`–`sm_120`, Vulkan subgroup kernels), or new speculative decoding drafters (`RAD-1`).
- **PATCH (`0.0.X`)**: Bug fixes, kernel numerical parity fixes, security patches, and documentation updates.

## 2. Pre-Release Quality & Determinism Gates

Every release candidate must pass the following automated and mathematical verification gates before a Git tag is signed:
1. **Bit-Exact Receipt Verification**: `swizzle-rt verify-receipt` must produce identical greedy token hashes (`X-Swizzle-Receipt-Hash`) across CPU (AVX2/F16C), NVIDIA CUDA, and Vulkan compute backends.
2. **Workspace Build & Test Suite**: `cargo test --workspace --locked` and `cargo clippy --workspace` must pass with zero errors.
3. **Security & License Audit**: All third-party crates in `Cargo.lock` must comply with permissive OSI-approved licenses (Apache-2.0 / MIT / BSD).

## 3. Release Artifacts & Distribution

When a release tag `vX.Y.Z` is pushed to `Swizzle-AI/swizzle-runtime`:
- Precompiled static binaries for Linux (`x86_64`, CUDA 12.8+ & Vulkan) and Windows (`x86_64`) are published to GitHub Releases.
- Multi-arch OCI container images are published to GitHub Container Registry (`ghcr.io/swizzle-ai/swizzle-runtime:vX.Y.Z` and `:latest`).
- Release notes and benchmark deltas (`TTFT`, `ITL p50/p95/p99`) are recorded in [`CHANGELOG.md`](CHANGELOG.md).
