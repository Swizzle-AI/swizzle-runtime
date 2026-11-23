# Swizzle AI Project Governance

Swizzle AI is an open-source organization building deterministic, high-throughput AI inference and robotics runtime infrastructure under the **Apache License 2.0**. This document defines the project's open governance model in alignment with **LF AI & Data Foundation**, **CNCF**, and **OpenSSF Best Practices** standards.

## 1. Principles

- **Open & Vendor-Neutral**: All source code, kernel specifications, technical reports, and verification suites are developed in the open under the Apache-2.0 license.
- **Meritocratic & Consensus-Driven**: Technical authority is earned through sustained, high-quality contributions to the codebase, documentation, benchmarks, and community reviews.
- **Deterministic Quality Gate**: Architectural changes affecting kernel execution, quantization formats, or speculative decoding must preserve bit-exact verification receipts (`swizzle-rt verify-receipt`) and pass automated CI checks.

## 2. Roles and Responsibilities

### Contributors
Anyone who contributes code, documentation, benchmark reproductions, issue reports, or code reviews. All contributors must abide by the [Code of Conduct](CODE_OF_CONDUCT.md) and sign off commits under the **Developer Certificate of Origin (DCO)** (`git commit -s`).

### Reviewers
Active contributors who have demonstrated expertise in one or more workspace crates (e.g., `swizzle-quant`, `swizzle-cuda`, `swizzle-vulkan`, `swizzle-engine`, `swizzle-server`). Reviewers triage issues and review pull requests in their domain.

### Maintainers
Maintainers have write access to Swizzle AI repositories and are listed in [`MAINTAINERS.md`](MAINTAINERS.md). Maintainers are responsible for:
- Reviewing and merging pull requests.
- Cutting signed releases and publishing container images and binary artifacts.
- Guiding the technical roadmap ([`ROADMAP.md`](ROADMAP.md)) and mentoring LFX / GSoC contributors.

### Technical Steering Committee (TSC)
The TSC consists of the lead maintainers across Swizzle AI's core repositories (`swizzle-runtime`, `swizzle-gateway`, `swizzle-kernels`, `swizzle-router`, `swizzle-robotics`). The TSC makes decisions on:
- Project-wide architectural standards and wire-protocol compatibility.
- Addition or removal of maintainers (by lazy consensus or 2/3 majority vote).
- Foundation hosting, grant allocations (LFX, NLnet, FLOSS/fund), and security disclosures.

## 3. Decision-Making Process

Day-to-day technical decisions happen through **Lazy Consensus** on GitHub Issues and Pull Requests:
1. Non-trivial changes are proposed as an Issue or RFC Pull Request.
2. Any maintainer may approve and merge a PR once CI passes and review feedback is addressed.
3. For major architectural changes or breaking CLI/HTTP API changes, a proposal remains open for at least **72 hours** to allow all maintainers to review. If a consensus cannot be reached, the TSC resolves the matter by a simple majority vote.

## 4. Becoming a Maintainer

A contributor may be nominated by any existing maintainer after landing at least **5 non-trivial pull requests** and demonstrating sound technical judgment in code reviews. Nominations are approved by lazy consensus among current maintainers within 7 days.
