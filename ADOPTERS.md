# Adopters & Ecosystem Integrations

This file lists organizations, projects, developer tools, and research workflows using or integrating **Swizzle Runtime (`swizzle-rt`)**.

If you use `swizzle-rt` in research, local AI development, edge robotics, or production, please open a Pull Request to add your project or organization below!

## Ecosystem & Tooling Integrations

Because `swizzle-rt` exposes a zero-configuration **OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)**, **llama.cpp-compatible**, and **Ollama-compatible** HTTP wire protocol with deterministic receipt headers (`X-Swizzle-Receipt-Hash`), it acts as a drop-in local inference engine for:

| Project / Workflow | Integration Type | Description |
|---|---|---|
| **Swizzle Gateway & Router** ([`Swizzle-AI/swizzle-gateway`](https://github.com/Swizzle-AI/swizzle-gateway)) | First-Party L7 Gateway | Sub-1ms Go L7 gateway with AST-diff prefix cache routing to `swizzle-rt` workers. |
| **Swizzle Robotics** ([`Swizzle-AI/swizzle-robotics`](https://github.com/Swizzle-AI/swizzle-robotics)) | Edge VLA Runtime | 80.5 Hz visual servoing and VLA policy execution backed by `swizzle-rt`. |
| **OpenAI-Compatible Coding Agents** (Aider, Continue, Open WebUI) | Local GGUF Provider | Drop-in local GGUF inference server via `OPENAI_API_BASE=http://127.0.0.1:8080/v1`. |
