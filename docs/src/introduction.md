# Swizzle Runtime

**Swizzle Runtime: open engine for running AI locally. Runs your GGUF models today,
and .ktc models verified and tuned for your hardware.**

Swizzle Runtime runs language, embedding and reranking models on your own computer,
from one binary with an OpenAI-compatible server. It is written in Rust,
runs on Windows and Linux (x86-64), on the CPU, on NVIDIA GPUs through CUDA
and on AMD and Intel GPUs through Vulkan. Swizzle Runtime
never connects to the network, except when you run `kinetic pull`.

```sh
kinetic run Qwen3.5-4B-Q4_K_M.gguf "Explain recursion in two sentences."
kinetic serve Qwen3.5-4B-Q4_K_M.gguf     # OpenAI API at http://127.0.0.1:8080/v1
```

## Why Swizzle Runtime

- **Your files, unchanged.** Swizzle Runtime reads GGUF files as llama.cpp writes
  them, from Qwen3.5, Gemma 4 and LFM3 to Llama, Gemma 3, Granite and Falcon-H1, in
  every common quantization.
- **Faster without changing the answer.** Speculative decoding, prompt
  caching and a shortcut through the output layer are on by default and
  exact: greedy output is token for token what plain decoding gives, and a
  seeded sampled answer is the same answer for the same seed. There is
  nothing to tune and no draft model to pick. See
  [How Swizzle Runtime is fast](how-it-is-fast.md).
- **Large mixture-of-experts models on small GPUs.** Experts that do not fit
  in video memory stay in system RAM, and Swizzle Runtime decides layer by layer
  where each one runs.
- **Private by design.** Swizzle Runtime never connects to the network, except
  when you run `kinetic pull`. No telemetry, no update checks, no accounts;
  `run`, `chat` and `serve` never download anything. The server listens on
  `127.0.0.1` and refuses requests from web pages.
- **Drop-in for llama.cpp users.** The common flags work as they do in
  llama.cpp, the ones Swizzle Runtime does not need are accepted, and the few with
  no equivalent stop with a line saying what to use instead. See [llama.cpp flags](llama-flags.md).
- **Verifiable models.** A `.ktc` file carries weights, tokenizer and chat
  template with a hash of every byte and an optional signature, and
  `kinetic verify` checks them.

## Where to go next

- [Install](install.md) Swizzle Runtime, then follow the [quick start](quick-start.md).
- Look up a command in the [command line reference](cli.md) or an endpoint
  in the [server API](server.md).
- Check whether your model runs: [Models and formats](models.md).
- Use Swizzle Runtime from [Python, C or Rust](bindings.md).
- Something wrong? [FAQ and troubleshooting](faq.md).

Swizzle Runtime is open source under the Apache License 2.0, made by
[Swizzle AI](https://github.com/Swizzle-AI). The source is at
[github.com/Swizzle-AI/swizzle-runtime](https://github.com/Swizzle-AI/swizzle-runtime).
