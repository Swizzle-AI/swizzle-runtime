# Good first issues (drafts)

Drafts to file on GitHub with the `good first issue` label. Each is small,
self-contained and has a clear way to check it is done. Paths are relative to
the repository root.

## 1. `swizzle-rt info`: recognize vision projectors

`swizzle-rt info Qwen3.5-4B-mmproj-F16.gguf` prints `architecture clip (not
supported yet)` and a tokenizer error, although Swizzle Runtime uses the file with
`swizzle-rt serve --mmproj`. Print that it is a vision projector, its projector
type and whether Swizzle Runtime supports it (`clip.projector_type`), and skip the
tokenizer line.

- Where: `crates/swizzle-cli/src/info.rs`, `crates/swizzle-vision/src/encoder.rs`
- Done when: the mmproj file shows `vision projector qwen3vl_merger` and no
  error line.

## 2. `swizzle-rt info`: say whether the model loads, and where

`info` only checks the architecture name, so a file can show as supported
and still fail to load (rotary scaling, a pre-tokenizer). Add a line that
builds the model and reports `loads: yes` or the load error, and with the
`cuda` feature whether it would run on the GPU.

- Where: `crates/swizzle-cli/src/info.rs`
- Done when: a Qwen2.5 file shows that it runs on the CPU only, with the reason.

## 3. `swizzle-rt info --json`

The same facts as one JSON object, for scripts and issue reports, as
`swizzle-rt bench --json` does.

- Where: `crates/swizzle-cli/src/info.rs`, `main.rs`
- Done when: the output parses with `python -m json.tool` and has a test.

## 4. `swizzle-rt run --stop <TEXT>`

The engine already supports stop strings (the server's `stop` field). Add a
repeatable `--stop` flag to `run` and `chat`.

- Where: `crates/swizzle-cli/src/main.rs` (`GenArgs::request`)
- Done when: `swizzle-rt run m.gguf "Count to ten" --stop 5` ends before 5.

## 5. `-f`/`--file` for `swizzle-rt run`

llama.cpp users type `-f prompt.txt`; Swizzle Runtime answers that they should pipe
the prompt in. Accept the flag and read the file instead, and remove it from
the list of unsupported flags.

- Where: `crates/swizzle-cli/src/main.rs`, `llama_flags.rs`, then
  `python scripts/llama_flags_doc.py`
- Done when: `swizzle-rt run m.gguf -f prompt.txt` works and the docs table is
  regenerated.

## 6. Constrained output in `swizzle-rt run`

The server takes `grammar` and `response_format`; the command line does not.
Add `--grammar-file <FILE>` and `--json-schema <JSON>` (llama.cpp's names),
built with `swizzle_grammar::Grammar` as `crates/swizzle-server/src/api.rs`
does.

- Where: `crates/swizzle-cli/src/main.rs`, `llama_flags.rs`
- Done when: `swizzle-rt run m.gguf "Is the sky blue?" --grammar-file yesno.gbnf`
  prints `yes` or `no`.

## 7. Python: more sampling arguments

`Model.generate` and `Model.chat` take `max_tokens`, `temperature` and
`seed`. Add `top_k`, `top_p`, `min_p`, `repeat_penalty` and `stop`.

- Where: `bindings/python/src/lib.rs`, `bindings/python/tests/test_model.py`
- Done when: the new arguments are tested and listed in `bindings/python/README.md`.

## 8. Python: model properties

Expose `context_length`, `is_embedding` and `embedding_size` on
`swizzle.Model`; the Rust crate has them.

- Where: `bindings/python/src/lib.rs`
- Done when: each has a test.

## 9. C API: start a session over

Add `swizzle_session_clear(session)`, which forgets the turns and keeps the
system prompt, so a caller need not free and recreate sessions.

- Where: `crates/swizzle-capi/src/lib.rs`, `include/swizzle.h`, `tests/api.rs`
- Done when: tested, and documented in the header.

## 10. `swizzle-rt bench -p` and `-n`

Let `bench` take the prompt and generation lengths, as `llama-bench -p 512
-n 128` does, defaulting to today's 512 and 128.

- Where: `crates/swizzle-cli/src/bench.rs`
- Done when: `swizzle-rt bench m.gguf -p 1024 -n 64` prints `pp1024` and `tg64`.

## 11. Warn about `--device` with embedding models

Embedding and reranking models always run on the CPU, but `swizzle-rt serve
bge.gguf --device cuda` silently ignores the flag. Fail for `cuda`, as text
models do when the GPU is not usable.

- Where: `crates/swizzle-cli/src/main.rs` (`Command::Serve`)
- Done when: the command exits with 1 and a clear message.

## 12. A new BPE pre-tokenizer

Files whose `tokenizer.ggml.pre` Swizzle Runtime does not know fail to load. Add one
that a popular model uses (for example `deepseek-v3`, `tekken` or
`gpt-4o`), with its split pattern.

- Where: `crates/swizzle-tokenizer/src/lib.rs`, `pretokenize.rs`
- Done when: a test compares the token ids of a few lines of mixed text,
  numbers and code with `llama-tokenize` on the same file.

## 13. Docker: keep Warm Start's cache

The images run as a system user made with `--no-create-home`, so the
default cache directory is probably not writable and Warm Start falls back
to memory, with a warning at start-up (not checked in a container yet). Give the user a writable cache
directory (for example a `/cache` volume passed as `--cache-dir`).

- Where: `docker/Dockerfile`, `docker/Dockerfile.cuda`, `docs/src/install.md`
- Done when: `docker run ... -v swizzle-cache:/cache` keeps states across
  restarts, without a warning.

## 14. Shell completions

`swizzle completions bash|zsh|fish|powershell` printing a completion
script, with `clap_complete` (MIT or Apache-2.0).

- Where: `crates/swizzle-cli/src/main.rs`
- Done when: the scripts complete subcommands and flags, and the docs say how
  to install them.

## 15. `swizzle-rt chat`: save a conversation

`/save FILE` writes the conversation as JSON messages, in the shape the
server takes, so a chat can be replayed with `curl`.

- Where: `crates/swizzle-cli/src/chat.rs`
- Done when: the saved file sent to `/v1/chat/completions` continues the
  conversation.

## 16. Fused QKV weights (Phi-3 and others)

Some transformers store one `attn_qkv.weight` instead of three matrices;
`arch/transformer.rs` rejects them. Split the fused matrix into the three
views at load time.

- Where: `crates/swizzle-models/src/arch/transformer.rs`, `loader.rs`
- Done when: a Phi-3 mini GGUF loads, matches llama.cpp's greedy output for
  the first 32 tokens, and passes the `architectures` identity tests.

## 17. Docs: using Swizzle Runtime from editors and chat apps

A page with tested settings for a few popular OpenAI-compatible clients
(an editor assistant, a chat UI, an agent framework): base URL, model name,
what works and what does not.

- Where: `docs/src/`, `docs/src/SUMMARY.md`
- Done when: each setting was tried against `swizzle-rt serve`.

## 18. `/v1/models`: more about the model

Add the architecture, the device and whether the server takes images to the
`/v1/models` entry (as extra fields, which OpenAI clients ignore).

- Where: `crates/swizzle-server/src/server.rs` (`models`)
- Done when: covered by `crates/swizzle-server/tests/server.rs`.
