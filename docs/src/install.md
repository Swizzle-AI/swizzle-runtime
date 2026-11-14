# Install

Swizzle Runtime runs on Windows 10 and 11 and on Linux, on x86-64. Pick one way to
install it.

## Install script

```powershell
irm https://raw.githubusercontent.com/Swizzle-AI/swizzle-runtime/main/scripts/install.ps1 | iex
```

```sh
curl -fsSL https://raw.githubusercontent.com/Swizzle-AI/swizzle-runtime/main/scripts/install.sh | sh
```

The scripts download the latest release (the full build, which runs on NVIDIA
GPUs through CUDA, on AMD and Intel GPUs through Vulkan and on the CPU), check it against the release's `SHA256SUMS` and install it for the
current user: into `%LOCALAPPDATA%\Programs\swizzle`, put on the user `PATH`, on
Windows, and into `~/.local/bin` on Linux. `-Version 0.1.0` / `--version 0.1.0`
picks a release and `-Cpu` / `--cpu` the smaller CPU-only build.

## Release binaries

From the [releases page](https://github.com/Swizzle-AI/swizzle-runtime/releases),
take the archive for your system:

| Archive | For |
|---|---|
| `swizzle-<version>-windows-x86_64-cuda.zip` | Windows, with NVIDIA GPU support |
| `swizzle-<version>-windows-x86_64-cpu.zip` | Windows, CPU only |
| `swizzle-<version>-linux-x86_64-cuda.tar.gz` | Linux, with NVIDIA GPU support |
| `swizzle-<version>-linux-x86_64-cpu.tar.gz` | Linux, CPU only |

The `cuda` archives also run on machines without a GPU, on the CPU; they are
the right choice unless you want the smallest download. They need only the
NVIDIA driver, not the CUDA toolkit.

Each archive holds the `swizzle` binary, the C library and header
(`lib/`, `include/`), the license files and a `SHA256SUMS` of its contents.
The release also has a `SHA256SUMS` for the archives themselves. Check the
download before you unpack it:

```sh
# Linux
sha256sum -c SHA256SUMS --ignore-missing
tar -xzf swizzle-0.1.0-linux-x86_64-cuda.tar.gz
sudo install swizzle-0.1.0-linux-x86_64-cuda/swizzle /usr/local/bin/
```

```powershell
# Windows (PowerShell): compare the hash with the line in SHA256SUMS
Get-FileHash .\swizzle-0.1.0-windows-x86_64-cuda.zip -Algorithm SHA256
Expand-Archive .\swizzle-0.1.0-windows-x86_64-cuda.zip -DestinationPath $env:LOCALAPPDATA\Programs
```

Then put the unpacked folder on your `PATH` and check:

```sh
swizzle --version
```

On Windows, SmartScreen may warn about a new download the first time you run
it; see [FAQ](faq.md#windows-says-the-file-is-not-commonly-downloaded).

## Python

```sh
pip install swizzle-engine
```

Wheels are built for Windows and Linux (manylinux 2.28) on x86-64, for
Python 3.9 and newer. They run models on the CPU; to use an NVIDIA GPU from
Python, [build the package from source](building.md#the-libraries) with the
CUDA toolkit installed, or run `swizzle-rt serve` and use any OpenAI client.
See [Python, C and Rust](bindings.md) for the API.

## Docker

The images run `swizzle-rt serve` on port 8080 as an unprivileged user, with
your models folder mounted read-only. Build them from a clone of the
repository:

```sh
docker build -f docker/Dockerfile -t swizzle .
docker run --rm -p 8080:8080 -v "$PWD/models:/models:ro" swizzle /models/model.gguf
```

For NVIDIA GPUs, with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host:

```sh
docker build -f docker/Dockerfile.cuda -t swizzle:cuda .
docker run --rm --gpus all -p 8080:8080 -v "$PWD/models:/models:ro" swizzle:cuda /models/model.gguf
```

Inside the container the server listens on all interfaces, so that
`-p 8080:8080` reaches it; `-p 127.0.0.1:8080:8080` keeps it on your own
machine. Arguments after the image name go to `swizzle-rt serve`, so
`swizzle /models/model.gguf -c 8192` works as on the command line.

## From source

With Rust 1.91 or newer:

```sh
git clone https://github.com/Swizzle-AI/swizzle-runtime
cargo install --path swizzle/crates/swizzle-cli
```

For GPU support, install the CUDA toolkit 12.8 or newer first. See
[Building from source](building.md).

## Get a model

Any `.gguf` file of a [supported architecture](models.md) works; `run`,
`chat` and `serve` only read files from your disk. Download one from Hugging
Face with `swizzle-rt pull <org>/<repo>` (see [pull](cli.md#pull)) or your
browser. Models Ollama or LM Studio already downloaded run by name; see
[models](cli.md#models).
To try it out:

| Model | Size | Good for |
|---|---:|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF) | 0.3 GB | checking that everything works |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | 0.6 GB | a small model that thinks and calls tools |
| [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF) | 0.7 GB | fast answers on a laptop CPU |
| [Qwen3.5-4B](https://huggingface.co/models?search=Qwen3.5-4B%20GGUF) Q4_K_M | 2.8 GB | a capable everyday model, on a GPU with 4 GB or more |

`swizzle-rt info model.gguf` shows what a file contains and whether Swizzle Runtime
supports it.
