class SwizzleRt < Formula
  desc "Exact, high-throughput pure-Rust inference engine for GGUF, Safetensors & .ktc models"
  homepage "https://github.com/Swizzle-AI/swizzle-runtime"
  version "0.2.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/Swizzle-AI/swizzle-runtime/releases/download/v0.2.0/swizzle-rt-0.2.0-macos-arm64-cpu.tar.gz"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/Swizzle-AI/swizzle-runtime/releases/download/v0.2.0/swizzle-rt-0.2.0-linux-x86_64-cuda.tar.gz"
    end
  end

  def install
    bin.install "swizzle-rt"
    include.install "include/swizzle.h" if File.exist?("include/swizzle.h")
  end

  test do
    assert_match "swizzle-rt", shell_output("#{bin}/swizzle-rt --version")
  end
end
