#!/usr/bin/env sh
set -eu

VERSION="${SWIZZLE_VERSION:-v0.2.0}"
REPO="Swizzle-AI/swizzle-runtime"
INSTALL_DIR="${SWIZZLE_INSTALL_DIR:-$HOME/.local/bin}"

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
    echo "Prebuilt binaries currently target Linux x86_64 (CUDA 12.8)."
    echo "Build from source on $OS $ARCH with:"
    echo "  git clone https://github.com/$REPO.git && cd swizzle-runtime && cargo build --release --locked -p swizzle-cli"
    exit 1
fi

ARCHIVE="swizzle-rt-${VERSION}-linux-x86_64-cuda12.8.tar.gz"
BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading Swizzle Runtime ${VERSION} (${ARCHIVE})..."
curl -fsSL "${BASE_URL}/${ARCHIVE}" -o "${TMP_DIR}/${ARCHIVE}"

if command -v sha256sum >/dev/null 2>&1; then
    echo "Verifying SHA256 checksum..."
    curl -fsSL "${BASE_URL}/SHA256SUMS.txt" -o "${TMP_DIR}/SHA256SUMS.txt"
    (cd "$TMP_DIR" && sha256sum -c SHA256SUMS.txt --ignore-missing)
fi

mkdir -p "$INSTALL_DIR"
tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "$TMP_DIR"

BIN_PATH="$(find "$TMP_DIR" -type f -name 'swizzle-rt' | head -n 1)"
if [ -z "$BIN_PATH" ]; then
    echo "Error: swizzle-rt binary not found in archive."
    exit 1
fi

install -m 0755 "$BIN_PATH" "${INSTALL_DIR}/swizzle-rt"
echo "Installed swizzle-rt to ${INSTALL_DIR}/swizzle-rt"

case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        echo "Add ${INSTALL_DIR} to your PATH:"
        echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
        ;;
esac

echo "Run 'swizzle-rt --help' to get started."
