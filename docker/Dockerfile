# Swizzle Runtime on the CPU, serving an OpenAI-compatible API on port 8080.
#
#   docker build -f docker/Dockerfile -t swizzle-rt .
#   docker run --rm -p 8080:8080 -v "$PWD/models:/models:ro" swizzle-rt /models/model.gguf

FROM rust:1.96-bookworm AS build
WORKDIR /src
# Only the sources: rust-toolchain.toml would make rustup fetch another toolchain.
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN cargo build --release --locked -p swizzle-cli --no-default-features

FROM debian:bookworm-slim
COPY --from=build /src/target/release/swizzle-rt /usr/local/bin/swizzle-rt
COPY LICENSE NOTICE README.md /usr/share/doc/swizzle-runtime/
RUN useradd --system --uid 10001 --no-create-home swizzle
USER swizzle
VOLUME /models
EXPOSE 8080
ENTRYPOINT ["swizzle-rt", "serve", "--host", "0.0.0.0", "--port", "8080"]
CMD ["/models/model.gguf"]
