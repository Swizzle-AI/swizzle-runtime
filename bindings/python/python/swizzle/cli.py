"""Python CLI launcher for `swizzle-engine` (`swizzle-py`)."""

import argparse
import sys

from . import Model, __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="swizzle-py",
        description=f"Swizzle Runtime Python CLI (v{__version__})",
    )
    parser.add_argument("--version", action="version", version=f"swizzle-engine {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run a single prompt against a local .gguf or .ktc model")
    run_p.add_argument("model", help="Path to the .gguf or .ktc model")
    run_p.add_argument("-p", "--prompt", required=True, help="Prompt string")
    run_p.add_argument("-n", "--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    run_p.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        m = Model(args.model)
        for piece in m.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temp):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
