"""CLI for Swizzle Weight-Space Programming (`swizzle-wsp`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inspector import WeightInspector


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="swizzle-wsp",
        description="Swizzle Weight-Space Programming: Inspect, Audit & Surgically Patch Neural Weights",
    )
    sub = parser.add_subparsers(dest="cmd")

    ins = sub.add_parser("inspect", help="Inspect .safetensors weights and print spectral/outlier digest")
    ins.add_argument("path", type=str, help="Path to .safetensors file")
    ins.add_argument("--top", type=int, default=8, help="Top tensors sorted by kurtosis")

    args = parser.parse_args()
    if args.cmd == "inspect":
        digests = WeightInspector.inspect_safetensors(Path(args.path))
        digests.sort(key=lambda d: d.kurtosis, reverse=True)
        out = [
            {
                "name": d.name,
                "shape": d.shape,
                "kurtosis": d.kurtosis,
                "max_abs": d.max_abs,
                "sv_top3": d.top_singular_values[:3],
                "outlier_rows": d.outlier_row_indices[:6],
            }
            for d in digests[: args.top]
        ]
        print(json.dumps(out, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
