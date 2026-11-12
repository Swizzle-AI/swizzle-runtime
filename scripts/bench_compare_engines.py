#!/usr/bin/env python3
"""Head-to-head real-hardware benchmark & parity harness for swizzle-rt.

Runs `<engine> bench <model> --device <cpu|cuda> --repetitions <N> --json`
sequentially against an optional comparison binary while monitoring GPU
temperature via `nvidia-smi` so laptop/workstation hardware stays cool.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def gpu_telemetry() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 3:
        return None
    return {
        "temp_c": float(parts[0]),
        "power_w": float(parts[1]) if parts[1] not in ("", "[N/A]") else None,
        "vram_mib": float(parts[2]),
    }


def wait_for_cool_gpu(max_temp_c: float, cooldown_secs: float) -> dict[str, Any] | None:
    if cooldown_secs > 0:
        time.sleep(cooldown_secs)
    deadline = time.monotonic() + 120.0
    while True:
        tel = gpu_telemetry()
        if tel is None or tel["temp_c"] <= max_temp_c:
            return tel
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU stayed at {tel['temp_c']:.1f}C (> {max_temp_c:.1f}C limit); aborting for safety."
            )
        time.sleep(3.0)


def run_bench(
    binary: str,
    model: str,
    device: str,
    repetitions: int,
    max_temp_c: float,
    cooldown_secs: float,
) -> dict[str, Any]:
    pre = wait_for_cool_gpu(max_temp_c, cooldown_secs)
    cmd = [
        binary,
        "bench",
        model,
        "--device",
        device,
        "--repetitions",
        str(repetitions),
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    post = gpu_telemetry()
    if not data.get("edit_identical", False):
        raise RuntimeError(f"{binary} failed edit_identical on {model} ({device})")
    return {"bench": data, "gpu_before": pre, "gpu_after": post}


def check_parity_and_telemetry(
    swizzle_bin: str,
    baseline_bin: str | None,
    model: str,
    device: str,
    port_base: int,
) -> dict[str, Any]:
    prompt = "Explain why cache-aligned tile strides improve dense GEMM efficiency in 3 concise points:"

    def query_server(binary: str, port: int) -> tuple[str, dict[str, str]]:
        proc = subprocess.Popen(
            [binary, "serve", model, "--device", device, "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            health_url = f"http://127.0.0.1:{port}/health"
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                        if resp.status == 200:
                            break
                except OSError:
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"server {binary} did not become healthy on port {port}")

            payload = json.dumps(
                {
                    "prompt": prompt,
                    "max_tokens": 48,
                    "temperature": 0.0,
                    "top_k": 1,
                    "seed": 42,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["text"]
            return text, hdrs
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    swizzle_text, swizzle_hdrs = query_server(swizzle_bin, port_base)
    required_headers = [
        "x-swizzle-backend",
        "x-swizzle-ttft-ms",
        "x-swizzle-tokens-per-sec",
        "x-swizzle-gen-tokens",
    ]
    missing = [h for h in required_headers if h not in swizzle_hdrs]
    if missing:
        raise RuntimeError(f"missing telemetry headers in swizzle-rt response: {missing}")

    result: dict[str, Any] = {
        "model": Path(model).name,
        "device": device,
        "telemetry_headers": {h: swizzle_hdrs[h] for h in required_headers},
        "swizzle_preview": swizzle_text[:120],
    }
    if baseline_bin:
        base_text, _ = query_server(baseline_bin, port_base + 1)
        exact = swizzle_text == base_text
        result["greedy_parity"] = exact
        if not exact:
            result["baseline_preview"] = base_text[:120]
    return result


def pct_delta(new: float, old: float) -> str:
    if old <= 0:
        return "n/a"
    return f"{((new - old) / old) * 100.0:+.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swizzle", required=True, help="Path to swizzle-rt binary")
    ap.add_argument("--baseline", help="Optional baseline binary for A/B comparison")
    ap.add_argument("--model", action="append", required=True, help="GGUF model path (repeatable)")
    ap.add_argument("--device", action="append", choices=["cuda", "cpu"], help="Devices (default: cuda cpu)")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--max-temp-c", type=float, default=75.0)
    ap.add_argument("--cooldown-secs", type=float, default=2.0)
    ap.add_argument("--verify-server", action="store_true")
    ap.add_argument("--port-base", type=int, default=18080)
    ap.add_argument("--out", help="Optional path to write full JSON report")
    args = ap.parse_args()

    devices = args.device or ["cuda", "cpu"]
    rows: list[dict[str, Any]] = []
    parity_checks: list[dict[str, Any]] = []

    for model in args.model:
        for dev in devices:
            base_res = None
            if args.baseline:
                print(f"[bench] baseline {Path(model).name} ({dev})...", file=sys.stderr, flush=True)
                base_res = run_bench(
                    args.baseline, model, dev, args.repetitions, args.max_temp_c, args.cooldown_secs
                )
            print(f"[bench] swizzle-rt {Path(model).name} ({dev})...", file=sys.stderr, flush=True)
            swiz_res = run_bench(
                args.swizzle, model, dev, args.repetitions, args.max_temp_c, args.cooldown_secs
            )
            rows.append(
                {
                    "model": Path(model).name,
                    "device": dev,
                    "baseline": base_res,
                    "swizzle": swiz_res,
                }
            )
            if args.verify_server:
                print(f"[server] parity+telemetry {Path(model).name} ({dev})...", file=sys.stderr, flush=True)
                parity_checks.append(
                    check_parity_and_telemetry(
                        args.swizzle, args.baseline, model, dev, args.port_base
                    )
                )

    report = {"benchmarks": rows, "server_checks": parity_checks}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("| Model | Device | Test | Baseline | Swizzle | Delta | Edit Exact |")
    print("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for r in rows:
        sb = r["swizzle"]["bench"]
        bb = r["baseline"]["bench"] if r["baseline"] else None
        for label, key in [
            ("pp512", "prefill_tokens_per_second"),
            ("tg128", "decode_tokens_per_second"),
            ("edit256", "edit_tokens_per_second"),
        ]:
            sv = sb[key]
            bv = bb[key] if bb else 0.0
            delta = pct_delta(sv, bv) if bb else "n/a"
            bstr = f"{bv:.2f}" if bb else "-"
            exact = "yes" if sb.get("edit_identical") else "NO"
            print(f"| {r['model']} | {r['device']} | {label} | {bstr} | {sv:.2f} | {delta} | {exact} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

