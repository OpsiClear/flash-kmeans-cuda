"""One-shot metric extractor for the auto-tune loop.

Runs FKC_AUTOTUNE_VERBOSE=1 once over the target D set, then a stabilized
bench. Reports the SUM-OF-TFLOPS across all target D values as the single
optimization metric (higher = better).

Output line is exactly: TOTAL_TFLOPS=<float>

The verbose probe lines are logged to stderr; the only thing on stdout is
the TOTAL_TFLOPS=... line, so the auto-tune loop can grep it directly.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

D_LIST = [128, 192, 224, 256, 320, 384]
N = 32768
K = 8192
ROUNDS = 50
WARMUP = 10
OUTER = 3


def main() -> int:
    env = os.environ.copy()
    env.pop("FKC_AUTOTUNE_VERBOSE", None)

    cmd = [
        sys.executable,
        "benchmarks/bench_d_sweep.py",
        "--n", str(N),
        "--k", str(K),
        "--d", *[str(d) for d in D_LIST],
        "--rounds", str(ROUNDS),
        "--warmup", str(WARMUP),
        "--outer", str(OUTER),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"BENCH_FAILED rc={proc.returncode}\n{proc.stderr[-2000:]}\n")
        return 1

    line_re = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+")
    tflops_by_d: dict[int, float] = {}
    for line in proc.stdout.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        d = int(m.group(1))
        if d in D_LIST:
            tflops_by_d[d] = float(m.group(3))

    missing = [d for d in D_LIST if d not in tflops_by_d]
    if missing:
        sys.stderr.write(f"MISSING_D={missing}\n{proc.stdout}\n")
        return 2

    total = sum(tflops_by_d[d] for d in D_LIST)
    sys.stderr.write("Per-D TFLOPS: " + ", ".join(
        f"D{d}={tflops_by_d[d]:.1f}" for d in D_LIST
    ) + "\n")
    print(f"TOTAL_TFLOPS={total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
