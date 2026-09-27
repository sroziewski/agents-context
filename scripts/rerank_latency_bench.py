#!/usr/bin/env python3
"""Rerank latency: local CPU vs remote GPU, network round-trip included (DEV-35).

What is measured vs modelled
----------------------------
- Network RTT to the remote box: measured (ICMP ping).
- Per-request transport over the real SSH channel (the DEV-34 `compute` path uses a
  persistent process, so the per-request cost is one round-trip on an already-open
  connection, not a fresh handshake): measured when the SSH key authenticates, else
  reported as unavailable and modelled from RTT + payload / bandwidth.
- Model compute (CPU and GPU pairs/sec): NOT measured here — there is no reranker built
  yet (DEV-28). Supplied as rates and clearly labelled estimates. Replace them with the
  DEV-28 numbers and re-run; the transport half stays real.

The point is the shape of the trade-off and the crossover N, not false precision.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HOST_DEFAULT = "sroziewski@192.0.2.10"

# --- estimates (edit or pass on the CLI); label them as such in any report ---
# A cross-encoder reranker scores one (query, document) pair. Rates are pairs/second at
# ~512 tokens. Defaults are order-of-magnitude guesses for a 278M-param multilingual model
# (jina-v2) — a 568M model (bge-v2-m3) is roughly half these. Do not quote as measured.
CPU_PAIRS_PER_SEC = 30.0   # 12-core CPU via onnxruntime, batched — GUESS, verify in DEV-28
GPU_PAIRS_PER_SEC = 400.0  # a datacentre GPU, batched — GUESS, verify in DEV-28
BYTES_PER_DOC = 2048       # ~512 tokens of text
QUERY_BYTES = 200
BANDWIDTH_MBIT = 100.0     # fallback only, when the SSH leg cannot be measured


def ping_rtt_ms(host_ip: str) -> float | None:
    try:
        out = subprocess.run(["ping", "-n", "5", host_ip], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"Average = (\d+)ms", out) or re.search(r"min/avg[^=]*= [\d.]+/([\d.]+)", out)
    return float(m.group(1)) if m else None


def ssh_per_request_ms(host: str, key: str, payload_bytes: int, reps: int = 5) -> tuple[float | None, str]:
    """Min round-trip of one request over a reused SSH channel (ControlMaster master socket).

    Sends payload_bytes up, gets a tiny reply, mirroring the compute RPC. Returns
    (ms, note); ms is None when the key does not authenticate.
    """
    sock = str(Path(subprocess.run(["cygpath", "-u", "/tmp"], capture_output=True, text=True).stdout.strip() or "/tmp") / f"ctx_ctl_{time.time_ns()}")
    common = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    master = subprocess.run(["ssh", *common, "-i", key, "-M", "-S", sock, "-fN", host], capture_output=True, text=True, timeout=20)
    if master.returncode != 0:
        return None, master.stderr.strip().splitlines()[-1] if master.stderr.strip() else "ssh master failed"
    try:
        payload = b"x" * payload_bytes
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            subprocess.run(["ssh", *common, "-S", sock, host, "wc -c >/dev/null"], input=payload, capture_output=True, timeout=20)
            dt = (time.perf_counter() - t0) * 1000
            best = dt if best is None else min(best, dt)
        return best, "measured over reused channel"
    finally:
        subprocess.run(["ssh", "-S", sock, "-O", "exit", host], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST_DEFAULT)
    ap.add_argument("--ip", default="192.0.2.10")
    ap.add_argument("--key", default="D:/dev_tools/moba/.ssh/id_rsa")
    ap.add_argument("--cpu-rate", type=float, default=CPU_PAIRS_PER_SEC)
    ap.add_argument("--gpu-rate", type=float, default=GPU_PAIRS_PER_SEC)
    ap.add_argument("--ns", default="10,30,50")
    ap.add_argument("--no-ssh", action="store_true", help="skip the SSH leg, model transport from RTT")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]

    rtt = ping_rtt_ms(args.ip)
    print(f"ICMP RTT to {args.ip}: {rtt:.0f} ms (measured)" if rtt else "ICMP RTT: unreachable")
    rtt = rtt if rtt else 33.0

    # per-request transport at the middle N, measured if the key works
    mid = sorted(ns)[len(ns) // 2]
    payload = QUERY_BYTES + mid * BYTES_PER_DOC
    transport_note = "modelled from RTT + payload/bandwidth"
    if args.no_ssh:
        req_ms = rtt + (payload * 8 / (args.__dict__.get("bw", BANDWIDTH_MBIT) * 1e6)) * 1000
    else:
        measured, note = ssh_per_request_ms(args.host, args.key, payload)
        if measured is not None:
            req_ms, transport_note = measured, f"MEASURED over SSH ({note})"
        else:
            req_ms = rtt + (payload * 8 / (BANDWIDTH_MBIT * 1e6)) * 1000
            transport_note = f"SSH unavailable ({note}); modelled from RTT + {BANDWIDTH_MBIT:.0f} Mbit/s"
    print(f"per-request transport (~{payload/1024:.0f} KB up at N={mid}): {req_ms:.0f} ms ({transport_note})")
    print(f"compute rates (ESTIMATES, verify in DEV-28): CPU {args.cpu_rate:.0f} pairs/s, GPU {args.gpu_rate:.0f} pairs/s\n")

    print(f"{'N':>4} | {'local CPU':>12} | {'remote GPU':>12} | winner")
    print("-" * 52)
    for n in ns:
        local_ms = n / args.cpu_rate * 1000
        remote_ms = req_ms + n / args.gpu_rate * 1000
        win = "remote" if remote_ms < local_ms else "local"
        print(f"{n:>4} | {local_ms:>9.0f} ms | {remote_ms:>9.0f} ms | {win}")

    # crossover: local_ms == remote_ms  ->  N/cpu = req + N/gpu
    denom = 1000 / args.cpu_rate - 1000 / args.gpu_rate
    if denom > 0:
        n_star = req_ms / denom
        print(f"\ncrossover: remote wins for N >= {n_star:.1f} candidates (under these estimates)")
    else:
        print("\ncrossover: CPU rate >= GPU rate — check the estimates")
    print("\nNote: transport is real once the SSH key authenticates; compute is modelled until DEV-28.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
