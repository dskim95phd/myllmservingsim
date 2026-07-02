#!/usr/bin/env python3
"""Run Qwen3-32B sparse-LPDDR simulator sweeps.

This script generates cluster configs for three comparisons:

* dense_cpu: dense attention, existing CPU fallback
* sparse_cpu: sparse attention, no LPDDR tier
* sparse_lpddr: sparse attention, per-NPU LPDDR tier

Run it from the repository root, inside the simulator environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean


DEFAULT_HBM_SIZES = "40,44,48,56,64"
DEFAULT_LPDDR_SIZES = "0,32,64,128"
DEFAULT_SPARSE_KS = "128,256,512"
DEFAULT_MODES = "dense_cpu,sparse_cpu,sparse_lpddr"


def _parse_csv_numbers(value: str, cast):
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(cast(item))
    return out


def _mean_column(rows: list[dict[str, str]], name: str) -> float | None:
    values = []
    for row in rows:
        raw = row.get(name)
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return mean(values) if values else None


def _sum_column(rows: list[dict[str, str]], name: str) -> float:
    total = 0.0
    for row in rows:
        raw = row.get(name)
        if raw in (None, ""):
            continue
        try:
            total += float(raw)
        except ValueError:
            continue
    return total


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _label(value) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _base_config(args: argparse.Namespace) -> dict:
    return {
        "num_nodes": 1,
        "link_bw": args.link_bw,
        "link_latency": args.link_latency,
        "nodes": [
            {
                "num_instances": 1,
                "cpu_mem": {
                    "mem_size": args.cpu_mem_size,
                    "mem_bw": args.cpu_mem_bw,
                    "mem_latency": args.cpu_mem_latency,
                },
                "instances": [
                    {
                        "model_name": args.model,
                        "hardware": args.hardware,
                        "npu_mem": {
                            "mem_size": args.hbm_sizes[0],
                            "mem_bw": args.npu_mem_bw,
                            "mem_latency": args.npu_mem_latency,
                        },
                        "num_npus": args.num_npus,
                        "tp_size": args.tp_size,
                        "pd_type": None,
                        "enable_prefix_caching": False,
                        "enable_sparse_attention": True,
                        "sparse_k": args.sparse_ks[0],
                        "sparse_selection_policy": args.sparse_selection_policy,
                        "kv_placement_policy": args.kv_placement_policy,
                    }
                ],
            }
        ],
    }


def _iter_cases(args: argparse.Namespace):
    for mode in args.modes:
        for hbm in args.hbm_sizes:
            for sparse_k in args.sparse_ks:
                for lpddr in args.lpddr_sizes:
                    if mode != "sparse_lpddr" and lpddr != 0:
                        continue
                    if mode == "dense_cpu" and sparse_k != args.sparse_ks[0]:
                        continue
                    yield {
                        "mode": mode,
                        "hbm": hbm,
                        "lpddr": lpddr,
                        "sparse_k": sparse_k,
                    }


def _write_case_config(args: argparse.Namespace, case: dict) -> Path:
    cfg = _base_config(args)
    inst = cfg["nodes"][0]["instances"][0]
    inst["npu_mem"]["mem_size"] = case["hbm"]
    inst["sparse_k"] = case["sparse_k"]

    mode = case["mode"]
    if mode == "dense_cpu":
        inst["enable_sparse_attention"] = False
        inst.pop("lpddr_mem", None)
    elif mode == "sparse_cpu":
        inst["enable_sparse_attention"] = True
        inst.pop("lpddr_mem", None)
    elif mode == "sparse_lpddr":
        inst["enable_sparse_attention"] = True
        inst["lpddr_mem"] = {
            "mem_size": case["lpddr"],
            "mem_bw": args.lpddr_mem_bw,
            "mem_latency": args.lpddr_mem_latency,
            "hbm_link_bw": args.hbm_lpddr_bw,
        }
    else:
        raise ValueError(f"unknown mode: {mode}")

    args.config_dir.mkdir(parents=True, exist_ok=True)
    hbm_label = _label(case["hbm"])
    lpddr_label = _label(case["lpddr"])
    path = args.config_dir / (
        f"qwen3_{mode}_hbm{hbm_label}_lpddr{lpddr_label}_k{case['sparse_k']}.json"
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return path


def _output_path(args: argparse.Namespace, case: dict) -> Path:
    hbm_label = _label(case["hbm"])
    lpddr_label = _label(case["lpddr"])
    return args.output_dir / (
        f"qwen3_{case['mode']}_hbm{hbm_label}_lpddr{lpddr_label}_k{case['sparse_k']}.csv"
    )


def _log_path(output_path: Path) -> Path:
    return output_path.with_suffix(".log")


def _run_case(args: argparse.Namespace, config_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "serving",
        "--cluster-config",
        str(config_path),
        "--dtype",
        args.dtype,
        "--block-size",
        str(args.block_size),
        "--dataset",
        args.dataset,
        "--output",
        str(output_path),
        "--num-req",
        str(args.num_req),
        "--log-level",
        args.log_level,
    ]
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0

    if args.jobs > 1:
        log_path = _log_path(output_path)
        print(f"  log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            return subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT).returncode

    return subprocess.run(cmd).returncode


def _run_case_item(args: argparse.Namespace, item: tuple[dict, Path, Path]) -> tuple[dict, int]:
    case, config_path, output_path = item
    print(
        f"RUN mode={case['mode']} hbm={case['hbm']}GB "
        f"lpddr={case['lpddr']}GB k={case['sparse_k']}",
        flush=True,
    )
    return case, _run_case(args, config_path, output_path)


def summarize(args: argparse.Namespace) -> Path:
    rows = []
    for request_csv in sorted(args.output_dir.glob("qwen3_*.csv")):
        if request_csv.name.endswith(".sparse.csv") or request_csv.name == "summary.csv":
            continue

        parts = request_csv.stem.split("_")
        mode = "_".join(parts[1:-3])
        hbm = parts[-3].removeprefix("hbm")
        lpddr = parts[-2].removeprefix("lpddr")
        sparse_k = parts[-1].removeprefix("k")

        req_rows = _read_csv(request_csv)
        sparse_rows = _read_csv(request_csv.with_suffix(".sparse.csv"))
        row = {
            "run": request_csv.stem,
            "mode": mode,
            "hbm_gb": hbm,
            "lpddr_gb": lpddr,
            "sparse_k": sparse_k,
            "request_rows": str(len(req_rows)),
            "sparse_rows": str(len(sparse_rows)),
        }

        for src, dst in (("TTFT", "ttft_ms"), ("TPOT", "tpot_ms"), ("latency", "latency_ms")):
            value = _mean_column(req_rows, src)
            row[dst] = "" if value is None else f"{value / 1e6:.6f}"

        if sparse_rows:
            hbm_hit_rate = _mean_column(sparse_rows, "hbm_hit_rate")
            row.update(
                {
                    "hbm_hit_rate": "" if hbm_hit_rate is None else f"{hbm_hit_rate:.6f}",
                    "lpddr_hit_blocks": f"{_sum_column(sparse_rows, 'lpddr_hit_blocks'):.0f}",
                    "cpu_hit_blocks": f"{_sum_column(sparse_rows, 'cpu_hit_blocks'):.0f}",
                    "lpddr_promote_gb": f"{_sum_column(sparse_rows, 'lpddr_to_hbm_promotion_bytes') / 1e9:.6f}",
                    "hbm_to_lpddr_gb": f"{_sum_column(sparse_rows, 'hbm_to_lpddr_eviction_bytes') / 1e9:.6f}",
                    "copy_time_ms": f"{_sum_column(sparse_rows, 'copy_time_ns') / 1e6:.6f}",
                }
            )
        rows.append(row)

    summary_path = args.output_dir / "summary.csv"
    if not rows:
        print(f"No completed request CSVs found under {args.output_dir}")
        return summary_path

    fieldnames = [
        "run",
        "mode",
        "hbm_gb",
        "lpddr_gb",
        "sparse_k",
        "request_rows",
        "sparse_rows",
        "ttft_ms",
        "tpot_ms",
        "latency_ms",
        "hbm_hit_rate",
        "lpddr_hit_blocks",
        "cpu_hit_blocks",
        "lpddr_promote_gb",
        "hbm_to_lpddr_gb",
        "copy_time_ms",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--hardware", default=os.environ.get("HARDWARE", "RTXPRO6000"))
    p.add_argument("--dataset", default="workloads/sharegpt-qwen3-32b-300-sps10.jsonl")
    p.add_argument("--config-dir", type=Path, default=Path("configs/cluster/generated/qwen3_lpddr"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/sweep/qwen3_lpddr"))
    p.add_argument("--num-req", type=int, default=300)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-npus", type=int, default=2)
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--hbm-sizes", default=DEFAULT_HBM_SIZES)
    p.add_argument("--lpddr-sizes", default=DEFAULT_LPDDR_SIZES)
    p.add_argument("--sparse-ks", default=DEFAULT_SPARSE_KS)
    p.add_argument("--modes", default=DEFAULT_MODES)
    p.add_argument("--npu-mem-bw", type=float, default=1597)
    p.add_argument("--npu-mem-latency", type=float, default=0)
    p.add_argument("--cpu-mem-size", type=float, default=512)
    p.add_argument("--cpu-mem-bw", type=float, default=256)
    p.add_argument("--cpu-mem-latency", type=float, default=0)
    p.add_argument("--lpddr-mem-bw", type=float, default=1024)
    p.add_argument("--lpddr-mem-latency", type=float, default=300)
    p.add_argument("--hbm-lpddr-bw", type=float, default=2048)
    p.add_argument("--link-bw", type=float, default=16)
    p.add_argument("--link-latency", type=float, default=20000)
    p.add_argument("--sparse-selection-policy", default="hybrid")
    p.add_argument("--kv-placement-policy", default="lfu_hotness")
    p.add_argument("--log-level", default="WARNING")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of simulator runs to execute concurrently. "
            "Use >1 on multicore CPU servers; parallel logs are written next to each CSV."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Generate configs and print commands without running.")
    p.add_argument("--summary-only", action="store_true", help="Only rebuild summary.csv from existing outputs.")
    p.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed simulator run.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.hbm_sizes = _parse_csv_numbers(args.hbm_sizes, float)
    args.lpddr_sizes = _parse_csv_numbers(args.lpddr_sizes, float)
    args.sparse_ks = _parse_csv_numbers(args.sparse_ks, int)
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.jobs = max(1, args.jobs)

    if args.summary_only:
        summarize(args)
        return 0

    items = []
    for case in _iter_cases(args):
        config_path = _write_case_config(args, case)
        output_path = _output_path(args, case)
        items.append((case, config_path, output_path))

    failures = []
    if args.jobs == 1 or args.dry_run:
        for item in items:
            case, rc = _run_case_item(args, item)
            if rc != 0:
                failures.append((case, rc))
                print(f"FAILED rc={rc}: {case}", flush=True)
                if args.stop_on_error:
                    break
    else:
        print(f"Running {len(items)} case(s) with jobs={args.jobs}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_item = {
                executor.submit(_run_case_item, args, item): item for item in items
            }
            for future in concurrent.futures.as_completed(future_to_item):
                case, _config_path, output_path = future_to_item[future]
                try:
                    completed_case, rc = future.result()
                except Exception as exc:  # pragma: no cover - defensive for runner failures
                    completed_case, rc = case, 1
                    print(f"FAILED exception={exc!r}: {case}", flush=True)
                if rc != 0:
                    failures.append((completed_case, rc))
                    print(
                        f"FAILED rc={rc}: {completed_case}; see {_log_path(output_path)}",
                        flush=True,
                    )
                    if args.stop_on_error:
                        for pending in future_to_item:
                            pending.cancel()
                        break

    if not args.dry_run:
        summarize(args)
    if failures:
        print(f"{len(failures)} run(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
