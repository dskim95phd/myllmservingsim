#!/usr/bin/env python3
"""Analyze Qwen3 sparse-LPDDR sweep outputs.

Reads the ``summary.csv`` produced by ``run-qwen3-lpddr-sweep.py`` and writes:

* comparisons.csv: sparse CPU vs dense, sparse LPDDR vs sparse CPU
* report.md: compact human-readable summary
* optional plots when matplotlib is available
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = ("ttft_ms", "tpot_ms", "latency_ms")
TRAFFIC_METRICS = (
    "hbm_hit_rate",
    "lpddr_hit_blocks",
    "cpu_hit_blocks",
    "lpddr_promote_gb",
    "hbm_to_lpddr_gb",
    "copy_time_ms",
)


def _float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100.0:.2f}%"


def _speedup(base: float | None, new: float | None) -> float | None:
    if base is None or new is None or new <= 0:
        return None
    return base / new


def _reduction(base: float | None, new: float | None) -> float | None:
    if base is None or base <= 0 or new is None:
        return None
    return (base - new) / base


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _key(row: dict[str, str], *cols: str) -> tuple[str, ...]:
    return tuple(row.get(col, "") for col in cols)


def _index(rows: list[dict[str, str]], *cols: str) -> dict[tuple[str, ...], dict[str, str]]:
    out = {}
    for row in rows:
        out[_key(row, *cols)] = row
    return out


def _comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dense = _index([r for r in rows if r.get("mode") == "dense_cpu"], "hbm_gb")
    sparse_cpu = _index(
        [r for r in rows if r.get("mode") == "sparse_cpu"], "hbm_gb", "sparse_k"
    )

    comparisons: list[dict[str, str]] = []
    for row in rows:
        mode = row.get("mode")
        hbm = row.get("hbm_gb", "")
        sparse_k = row.get("sparse_k", "")
        lpddr = row.get("lpddr_gb", "")

        if mode == "sparse_cpu":
            base = dense.get((hbm,))
            baseline = "dense_cpu"
        elif mode == "sparse_lpddr":
            base = sparse_cpu.get((hbm, sparse_k))
            baseline = "sparse_cpu"
        else:
            continue

        if not base:
            continue

        out = {
            "mode": mode,
            "baseline": baseline,
            "hbm_gb": hbm,
            "lpddr_gb": lpddr,
            "sparse_k": sparse_k,
        }
        for metric in METRICS:
            base_value = _float(base, metric)
            value = _float(row, metric)
            out[metric] = _fmt(value)
            out[f"{metric}_baseline"] = _fmt(base_value)
            out[f"{metric}_speedup"] = _fmt(_speedup(base_value, value))

        base_cpu_hits = _float(base, "cpu_hit_blocks")
        cpu_hits = _float(row, "cpu_hit_blocks")
        out["cpu_hit_blocks"] = _fmt(cpu_hits, 0)
        out["cpu_hit_reduction"] = _pct(_reduction(base_cpu_hits, cpu_hits))

        for metric in TRAFFIC_METRICS:
            out.setdefault(metric, _fmt(_float(row, metric)))

        comparisons.append(out)
    return comparisons


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top(rows: list[dict[str, str]], metric: str, limit: int) -> list[dict[str, str]]:
    def score(row: dict[str, str]) -> float:
        try:
            return float(row.get(metric, "") or "-inf")
        except ValueError:
            return float("-inf")

    return sorted(rows, key=score, reverse=True)[:limit]


def _write_report(path: Path, rows: list[dict[str, str]], limit: int) -> None:
    lpddr_rows = [r for r in rows if r.get("mode") == "sparse_lpddr"]
    sparse_rows = [r for r in rows if r.get("mode") == "sparse_cpu"]
    lines = [
        "# Qwen3 LPDDR Sparse Sweep Analysis",
        "",
        f"- comparison rows: {len(rows)}",
        f"- sparse CPU vs dense rows: {len(sparse_rows)}",
        f"- sparse LPDDR vs sparse CPU rows: {len(lpddr_rows)}",
        "",
        "## Best LPDDR Latency Speedups",
        "",
        "| hbm_gb | lpddr_gb | sparse_k | latency_speedup | tpot_speedup | hbm_hit_rate | cpu_hit_reduction | copy_time_ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _top(lpddr_rows, "latency_ms_speedup", limit):
        lines.append(
            "| {hbm_gb} | {lpddr_gb} | {sparse_k} | {latency_ms_speedup} | "
            "{tpot_ms_speedup} | {hbm_hit_rate} | {cpu_hit_reduction} | {copy_time_ms} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Best Sparse Attention Speedups",
            "",
            "| hbm_gb | sparse_k | latency_speedup_vs_dense | tpot_speedup_vs_dense |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in _top(sparse_rows, "latency_ms_speedup", limit):
        lines.append(
            "| {hbm_gb} | {sparse_k} | {latency_ms_speedup} | {tpot_ms_speedup} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `sparse_cpu` rows compare sparse attention against dense CPU fallback at the same HBM size.",
            "- `sparse_lpddr` rows compare LPDDR tiering against sparse CPU fallback at the same HBM size and sparse K.",
            "- LPDDR is useful when latency speedup is above 1.0 and CPU hit reduction or LPDDR hit traffic is non-zero.",
            "- If all LPDDR traffic fields are zero, the run likely did not create enough HBM pressure.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_metric(rows: list[dict[str, str]], output_dir: Path, metric: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    lpddr_rows = [r for r in rows if r.get("mode") == "sparse_lpddr"]
    if not lpddr_rows:
        return

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in lpddr_rows:
        groups.setdefault((row.get("hbm_gb", ""), row.get("sparse_k", "")), []).append(row)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (hbm, sparse_k), group in sorted(groups.items(), key=lambda x: (float(x[0][0]), int(x[0][1]))):
        points = []
        for row in group:
            x = _float(row, "lpddr_gb")
            y = _float(row, metric)
            if x is not None and y is not None:
                points.append((x, y))
        if not points:
            continue
        points.sort()
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=f"HBM {hbm}GB, K={sparse_k}",
        )

    ax.set_xlabel("LPDDR per NPU (GB)")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}.png", dpi=160)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    summary_path = args.summary_csv or (args.output_dir / "summary.csv")
    rows = _read_summary(summary_path)
    comparisons = _comparison_rows(rows)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    comparisons_path = args.analysis_dir / "comparisons.csv"
    report_path = args.analysis_dir / "report.md"
    _write_csv(comparisons_path, comparisons)
    _write_report(report_path, comparisons, args.top)

    if args.plots:
        _plot_metric(comparisons, args.analysis_dir, "latency_ms_speedup")
        _plot_metric(comparisons, args.analysis_dir, "tpot_ms_speedup")
        _plot_metric(comparisons, args.analysis_dir, "hbm_hit_rate")

    print(f"Wrote {comparisons_path}")
    print(f"Wrote {report_path}")
    if args.plots:
        print(f"Wrote plots under {args.analysis_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/sweep/qwen3_lpddr"))
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--analysis-dir", type=Path, default=Path("outputs/sweep/qwen3_lpddr/analysis"))
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--plots", action="store_true", help="Also generate PNG plots if matplotlib is installed.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
