"""Generate offline SVG presentation artifacts from measured A3 round results."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import numpy as np

DEFAULT_SUMMARY = Path(__file__).with_name("results") / "summary.json"
DEFAULT_OUTPUT = Path(__file__).with_name("results") / "presentation"
WIDTH = 960
HEIGHT = 540


def _svg_document(title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">'
        '<rect width="100%" height="100%" fill="#101820"/>'
        f'<text x="48" y="52" fill="#ffffff" font-size="26" font-family="Arial" '
        f'font-weight="700">{escape(title)}</text>{body}</svg>'
    )


def _line_chart(title: str, series: dict[str, list[float | None]], *, percent: bool) -> str:
    left, top, right, bottom = 90, 90, 900, 455
    colors = ("#ff5f00", "#f5c542", "#54c2ff")
    body = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#73808c"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#73808c"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        label = f"{value:.1f}" if not percent else f"{value * 100:.0f}%"
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#273540"/>')
        body.append(f'<text x="40" y="{y + 5:.1f}" fill="#b8c4cc" font-size="14" font-family="Arial">{label}</text>')
    x_values = np.linspace(left + 60, right - 60, 3)
    for index, x in enumerate(x_values):
        body.append(f'<text x="{x - 28:.1f}" y="490" fill="#ffffff" font-size="16" font-family="Arial">Round {index}</text>')
    for series_index, (name, values) in enumerate(series.items()):
        points = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = x_values[index]
            y = bottom - float(value) * (bottom - top)
            points.append((x, y, value))
        color = colors[series_index % len(colors)]
        if len(points) > 1:
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            body.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, y, value in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
            body.append(f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" fill="{color}" font-size="14" font-family="Arial">{value:.3f}</text>')
        legend_x = 110 + series_index * 245
        body.append(f'<rect x="{legend_x}" y="65" width="18" height="5" fill="{color}"/>')
        body.append(f'<text x="{legend_x + 26}" y="72" fill="#ffffff" font-size="14" font-family="Arial">{escape(name)}</text>')
    return _svg_document(title, "".join(body))


def _throughput_chart(values: list[float]) -> str:
    maximum = max(values) * 1.1
    normalized = [value / maximum for value in values]
    svg = _line_chart("Attacker throughput: measured cards per hour", {"Actual throughput": normalized}, percent=False)
    for value, normalized_value in zip(values, normalized, strict=True):
        svg = svg.replace(f">{normalized_value:.3f}</text>", f">{value:.0f} cards/hour</text>", 1)
    return svg


def _heatmap(signal_rows: list[dict]) -> str:
    left, top = 350, 95
    cell_width, cell_height = 165, 42
    body = []
    for column, label in enumerate(("Round 0", "Round 1", "Round 2")):
        body.append(f'<text x="{left + column * cell_width + 45}" y="78" fill="#ffffff" font-size="15" font-family="Arial">{label}</text>')
    for row_index, row in enumerate(signal_rows):
        y = top + row_index * cell_height
        body.append(f'<text x="30" y="{y + 26}" fill="#ffffff" font-size="13" font-family="Arial">{escape(row["feature"])}</text>')
        for column, key in enumerate(("round_0_auc", "round_1_auc", "round_2_auc")):
            value = float(row[key])
            strength = max(0.0, min(1.0, (value - 0.5) * 2))
            red = int(45 + 210 * strength)
            green = int(75 + 70 * (1 - strength))
            x = left + column * cell_width
            body.append(f'<rect x="{x}" y="{y}" width="150" height="34" rx="4" fill="rgb({red},{green},55)"/>')
            body.append(f'<text x="{x + 55}" y="{y + 23}" fill="#ffffff" font-size="14" font-family="Arial">{value:.3f}</text>')
    return _svg_document("A3 signal mortality — direction-normalized ROC AUC", "".join(body))


def _bin_statistics(summary: dict) -> list[dict]:
    statistics = []
    for round_number, round_result in enumerate(summary["rounds"]):
        cards = int(round_result["params"]["cards_per_hour"])
        bins = min(cards, int(round_result["params"]["n_bins"]))
        counts = np.full(bins, cards // bins, dtype=int)
        counts[: cards % bins] += 1
        probabilities = counts / counts.sum()
        entropy = 0.0 if bins == 1 else float(-(probabilities * np.log(probabilities)).sum() / np.log(bins))
        entropy = min(1.0, max(0.0, entropy))
        statistics.append(
            {
                "round": round_number,
                "cards": cards,
                "distinct_bins": bins,
                "dominant_bin_share": float(counts.max() / cards),
                "normalized_bin_entropy": entropy,
                "bin_cooccurrence_auc": float(
                    round_result["per_feature_auc"]["a3_bin_cooccurrence_score"][
                        "direction_normalized_auc"
                    ]
                ),
            }
        )
    return statistics


def build_presentation(summary_path: str | Path, output_dir: str | Path) -> dict:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rounds = summary["rounds"]
    metrics = {
        "active_recall": [float(row["detector_metrics"]["recall"]) for row in rounds],
        "active_roc_auc": [float(row["detector_metrics"]["roc_auc"]) for row in rounds],
        "frozen_previous_recall": [
            None,
            float(rounds[1]["frozen_previous_detector_metrics"]["recall"]),
            float(rounds[2]["frozen_previous_detector_metrics"]["recall"]),
        ],
    }
    throughput = [float(row["actual"]) for row in summary["throughput"]]
    bin_statistics = _bin_statistics(summary)
    artifacts = {
        "source": Path(summary_path).name,
        "metrics_across_rounds": metrics,
        "throughput_cards_per_hour": throughput,
        "signal_mortality": summary["signal_mortality"],
        "bin_clustering": bin_statistics,
    }
    files = {
        "round_metrics.svg": _line_chart(
            "Detector performance across adaptive rounds",
            {
                "Active recall": metrics["active_recall"],
                "Active ROC AUC": metrics["active_roc_auc"],
                "Frozen previous-head recall": metrics["frozen_previous_recall"],
            },
            percent=True,
        ),
        "throughput.svg": _throughput_chart(throughput),
        "signal_mortality.svg": _heatmap(summary["signal_mortality"]),
        "bin_clustering.svg": _line_chart(
            "BIN clustering and detection-signal persistence",
            {
                "Dominant BIN share": [row["dominant_bin_share"] for row in bin_statistics],
                "BIN co-occurrence AUC": [row["bin_cooccurrence_auc"] for row in bin_statistics],
                "Normalized BIN entropy": [row["normalized_bin_entropy"] for row in bin_statistics],
            },
            percent=True,
        ),
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    (output / "presentation.json").write_text(
        json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_presentation(args.summary, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "charts": 4, "rounds": len(result["metrics_across_rounds"]["active_recall"])}, indent=2))


if __name__ == "__main__":
    main()
