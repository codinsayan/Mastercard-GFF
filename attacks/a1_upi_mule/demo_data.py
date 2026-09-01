"""Deterministic, replay-only view models for the A1 Streamlit demo."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROLE_COLOR = {
    "VICTIM": "#3B82F6",
    "MULE_L1": "#F59E0B",
    "MULE_L2": "#8B5CF6",
    "CASHOUT": "#EF4444",
    "UNKNOWN": "#94A3B8",
}
ROLE_LABEL = {
    "VICTIM": "Victim",
    "MULE_L1": "Layer-1 mule",
    "MULE_L2": "Layer-2 mule",
    "CASHOUT": "Cash-out",
    "UNKNOWN": "Unknown",
}

REQUIRED_FILES = (
    "events.parquet",
    "features.parquet",
    "scores.parquet",
    "metrics.json",
    "redteam_reasoning.json",
)
REQUIRED_EVENT_COLUMNS = {
    "event_id", "timestamp", "attack_id", "customer_id", "counterparty_id",
    "counterparty_account_age_days", "amount", "meta",
}
REQUIRED_SCORE_COLUMNS = {"event_id", "risk_score"}


class ArtifactError(ValueError):
    """Actionable cached-artifact validation failure."""


@dataclass(frozen=True)
class RoundArtifacts:
    round_number: int
    events: pd.DataFrame
    features: pd.DataFrame
    scores: pd.DataFrame
    metrics: dict[str, Any]
    reasoning: dict[str, Any]


@dataclass(frozen=True)
class GraphViewModel:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    sampled: bool


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected a JSON object in {path}")
    return value


def load_round_artifacts(results_root: str | Path, round_number: int) -> RoundArtifacts:
    """Load and validate one frozen round without generation or model fitting."""
    round_dir = Path(results_root) / f"round_{round_number}"
    missing = [name for name in REQUIRED_FILES if not (round_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise ArtifactError(
            f"Round {round_number} is missing: {joined}. Precompute it with "
            f"`python -m loop.run_rounds --rounds 0 1 2 --seed 42`."
        )
    artifacts = RoundArtifacts(
        round_number=round_number,
        events=pd.read_parquet(round_dir / "events.parquet"),
        features=pd.read_parquet(round_dir / "features.parquet"),
        scores=pd.read_parquet(round_dir / "scores.parquet"),
        metrics=_read_json(round_dir / "metrics.json"),
        reasoning=_read_json(round_dir / "redteam_reasoning.json"),
    )
    validate_artifacts(artifacts)
    return artifacts


def validate_artifacts(artifacts: RoundArtifacts) -> None:
    missing_events = REQUIRED_EVENT_COLUMNS.difference(artifacts.events.columns)
    missing_scores = REQUIRED_SCORE_COLUMNS.difference(artifacts.scores.columns)
    if missing_events:
        raise ArtifactError(f"events.parquet is missing columns: {sorted(missing_events)}")
    if missing_scores:
        raise ArtifactError(f"scores.parquet is missing columns: {sorted(missing_scores)}")
    if artifacts.events["event_id"].duplicated().any():
        raise ArtifactError("events.parquet contains duplicate event_id values")
    if artifacts.scores["event_id"].duplicated().any():
        raise ArtifactError("scores.parquet contains duplicate event_id values")
    if "per_attack" not in artifacts.metrics or "A1" not in artifacts.metrics["per_attack"]:
        raise ArtifactError("metrics.json is missing per_attack.A1")


def _safe_meta(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_a1_campaigns(events: pd.DataFrame) -> list[str]:
    """Return stable A1 campaign identifiers ordered by first event time."""
    a1 = events.loc[events["attack_id"].eq("A1")].copy()
    a1["campaign_id"] = a1["meta"].map(
        lambda value: str(_safe_meta(value).get("a1_campaign_id", ""))
    )
    a1 = a1.loc[a1["campaign_id"].ne("")]
    if a1.empty:
        return []
    first = a1.groupby("campaign_id", sort=False)["timestamp"].min().sort_values(kind="stable")
    return first.index.astype(str).tolist()


def campaign_events(events: pd.DataFrame, campaign_id: str) -> pd.DataFrame:
    """Select only one synthetic A1 campaign and sort it causally."""
    campaign = events.loc[
        events["attack_id"].eq("A1")
        & events["meta"].map(
            lambda value: str(_safe_meta(value).get("a1_campaign_id", "")) == campaign_id
        )
    ].copy()
    return campaign.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)


def build_replay_steps(events: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Join cached risk by event id and expose safe replay fields."""
    ordered = events.sort_values(["timestamp", "event_id"], kind="stable").copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])
    score_map = scores.set_index(scores["event_id"].astype(str))["risk_score"]
    ordered["risk_score"] = ordered["event_id"].astype(str).map(score_map)
    ordered["source_role"] = ordered["meta"].map(
        lambda value: str(_safe_meta(value).get("a1_source_role", "UNKNOWN"))
    )
    ordered["destination_role"] = ordered["meta"].map(
        lambda value: str(_safe_meta(value).get("a1_destination_role", "UNKNOWN"))
    )
    ordered["call_active"] = ordered["meta"].map(
        lambda value: bool(_safe_meta(value).get("a1_call_active", False))
    )
    start = ordered["timestamp"].min() if not ordered.empty else pd.Timestamp(0)
    ordered["elapsed_seconds"] = (ordered["timestamp"] - start).dt.total_seconds()
    ordered["stage"] = ordered["destination_role"].map(
        {"MULE_L1": "Victim payment", "MULE_L2": "L1 fan-out", "CASHOUT": "Cash-out"}
    ).fillna("Transfer")
    ordered["decision"] = np.where(
        ordered["risk_score"].isna(), "Score unavailable",
        np.where(ordered["risk_score"].ge(0.5), "Flagged", "Observed"),
    )
    return ordered.reset_index(drop=True)


def build_campaign_graph(events: pd.DataFrame, max_nodes: int = 80) -> GraphViewModel:
    """Build a deterministic, bounded lane layout for a selected campaign."""
    if max_nodes < 4:
        raise ValueError("max_nodes must be at least 4")
    ordered = events.sort_values(["timestamp", "event_id"], kind="stable").copy()
    ordered["source_role"] = ordered["meta"].map(
        lambda value: str(_safe_meta(value).get("a1_source_role", "UNKNOWN"))
    )
    ordered["destination_role"] = ordered["meta"].map(
        lambda value: str(_safe_meta(value).get("a1_destination_role", "UNKNOWN"))
    )
    all_nodes = sorted(set(ordered["customer_id"].astype(str)) | set(ordered["counterparty_id"].astype(str)))
    sampled = len(all_nodes) > max_nodes
    if sampled:
        totals: dict[str, float] = {}
        for row in ordered.itertuples():
            totals[str(row.customer_id)] = totals.get(str(row.customer_id), 0.0) + float(row.amount)
            totals[str(row.counterparty_id)] = totals.get(str(row.counterparty_id), 0.0) + float(row.amount)
        keep = set(sorted(all_nodes, key=lambda node: (-totals.get(node, 0.0), node))[:max_nodes])
        ordered = ordered.loc[
            ordered["customer_id"].astype(str).isin(keep)
            & ordered["counterparty_id"].astype(str).isin(keep)
        ]

    role_map: dict[str, str] = {}
    for row in ordered.itertuples():
        role_map.setdefault(str(row.customer_id), str(row.source_role))
        role_map.setdefault(str(row.counterparty_id), str(row.destination_role))
    lane = {"VICTIM": 0.0, "MULE_L1": 1.0, "MULE_L2": 2.0, "CASHOUT": 3.0}
    node_rows: list[dict[str, Any]] = []
    for role in ("VICTIM", "MULE_L1", "MULE_L2", "CASHOUT", "UNKNOWN"):
        members = sorted(node for node, node_role in role_map.items() if node_role == role)
        for index, node in enumerate(members):
            node_rows.append(
                {"node_id": node, "role": role, "x": lane.get(role, 1.5),
                 "y": index - (len(members) - 1) / 2}
            )
    nodes = pd.DataFrame(node_rows)
    edges = ordered.loc[:, ["event_id", "timestamp", "customer_id", "counterparty_id", "amount"]].copy()
    edges = edges.rename(columns={"customer_id": "source", "counterparty_id": "target"})
    return GraphViewModel(nodes=nodes, edges=edges.reset_index(drop=True), sampled=sampled)


def build_round_comparison(metrics_by_round: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for round_number in sorted(metrics_by_round):
        metrics = metrics_by_round[round_number]
        attack = metrics["per_attack"]["A1"]
        rows.append({"round": round_number, **metrics.get("params", {}), **attack})
    return pd.DataFrame(rows)


def build_signal_mortality(metrics_by_round: dict[int, dict]) -> pd.DataFrame:
    """Return measured AUC values only; never synthesize missing signals."""
    rows = []
    for round_number in sorted(metrics_by_round):
        for signal, auc in metrics_by_round[round_number].get("signal_auc", {}).items():
            rows.append({"round": round_number, "signal": signal, "auc": float(auc)})
    return pd.DataFrame(rows)


def load_external_metrics(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path))


def format_currency(value: float) -> str:
    """Format a rupee amount, switching to lakh notation above 100,000."""
    value = float(value)
    if value >= 100_000:
        return f"₹{value / 100_000:.2f} lakh"
    return f"₹{value:,.0f}"


def format_duration(seconds: float) -> str:
    """Format an elapsed-time offset as a compact, human string."""
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def short_node_label(node_id: str) -> str:
    """Compress a synthetic account id to a short on-graph label."""
    digits = re.findall(r"\d+", node_id)
    if digits:
        return f"#{digits[-1][-4:]}"
    return node_id[-4:]


def render_graph_svg(graph: GraphViewModel, active_edges: int, width: int = 960) -> str:
    """Render a deterministic, self-contained SVG of the campaign money flow.

    Built by hand instead of a charting library so role columns, labels, and
    the currently active edge never overlap regardless of viewer theme.
    """
    left_margin, right_margin = 74, 74
    top_margin, bottom_margin = 56, 40
    row_gap = 58
    plot_width = max(width - left_margin - right_margin, 1)

    if graph.nodes.empty:
        height = 200
        return (
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            'font-family="system-ui,sans-serif" font-size="14" fill="#667085">'
            "No graph data for this campaign.</text></svg>"
        )

    nodes = graph.nodes.copy()
    max_lane = max(float(nodes["x"].max()), 1.0)
    min_y, max_y = float(nodes["y"].min()), float(nodes["y"].max())
    height = top_margin + bottom_margin + max(max_y - min_y, 0.0) * row_gap + 40

    def px(x: float) -> float:
        return left_margin + (x / max_lane) * plot_width

    def py(y: float) -> float:
        return top_margin + (y - min_y) * row_gap + 26

    positions = {
        str(row.node_id): (px(float(row.x)), py(float(row.y)), str(row.role))
        for row in nodes.itertuples()
    }

    header_x = nodes.groupby("role")["x"].first()
    parts: list[str] = [
        (
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            'xmlns="http://www.w3.org/2000/svg" '
            'font-family="system-ui,-apple-system,Segoe UI,sans-serif">'
        ),
        (
            "<defs>"
            '<marker id="arrowActive" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#EF4B39"/></marker>'
            '<marker id="arrowIdle" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#CBD2DC"/></marker>'
            "</defs>"
        ),
        (
            "<style>"
            ".node-label{font-size:11px;fill:#475467;font-weight:600}"
            ".lane-label{font-size:11.5px;fill:#667085;font-weight:700;letter-spacing:.04em;"
            "text-transform:uppercase}"
            ".amt-label{font-size:11.5px;fill:#0B1220;font-weight:700}"
            "@keyframes pulseRing{0%{r:15;opacity:.55}100%{r:30;opacity:0}}"
            ".pulse{animation:pulseRing 1.6s ease-out infinite;fill:none;stroke:#EF4B39;"
            "stroke-width:2}</style>"
        ),
    ]

    for role, lane_x in header_x.items():
        parts.append(
            f'<text class="lane-label" x="{px(float(lane_x)):.1f}" y="24" text-anchor="middle">'
            f"{html.escape(ROLE_LABEL.get(str(role), str(role)))}</text>"
        )

    last_active_index = active_edges - 1
    for index, edge in enumerate(graph.edges.itertuples()):
        source = positions.get(str(edge.source))
        target = positions.get(str(edge.target))
        if not source or not target:
            continue
        active = index < active_edges
        is_current = index == last_active_index
        color = "#EF4B39" if active else "#D9DEE5"
        width_px = 2.6 if is_current else (1.8 if active else 1.2)
        marker = "arrowActive" if active else "arrowIdle"
        opacity = 1.0 if is_current else (0.85 if active else 0.55)
        sx, sy, _ = source
        tx, ty, _ = target
        # Shrink the line so the arrowhead lands before the node circle, not under it.
        dx, dy = tx - sx, ty - sy
        distance = max((dx**2 + dy**2) ** 0.5, 1e-6)
        trim = 20
        ex, ey = tx - dx / distance * trim, ty - dy / distance * trim
        parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{color}" stroke-width="{width_px}" stroke-linecap="round" '
            f'marker-end="url(#{marker})" opacity="{opacity}"/>'
        )
        if is_current:
            mx, my = (sx + tx) / 2, (sy + ty) / 2 - 10
            label = f"{format_currency(float(edge.amount))}"
            bubble_w = 22 + len(label) * 6.4
            parts.append(
                f'<rect x="{mx - bubble_w / 2:.1f}" y="{my - 12:.1f}" width="{bubble_w:.1f}" height="20" '
                'rx="10" fill="#FFFFFF" stroke="#EF4B39" stroke-width="1.2"/>'
                f'<text class="amt-label" x="{mx:.1f}" y="{my + 3:.1f}" text-anchor="middle">'
                f"{html.escape(label)}</text>"
            )

    current_target = None
    if 0 <= last_active_index < len(graph.edges):
        current_target = str(graph.edges.iloc[last_active_index]["target"])

    for node_id, (x, y, role) in positions.items():
        radius = 18 if role in ("VICTIM", "CASHOUT") else 14
        color = ROLE_COLOR.get(role, ROLE_COLOR["UNKNOWN"])
        if node_id == current_target:
            parts.append(f'<circle class="pulse" cx="{x:.1f}" cy="{y:.1f}" r="15"/>')
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            'stroke="#FFFFFF" stroke-width="2.5"/>'
        )
        parts.append(
            f'<text class="node-label" x="{x:.1f}" y="{y + radius + 15:.1f}" text-anchor="middle">'
            f"{html.escape(short_node_label(node_id))}</text>"
        )

    parts.append("</svg>")
    return "".join(parts)
