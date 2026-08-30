"""Stage-1 LLM reasoner with strict parsing, bounds, retry, cache, and CEM fallback."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

LOGGER = logging.getLogger(__name__)
REQUIRED_PROPOSAL_KEYS = {"reasoning", "hypotheses", "proposed_params"}


def _parse(raw: str | dict) -> dict:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict) or not REQUIRED_PROPOSAL_KEYS.issubset(value):
        raise ValueError(f"Proposal must contain {sorted(REQUIRED_PROPOSAL_KEYS)}")
    if not isinstance(value["reasoning"], str):
        raise TypeError("reasoning must be a string")
    if not isinstance(value["hypotheses"], list):
        raise TypeError("hypotheses must be a list")
    if not isinstance(value["proposed_params"], dict):
        raise TypeError("proposed_params must be an object")
    return value


def clamp_params(proposed: dict, param_space: dict[str, tuple[float, float]]) -> dict:
    """Drop unknown values and clamp known numeric parameters into their frozen bounds."""
    result = {}
    for name, value in proposed.items():
        if name not in param_space or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        low, high = param_space[name]
        result[name] = max(low, min(high, value))
    return result


def propose(
    report: dict,
    *,
    invoke: Callable[[str], str | dict] | None = None,
    fallback: Callable[[dict], dict] | None = None,
    budget_check: Callable[[dict], bool] | None = None,
    cache_path: str | Path | None = None,
    attempts: int = 3,
) -> dict:
    """Produce a safe proposal; no API dependency is imported and cached mode is supported."""
    space = report["param_space"]
    prompt = render_prompt(report)
    failures: list[str] = []
    if invoke is not None:
        for _ in range(attempts):
            try:
                parsed = _parse(invoke(prompt))
                parsed["proposed_params"] = clamp_params(parsed["proposed_params"], space)
                if budget_check is None or budget_check(parsed["proposed_params"]):
                    return _cache(parsed, report, cache_path, failures)
                failures.append("proposal rejected by realism or utility budget")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning("LLM proposal rejected: %s", exc)
                failures.append(str(exc))
    fallback_params = fallback(report) if fallback is not None else report["attack_state"]["current_params"]
    result = {
        "reasoning": "LLM unavailable or invalid; using the deterministic CEM fallback region.",
        "hypotheses": [],
        "proposed_params": clamp_params(fallback_params, space),
        "fallback_used": True,
    }
    return _cache(result, report, cache_path, failures)


def render_prompt(report: dict) -> str:
    template_path = Path(__file__).with_name("prompts") / "evasion.txt"
    return template_path.read_text(encoding="utf-8").replace(
        "{{DETECTOR_REPORT_JSON}}", json.dumps(report, indent=2, default=str)
    )


def _cache(result: dict, report: dict, cache_path: str | Path | None, failures: list[str]) -> dict:
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"report": report, "proposal": result, "failures": failures}, indent=2, default=str),
            encoding="utf-8",
        )
    return result
