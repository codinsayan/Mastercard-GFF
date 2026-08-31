"""Codex reasoner adapter with strict parsing, retries, caching, and fallback."""

from __future__ import annotations

import json
import logging
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
REQUIRED_PROPOSAL_KEYS = {"reasoning", "hypotheses", "proposed_params"}
REQUIRED_HYPOTHESIS_KEYS = {"signal_targeted", "action", "cost_to_attacker"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = Path(__file__).with_name("proposal.schema.json")


@dataclass(frozen=True)
class CodexResponse:
    """Final Codex response plus enough metadata to audit how it was produced."""

    raw: str
    metadata: dict[str, Any]


def invoke_codex(
    prompt: str,
    *,
    executable: str = "codex",
    model: str | None = None,
    timeout: float = 120,
    cwd: str | Path = PROJECT_ROOT,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> CodexResponse:
    """Invoke the ChatGPT-authenticated Codex CLI without a shell or API key."""
    command = [
        executable,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(Path(schema_path).resolve()),
        "--color",
        "never",
    ]
    if model:
        command.extend(["--model", model])
    command.append("-")

    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=Path(cwd),
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RuntimeError(f"codex exec failed: {detail}")
    if not completed.stdout.strip():
        raise RuntimeError("codex exec returned an empty response")

    try:
        version = subprocess.run(
            [executable, "--version"],
            text=True,
            capture_output=True,
            cwd=Path(cwd),
            timeout=min(timeout, 10),
            check=False,
        )
        cli_version = version.stdout.strip() if version.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        cli_version = "unknown"
    return CodexResponse(
        raw=completed.stdout.strip(),
        metadata={
            "runner": "codex_cli",
            "cli_version": cli_version,
            "model": model or "configured-default",
        },
    )


def _parse(raw: str | dict[str, Any]) -> dict[str, Any]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict) or set(value) != REQUIRED_PROPOSAL_KEYS:
        raise ValueError(f"Proposal fields must be exactly {sorted(REQUIRED_PROPOSAL_KEYS)}")
    if not isinstance(value["reasoning"], str):
        raise TypeError("reasoning must be a string")
    if not isinstance(value["hypotheses"], list):
        raise TypeError("hypotheses must be a list")
    for hypothesis in value["hypotheses"]:
        if not isinstance(hypothesis, dict) or set(hypothesis) != REQUIRED_HYPOTHESIS_KEYS:
            raise ValueError(
                f"Hypothesis fields must be exactly {sorted(REQUIRED_HYPOTHESIS_KEYS)}"
            )
        if not all(isinstance(item, str) for item in hypothesis.values()):
            raise TypeError("hypothesis values must be strings")
    if not isinstance(value["proposed_params"], dict):
        raise TypeError("proposed_params must be an object")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value["proposed_params"].values()
    ):
        raise TypeError("proposed parameter values must be finite numbers")
    return value


def clamp_params(
    proposed: dict[str, Any], param_space: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Drop unknown values and clamp known numeric parameters to frozen bounds."""
    result = {}
    for name, value in proposed.items():
        if (
            name not in param_space
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            continue
        low, high = param_space[name]
        result[name] = max(low, min(high, value))
    return result


def propose(
    report: dict[str, Any],
    *,
    invoke: Callable[[str], str | dict[str, Any] | CodexResponse] = invoke_codex,
    fallback: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    budget_check: Callable[[dict[str, float]], bool] | None = None,
    cache_path: str | Path | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    """Return a bounded strict proposal, falling back if Codex cannot produce one."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    space = report["param_space"]
    prompt = render_prompt(report)
    retry_prompt = prompt
    failures: list[str] = []
    raw_responses: list[str | dict[str, Any]] = []
    metadata: dict[str, Any] = {"runner": "custom", "model": "unknown"}

    for _ in range(attempts):
        try:
            response = invoke(retry_prompt)
            if isinstance(response, CodexResponse):
                raw = response.raw
                metadata = response.metadata
            else:
                raw = response
            raw_responses.append(raw)
            parsed = _parse(raw)
            parsed["proposed_params"] = clamp_params(parsed["proposed_params"], space)
            if budget_check is None or budget_check(parsed["proposed_params"]):
                return _cache(
                    parsed,
                    report,
                    prompt,
                    raw_responses,
                    cache_path,
                    failures,
                    metadata,
                    fallback_used=False,
                )
            failure = "proposal rejected by realism or utility budget"
        except (
            json.JSONDecodeError,
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
            TypeError,
            ValueError,
        ) as exc:
            failure = str(exc)
            LOGGER.warning("Codex proposal rejected: %s", exc)
        failures.append(failure)
        retry_prompt = (
            f"{prompt}\n\nThe previous response was rejected: {failure}. "
            "Return a corrected JSON proposal that satisfies every constraint."
        )

    fallback_params = (
        fallback(report) if fallback is not None else report["attack_state"]["current_params"]
    )
    result = {
        "reasoning": "Codex was unavailable or invalid; using the deterministic CEM seed.",
        "hypotheses": [],
        "proposed_params": clamp_params(fallback_params, space),
    }
    return _cache(
        result,
        report,
        prompt,
        raw_responses,
        cache_path,
        failures,
        metadata,
        fallback_used=True,
    )


def render_prompt(report: dict[str, Any]) -> str:
    template_path = Path(__file__).with_name("prompts") / "evasion.txt"
    return template_path.read_text(encoding="utf-8").replace(
        "{{DETECTOR_REPORT_JSON}}", json.dumps(report, indent=2, default=str)
    )


def _cache(
    result: dict[str, Any],
    report: dict[str, Any],
    prompt: str,
    raw_responses: list[str | dict[str, Any]],
    cache_path: str | Path | None,
    failures: list[str],
    metadata: dict[str, Any],
    *,
    fallback_used: bool,
) -> dict[str, Any]:
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "runner": metadata,
                    "fallback_used": fallback_used,
                    "report": report,
                    "prompt": prompt,
                    "raw_responses": raw_responses,
                    "proposal": result,
                    "failures": failures,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return result
