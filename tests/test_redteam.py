import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from redteam.cem import refine
from redteam.llm_agent import CodexResponse, clamp_params, invoke_codex, propose
from redteam.realism import BudgetViolation, check_budgets, enforce, ks_distance
from redteam.report import build_detector_report

REPORT = {
    "param_space": {"split_factor": (1, 20), "mule_age_days": (1, 90)},
    "attack_state": {"current_params": {"split_factor": 3, "mule_age_days": 5}},
}


def _proposal(**overrides):
    value = {
        "reasoning": "test",
        "hypotheses": [],
        "proposed_params": {"split_factor": 8},
    }
    value.update(overrides)
    return value


def test_clamp_drops_unknowns_non_finite_values_and_enforces_bounds():
    assert clamp_params(
        {"split_factor": 99, "mule_age_days": float("nan"), "unknown": 2},
        REPORT["param_space"],
    ) == {"split_factor": 20}


def test_invalid_codex_response_uses_strict_fallback():
    result = propose(
        REPORT,
        invoke=lambda _: "not-json",
        fallback=lambda _: {"split_factor": 7},
    )
    assert set(result) == {"reasoning", "hypotheses", "proposed_params"}
    assert result["proposed_params"] == {"split_factor": 7}


def test_extra_fields_and_malformed_hypotheses_are_rejected():
    invalid = _proposal(extra="not allowed", hypotheses=[{"action": "missing fields"}])
    result = propose(REPORT, invoke=lambda _: invalid, attempts=1)
    assert result["proposed_params"] == REPORT["attack_state"]["current_params"]


def test_valid_codex_response_is_strict_and_clamped():
    raw = json.dumps(
        _proposal(proposed_params={"split_factor": 100, "unknown": 4})
    )
    result = propose(REPORT, invoke=lambda _: raw)
    assert result["proposed_params"] == {"split_factor": 20}


def test_budget_rejection_retries_then_falls_back():
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        return _proposal()

    result = propose(REPORT, invoke=invoke, budget_check=lambda _: False, attempts=3)
    assert len(calls) == 3
    assert "previous response was rejected" in calls[1]
    assert result["proposed_params"] == REPORT["attack_state"]["current_params"]


def test_timeout_falls_back_without_breaking_loop():
    def timeout(_):
        raise subprocess.TimeoutExpired("codex", 1)

    result = propose(REPORT, invoke=timeout, attempts=1)
    assert result["proposed_params"] == REPORT["attack_state"]["current_params"]


def test_codex_adapter_uses_safe_noninteractive_command(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="codex-cli 1.2.3\n", stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(_proposal()), stderr="progress")

    monkeypatch.setattr(subprocess, "run", fake_run)
    schema = tmp_path / "proposal.schema.json"
    schema.write_text("{}", encoding="utf-8")
    response = invoke_codex(
        "prompt", executable="codex-test", model="gpt-test", cwd=tmp_path, schema_path=schema
    )
    command, kwargs = calls[0]
    assert command == [
        "codex-test",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema),
        "--color",
        "never",
        "--model",
        "gpt-test",
        "-",
    ]
    assert kwargs["input"] == "prompt" and "shell" not in kwargs
    assert response.metadata == {
        "runner": "codex_cli",
        "cli_version": "codex-cli 1.2.3",
        "model": "gpt-test",
    }


def test_full_codex_exchange_is_cached(tmp_path):
    cache = tmp_path / "redteam_reasoning.json"
    response = CodexResponse(
        json.dumps(_proposal()),
        {"runner": "codex_cli", "cli_version": "1.2.3", "model": "gpt-test"},
    )
    propose(REPORT, invoke=lambda _: response, cache_path=cache)
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["runner"]["model"] == "gpt-test"
    assert saved["fallback_used"] is False
    assert saved["prompt"] and saved["raw_responses"] and saved["proposal"]


class DummyAttack:
    attack_id = "A1"

    def default_params(self):
        return {"x": 0.0, "n": 1}

    def param_space(self):
        return {"x": (-5.0, 5.0), "n": (0, 10)}

    def utility(self, events):
        return float(events.attrs.get("utility", 0.5))

    def describe_state(self, params):
        return {
            "attack_summary": "bounded synthetic test",
            "param_semantics": {"x": "continuous", "n": "integer"},
            "defender_signals": {},
            "current_params": params,
        }


def test_cem_refines_toward_objective_and_preserves_types():
    objective = lambda params: -(params["x"] - 2.0) ** 2 - (params["n"] - 7) ** 2
    result = refine(
        {"proposed_params": {"x": -2.0, "n": 2}},
        DummyAttack(),
        objective,
        seed=9,
        samples=100,
        iterations=6,
    )
    assert result["x"] == pytest.approx(2.0, abs=0.3)
    assert isinstance(result["n"], int) and result["n"] == 7


def test_realism_and_utility_budgets():
    reference = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0]})
    candidate = reference.copy()
    accepted = check_budgets(DummyAttack(), candidate, reference)
    assert accepted.accepted and accepted.max_ks == 0
    assert enforce(DummyAttack(), candidate, reference) is candidate

    unrealistic = pd.DataFrame({"amount": [100.0, 200.0, 300.0]})
    with pytest.raises(BudgetViolation, match="KS distance"):
        enforce(DummyAttack(), unrealistic, reference, max_ks=0.2)
    candidate.attrs["utility"] = 0.01
    with pytest.raises(BudgetViolation, match="utility"):
        enforce(DummyAttack(), candidate, reference, min_utility=0.1)


def test_ks_distance_is_exact_for_separated_samples():
    assert ks_distance(pd.Series([1, 2]), pd.Series([3, 4])) == 1.0


def test_detector_report_uses_current_round_params():
    report = build_detector_report(
        DummyAttack(),
        {"auc": 0.8},
        pd.DataFrame({"event_id": ["e1"], "amount": [12.0]}),
        params={"x": 3.0, "n": 8},
    )
    assert report["attack_state"]["current_params"] == {"x": 3.0, "n": 8}
