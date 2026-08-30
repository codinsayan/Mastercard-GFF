import json

from redteam.llm_agent import clamp_params, propose

REPORT = {
    "param_space": {"split_factor": (1, 20), "mule_age_days": (1, 90)},
    "attack_state": {"current_params": {"split_factor": 3, "mule_age_days": 5}},
}


def test_clamp_drops_unknowns_and_enforces_bounds():
    assert clamp_params({"split_factor": 99, "unknown": 2}, REPORT["param_space"]) == {
        "split_factor": 20
    }


def test_invalid_llm_response_uses_fallback():
    result = propose(REPORT, invoke=lambda _: "not-json", fallback=lambda _: {"split_factor": 7})
    assert result["fallback_used"] is True
    assert result["proposed_params"] == {"split_factor": 7}


def test_valid_llm_response_is_strict_and_clamped():
    raw = json.dumps({
        "reasoning": "test",
        "hypotheses": [],
        "proposed_params": {"split_factor": 100, "unknown": 4},
    })
    result = propose(REPORT, invoke=lambda _: raw)
    assert result["proposed_params"] == {"split_factor": 20}
