import numpy as np
import pandas as pd
import pytest

from core.features import build_core_features
from core.population import BasePopulation, PopulationConfig
from core.schema import REQUIRED_COLUMNS, validate
from loop.run_rounds import check_phase0


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [8.0, 10.0, 25.5, 49.0, 99.0, 500.0],
            "mcc": ["5411", "5411", "5812", "5812", "5732", "5999"],
        }
    )


def _feature_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e0", "e1", "e2", "e3"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:30:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 00:45:00",
                ]
            ),
            "customer_id": ["c1", "c1", "c1", "c2"],
            "counterparty_id": ["m1", "m2", "m1", "m1"],
            "device_id": ["d1", "d1", "d2", "d3"],
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_schema_accepts_reference_calibrated_population():
    population = BasePopulation(_reference(), PopulationConfig(events=200)).generate(seed=7)
    validate(population)
    assert set(REQUIRED_COLUMNS).issubset(population)
    assert len(population) == 200
    assert population["auth_result"].eq("DECLINED").mean() == pytest.approx(0.04, abs=0.04)
    assert set(population.loc[population["channel"].eq("UPI_P2P"), "counterparty_type"]) == {
        "BENEFICIARY"
    }


def test_population_defaults_match_frozen_scale():
    assert PopulationConfig() == PopulationConfig(
        customers=20_000,
        merchants=2_000,
        vpas=5_000,
        days=90,
        events=3_000_000,
        decline_rate=0.04,
    )


def test_population_is_seed_deterministic():
    generator = BasePopulation(_reference(), PopulationConfig(events=40))
    pd.testing.assert_frame_equal(generator.generate(4), generator.generate(4))


def test_population_has_diurnal_and_weekly_cycles():
    config = PopulationConfig(customers=100, merchants=100, vpas=100, events=50_000)
    events = BasePopulation(_reference(), config).generate(19)
    hourly = events.groupby(events["timestamp"].dt.hour).size()
    assert hourly.loc[18:21].mean() > 3 * hourly.loc[1:4].mean()
    daily = events.groupby(events["timestamp"].dt.normalize()).size()
    weekend = daily[daily.index.dayofweek >= 5].mean()
    weekday = daily[daily.index.dayofweek < 5].mean()
    assert weekend > weekday


def test_reference_aliases_are_normalized(tmp_path):
    source = tmp_path / "sparkov.csv"
    pd.DataFrame({"amt": [10, 25], "category": ["food", "travel"]}).to_csv(
        source, index=False
    )
    population = BasePopulation.from_path(source, PopulationConfig(events=5)).generate(2)
    validate(population)


def test_reference_amount_is_required_and_must_be_positive():
    with pytest.raises(ValueError, match="amount"):
        BasePopulation(pd.DataFrame({"other": [1]}))
    with pytest.raises(ValueError, match="positive"):
        BasePopulation(pd.DataFrame({"amount": [0, -1, np.nan]}))


def test_core_features_are_prior_only_and_correct():
    features = build_core_features(_feature_events())
    assert list(features.index) == ["e0", "e1", "e2", "e3"]
    assert features.loc["e1", "customer_count_1h"] == 1
    assert features.loc["e2", "customer_count_1h"] == 0
    assert features.loc["e2", "customer_count_24h"] == 2
    assert features.loc["e2", "customer_distinct_counterparties_24h"] == 2
    assert features.loc["e2", "customer_distinct_devices_24h"] == 1
    assert features.loc["e3", "counterparty_distinct_customers_1h"] == 1
    assert features.loc["e2", "time_since_previous_s"] == 5_400
    assert features.loc["e2", "amount_zscore_30d"] == pytest.approx(2.12132)


def test_future_event_does_not_change_existing_features():
    events = _feature_events()
    before = build_core_features(events)
    future = pd.DataFrame(
        {
            "event_id": ["future"],
            "timestamp": pd.to_datetime(["2026-02-01"]),
            "customer_id": ["c1"],
            "counterparty_id": ["m9"],
            "device_id": ["d9"],
            "amount": [9_999.0],
        }
    )
    after = build_core_features(pd.concat([events, future], ignore_index=True))
    pd.testing.assert_frame_equal(before, after.loc[before.index])


def test_same_timestamp_events_do_not_leak_into_rolling_windows():
    events = _feature_events().iloc[:2].copy()
    events.loc[1, "timestamp"] = events.loc[0, "timestamp"]
    features = build_core_features(events)
    assert features["customer_count_1h"].eq(0).all()


def test_phase0_readiness_check_runs():
    assert check_phase0(11) == {"events": 250, "features": 52}
