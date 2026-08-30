import pandas as pd
import pytest

from core.population import BasePopulation, PopulationConfig
from core.schema import REQUIRED_COLUMNS, validate


def test_schema_accepts_reference_calibrated_population():
    reference = pd.DataFrame({"amount": [10.0, 25.5, 99.0], "mcc": ["5411", "5812", "5732"]})
    population = BasePopulation(reference, PopulationConfig(events=20)).generate(seed=7)
    validate(population)
    assert set(REQUIRED_COLUMNS).issubset(population)
    assert len(population) == 20


def test_population_is_seed_deterministic():
    reference = pd.DataFrame({"amount": [10.0, 25.5], "mcc": ["5411", "5812"]})
    generator = BasePopulation(reference, PopulationConfig(events=10))
    pd.testing.assert_frame_equal(generator.generate(4), generator.generate(4))


def test_reference_amount_is_required():
    with pytest.raises(ValueError, match="amount"):
        BasePopulation(pd.DataFrame({"other": [1]}))
