# Attack package integration contract

The authoritative requirements are in `00_TEAM_BRIEF_AND_INTEGRATION (1).md`.
Work only in your assigned `attacks/aN_*` directory.

Each package must contain `generator.py`, `features.py`, `detector.py`, `card.json`,
and `README.md`. Its `__init__.py` must export instantiated objects:

```python
from .detector import Detector
from .features import Features
from .generator import Generator

GENERATOR = Generator()
FEATURE_BLOCK = Features()
DETECTOR = Detector()
```

The generator must implement `describe_state()`. The returned `param_semantics` must
cover every parameter, and `defender_signals` must cover every emitted feature exactly.
Keep both mappings synchronized with `card.json`.

Run your merge gate with:

```powershell
python -m pytest tests/test_contract.py -k a1
```

A skipped test means the three package exports are not wired yet; a merge-ready attack
must pass, not skip.

