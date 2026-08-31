# A3 orchestrator integration status

The shared `loop/run_rounds.py` is intentionally still a Phase-0 readiness stub. It validates the
base population and shared features, but its non-dry-run path exits with `No attacks are registered
yet; merge A1-A4 before running rounds`. It has no round execution API or artifact-writer hook into
which A3 can be registered without redesigning shared architecture.

A3 therefore remains independently executable through:

```powershell
python -m attacks.a3_card_testing.evaluation
python -m attacks.a3_card_testing.presentation
python -m attacks.a3_card_testing.fidelity --dataset <local-file> --format <sparkov|ieee-cis>
```

Once the shared owner completes the orchestrator, integration requires it to:

1. import the existing A3 `GENERATOR`, `FEATURE_BLOCK`, and `DETECTOR` package exports;
2. call the already implemented A3 evaluation stages or reproduce their deterministic split and
   threshold conventions;
3. route shared round artifacts to the frozen `results/round_{r}` layout; and
4. preserve the committed A3-specific Codex proposal schema because the current generic schema's
   dynamically keyed `proposed_params` object is rejected by strict Codex structured output.

No shared file is modified by A3 while that integration surface is absent.
