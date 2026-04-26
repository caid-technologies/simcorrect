# Contributing

## Development Setup

SimCorrect is intentionally lightweight at the contract-test layer. Use uv for Python commands so the local package is installed consistently.

```bash
uv sync --extra test
uv run --no-sync python -m unittest discover -s tests
```

The full demos require MuJoCo and rendering dependencies documented in the README. Keep pure contract helpers free of MuJoCo imports so they remain fast to test.

For a full demo environment, install the optional rendering dependencies:

```bash
uv sync --extra demo
```

When running from a problem subdirectory, point uv at the repository project:

```bash
uv run --project .. python render_demo.py
```

## CAID Contract

SimCorrect consumes OpenCAD's CAID design artifact and returns parameter patches.

- Use `caid_contract.py` for artifact loading, parameter resolution, and patch creation.
- Use `mjcf_correction.py` for local MJCF body mass or joint reference edits.
- Keep problem-specific glue in the problem folder, for example `Problem1_ForearmLength/caid_loop.py`.
- Map simulation names to design names through `simulation_tags` instead of hardcoding aliases in diagnosis code.

The written contract lives in `docs/CAID_ARTIFACT_CONTRACT.md`.

`opencad.py` is a compatibility facade for older scripts. New code should avoid importing it directly unless compatibility is the goal.

## Generated Files

Do not commit `__pycache__`, rendered videos, or temporary correction artifacts unless they are intentional demo outputs.
