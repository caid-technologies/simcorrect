# Contributing

## Development Setup

SimCorrect is intentionally lightweight at the contract-test layer. Use uv for Python commands.

```bash
uv run --no-project python -m unittest discover -s tests
```

The full demos require MuJoCo and rendering dependencies documented in the README. Keep pure contract helpers free of MuJoCo imports so they remain fast to test.

## CAID Contract

SimCorrect consumes OpenCAD's CAID design artifact and returns parameter patches.

- Use `caid_contract.py` for artifact loading, parameter resolution, and patch creation.
- Keep problem-specific glue in the problem folder, for example `Problem1_ForearmLength/caid_loop.py`.
- Map simulation names to design names through `simulation_tags` instead of hardcoding aliases in diagnosis code.

## Generated Files

Do not commit `__pycache__`, rendered videos, or temporary correction artifacts unless they are intentional demo outputs.
