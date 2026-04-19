# Contributing to PLN-THRML

Contributions are welcome! This document covers the essentials.

## Development setup

```bash
git clone --recurse-submodules https://github.com/xiaohanma-oss/PLN-THRML.git
cd PLN-THRML
pip install -e ".[dev]"          # installs thrml, jax, pytest, hyperon
git submodule update --init      # fetch upstream PLN test baselines
```

## Running tests

```bash
pytest tests/ -v                 # all tests (~2 min)
pytest -m slow -v                # scalability tests only (long chains, upstream PLN examples)
```

## Code conventions

- **Docstrings**: all public functions should have a docstring with at least
  a one-line summary and a `Parameters` section for non-trivial signatures.
- **Test tolerances**: strength ±0.05 at K=16 (see `tests/conftest.py` for
  K-dependent tolerances); confidence is closed-form on CPU so tests use ≤1e-3.
- **README sync**: if your change modifies the module structure, public API,
  or test file names, update `README.md` to match.
- **No MeTTa invention**: MeTTa code must follow upstream
  [trueagi-io/PLN](https://github.com/trueagi-io/PLN) conventions.
  When in doubt, check `vendor/PLN/`.

## Pull request process

1. Fork the repo and create a feature branch.
2. Make your changes — keep commits focused.
3. Run `pytest tests/ -v` and ensure all tests pass.
4. Open a PR against `main` with a short description of what and why.

## What to contribute

- New PLN rules (intensional, higher-order, temporal)
- Accuracy improvements (better Beta discretization, adaptive K)
- Hardware deployment experiments (TSU benchmarks)
- Documentation and examples
- Bug reports and fixes

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE).
