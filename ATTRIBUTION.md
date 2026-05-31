# Third-Party Attributions

PLN-THRML bundles or depends on the following third-party components.

## vendor/PLN (git submodule)

- **Source**: https://github.com/trueagi-io/PLN
- **Pinned at**: commit `4405956` (tag `v0.9`)
- **License**: MIT
- **Copyright**: 2025 NARTECH (see `vendor/PLN/LICENSE`)
- **Usage**: Reference implementation of PLN inference rules used for
  cross-validation in our test suite (`tests/test_hybrid.py`,
  `tests/test_unified.py`).

## thrml (runtime dependency)

- **Source**: https://github.com/extropic-ai/thrml
- **License**: see upstream
- **Usage**: Boltzmann/Ising sampling primitives (SpinNode,
  SpinEBMFactor, BlockGibbsSpec) on which PLN-THRML's strength path
  is built.

## jax (runtime dependency)

- **License**: Apache 2.0
- **Usage**: Numerical backend for thrml sampling.
