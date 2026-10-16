# Contributing

Bug reports, scenario proposals and pull requests are welcome. Please read the
[code of conduct](CODE_OF_CONDUCT.md) first.

## Setup

```bash
git clone https://github.com/jamilhammoudeh/vbar
cd vbar
pip install -e ".[dev]"
```

V-BAR supports Python 3.12 and later.

## Checks

A pull request must pass the same checks as CI:

```bash
python -m pytest -q
ruff check
ruff format --check
mypy
```

The suite includes `tests/test_docs.py`, which runs every python example in README.md and
`docs/`, so a change that breaks an example fails the tests. CI also reports coverage (it fails
below 95%) and prints the throughput from `python benchmarks/steps.py`; compare that number before
and after a change on the same machine. A weekly mutation-testing run (`mutmut run`, configured in
`pyproject.toml`) plants small changes in the core modules and lists the ones no test catches.

## Conventions

- Frame: the Hill frame, centred on the target. x is radial outward (R-bar), y along-track
  (V-bar), z along the orbit normal. Relative velocities are rates in the rotating frame.
- Units: SI throughout, angles in radians.
- Physical constants are defined only in `src/vbar/conventions.py` and
  `src/vbar/dynamics/bodies.py`.
- Agents see the output of the sensor model, never the true state.

## Physics, metrics and scenarios

Changes to dynamics, metrics or scenarios come with the tests that justify them: a
known-answer case (a closed-form solution or a textbook result), a convergence or invariance
check where one applies, and the tolerance stated in the test.

The golden trajectories in `tests/golden/` freeze the current outputs. If a change is meant to
alter them, regenerate them with `python tests/test_golden.py` and explain the diff in the pull
request.

## Pull requests

- One change per pull request, with its tests.
- A change to a feature updates its page in `docs/` in the same pull request.
- Commit messages are short, lowercase and in the past tense: `added drag model`,
  `fixed keep-out check`.
- Do not copy code from other projects unless its license allows it, and say where it came from.
- Contributions are released under the MIT license of this repository.

## Versions and deprecations

Versions follow Semantic Versioning. Before 1.0, a minor release (0.x) may change the public API;
every change goes in CHANGELOG.md under Changed or Removed, with what to do instead. From 1.0, a
public function, class or argument that is going away first raises a `DeprecationWarning` for at
least one minor release. The public API is every name a package lists in `__all__` or the
documentation uses, the registered Gymnasium IDs and the documented configuration fields; names
with a leading underscore are private.

## Issues

Use the templates for bug reports, feature requests and scenario proposals.
