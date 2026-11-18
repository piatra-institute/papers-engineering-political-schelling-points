# Simulation: exact equilibria of a regime-change global game with an amplified message

This directory computes every number and figure in the paper. Nothing here is fitted to data; the model is solved exactly (one-dimensional root finding to machine precision) and checked by independent numerical methods.

## Files

- `model.py`: the static game. Citizens receive private signals about a fundamental; a message reaches a share of them (the reach), and receivers may perceive its precision differently (provenance beliefs). Solves for all monotone equilibria, gives the closed-form evidential channel, the neutral message, the exact coordination multiplier and its closed form at the neutral message, and two certificates: indifference checked by numerical integration of the posterior, and iterated best responses (computed by quadrature only) from "nobody participates" and "everybody participates".
- `aware.py`: the same game when non-receivers know a message was shown to a share of the population but not its content (a common-prior Bayesian game). Used to check the simpler benchmark in which non-receivers ignore the message.
- `dynamics.py`: repeated play in which participation erodes the status quo's expected strength. Steady states, the closed-form feedback threshold for bistability, critical campaign content for persistence, and residence times when the fundamental is hit by shocks.
- `experiment.py`: the laboratory design (bulletin visibility by incentive), model-derived cell predictions, simulated groups with attenuation of the coordination component, decision noise and heterogeneity, and the power analysis.
- `finite_group.py`: the symmetric equilibrium of the public-bulletin game with a finite group, to check the continuum approximation used for the design.
- `run_all.py`: runs the unit tests (aborts on failure), all computations and verification grids, writes `output/results.json` and `../paper/figures/{channels,dynamics,design}.pdf`.
- `test_model.py`: twelve unit tests.

## Reproduce

From this directory with Python 3.12 and uv:

```sh
uv sync --frozen
uv run --frozen python run_all.py
```

Tests alone:

```sh
uv run --frozen python -m unittest -v test_model
```

The run takes a few minutes on a laptop. All random processes (simulated experiments, shocks, Monte Carlo checks) use the seeds in `CONFIG` in `run_all.py`.

## Workspace evidence record

The recorded execution is made from the archive root with

```sh
./papers run engineering-political-schelling-points --id model --cwd simulation --artifact simulation/output/results.json -- uv run --frozen python run_all.py
```

and its receipt is stored under `verification/`.
