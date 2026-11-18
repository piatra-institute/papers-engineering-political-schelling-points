# Simulation: exact equilibria of a regime-change global game with an amplified message

This directory computes every number and figure in the paper. Nothing here is fitted to data; the model is solved exactly (one-dimensional root finding to machine precision) and checked by independent numerical methods.

## Files

- `model.py`: the static game. Citizens receive private signals about a fundamental; receivers of a message commonly believe that a share of citizens received it (the commonly believed reach), which may differ from the share that actually did, and receivers may perceive its precision differently (provenance beliefs). Solves for all monotone equilibria, gives the closed-form evidential channel, the neutral message, the exact coordination multiplier, its closed form 1/(1 - kappa v) at the neutral message, realized participation when actual and believed receipt differ, the share of contents with multiple equilibria outside the uniqueness bound, and two certificates: indifference checked by numerical integration of the posterior, and iterated best responses (computed by quadrature only) from "nobody participates" and "everybody participates".
- `aware.py`: the same game when non-receivers know a message was shown to a share of the population but not its content (a common-prior Bayesian game). Used to check the simpler benchmark in which non-receivers ignore the message.
- `dynamics.py`: repeated play in which participation erodes the status quo's expected strength. Steady states, the closed-form feedback threshold for bistability, critical campaign content for persistence in the noise-free skeleton, residence times under belief-consistent shocks, and Monte Carlo paths of a temporary campaign under those shocks.
- `experiment.py`: the laboratory design (bulletin visibility by incentive, plus cells that hold receipt fixed and vary only public acknowledgment), model-derived cell predictions, simulated groups with attenuation of the coordination component, decision noise and heterogeneity, and a power analysis that searches for the number of groups whose simulated power reaches the target.
- `finite_group.py`: the symmetric equilibrium of the public-bulletin game and of the partial-visibility game (receivers and informed non-receivers) with a finite group, to check the continuum approximation used for the design.
- `run_all.py`: runs the unit tests (aborts on failure), all computations and verification grids, writes `output/results.json` and `../paper/figures/{channels,dynamics,design}.pdf`.
- `test_model.py`: fourteen unit tests.

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

The run takes about fifteen minutes on a laptop. All random processes (simulated experiments, shocks, Monte Carlo checks) use the seeds in `CONFIG` in `run_all.py`.

## Workspace evidence record

The recorded execution is made from the archive root with

```sh
./papers run engineering-political-schelling-points --id model --cwd simulation --artifact simulation/output/results.json -- uv run --frozen python run_all.py
```

and its receipt is stored under `verification/`.
