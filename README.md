# Engineering Political Schelling Points

When many accounts amplify a call to collective action, the message can matter as evidence about whether the action can succeed and as a shared reference that lets each person predict what others will do. This paper models both in a regime-change global game in which the message's provenance (how independent its sources are believed to be) and its reach (whether receipt is common knowledge) are separate variables. It decomposes the message's effect exactly into an evidential and a coordination channel, gives exact conditions under which disclosing a common source raises, leaves unchanged or lowers participation, derives a closed-form threshold for a temporary message to leave a persistent change, shows which data can and cannot separate persuasion, coordination and the majority illusion, and designs a laboratory experiment with model-derived effect sizes and sample sizes.

[Read the paper](https://piatra.institute/papers/engineering-political-schelling-points.pdf)

## Build

```bash
uv run build.py          # -> paper/PAPER.pdf  (vendored canonical recipe)
```

Requires `pandoc` and `xelatex` on PATH. From the workspace you can also run `papers build engineering-political-schelling-points`.

## Simulation

The equilibrium computations, verification grids, dynamics and the experimental power analysis are in `simulation/`; see `simulation/README.md`. Reproduce with `uv sync --frozen && uv run --frozen python run_all.py` from that directory.
