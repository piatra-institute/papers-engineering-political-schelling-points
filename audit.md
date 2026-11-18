# Audit

Dated log of editorial passes and verification runs. Newest first.

## 2026-11-18: first build

Scope: new paper built from the seed material in `chats/` (used as leads only; nothing executed from it).

Built:
- `simulation/`: exact equilibrium solver for a regime-change global game with an amplified message whose perceived precision depends on believed source independence and whose reach measures common knowledge of receipt (`model.py`); common-prior variant with informed non-receivers (`aware.py`); repeated game with erosion (`dynamics.py`); laboratory design, behavioural simulation and power (`experiment.py`); finite-group check (`finite_group.py`); twelve unit tests; `run_all.py`.
- Recorded run `model` (`verification/model.json`), artifacts `simulation/output/results.json` and the three figures. The run was recorded twice; the second recording changed only the legend placement of Figure 1d after page inspection showed it overlapping a curve. Numerical results are identical across the two recordings.
- `paper/PAPER.md`, `references.yaml` (36 entries), `claims.yaml` (computation, source, definition, assumption, interpretation and normative claims; source claims all `verified: false`), `sources.md`, `research.md`, `brief.md`, `editorial.md`, `metadata.yaml`, `README.md`, `CITATION.cff`.

Verification:
- Equilibria certified by quadrature-based indifference checks and iterated best responses from both extreme profiles (20 configurations); Proposition 1, the multiplier formula and the precision lemma checked on grids without failures; the first-order disclosure criterion validated against exact equilibria; bistability characterisation checked on a 1640-point grid.
- Literature: full texts or abstracts obtained for the closest prior work; several seed readings corrected (Bueno de Mesquita 2010 is informational, not focal; Enke and Zimmermann report partial neglect on average; the Morris-Shin 2003 uniqueness condition is for a linear game, so the regime-change condition is cited to Angeletos, Hellwig and Pavan 2007).
- refs: all cited IDs resolve; registry identity verified or advisory for every DOI; entries without DOI are left to the referee.
- Coverage self-audit of numerals and spelled-out quantities against claims.yaml (script kept outside the repository; counts in the author's report).
- Remaining for the independent reviewer: source-claim verification, claims coverage stamp, editorial and visual review records.
