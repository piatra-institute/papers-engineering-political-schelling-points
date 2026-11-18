# Audit

Dated log of editorial passes and verification runs. Newest first.

## 2026-11-18: revision round 1 (fresh-context Claude referee, 9 majors, 11 minors)

Scope: every finding fixed or answered; per-finding record in revision/2026-11/ledger/engineering-political-schelling-points/fix-round1.json.

Changes:
- Model: actual receipt separated from commonly believed receipt (Proposition 3, `model.realized`); ignorance of source structure separated from neglect of a known correlation (post-disclosure neglect chi'); M0 = 1/(1 - kappa v) stated and checked; uniqueness bound stated in campaign size, with the multiplicity region computed for larger campaigns.
- Disclosure: labels modelled as noticed with probability p; both participation and cutoff scales reported; "more than in proportion" with E and C falls; paradox threshold now in chi'.
- Persistence: new baseline (gamma 0.6, theta_bar 0.8) with both steady states at positive strength; belief-consistent shocks reported (residence times, stationary share, Monte Carlo campaign paths) at baseline and in a precise environment (alpha 16, beta 100).
- Identification: Proposition 5(ii) restated as local; weak global identification quantified (Kullback-Leibler per choice); second majority-illusion variant; signatures labelled as modelling consequences.
- Design: vectorised simulation; sample sizes found by simulated power (2000 replications); acknowledgment cell holding receipt fixed (after Thomas et al. 2014); provenance arm with chi', Pi from the aware model, no-feedback protocol; finite-group check extended to the partial cell.
- Literature: six references added after Crossref and OpenAlex checks (Morris and Shin 2007; Myatt and Wallace 2012; Thomas et al. 2014; De Freitas et al. 2019; Shadmehr and Bernhardt 2011; Levy and Razin 2015).
- Title subtitle shortened to "Disclosure Removes Coordination Only When Commonly Known"; abstract rewritten (248 words); scope disclaimer in the introduction removed.

Verification:
- Recorded run `model` re-recorded after the code changes (a first attempt was stopped to fix figure layout; its partial state did not replace the receipt). Fourteen unit tests pass.
- claims: 374 claims (298 computation, 43 source, rest definition/assumption/interpretation/normative); no FAIL; source claims stay verified: false.
- Coverage self-audit: 390 numerals, 324 covered by claims, 35 exempt labels or years, 30 formula constants, 1 version identifier, 0 uncovered; 113 spelled-out quantities, 32 covered, 81 structural or mathematical uses reviewed by hand.
- build: 27 pages, no missing characters; every page rendered and inspected.
- refs --verify: no mismatch; standards: PASS; check: no FAIL (NEEDS_REVIEW for source claims, coverage, editorial and visual records).

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
