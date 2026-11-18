# Editorial record

Status: author's full read of the round-1 revision complete; independent editorial review pending.

## Contribution and reader

Question: how much of an amplified message's effect on collective action is evidence about the fundamental and how much is coordination, and what does disclosing a common source do to each? Reader: political economists and political scientists working on protest, propaganda and information design; platform-integrity and election-integrity researchers who design disclosures and audits. Confidence: the results are theorems and exact computations in a stylised model (normal information, one binary action, continuum, cutoff strategies); magnitudes describe the baseline, not any population. The experiment is a design.

The residual over prior work, stated in the paper's "What this paper adds": provenance and visibility as separate variables in a regime-change global game; an exact decomposition with a closed-form evidential channel and a reach-dependent multiplier; a disclosure theorem distinguishing isolated, common-knowledge and public disclosure (the publicity paradox, signed by the publicity-credibility ratio); a closed-form bistability threshold with erosion; an identification result; a design with effect sizes and sample sizes. Not claimed as new: public-signal overweighting and the publicity multiplier (Morris and Shin), signal manipulation by regimes (Edmond), the credulity mechanism (Little 2017), rumours as public topics (Chen, Lu and Suen), correlation neglect (Enke and Zimmermann; DeMarzo, Vayanos and Zwiebel), mean-field bistability (Brock and Durlauf).

## Material worth preserving

- The opening case of eighty accounts and a later disclosure, which states the problem without operational detail.
- Proposition 2: coordination value is a multiplier on evidential value; a commonly known uninformative message is ignored, while an isolated disclosure leaves the coordination channel in place.
- Theorem 1(c) and the ratio Pi, which turn the "publicity paradox" into a checkable condition, and its validation against exact equilibria.
- Proposition 4(ii): the iso-response curve showing that a credible privately seen message and a discredited public one are observationally equivalent.
- The provenance arm's prediction of a sign reversal at a neglect level the same experiment measures.

## Reverse outline (updated in round 1: adds 'How large a campaign the analysis covers', Proposition 3 on receipt versus common belief, label analysis, belief-consistent persistence, the acknowledgment cell)

1. A call seen by many and sent by one: the case and the two values; scope (analysis, measurement, defence; no production).
2. What is known: global games and public information; regime change and manipulation; information design; common knowledge and thresholds; correlation neglect and the majority illusion; empirical evidence; the residual.
3. The model: payoffs, provenance as perceived precision, reach as shared visibility, Lemma 1 (one-equation reduction), baseline.
4. Evidential and coordination channels: Proposition 1 (neutral message, closed-form E, sign and monotonicity of C, multiplier), numbers, verification grid and certificates; Proposition 2 (parasitic coordination, isolated receiver).
5. Disclosure: Theorem 1 (a)-(d); isolated versus common versus public disclosure at baseline; three versus five accounts; neglect threshold; first-order criterion validation.
6. Persistence: erosion dynamics, Proposition 3 (gamma_c, persistence criterion), numbers, disclosure timing, shocks.
7. Identification: Proposition 4 and Table 1; implication for exposure studies.
8. Experiment: design, predictions, power (Table 2), size check, provenance arm, approximations (aware non-receivers, finite groups).
9. Implications for measurement, detection, disclosure and institutional design.
10. Scope.
Appendix: proofs.

## Strongest rival

Rival 1: the amplification effect is persuasion or a majority illusion, and Morris and Shin's public-signal overweighting already covers any amplification; provenance and reach add nothing. Response: Proposition 4 shows these are observationally equivalent in exposure data, and the paper's contribution is to separate them in a design and to sign disclosure effects, which the overweighting result alone does not do. Rival 2: in a unique-equilibrium global game a message without information cannot coordinate, so "coordination without evidence" is a slogan. Response: accepted and made a result (Proposition 2(i)); coordination without evidence survives only off the common-knowledge diagonal (Proposition 2(iii)), which is the mechanism of Little (2017) recast in terms of provenance.

## Decisions and remaining checks

- Regime-change uniqueness condition is cited to Angeletos, Hellwig and Pavan (2007, Prop. 1, p. 719); the Morris-Shin (2003) working paper states its condition for a linear investment game.
- Bueno de Mesquita (2010) is described as informational, following the paper's own statement that focal points are ruled out.
- Enke and Zimmermann (2019) is quoted as "partially as independent".
- Disclosure shares are reported on the participation scale at theta0 throughout.
- No public-data empirical check was attempted: no public dataset measures beliefs about common-knowledge reach.
- Source claims remain verified: false pending independent review; editorial-review.json and visual-review.json are left to the independent reviewer.

## Revision round 1 (2026-11-18): decisions

Responding to a fresh-context Claude referee (nine majors, eleven minors; per-finding responses in revision/2026-11/ledger/engineering-political-schelling-points/fix-round1.json).

- Title: the head stays; the subtitle is now "Disclosure Removes Coordination Only When Commonly Known" (7 words), which states the main disclosure result.
- Focal points (F1): the introduction now says that under uniqueness a commonly known uninformative message has no effect, that the coordination channel is the Morris-Shin publicity effect, and that Schelling-style selection appears only outside the uniqueness bound (now quantified) and in the repeated game.
- Campaign size (F2): the opening example uses five accounts; a new subsection states the bound (8 reports believed independent at baseline) and reports the multiplicity region for larger campaigns.
- Neglect (F3): ignorance of the source structure and neglect of a known correlation are separate; disclosure results are reported as functions of post-disclosure neglect chi'.
- Persistence (F4): new baseline with both steady states at positive status-quo strength; belief-consistent shocks reported (exit times, stationary shares, campaign paths); a precise environment shows where persistence survives; abstract and implications restated at that scale.
- Receipt versus common belief (F5): actual receipt v_a separated from commonly believed reach v (Proposition 3); new laboratory cell holding receipt fixed and varying public acknowledgment, after Thomas et al. (2014), with predicted effects and simulated power.
- Labels (F6): modelled as noticed with probability p; the isolated figure is stated as a lower bound.
- Identification (F7): Proposition 5(ii) restated as local; weak global identification quantified by Kullback-Leibler divergence.
- Literature (F8, F9, F18): Morris and Shin (2007), Myatt and Wallace (2012), Thomas et al. (2014), De Freitas et al. (2019), Shadmehr and Bernhardt (2011), Levy and Razin (2015) added and verified; the contribution restated as the regime-change, disclosure-framed version with closed forms.
- Minors: both scales reported for disclosure shares; "more than in proportion"; M0 = 1/(1 - kappa v); wording of the coordination channel; sample sizes found by simulated power; finite-group check for the partial cell; signatures labelled as modelling consequences with a second illusion variant; provenance-arm Pi from the aware model and a no-feedback protocol; abstract's last sentence states the binding limit; the scope disclaimer in the introduction removed.
