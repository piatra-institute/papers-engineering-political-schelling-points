# Research

Findings, tiered by source proximity. T1 primary, T2 authoritative secondary, T3 reference, T4 general web (leads only). Locators refer to the versions listed in sources.md.

## Findings

### The closest prior work and what it already establishes

- [T1] Static regime-change global game with normal prior (precision alpha) and private signals (precision beta): unique equilibrium iff beta >= alpha^2/(2 pi), in monotone strategies; threshold tends to 1 - c as beta grows. Angeletos, Hellwig and Pavan (2007), Proposition 1, p. 719. Supports Lemma 1's no-message benchmark and the uniqueness bound used throughout.
- [T1] Public signals receive more weight than their relative precision in a beauty contest because of the coordination motive; welfare falls with public precision when r > 1/2 and private information is precise. Morris and Shin (2002), CFDP 1312 pp. 12-16. The coordination multiplier of Proposition 1(iv) is the regime-change analogue with reach.
- [T1] "Publicity" effect: public information has disproportionate impact; the multiplier is largest just before the multiplicity zone. Morris and Shin (2003), CFDP 1275R pp. 30-32. Consequence: the paper does not claim the existence of a public-information multiplier as new; it claims the reach-dependent closed form and its use for disclosure.
- [T1] Uniqueness condition in the 2003 survey's worked example is for a linear investment game (gamma-tilde <= 2 pi), not regime change. Recorded so the regime-change formula is cited to AHP (2007).
- [T1] Policy interventions generate endogenous information that restores multiplicity, without relying on the policy being a public signal. Angeletos, Hellwig and Pavan (2006), abstract pp. 452-454.
- [T1] Credulity: non-credulous citizens respond to propaganda through the coordination motive and mimic the credulous when it dominates. Little (2017), abstract and Proposition 2. Closest predecessor of Proposition 2(iii) (isolated disclosure); the difference is that here the "credulous" are receivers who have not seen the provenance information, and the analysis is in a regime-change game with reach.
- [T1] Communication technology: publicizing grievances has an ambiguous effect on protest; logistics information raises it. Little (2016), abstract.
- [T1/T2] Rumours as public topics; discussion can make a rumour more mobilizing than full belief; extreme rumours backfire. Chen, Lu and Suen (2016), published abstract; WP Proposition 3. Related to the neutral-message result (a message reporting a stronger status quo than z0 demobilizes).
- [T1] Regime manipulation of signals; comparative statics in signal precision and number of signals. Edmond (2013), published abstract.
- [T1] Vanguard violence manipulates beliefs about anti-government sentiment; mechanism informational; focal points ruled out by the solution concept. Bueno de Mesquita (2010), pp. 446, 448. The seed draft's reading (violence as coordinating public signal) is contradicted and was not used.
- [T1] Protests as costly, informative signals of willingness to act. Barberà and Jackson (2020), WP abstract.
- [T1] Information design in binary-action supermodular games: sequential obedience for smallest-equilibrium implementation; some outcomes require private signals. Morris, Oyama and Takahashi (2024), pp. 775, 782, 784.
- [T1] Adversarial public information design: optimal policy induces perfect coordination while preserving structural uncertainty; analysis confined to nondiscriminatory disclosures. Inostroza and Pavan (2025), pp. 763-764, 793.
- [T1] Partial publicity in a beauty contest: public signal reaches each agent with probability P; optimal P can be below one. Cornand and Heinemann (2008), WP pp. 8, 12. Closest predecessor for "reach"; the present paper differs in game (regime change), in the separation of reach from provenance, and in the disclosure question.

### Correlation neglect, the majority illusion and evidence

- [T1] Correlation neglect: on average partial; bimodal distribution between rational and full neglect. Enke and Zimmermann (2019), pp. 314, 321. Motivates treating neglect chi as a parameter in [0, 1] to be measured.
- [T1] Persuasion bias: failure to account for repetition. DeMarzo, Vayanos and Zwiebel (2003), abstract.
- [T1] Majority illusion definition and degree disassortativity. Lerman, Yan and Wu (2016), p. 3.
- [T1] Exposure to IRA accounts concentrated (1% of users, 70% of exposures); no evidence of a meaningful relationship with attitudes, polarization or voting; n = 1496. Eady et al. (2023), pp. 1, 9.
- [T1] Lab global games: threshold strategies; predictability similar for public and private information. Heinemann, Nagel and Ockenfels (2004), abstract and Result 1.
- [T1/T2] Subjects weight public signals more than private but less than equilibrium. Cornand and Heinemann (2014), published abstract. Motivates the attenuation parameter omega in the design.
- [T2] Misperceived norms: correcting beliefs about others' support changes costly behaviour. Bursztyn, González and Yanagizawa-Drott (2020), abstract.
- [T2] Focal points lose power with small payoff asymmetries. Crawford, Gneezy and Rottenstreich (2008), abstract.
- [T2] Coordination-network detection from shared behavioural traces. Pacheco et al. (2021), abstract.
- [T2] Mean-field social interactions: multiplicity above a threshold (beta J > 1). Brock and Durlauf (2001), published abstract; 1995 WP Proposition 2.

### Model results (computed, recorded run `model`)

- Lemma 1 reduces any number of receiver groups to one equation; certificates by quadrature and iterated best responses agree with the analytic cutoffs to 1e-9.
- Proposition 1 and the multiplier formula hold on a 2592-point grid; Proposition 2's limit and closed forms hold to machine precision.
- Theorem 1: isolated disclosure removes a small share of the effect at full reach; common-knowledge disclosure removes most; public disclosure that raises reach amplifies iff Pi > 1 (first-order criterion validated against exact equilibria).
- Proposition 3: gamma_c closed form for c = 1/2, verified numerically; persistence threshold sharp; shocks of the prior's size make the states metastable.
- Identification: four explanations of one exposure effect are observationally equivalent; the design separates them.
- Design: interaction effect sizes and group counts for omega in {1, 0.5, 0.25}; test size near nominal.

### Open questions

- Whether real platform audiences hold beliefs about reach that correspond to a common-knowledge share; no public dataset measures this.
- Whether the disclosure effects survive competing messages and strategic senders; not modelled.
