# Metric cards

One card per metric in `vbar.metrics.METRICS`, grouped by family in the files beside this one.
Each card has these fields:

```
## <ID> <name> (v<version>)
- Definition:         exact formula, units, inputs (truth or sensor), when computed
- Summary:            how it ranks across episodes (interquartile mean, unsafe tail or rate)
- Why it matters:     the question it answers
- Known-answer tests: cases with expected values and tolerance
- Convergence:        behaviour as the step and the sampling shrink
- Invariance:         transformations that must not change it
- Mutation detection: broken controllers it catches, and their rank order
- Anti-gaming:        exploits tried, results, fixes
- Reproducibility:    cross-platform tolerance
- Power:              episodes needed to detect a stated change on the baselines
- Validity:           evidence it predicts what it claims, where applicable
- Changelog:          version history
```

## Resolution of a success rate

Half-width of the 95% Clopper-Pearson interval (`vbar.metrics.clopper_pearson`) for a rate measured
as a fraction p of n episodes, from the interval's beta quantiles at p n successes, which need not
be a whole number (49.5 of 50 for p = 0.99). A 100-episode cell resolves a 90% success rate to
about 6 points (90 of 100 gives 0.824 to 0.951). The normal approximation, with the standard error
sqrt(p (1 - p) / n) of Kochenderfer et al. (*Algorithms for Decision Making*, 2022, section 14.1,
eq. 14.5), asks for 385 episodes for a half-width of 0.03 at p = 0.9; the exact half-width there
is 0.031. The book prints only the normal interval, which goes below 0 for rare failures. Two
rates closer than these widths cannot be told apart at that n. The table applies to every card
whose summary is a rate.

| n | p = 0.5 | p = 0.9 | p = 0.95 | p = 0.99 |
|---|---|---|---|---|
| 50 | 0.145 | 0.092 | 0.072 | 0.045 |
| 100 | 0.102 | 0.064 | 0.048 | 0.027 |
| 200 | 0.071 | 0.044 | 0.033 | 0.017 |
| 385 | 0.051 | 0.031 | 0.023 | 0.012 |
| 1000 | 0.031 | 0.019 | 0.014 | 0.007 |
