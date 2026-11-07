# Metric definitions

All evaluation CSVs are derived from held-out 1-Hz predictions in original target
units, after full-window warm-up. Constant-series correlations are undefined and
reported as missing values. Finite-pair support is never silently fabricated.

| Metric | Definition | Aggregation |
| --- | --- | --- |
| RMSE | sqrt(mean((prediction-target)^2)) | Pooled time points in each fold/seed |
| Pearson r | Sample Pearson correlation | Pooled time points in each fold/seed |
| CCC | 2 population covariance / (var(target)+var(prediction)+(mean difference)^2) | Pooled time points in each fold/seed |
| TCC | Same CCC formula, applied per session | Equal-session mean within fold/seed |
| MS-DTW | Operational multiscale DTW described below | Equal-session mean within fold/seed |
| TEC | Agreement of signs of deviations from the selected reference | Equal-session mean within fold/seed |

## Operational MS-DTW

At 1 Hz, nonoverlapping block averages use scales `[1,5,10]` samples, corresponding
to 1, 5 and 10 seconds. Drop trailing incomplete blocks per scale. Scales with fewer
than two blocks are undefined and omitted. The DTW local cost is absolute score
difference; horizontal/vertical moves add a 0.01 score-unit penalty; diagonal
moves add no penalty. A Sakoe-Chiba band uses ceil(10% of the longer sequence)
and at least one sample. Normalize cumulative cost by the selected path length.
Take the arithmetic mean across available scales. The metric depends on target
units. No local target normalization is performed.

## TEC

Default `centered`: compare sign(y_t - session mean y) with sign(p_t - session mean p).
Exact zero is a third tie category. This implements the session-mean-centered
reading of Eq 18 and must be described as centered temporal agreement.

Optional `initial`: replace each session mean with its own first 10 available
post-warm-up 1-Hz values. This is a separate initial-reference variant, not the
reported clinical validation. Neither variant assigns clinical improvement labels.
Kendall tau-b against clinical trajectories is not calculated unless actual
independent clinical labels and their mapping are supplied; the manuscript alone
cannot provide the necessary session-level pairs.

## Across runs and comparisons

Average the three seeds inside each fold. Report mean, sample SD (ddof=1), and
95% Student-t confidence intervals across the five fold means. Repeated seeds are
not counted as 15 independent participant partitions. Paired two-sided t-tests
compare matched fold means; Holm correction covers all available baseline-by-
metric comparisons together. Effect size is paired Cohen dz=mean(paired difference)
/SD(paired difference). Lower is better for RMSE and MS-DTW, higher for the other
metrics. The sign of the paired effect is retained. Fold-level inference has only
five pairs and overlapping training sets; it does not establish clinical efficacy.
No p-value is inferred from rounded published means/SDs.

## Memory horizon conventions

For each neuron/layer/window, average the dynamic forget-gate tau across the 300
steps, then apply Eq 11. Average these window-level horizons per session; average
sessions equally per neuron/run for GMM. This is explicitly different from first
averaging all tau then applying a nonlinear formula. Also export
`retention_horizon_seconds=-dt/log(mean(forget_gate))` separately. The GMM works
on raw neuron values in seconds, does not fit binned histogram counts, does not
force K=3 and does not impose the rapid/intermediate/long-term cutoffs.
