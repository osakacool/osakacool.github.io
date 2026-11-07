# Correspondence to the supplied manuscript

This repository is a newly prepared, equation-based implementation. The supplied
PDF contains descriptions and aggregate results, but no original source code,
trained checkpoints, session records, fold assignments, or raw predictions.

| Component | Manuscript source | Implementation and status |
| --- | --- | --- |
| Adaptive tau | Section 4.1, Eq 2 | Dynamic input/hidden-state sigmoid map, bounded 0.1 to 10 s |
| Gate modulation | Eqs 3-6, 9 | Each input/forget/output gate multiplied by exp(-dt/tau) |
| Network | Section 5.4 | Two layers, 128 neurons each, interlayer dropout 0.2, prelayer batch norm |
| Output | Eq 7, Section 5.4 | Linear scalar head, no sigmoid or clipping; linear outputs are not bounded to [0,1] |
| Optimization | Section 4.4 | MSE, Adam 0.001, L2 on weight matrices 1e-4, batch size 32 |
| LR and stopping | Sections 4.4 and 5.4 | ReduceLROnPlateau factor 0.1 and patience 5; early stopping patience 10 |
| Feature reduction | Section 5.1 | Training-only scaling and fixed 128-component PCA; measured variance is recorded |
| Sequences | Section 5.2 | 300 frames; training endpoints every 10 frames, prediction endpoints every frame |
| Standard LSTM | Section 5.2 | Two 128-unit layers, as an explicit matched-depth choice |
| Attention LSTM | Section 5.2 | Additive temporal attention over last 32 recurrent states |
| TCN | Section 5.2 | Eight causal residual convolutions, kernel 3, dilations 1 through 128 |
| Psychoacoustic model | Section 5.2 | Ridge on mean/SD of named descriptors within the same input window |
| Validation | Sections 5.3 and 6 | Five participant-disjoint folds, three training seeds; disjoint inner validation |
| GMM | Section 6, Figure 5 and Supplement S1 | K=1..10 on raw per-neuron horizon values; select BIC minimum |
| Attribution | Eq 19 | Squared input gradients mapped through fitted scaler/PCA, grouped by input lag |

## Details not specified by the manuscript

The following are explicit implementation choices, not recovered original settings:
inner validation fraction 0.2; split seed 2026; training seeds 17, 29, 43; maximum
200 epochs; gradient clipping norm 1; training stride 10 frames; training-only
PCA fit sample capped at 100,000 frames; per-fold PCA randomized solver;
attention scoring network; single-convolution residual TCN blocks; ridge alpha
validation grid; one-second prediction pooling alignment; MS-DTW choices;
TEC reference conventions; gradient attribution window sample. The extractor
requires an explicit acoustic analysis-window duration because it is not specified.
PCA sampling is deterministic and includes only training participants. Increase
`pca_fit_max_frames` or set it to null to fit every training frame if memory permits.

The 6,373 ComParE values are functionals, not 6,373 instantaneous low-level
measurements. Script 03 calculates them in trailing windows at 100-ms hops.
The window duration must be chosen and reported, e.g. 1 s. This can be expensive
for 1,200 sessions and creates additional acoustic context beyond the 300 feature
endpoints. It is not evidence that this exact extraction was used for the paper.

## Points requiring resolution for an exact replication

1. **Target definition.** The PDF does not specify a reproducible clinical
   regulation-score construction rule. The primary pipeline requires actual
   timestamped regulation_score observations; it will not synthesize them.
2. **Dataset contents.** The manuscript-listed Kaggle landing page did not expose
   a readable file schema during preparation. Actual counts, modalities, licenses,
   annotation provenance and sample rates require inspection of the upstream files.
   No original data or credentials are distributed here.
3. **PCA variance.** Fixed 128 components cannot guarantee 95% retained variance.
   The actual fraction is exported for every fold; it is not overwritten with 95%.
4. **Horizon bound.** Eq 11 gives H=dt/(1-exp(-dt/tau)). At dt=0.1 s and tau<=10 s,
   H<=10.0500833 s. Therefore 20.85 s cannot follow from these equations and bounds.
   The code preserves the stated bound and exports measured horizons. It also
   exports a separate horizon based on actual forget-gate retention. This second
   quantity is not substituted for Eq 11. Nonlinear averaging order is documented.
5. **CCC versus TCC.** Eq 16 is the CCC formula. Applying both to identical samples
   must give the same value. To expose an explicit distinction, CCC is pooled
   across a fold, while TCC is the equal-session mean CCC. This aggregation choice
   is not specified in the PDF and must be reconciled before claiming exact tables.
6. **MS-DTW.** Eq 17 omits operational scales, local distance, path penalty and
   normalization. The implemented operational definition is fully specified in
   metric_definitions.md and must not be represented as the recovered original.
7. **TEC.** Centered trend agreement in Eq 18 is not inherently clinical
   improvement/deterioration. Both centered and initial-reference options are
   available. Clinical-label validation requires separate observed labels.
8. **Session across folds.** A participant/session can be in the test partition of
   only one of the five outer folds. A session may be evaluated across seeds of
   that fold, not treated as independent test data in all five models.
9. **Demographic balance.** GroupKFold balances group sizes, not demographic strata.
   The current manifest has no verified age/gender records, so balanced demographic
   splits are not claimed. The actual split participant lists are saved.
10. **Figures and statistics.** Some supplied PDF captions describe reconstructed
    distributions/scenarios. This repository plots only observed run outputs.
    Fold values, p-values, confidence intervals and neuron clusters are recalculated;
    no outputs are constructed to match the reported summary values.
11. **Hyperparameter budgets.** The paper does not provide the baseline tuning
    search spaces. Neural models use fixed, documented settings and the same epoch
    cap; ridge uses a validation grid. Equal tuning budgets are not claimed.

The default tau limits can be changed for a separately identified sensitivity
analysis. Raising tau_max solely to reproduce a reported cluster is not an exact
replication of the manuscript configuration.

## Memory and cache handling

Scaling/PCA reads raw sessions sequentially. Transformed per-session arrays are
stored as memory-mapped NPY files under a fingerprinted preprocessing cache per
fold, shared by all neural models/seeds in the same output folder. Cache keys
include input hashes, splits, target and transform settings. PCA fitting still
requires memory for up to 100,000 x 6,373 sampled values and solver workspace;
reduce the explicit fit cap if needed and report that choice. Script 04 produces
standalone diagnostic exports; the training runner manages its own cache.
