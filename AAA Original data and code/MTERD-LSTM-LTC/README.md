# MTERD-LSTM-LTC

Code for the manuscript **“Multi-scale Auditory Psychological Temporal Analysis: Quantifying Emotional Regulation in Music Therapy via LSTM-LTC Modeling”** (Zhang, Zhang & Zhan).

This repository folder contains the analysis scripts numbered 11–18, covering evaluation metrics, participant-wise cross-validation, leakage-safe feature preprocessing, the LSTM-LTC model, baseline models, the training loop, temporal-cluster analysis, and fold-level statistical inference.

- Dataset: Music Therapy Emotional Response Dataset (MTERD), publicly available on Kaggle — <https://www.kaggle.com/datasets/colabsss/music-therapy-emotional-response-dataset>
- Code repository path: `AAA Original data and code/MTERD-LSTM-LTC`

---

## 1. Contents

| # | File | Role |
|---|------|------|
| 11 | `11-Advanced Metrics Calculator.py` | Regression and temporal-alignment metrics (RMSE, Pearson r, CCC, TCC, MS-DTW, TEC, Kendall τ<sub>b</sub>) |
| 12 | `12-Cross-validation partitioning.py` | Participant-disjoint 5-fold split (`GroupKFold` / `StratifiedGroupKFold`) |
| 13 | `13-Data Preprocessing.py` | Train-only `StandardScaler` + PCA (128 components); 100 ms → 1 Hz pooling |
| 14 | `14-LSTM-LTC Model.py` | LSTM cell with input-dependent, neuron-specific time constants; stacked model with τ logging |
| 15 | `15-Baseline Models.py` | Standard LSTM, attention LSTM, dilated TCN, psychoacoustic ridge regression |
| 16 | `16-Model Training.py` | Supervised training loop (Adam, LR plateau schedule, gradient clipping, early stopping) |
| 17 | `17-Temporal Analysis.py` | GMM/BIC clustering of learned memory horizons with internal validity indices |
| 18 | `18-Statistical Comparison.py` | Paired *t*-tests, Cohen’s d<sub>z</sub>, Holm–Bonferroni adjustment, *t*-based 95% CIs |

---

## 2. Requirements

```
python >= 3.8
numpy
scipy
pandas
scikit-learn
torch
statsmodels
```

```bash
pip install numpy scipy pandas scikit-learn torch statsmodels
```

A CUDA-capable GPU is optional; pass `device="cuda"` to the training function. Training the full model on the complete MTERD feature set is the most compute-intensive step in the pipeline.

---

## 3. Data assumptions

The scripts operate on pre-extracted features, not on raw audio. They expect:

- **Input features** — acoustic descriptors extracted from session audio at 100 ms frame resolution. In the paper these are openSMILE vectors of 6,373 dimensions per frame, reduced to 128 dimensions by PCA. Any frame-level feature matrix with a consistent column layout can be used.
- **Targets** — continuous emotional-regulation annotations resampled to the same frame grid (the original annotations are provided at 1 Hz; use `pool_100ms_to_1hz` on predictions when comparing against 1 Hz ground truth).
- **Tensors** — `X` has shape `[N, T, F]` (sessions × timesteps × features) and `y` has shape `[N, T]`; the model emits one scalar per timestep.
- **Metadata table** — one row per frame or per session with a participant identifier column (`participant_id` by default) used for grouping.

The 1,200 sessions × 300 participants protocol, 15–45 min session lengths, and 30 s (300-frame) evaluation windows follow the manuscript.

---

## 4. Pipeline

Scripts are numbered in execution order.

```bash
# 1) Participant-disjoint folds (adds a `fold` column: 0..4 = test fold, -1 = training row)
python "12-Cross-validation partitioning.py" \
  --input mterd_metadata.csv \
  --output mterd_folds.csv \
  --participant-col participant_id

# 2) Per fold: fit preprocessing on training frames only, then transform validation/test frames
#    AcousticPreprocessor(n_components=128).fit_transform(train) / .transform(val)

# 3) Train LSTM-LTC and each baseline on identical folds and windows
#    train_model(model, x_train, y_train, x_val, y_val, ...)

# 4) Score held-out sessions
#    summary(y_true, y_pred) for RMSE / Pearson r / CCC / TCC / MS-DTW
#    therapeutic_effect_consistency(...) and session_kendall_tau(...) on per-session sequences

# 5) Temporal analysis of learned time constants
#    analyze_horizons(horizons, components=range(1, 7))

# 6) Fold-level comparison against baselines
#    paired_comparisons(reference_fold_scores, {name: baseline_fold_scores})
```

Leakage rules enforced by the pipeline:

1. `AcousticPreprocessor.fit` / `fit_transform` must receive **training frames only**; use `transform` for validation and test frames.
2. Splitting is grouped on `participant_id`, so no participant appears in both training and test partitions. `create_folds` asserts group disjointness for every fold.
3. Model selection and early stopping use the validation partition; the test partition is scored once.

### Importing the scripts

The file names contain spaces and hyphens, so they are not importable as regular module names. Either rename them to valid identifiers (e.g. `metrics.py`) or load them by path:

```python
import importlib.util, pathlib

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

metrics = load("11-Advanced Metrics Calculator.py", "metrics")
print(metrics.summary(y_true, y_pred))
```

---

## 5. Script reference

### 11 — `11-Advanced Metrics Calculator.py`

All functions accept equal-length, finite, at-least-two-element arrays and raise `ValueError` otherwise.

| Function | Returns |
|---|---|
| `rmse(y_true, y_pred)` | Root mean squared error |
| `pearson_r(y_true, y_pred)` | Pearson correlation; `nan` if either input has zero variance |
| `ccc(y_true, y_pred)` | Lin’s concordance correlation coefficient; `nan` if the denominator is zero |
| `temporal_concordance(y_true, y_pred)` | Operational TCC — CCC computed on temporally aligned observations (identical to `ccc`) |
| `dtw_distance(a, b)` | Dynamic time warping cost, normalised by `max(len(a), len(b))` |
| `multiscale_dtw(y_true, y_pred, scales=(1, 5, 10))` | Mean normalised DTW across non-overlapping mean-pooled scales |
| `therapeutic_effect_consistency(y_true_sessions, y_pred_sessions)` | Fraction of sessions whose predicted and observed first-to-last trend signs agree (ties count as agreement) |
| `session_kendall_tau(y_true, y_pred)` | Kendall τ<sub>b</sub> for a single session |
| `summary(y_true, y_pred)` | Dict with `rmse`, `pearson_r`, `ccc`, `tcc`, `ms_dtw` |

`multiscale_dtw` truncates each series to a multiple of the pooling scale; sequences shorter than the largest scale raise `ValueError`. `therapeutic_effect_consistency` takes two equal-length lists of per-session sequences.

### 12 — `12-Cross-validation partitioning.py`

```python
create_folds(data, participant_col='participant_id', stratify_col=None, n_splits=5) -> DataFrame
```

- Uses `StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)` when `stratify_col` is given, otherwise `GroupKFold`.
- Adds an integer `fold` column: `0 … n_splits-1` for test rows, `-1` for training rows. Each row is a test row in exactly one fold.
- Raises `ValueError` if participant identifiers are missing, fewer distinct participants than folds exist, or the stratification column is absent, and asserts train/test group disjointness per fold.

CLI: `--input`, `--output`, `--participant-col`, `--stratify-col`.

### 13 — `13-Data Preprocessing.py`

```python
AcousticPreprocessor(n_components=128)
    .fit(training_frames)          # StandardScaler -> PCA, fitted on training data only
    .transform(frames)             # applies the stored scaler and PCA
    .fit_transform(training_frames)

pool_100ms_to_1hz(frame_predictions)
```

- PCA uses `svd_solver='randomized'`, `random_state=42`.
- `fit` requires a finite 2-D frame × feature matrix; `transform` rejects non-finite values.
- `pool_100ms_to_1hz` averages each complete block of ten consecutive frame predictions; fewer than ten frames raises `ValueError`, and any trailing partial block is dropped.

### 14 — `14-LSTM-LTC Model.py`

```python
LiquidLSTMCell(input_size, hidden_size, tau_min=0.1, tau_max=30.0, dt=0.1)
LSTMLTC(input_size=128, hidden_size=128, num_layers=2, dropout=0.2)
```

`LiquidLSTMCell.forward(x, state)` returns `((h, c), tau)` and computes:

- gate pre-activations from `[x; h]` via a single `Linear(·, 4·hidden)`;
- time constants `tau = tau_min + (tau_max - tau_min) · sigmoid(Linear_tau([x; h]))`, split into input/forget/output constants;
- decay factors `a_gate = 1 - exp(-dt / tau_gate)`, applied as `i = sigmoid(i_hat) · a_i`, `f = 1 - (1 - sigmoid(f_hat)) · a_f`, `o = sigmoid(o_hat) · a_o`;
- cell update `c = f · c + i · tanh(g)`, output `h = o · tanh(c)`.

`LSTMLTC.forward(x, return_taus=False)`:

- requires `x` of shape `[batch, time, features]`;
- applies `LayerNorm` to the layer input before each cell, with dropout between layers, and a `Linear(hidden, 1)` head;
- returns `[batch, time]` predictions, or `(predictions, taus)` with `taus` of shape `[batch, time, num_layers, 3·hidden]` when `return_taus=True`.

The returned τ values are the quantities analysed by script 17 (converted to effective memory horizons before clustering).

### 15 — `15-Baseline Models.py`

| Class | Description |
|---|---|
| `StandardLSTM(input_size=128, hidden_size=128)` | Two-layer `nn.LSTM` (`batch_first=True`, dropout 0.2) with a linear scalar head |
| `AttentionLSTM(input_size=128, hidden_size=128, window=32)` | Single-layer LSTM; at each timestep a softmax over the previous `window` states produces a local attention summary |
| `TCN(input_size=128, hidden_size=128)` | Eight dilated `Conv1d` blocks, kernel 3, dilations `[1, 2, 4, 8, 16, 32, 64, 128]`, each output cropped back to the input length, followed by a 1×1 convolution head |
| `PsychoacousticRidge(alpha=1.0)` | Ridge regression on handcrafted acoustic descriptors; `.fit(x, y)` / `.predict(x)` |

All models consume `[batch, time, features]` and return `[batch, time]`.

### 16 — `16-Model Training.py`

```python
train_model(model, x_train, y_train, x_val, y_val,
            epochs=100, batch_size=32, lr=0.001, patience=10, device='cpu')
    -> (model, {'best_validation_mse': float, 'epochs_completed': int})
```

- MSE loss, Adam, `ReduceLROnPlateau(mode='min', factor=0.1, patience=5)`, gradient clipping at norm 1.0.
- Shuffled mini-batches from a `TensorDataset`; inputs cast to `float32`.
- The best validation-MSE state dict is deep-copied and restored before returning; training stops after `patience` epochs without improvement.
- The training function does not set random seeds. Seed `torch`, `numpy` and the DataLoader generator externally if deterministic runs are required.

### 17 — `17-Temporal Analysis.py`

```python
analyze_horizons(horizons, components=range(1, 7), random_state=42) -> dict
```

- Fits one `GaussianMixture` per candidate component count and selects the minimum-BIC model.
- Requires positive, finite horizons and at least `max(components)` samples.
- Returns `selected_components`, `bic` (per candidate), `centers_seconds` (sorted component means), `silhouette` and `calinski_harabasz` (`None` when the selected model has a single component).

In the paper, K = 3 is selected by BIC, with cross-run centres of 0.38 ± 0.04 s, 4.71 ± 0.21 s, and 20.85 ± 1.12 s, a silhouette of 0.74, and a Calinski–Harabasz index of 138.6.

### 18 — `18-Statistical Comparison.py`

```python
paired_comparisons(reference, baselines) -> dict
mean_ci(values, confidence=0.95) -> (lower, upper)
```

- `reference` is a fold-level score vector; `baselines` maps model names to equally shaped fold-level vectors (minimum two folds).
- Per baseline: `mean_difference` (reference − baseline), `cohens_dz` (`mean(Δ) / sd(Δ)`, or `None` if `sd = 0`), `p_raw` from `scipy.stats.ttest_rel`, and `p_holm` after a Holm step-down adjustment across all baselines.
- `mean_ci` returns a *t*-distribution interval for the fold mean.

---

## 6. Notes on implementation

- **Time-constant range.** `LiquidLSTMCell` bounds τ by `tau_min=0.1` and `tau_max=30.0` seconds so that the long-term regime (> 10 s) reported in the paper is representable; the method section states an initialisation range of [0.1, 10.0]. Set `tau_max` explicitly if a narrower range is required.
- **Normalisation.** The implementation uses `LayerNorm` on each layer’s input. The manuscript describes batch-style normalisation of input features; both serve the same stabilising role, and the choice is not part of the reported results.
- **Temporal resolution.** `dt = 0.1` matches the 100 ms frame rate. Predictions are pooled to the 1 Hz annotation grid with `pool_100ms_to_1hz` before computing agreement with ground truth.
- **Evaluation window.** Comparisons use fixed-length inputs of 300 frames (30 s); longer sessions must be windowed, and session-level metrics are computed on the reconstructed per-session sequences.
- **Reproducibility.** Fixed seeds: fold assignment (42), PCA (42), GMM (42). Training runs are not seeded inside `train_model`.

---

## 7. Results reported in the paper

Five-fold participant-wise cross-validation on MTERD with 300-frame (30 s) windows.

LSTM-LTC: RMSE = 0.21 ± 0.02, Pearson r = 0.85 ± 0.02, CCC = 0.83 ± 0.03, TCC = 0.84 ± 0.03, MS-DTW = 0.42 ± 0.03, TEC = 0.91 ± 0.02. 95% CIs: RMSE [0.19, 0.23], Pearson r [0.83, 0.87], CCC [0.79, 0.87].

Relative to Attention-LSTM, the reported descriptive gains are 19.23% (RMSE), 10.39% (Pearson r) and 12.16% (CCC); MS-DTW falls by 23.4%. Attention-LSTM scores TCC = 0.76, MS-DTW = 0.55 and TEC = 0.85. All pairwise improvements are significant after Holm–Bonferroni correction (*p* < 0.001), with Cohen’s d<sub>z</sub> = 1.96 (RMSE), 3.14 (Pearson r) and 2.54 (CCC) against the strongest baseline.

Ablation: removing the liquid time-constant mechanism lowers TCC from 0.84 to 0.68 (18.7% relative); fixing all time constants to a single value lowers TEC by 15.2%.

The full per-model tables (standard regression metrics; TCC/MS-DTW/TEC; Kendall τ<sub>b</sub> validation) are reported in the manuscript’s Tables 2–3 and Supplementary Tables S1–S2.

---

## 8. Scope and limitations

- The learned time constants and feature attributions are data-driven statistical quantities. The mapping of the 0.38 s / 4.71 s / 20.85 s clusters onto rapid, intermediate and long-term auditory-emotional processing is a heuristic interpretation, not a causal neurophysiological measurement.
- The scripts model auditory features only; physiological channels (EEG, ECG, GSR) are not used as model inputs.
- Sessions longer than the evaluation window are analysed as independent windows; hierarchical session-level modelling is left to future work.
- Cultural and genre coverage is inherited from MTERD, which is weighted toward Western musical traditions.

---

## 9. Data and ethics

MTERD is a public, fully de-identified benchmark dataset; no new participants were recruited and no protected health information is included. Secondary analysis follows the Declaration of Helsinki. See the Kaggle page above for the original collection and annotation protocol.

## 10. Citation

```bibtex
@article{zhang_multiscale_auditory_lstmltc,
  title  = {Multi-scale Auditory Psychological Temporal Analysis: Quantifying
            Emotional Regulation in Music Therapy via LSTM-LTC Modeling},
  author = {Zhang, Linxi and Zhang, Jialu and Zhan, Zitong},
  note   = {Manuscript}
}
```

Contact: Zitong Zhan (corresponding author), `z_zitong_acad@163.com`.

## 11. License

No license file is included with these scripts and the manuscript does not state one. Contact the corresponding author (Zitong Zhan, `z_zitong_acad@163.com`) before redistributing or reusing the code beyond academic verification.

> （注：部分内容由豆包 AI 生成）