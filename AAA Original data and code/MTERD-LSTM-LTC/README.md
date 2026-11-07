# MTERD LSTM LTC

Code and research materials prepared from **Multi-scale Auditory Psychological
Temporal Analysis: Quantifying Emotional Regulation in Music Therapy via LSTM-LTC
Modeling**. This package contains an equation-based implementation, four baselines,
participant-disjoint training, metrics, memory-horizon analysis and figure scripts.

This is newly prepared code, not recovered original experiment code. The supplied
manuscript provides aggregate values but not the raw session data, original model
weights, target construction rule or complete metric specifications. The package
does not bundle MTERD recordings or claim to reproduce the published values.
Start with [implementation notes](docs/implementation_notes.md) and the
[reproducibility protocol](docs/reproducibility_protocol.md).

## Contents

| Path | Contents |
| --- | --- |
| requirements.txt, environment.yml | Python and conda environment definitions |
| configs/default.json | Manuscript settings and documented implementation choices |
| data/raw, data/processed | Locations for upstream and derived data, initially empty |
| scripts/01_download_dataset.py | Download the stated Kaggle resource and inventory files |
| scripts/02_validate_dataset.py | Validate actual session features, targets and manifest |
| scripts/03_extract_audio_features.py | Rolling ComParE functionals at 100-ms hops |
| scripts/04_preprocess_pca.py | Training-only foldwise scaler/PCA exports |
| scripts/05_train_lstm_ltc.py | Proposed model training and held-out prediction |
| scripts/06_train_baselines.py | Standard LSTM, TCN, Attention LSTM, psychoacoustic Ridge |
| scripts/07_cross_validation.py | Common five-fold experiment runner for all methods |
| scripts/08_evaluate_metrics.py | Regression and temporal metrics, paired comparisons |
| scripts/09_gmm_bic_analysis.py | Per-run neuron horizon clustering and BIC selection |
| scripts/10_generate_figures.py | Figures based exclusively on actual run outputs |
| models | LSTM LTC, Standard LSTM, Attention LSTM and causal TCN |
| mterd | Shared data, preprocessing, training, evaluation and attribution utilities |
| docs | Manifest schema, variable mapping, exact metric conventions and provenance |
| tests | Behavior tests and an artificial end-to-end smoke fixture |
| results/metrics, results/predictions, results/figures | Reserved output locations |

## Environment

Use Python 3.12, 64-bit. Run commands from the repository root.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, from the root: `conda env create -f environment.yml`, then
`conda activate mterd-lstm-ltc`. For a specific GPU/CUDA platform install the
appropriate PyTorch build using its official installation instructions. The
runtime chooses CUDA if available, otherwise CPU. Actual versions and hardware
are saved in run.json; compatible dependency ranges are not a historical lockfile.

## Data preparation

```bash
python scripts/01_download_dataset.py --hash-files
```

The downloader attempts the manuscript-listed Kaggle slug. Authentication, access
and the actual dataset file structure depend on the upstream release. Download
success is not evidence of the manuscript's participant counts or modalities.
If access fails, the script stops; it does not create replacement observations.

Populate `docs/dataset_manifest.csv` from the actual session files. Read
[variable_mapping.md](docs/variable_mapping.md) for the exact NPZ and annotation
CSV schema. Manifest paths are relative to the manifest itself. Required labels
are timestamped observed `regulation_score` values. If the dataset only contains
valence/arousal, resolve the target definition or conduct a separately labeled
valence/arousal experiment by changing `target` in a copied config.

For audio extraction, explicitly select the trailing functional-window duration:

```bash
python scripts/03_extract_audio_features.py --manifest docs/dataset_manifest.csv --window-seconds 1.0 --output-manifest docs/dataset_manifest_extracted.csv
python scripts/02_validate_dataset.py --manifest docs/dataset_manifest_extracted.csv --strict-manuscript
```

The 1.0-s window above is an explicit example choice, not a recovered manuscript
setting. Supply genuine sharpness, roughness and loudness arrays separately for
the psychoacoustic baseline, or run neural methods only until these are available.
Precomputed feature users may directly fill features_path and skip extraction.

## Full experiment

```bash
# Optional preprocessing inspection. Training refits the same foldwise transform.
python scripts/04_preprocess_pca.py --manifest docs/dataset_manifest.csv
# All models, five outer folds and three seed labels per model
python scripts/07_cross_validation.py --manifest docs/dataset_manifest.csv --out results/predictions/full
python scripts/08_evaluate_metrics.py --runs results/predictions/full --out results/metrics
python scripts/09_gmm_bic_analysis.py --runs results/predictions/full --out results/metrics/gmm
python scripts/10_generate_figures.py --runs results/predictions/full --metrics results/metrics --out results/figures
```

Use the extracted manifest path instead if script 03 produced your features.
The `full` run contains one directory per model/fold/seed, with predictions and
associated artifacts. The deterministic ridge fit is repeated under the same
seed labels for consistent matching. Seeds are averaged within fold before
statistical tests. The code refuses to overwrite an existing run directory.
Training all models produces 75 run directories with default settings.

Scripts 05 and 06 are alternatives for separate model training, using the same
runner, manifests and split settings:

```bash
python scripts/05_train_lstm_ltc.py --manifest docs/dataset_manifest.csv --out results/predictions/proposed
python scripts/06_train_baselines.py --manifest docs/dataset_manifest.csv --out results/predictions/baselines
```

For a neural-only run, use script 07 with
`--models lstm_ltc standard_lstm tcn attention_lstm`. `--folds 0` and
`--seeds 17` allow a preliminary fit. To modify settings, copy default.json and
pass `--config path/to/your_config.json` consistently to training and evaluation.
For a fixed-tau ablation set `fixed_tau` in a copied config and use a separate
output directory; compare with the dynamic run on the same held-out observations.

## Outputs

- **Training:** checkpoint.pt, preprocessor.joblib, training_history.csv, run.json,
  pca_report.json; ridge.joblib/ridge_validation.csv for the classical baseline.
- **Predictions:** one-second held-out target and prediction rows with participant,
  session, fold and seed IDs. No extrapolated labels or cross-session windows.
- **Metrics:** per-session, fold/seed, seed-averaged fold, summary and paired tests.
- **Interpretation:** memory_horizons.csv, attributions.csv, GMM BIC/components/validity.
- **Figures:** computed performance, temporal metrics, held-out session trajectory,
  measured horizon histogram, BIC curves and feature-gradient heatmap in PNG/PDF.

Figures 1 and 8 are conceptual diagrams in the manuscript, not numerical plots
recovered from data. The plotting script does not fabricate them or reconstruct
unavailable fold distributions. `docs/manuscript_reported_metrics.csv` transcribes
Tables 2 and 3 as a clearly labeled reference; it is never read by the training,
evaluation or plotting pipelines.

## Verification

```bash
python -m unittest discover -s tests -v
python tests/smoke_pipeline.py --out _smoke_check
```

The smoke command creates artificial, explicitly labeled fixtures and uses tiny
settings for five folds and all five methods. It checks the software pipeline,
not the paper's experimental results. Keep smoke output outside results/.
See [QA report](docs/qa_report.md) for checks actually run during preparation.

## Licensing and citation

CITATION.cff lists the manuscript's authors. A code license has not been assigned
on their behalf. Confirm the license before publication and comply separately
with upstream data and openSMILE terms. No credentials, recordings, original
participant records or original trained research checkpoints are included.
