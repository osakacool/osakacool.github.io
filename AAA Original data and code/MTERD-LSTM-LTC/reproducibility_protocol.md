# Reproducibility protocol

1. Archive the verified upstream dataset version, its license, files and hashes.
   Keep original bytes unchanged. Do not publish private identifiers or credentials.
2. Fill the session manifest from actual files. Verify session/participant linkage,
   the common time origin, label scale, missingness, recording offsets and all
   modality claims. The downloader inventory is not a session manifest.
3. Document the observed regulation-score definition. If only valence/arousal
   exists, use a separately named target experiment and describe it accurately.
4. Extract or supply feature matrices using an explicit acoustic window and hop.
   Inspect finite outputs; do not fill pathological functionals with arbitrary
   zeros. Supply separately verified psychoacoustic descriptors for the ridge model.
5. Run data validation and, for the manuscript dataset, the strict count/dimension
   check. Strict validation checks observed counts/dimensions, not clinical provenance.
6. Freeze the config and outer/inner participant lists. Keep all sessions from a
   participant in exactly one outer test fold. Never select a session after seeing
   model errors to present it as representative.
7. Fit scaling and PCA inside each training partition, with validation and test
   participants excluded. Check and report actual variance retained by 128 PCs.
8. Run each neural model with seeds 17, 29, 43 on every outer fold. The ridge model
   is deterministic; repeated seed labels are retained for matched artifact layout.
   Save the best validation checkpoint, full training history, package versions,
   input hashes, runtime and GPU memory usage. Never select by held-out performance.
9. Use held-out full-window predictions and one-second pooling. Recompute all
   metrics; do not paste values from the manuscript into results/. Check identical
   test points across every method and seed before inferential comparisons.
10. Select GMM component counts with BIC separately in every fold/seed. Report
    the actual selected counts and validity statistics. Do not force three modes.
11. Generate plots from saved outputs. Attribution gradients are in original
    acoustic feature coordinates after the scaler/PCA chain rule. Lag bins are
    computational context ranges, not proven clinical/physiological mechanisms.
12. Review implementation_notes.md against the final manuscript. Resolve the
    target, MS-DTW, TEC and memory-horizon discrepancies before describing results
    as an exact replication. The code can be used for a clearly documented reanalysis.

## Run provenance

Each run directory contains run.json with configuration, participant splits,
input SHA-256 hashes, package versions, seed, hardware and timings. Neural runs
also include checkpoint.pt, training_history.csv, preprocessor.joblib and
pca_report.json. LSTM-LTC additionally exports measured memory_horizons.csv and
attributions.csv. The top-level splits.json can be checked before training.
Joblib and torch checkpoints should only be loaded from trusted sources.

## Test data and empirical data

The tests create small artificial fixtures in explicitly named temporary or
_smoke directories. They test code behavior and do not reproduce the paper's
sample, p-values, therapeutic effects or numerical results. Test outputs are not
included under the empirical results directories. No trained research checkpoints
or original MTERD recordings are part of this archive.

## Publication preparation

README, code, config, manifest schema and this protocol are ready as repository
materials. Select a repository license before publishing; a software license has
not been assigned on the manuscript authors' behalf. Check the dataset and
openSMILE licenses separately. CITATION.cff uses the author names from the
provided title page and has no invented DOI or publication date. No remote
repository was modified when preparing this package.
