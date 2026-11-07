# Verification report

Prepared on 2026-11-07. All checks below use software test fixtures rather than
original MTERD participant data. Exact direct-package versions are recorded in
`tested_environment.txt`.

| Check | Result |
| --- | --- |
| Source compilation | Passed for all Python modules |
| Command entry points | All 10 numbered scripts accept --help |
| Unit checks | 8 passed using Python unittest |
| Five-model end-to-end run | Passed over five participant-disjoint folds, one seed, one epoch, tiny dimensions |
| Train/validation/test grouping | No participant overlap within a fold; each fixture participant tested in one outer fold |
| Window handling | Full windows do not cross sessions; trailing 1-Hz pooling matches expected timestamps |
| Preprocessing isolation | Scaler fit excludes unseen distribution; transform does not update fitted statistics |
| Model differentiation | All four neural models support finite input gradients; dynamic tau has trainable gradients |
| Tau and horizon bound | Stated tau range and Eq 11 upper bound checked numerically |
| Metrics | Identity-case RMSE, CCC, DTW and TEC verified; nonfinite and duplicate predictions rejected |
| Statistical grouping | Seeds averaged before five fold-level paired comparisons; Holm p-values checked |
| Actual openSMILE call | Passed: 11 x 6,373 finite functionals from a generated 44.1-kHz noise WAV, 1-s trailing windows, 0.1-s hops |
| GMM/BIC and figures | Completed on measured smoke-run horizons; selected K was not forced to 3 |
| Figure inspection | Smoke-run performance and attribution charts visually checked for readable layout |
| Kaggle client | Import checked; original dataset download and schema not validated |

The complete smoke test trained 25 tiny runs: five methods x five folds x one
seed. Its outputs were kept outside the empirical results directories and are not
shipped as research results. The GMM was run on eight test-model neurons per run,
not the paper's full 256 neurons. These fixture outputs must not be compared with
the manuscript's therapeutic findings.

## Not performed

- Training on the original 1,200 sessions or evaluation of original clinical labels.
- Recovery or validation of original participant demographics or physiological signals.
- Full-size 128-component/128-hidden-unit, 300-step, three-seed GPU experiments.
- Exact recovery of the unspecified score construction, extraction window or MS-DTW conventions.
- Upload to a remote GitHub repository.

## Repeat the checks

```bash
python -m unittest discover -s tests -v
python tests/smoke_pipeline.py --out _smoke_check
python tests/check_audio_extraction.py --out _audio_check
```

Use new output paths; the checks refuse to overwrite an existing fixture folder.
