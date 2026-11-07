# Variable mapping

Paths in the manifest are resolved relative to the manifest file. Session IDs must
be globally unique, participant IDs must remain stable across all their sessions.
The supplied manifest is a header-only schema, with no invented participant rows.

| Field | Type | Role |
| --- | --- | --- |
| participant_id | string | Outer and inner grouping key; never a predictive input |
| session_id | string | Unique session key; never a predictive input |
| audio_path | relative path | Optional for precomputed features; required by script 03 |
| features_path | relative path | NPZ with X, time, feature_names |
| annotations_path | relative path | CSV with timestamp and an observed target column |
| eeg_path, ecg_path, gsr_path | relative paths, optional | Retained metadata; excluded from the auditory model |
| condition | string, optional | Descriptive metadata; current GroupKFold does not stratify conditions |
| source_version | string | Verified upstream release identifier, if available |

## Features NPZ

- `X`: finite float matrix `[frames, features]`, normally 6,373 raw acoustic features.
- `time`: strictly increasing feature endpoint times in seconds, spaced by 0.1 s.
- `feature_names`: unique string array in the same feature order across all sessions.
- `psycho_X`: optional finite matrix `[frames, psycho_features]` required for the
  psychoacoustic baseline. It uses the same time axis.
- `psycho_feature_names`: names for psycho_X; default required names are
  `loudness`, `sharpness`, `roughness`. Supply measured, documented descriptors.

Script 03 extracts ComParE functionals only. It does not automatically implement
psychoacoustic sharpness or roughness; externally verified descriptors can be
added to each NPZ. The baseline fails if these are absent. ComParE names must not
be arbitrarily relabeled as psychoacoustic quantities. A different descriptor set
can be configured, but must be reported as a different baseline specification.

## Annotation CSV

Required: `timestamp` in seconds and the configured target, normally
`regulation_score`. Optional alternative targets: `valence`, `arousal`.
Every target value must be observed and have a documented scale. A valence or
arousal target is a separate experiment, not an interchangeable regulation score.
No formula converting valence/arousal to clinical regulation is supplied by the
manuscript, so the code does not invent one. All outputs retain target units.
A session-level scalar alone cannot support the temporal metrics implemented here.

If labels refer to interval centers rather than instants, convert their timestamps
using the upstream annotation protocol before running the pipeline. Resolve
annotation latency and recording offsets upstream; the code assumes a common
session clock. Missing/nonfinite labels are rejected rather than filled silently.

## Alignment

The extractor records trailing-window endpoints. The model uses 300 feature rows
ending at the prediction time, approximately 30 s of frame context. Targets are
linearly interpolated only inside their observed time range. Held-out frame
predictions are averaged over trailing 1 s bins `(k-1,k]` and compared with the
observed/interpolated target at second k. Incomplete bins are dropped. No context
crosses a session boundary. There is a full-window warm-up at each session start.
