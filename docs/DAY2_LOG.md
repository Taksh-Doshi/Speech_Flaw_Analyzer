# Day 2 Log: Speech Flaw Injectors and Dataset Gradient

## What was built

- Added `apply_flaws` with sample-based stitching: flaw specifications refer to
  the original timeline, modified spans are concatenated with untouched gaps,
  and returned labels use the resulting output timeline. Flaw labels retain the
  corresponding original span in `params` for validation.
- Added six flaw types:
  - `speed_up` → `too_fast`
  - `slow_down` → `too_slow`
  - `monotone`
  - `add_pause` → `long_pause`
  - `remove_pause` → `missing_pause`
  - `volume_drop` → `low_volume`
- Implemented deterministic dataset generation from prepared 16 kHz mono
  baselines and rough word alignments. Sentence and 4–12-word candidates are
  selected without overlap; choices use `set_global_seed(42)`.
- Added generated L0 metadata and L1–L5 WAV/JSON output. The levels contain
  1, 2, 3, 4, and 6 flaws, with the requested severity progression; L5
  includes a monotone span of at least 10 words.
- Added per-label checks for duration change and word rate, voiced F0 variation,
  silence duration/level, pause removal, and RMS level. A failed check retries
  once on another candidate span before the build exits unsuccessfully.
- Recorded generated level descriptions and file counts in
  `data/DATASET_CARD.md`.

## Validation results

The `speech01` build completed with every listed check passing:

| Level | Flaw type | Start (s) | End (s) | Severity | Check passed |
|---|---|---:|---:|---:|:---:|
| L1 | long_pause | 67.840 | 68.720 | 0.3 | yes |
| L2 | monotone | 19.080 | 24.620 | 0.5 | yes |
| L2 | too_slow | 62.980 | 68.147 | 0.4 | yes |
| L3 | too_fast | 16.840 | 17.963 | 0.6 | yes |
| L3 | missing_pause | 69.903 | 71.833 | 0.6 | yes |
| L3 | low_volume | 113.233 | 117.633 | 0.6 | yes |
| L4 | long_pause | 25.080 | 26.760 | 0.8 | yes |
| L4 | missing_pause | 31.700 | 35.340 | 0.8 | yes |
| L4 | too_slow | 54.120 | 62.914 | 0.8 | yes |
| L4 | monotone | 69.574 | 75.074 | 0.8 | yes |
| L5 | too_fast | 8.860 | 11.633 | 1.0 | yes |
| L5 | monotone | 28.633 | 33.273 | 1.0 | yes |
| L5 | low_volume | 40.313 | 44.893 | 1.0 | yes |
| L5 | missing_pause | 59.613 | 62.713 | 1.0 | yes |
| L5 | too_slow | 89.053 | 96.120 | 1.0 | yes |
| L5 | long_pause | 109.560 | 111.560 | 1.0 | yes |

The full test suite passed: **33 passed**, with two existing dependency
deprecation warnings (`audioop` and `python_multipart`). A repeated dataset
build produced byte-identical WAV and JSON outputs. The L5 monotone span covers
11 words.

## Problems found

- A short synthetic span was not measurable by `pyin` because it produced no
  voiced frames. The builder now treats that measurement error as a failed
  check and retries the flaw once on a different candidate span.
- The test run reports deprecation warnings from dependencies; they do not
  fail tests or affect this dataset build.

## Next steps

- Prepare and process additional baseline recordings to expand beyond
  `speech01`.
- Listen to the generated examples and review the automatic checks against
  perceptual quality, especially pause detection and pitch flattening.
- Consider reporting validation measurements alongside pass/fail results to
  make future dataset audits easier.
