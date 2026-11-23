# Paper cohort timing and outcome tables

These tables support the size-selected React-OT and GSM comparisons in *One-Shot Machine-Learned Hessian Initialization for Gaussian Transition-State Searches*. Each cohort contains100 reactions (17–21 total atoms). Selection sorts atom count descending and numeric reaction ID ascending. The cohorts are selected separately and have overlapping reaction identities; never pool them as independent chemistry tests.

## Files

| File | Contents |
|---|---|
| `cohort_ids.csv` | Exact selected reaction IDs, ranks and atom counts for both cohorts. |
| `optimization_outcomes.csv` | All800 selected first-attempt workflow outcomes, including nonacceptances. |
| `native_and_oneshot_frequency_fields.csv` | Unmodified canonical classification fields for the600 native-QM/OneShot records. |
| `recorded_timing_events.csv` | Original per-case timing TSV fields arranged in600 wide rows. Scientific event values are unchanged; node/job identifiers and local paths are omitted. Empty fields remain missing, not zero. |
| `recorded_search_times.csv` | The archived preparation-inclusive search metric used for native-QM/OneShot comparisons, with model-load share and acceptance. |
| `external_source_timing_records.csv` | All368 timing rows in the original native-CalcAll/External source table, plus the audited primary-acceptance flag. The original table was not a full400-attempt timing ledger. Use the flag when selecting accepted cases. Missing failed-case timings are not imputed. |
| `paired_timing_case_ratios.csv` | All joint-primary-success OneShot timing pairs with source fingerprints and descriptive tail flags. |
| `paired_timing_statistics_newaudit.csv` | Untrimmed paired geometric-mean ratios with reaction-bootstrap intervals. |
| `paired_optfreq_2x2.csv`, `paired_risk_difference_and_McNemar.csv` | Paired binary outcomes, conservative uncertainty intervals and exact tests. |
| `frequency_threshold_summary.csv` | Canonical −10/−50cm−1 sensitivity, retaining the original criteria. |
| `paired_timing_exclusions.csv` | Case IDs and reasons excluded from timing pairs. |
| `external_control_paired_time_ratios.csv` | Eligible native-CalcAll/External timing pairs. |
| `model/` | Checkpoint identifier and effective current-code model configuration; see its scope note. Checkpoint parts are in the repository's `ckpt/` directory. |
| `provenance.json` | Parser/source fingerprints, release scope and file integrity records. |
| `validate_tables.py` | Lightweight structural and numerical consistency checks on the published tables. |

## Workflow and outcome definitions

`Gaussian--CalcFC`, `Gaussian--CalcAll`, `OneShot` and `External--CalcAll` are the four reported workflows. The primary Opt+Freq rule is the repository's `analysis/audit_ts_freq_logs.py::classify` field `paper_ok_minus_10`: normal optimization and final frequency termination, stationary-point detection, all four convergence flags YES, and exactly one frequency below−10cm−1. Neither the criterion nor any source result is changed by this release. Exactly-one-mode tests at different thresholds need not be nested.

All100 inputs per dataset/workflow remain in `optimization_outcomes.csv`. SchemeE rescues, which request periodic QM Hessians, are excluded from these first-attempt results. The first-attempt acceptances are React-OT99/97/98/90 and GSM89/95/87/86 in CalcFC/CalcAll/OneShot/External order. These are optimization/frequency acceptances, not a uniformly current-checkpoint-validated IRC recovery count. No IRC outcome is inferred from optimization acceptance.

## Timing boundaries

Timing values are seconds. `recorded_timing_events.csv` preserves the original timer fields; for example, `gaussian_elapsed_blocks_s` separates linked Gaussian elapsed blocks, and wrapper/preparation totals can include final frequency work. Do not substitute those whole-job totals for the paper's search-only metric. `recorded_search_times.csv` provides the separately reconstructed metric used for the OneShot comparison.

That search metric includes the target-QM single point, recorded ML inference/loading share, checkpoint conversion/insertion/reconstruction, local preparation and TS optimization. It excludes final frequency/IRC, staging and queue time. Prediction-side tensor transfer/conversion/symmetrization/artifact saving outside the archived inference timer are also excluded. This is not complete batch makespan, isolated Hessian arithmetic cost, or cost per intended recovered pathway.

The External control uses Gaussian's first elapsed block and its separately recorded legacy model-loading allocation. Its GPU configuration differs from the native-QM CPU configuration. Do not interchange its timer convention with the native/OneShot wrapper-based metric, or interpret that control as hardware-normalized algorithmic speedup. The primary timing comparisons condition on both workflows passing Opt+Freq; all eligible pairs are retained. Bootstrap intervals describe variation across reactions, not repeated-run noise or statistical equivalence.

## Data provenance and reuse

The files are source tables transcribed or parsed from archived calculation records, with original numeric values retained; they are not raw Gaussian output logs. No new quantum calculations or timing runs are represented by this release. Original-log/timer fingerprints support traceability. Local filesystem prefixes and infrastructure identifiers are omitted from this public release. Model weights and input trees already in the repository remain separate. The current-code model configuration is not a reconstruction of unavailable historical uncommitted execution state or a verified training-reaction split.

Run `python data/paper_top100/validate_tables.py` from the repository root. Python3 standard library is sufficient.
