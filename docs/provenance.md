# Provenance and exclusion policy

This code-only repository is assembled from the research implementation while
leaving the original experiment and manuscript directories unchanged.

## Included

- checkpoint-oriented one-shot Hessian handoff;
- Gaussian External EIn/EOu adapter for controls;
- Link1-aware Opt+Freq classifier;
- minimal Gaussian and Slurm templates;
- unit tests and reproducibility documentation.

## Not included

- HORM/EquiformerV2 source code or model checkpoints;
- Transition1x, ReactBench, GSM, or React-OT datasets;
- Gaussian executables or licensed files;
- production gjf/chk/fchk/log/IRC trees;
- manuscript, SI, figures, figure source data, or unpublished tables;
- machine-specific absolute paths, scheduler accounting, or credentials.

The upstream code/data licenses must be followed independently. Before an
archival release, record the exact upstream commits, checkpoint SHA256, Python
environment, Gaussian revision, hardware, and repository commit.
