# Provenance and exclusion policy

This repository is assembled from the research implementation while leaving
the original experiment and manuscript directories unchanged.

## Included

- checkpoint-oriented one-shot Hessian handoff;
- Gaussian External EIn/EOu adapter for controls;
- Link1-aware Opt+Freq classifier;
- minimal Gaussian and Slurm templates;
- unit tests and reproducibility documentation;
- the raw 960-reaction benchmark input archive;
- release metadata and reconstruction instructions for the production
  EquiformerV2 checkpoint;
- pinned ReactBench and HORM source trees as Git submodules.

## Not included

- processed or derived benchmark datasets;
- GSM and React-OT generated TS guesses;
- Gaussian executables or licensed files;
- production gjf/chk/fchk/log/IRC trees;
- manuscript, SI, figures, figure source data, or unpublished tables;
- machine-specific absolute paths, scheduler accounting, or credentials.

The upstream code and data licenses apply independently. Exact upstream
commits, citations, licenses, and cryptographic checksums are recorded in
`THIRD_PARTY_NOTICES.md`, `data/README.md`, and `models/README.md`.
