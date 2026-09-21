# Raw 960-reaction benchmark input

`transition1x_960.tar.gz` contains the 960 raw reaction-pair XYZ files used to
define the benchmark cohort. Each file stores the reactant and product
structures for one reaction ID. The archive is the unmodified `ts1x.tar.gz`
distributed with the local ReactBench checkout and matches the expanded
ReactBench `data/ts1x/` directory file by file.

This directory intentionally contains no React-OT or GSM TS guesses, energies,
ML predictions, Gaussian inputs or outputs, optimization outcomes, IRC
classifications, timing tables, selected top-100 subsets, or other processed
data.

## Integrity

- reactions: 960
- archive SHA256:
  `bf39f131988328a9368f410dd6e98045925f7cad62df70a13fdd3cfa131c88a3`
- source ReactBench commit:
  `00065a96ab86cd5bcac2bc2b33885671bac11623`

`reaction_ids.txt` lists the included files. `XYZ_SHA256SUMS` records the
SHA256 checksum of every expanded XYZ file.

Extract with:

```bash
tar -xzf data/transition1x_960.tar.gz -C data
```

## Attribution

The reaction IDs follow the ReactBench study by Zhao et al. The corresponding
records originate from Transition1x. Cite ReactBench and Transition1x using the
entries in `../CITATIONS.bib`. The source terms and attribution notices are
summarized in `../THIRD_PARTY_NOTICES.md`.
