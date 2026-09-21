# Benchmark inputs and TS-guess datasets

The two prepared initial-guess collections are distributed separately:

| Directory | Initial-guess method | Structures |
|---|---|---:|
| `gsm/` | GSM on the MLIP PES | 871 |
| `react_ot/` | React-OT | 960 |

Each `rxn<ID>.xyz` contains one unoptimized TS guess with Cartesian coordinates
in angstrom. Files are copied without modification from the source population
used to construct the core-scaling calculation manifests. Atom ordering and
coordinates were checked against the corresponding Gaussian starting inputs.
The per-directory `manifest.csv` lists reaction ID, atom count, charge,
multiplicity, relative source path, and SHA256 of each structure.

GSM has 871 structures because 89 of the 960 reactions did not produce a
converged GSM path. These are complete available initial-guess collections,
not subsets selected by subsequent Gaussian optimization success. Matching
reaction IDs identify the same benchmark reaction across the two directories.

## Original reaction-pair archive

`transition1x_960.tar.gz` contains the 960 raw reaction-pair XYZ files used to
define the benchmark cohort. Each file stores the reactant and product
structures for one reaction ID. The archive is the unmodified `ts1x.tar.gz`
distributed with the local ReactBench checkout and matches the expanded
ReactBench `data/ts1x/` directory file by file.

The reaction-pair archive is distinct from the generated initial guesses in
`gsm/` and `react_ot/`. Neither initial-guess directory contains optimized TSs,
Gaussian calculation outputs, recovery classifications, or timing results.

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
