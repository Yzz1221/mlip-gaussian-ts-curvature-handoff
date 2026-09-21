---
name: mlip-gaussian-ts-validation
description: Validate this repository's Gaussian TS Opt+Freq outputs with the manuscript's single -10 cm^-1 imaginary-frequency criterion and classify bidirectional IRC endpoints with ReactBench connectivity. Use when computing recovery counts or auditing individual reactions.
---

# MLIP Gaussian TS validation

Use the included canonical implementations:

- `analysis/audit_ts_freq_logs.py::classify` for Opt+Freq. Report only
  `paper_ok_minus_10` as the success field.
- `analysis/classify_gaussian_irc_intended.py::classify_intended` for IRC
  endpoint identity. It imports the pinned ReactBench `table_generator`.
- `analysis/validate.py` for case-level results and fixed-denominator rates.

The Opt+Freq parser is copied from the manuscript audit script with SHA256
`0750dfa5b9e23546136ff2aa1be0720179edd156935771aab13c91ae3dd85e11`.
The IRC classifier is copied from the manuscript analysis script with SHA256
`91450aa8748137ecaba83492438a249e96138312d95b72d3511a969bf870f9b6`.
Do not infer success from termination alone. A passing calculation requires
normal Opt and final Freq termination without error, `Stationary point found`
after the last completed Opt, all four Opt convergence tests marked `YES`,
and exactly one frequency below **-10 cm^-1** in the final complete frequency
block. Earlier or incomplete frequency blocks do not count.

For IRC Intended, use the largest-numbered `Input orientation` geometry in
each FORWARD and REVERSE path. Compare their ReactBench adjacency matrices
with the two-frame Transition1x reference XYZ in either direction. The
reference frames are reactant first and product second in
`data/transition1x_960.tar.gz`. The ReactBench matching rule ignores
differences involving Zn, Mg, Li, Si, Ag, Cu, and B; identical raw IRC
endpoint adjacency matrices indicate `ConformationChange`, not Intended.
An IRC log with unextractable endpoints is `ClassificationError`.

Use the full dataset denominator: 871 for GSM and 960 for React-OT. Keep
missing and failed runs in the denominator. Count IRC Intended recovery only
for reactions that pass Opt+Freq and have both intended IRC endpoints; retain
IRC endpoint classification separately for existing logs from other cases.

For a run tree with `<runs>/rxn<ID>/opt_freq.log` and `irc.log`:

```bash
python analysis/validate.py --dataset react_ot --workflow oneshot \
  --runs runs/oneshot --output runs/oneshot_validation.json
```

Before reporting counts, verify that the parser and pinned ReactBench source
hashes match the versions described above and in
`analysis/classify_gaussian_irc_intended.py`. Do not modify Gaussian logs or
checkpoint files during analysis.
