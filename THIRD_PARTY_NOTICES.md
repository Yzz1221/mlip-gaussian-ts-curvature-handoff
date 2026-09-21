# Third-party notices

This repository combines original workflow code with third-party source code,
data, and a model checkpoint. Each third-party component remains subject to
its own license and attribution requirements.

## ReactBench

- upstream: <https://github.com/deepprinciple/ReactBench>
- pinned commit: `00065a96ab86cd5bcac2bc2b33885671bac11623`
- location: `third_party/ReactBench`
- license: CC BY-NC-SA 4.0, retained in the submodule and reproduced in
  `THIRD_PARTY_CC-BY-NC-SA-4.0.txt`
- use here: benchmark reaction IDs, framework reference, and optional endpoint
  connectivity utilities
- citation: Zhao et al., *Advanced Science* 2025,
  DOI `10.1002/advs.202506240`

The ReactBench submodule is pinned to the unmodified upstream commit. Local
research outputs and uncommitted modifications from the working ReactBench
checkout are not included.

## HORM and EquiformerV2 checkpoint

- upstream: <https://github.com/deepprinciple/HORM>
- pinned commit: `b4c2a35a28985c72ca47261bad0a96b2bc2ba084`
- source location: `horm/` (local copy of the tracked upstream source,
  excluding training datasets and checkpoints)
- locally used EquiformerV2 source: `horm/nets/equiformer_v2/equiformer_v2_oc20.py`
- modifications to the copied upstream file: CPU-safe device selection and
  removal of an unused internal Hessian call
- checkpoint location: `ckpt/eqv2.ckpt.*.part`; the repository setup command
  reconstructs `ckpt/eqv2.ckpt`. Provenance is in `ckpt/README.md`
- license: CC BY-NC-SA 4.0, retained in `horm/LICENSE` and reproduced in
  `THIRD_PARTY_CC-BY-NC-SA-4.0.txt`
- citation: Cui et al., *Scientific Data* 2026,
  DOI `10.1038/s41597-025-06350-5`

The upstream HORM README and LICENSE identify CC BY-NC-SA 4.0, while its
`pyproject.toml` lists MIT metadata. This repository preserves the upstream
files and attributes the source under the stated README/LICENSE terms.

## Transition1x benchmark inputs

- location: `data/transition1x_960.tar.gz`
- contents: 960 raw reactant/product XYZ pairs selected by ReactBench reaction
  ID
- source: Transition1x records distributed through the ReactBench benchmark
- citations: Grambow et al., *Scientific Data* 2020,
  DOI `10.1038/s41597-020-0460-4`; Schreiner et al., *Scientific Data* 2022,
  DOI `10.1038/s41597-022-01870-w`

Prepared GSM and React-OT TS guesses for these benchmark reactions are
included in the Gaussian input trees `data/Gaussian_GSM_eqV2/` and
`data/Gaussian_ReactOT_exact/`. These generated structures are not the
original Transition1x reference TSs. Their counts and provenance are
described in the main `README.md`.
Optimization results and manuscript source data are not included.
