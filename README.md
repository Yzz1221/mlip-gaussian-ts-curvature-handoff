# MLIP–Gaussian TS curvature handoff

Code for a one-shot transition-state workflow in which a Hessian-supervised
MLIP supplies the Cartesian curvature only at the initial TS-guess geometry.
The curvature is written to a Gaussian checkpoint and imported with `ReadFC`;
all subsequent energies, gradients, TS optimization, frequency analysis, and
IRC calculations can then be performed on the selected QM potential-energy
surface.

The repository includes the two workflows evaluated in the accompanying
study, the raw 960-reaction benchmark input archive, the prepared GSM (871)
and React-OT (960) TS-guess collections, release metadata for the
EquiformerV2 checkpoint used in production, and pinned upstream ReactBench and
HORM source trees. Processed analysis tables, figures, and Gaussian production
outputs are not included.

## Workflow

```text
TS guess (.gjf)
    │
    ├─ Gaussian fixed-geometry QM SP → initial .chk/.fchk
    ├─ HORM/EquiformerV2 Hessian at the checkpoint geometry
    ├─ symmetrize + convert eV Å⁻² → Eh a0⁻²
    ├─ inject Cartesian Force Constants → modified checkpoint
    ├─ Gaussian Opt=(TS,ReadFC) + Freq on the target QM PES
    └─ bidirectional Gaussian IRC + optional endpoint classification
```

The repository also includes a Gaussian `External` adapter used for continuous
MLIP controls. In those controls Gaussian retains the Berny/Freq/IRC drivers,
while the MLIP supplies energies and derivatives throughout the calculation.

## Repository layout

```text
src/mlip_gaussian_handoff/   reusable Python implementation
scripts/                     command-line wrappers
examples/                    Gaussian and Slurm templates
docs/                        workflow, configuration, and provenance notes
tests/                       tests that do not require Gaussian or HORM
data/                        raw reaction pairs, GSM and React-OT TS guesses
models/                      EquiformerV2 release metadata and reconstruction
third_party/                 pinned ReactBench and HORM Git submodules
```

The two study workflows map to the following code paths:

- **MLIP--OneShot--ReadFC:**
  `src/mlip_gaussian_handoff/workflow.py` and the
  `mlip-gaussian-prepare` command;
- **External--CalcAll:** `src/mlip_gaussian_handoff/external.py`,
  `scripts/horm_external.sh`, and
  `examples/gaussian/external_calcall.gjf`.

Clone the repository with its pinned upstream frameworks:

```bash
git clone --recurse-submodules \
  https://github.com/Yzz1221/mlip-gaussian-ts-curvature-handoff.git
cd mlip-gaussian-ts-curvature-handoff
```

Download the EquiformerV2 checkpoint assets from the
[`eqv2-model-v1` release](https://github.com/Yzz1221/mlip-gaussian-ts-curvature-handoff/releases/tag/eqv2-model-v1)
and reconstruct the checkpoint as described in
[`models/README.md`](models/README.md).

## External requirements

- Python 3.10+
- NumPy
- PyTorch and PyTorch Geometric versions compatible with the HORM checkout
- the pinned [HORM](third_party/HORM) checkout and the released EquiformerV2
  checkpoint
- Gaussian 16 plus `formchk` and `unfchk`
- optional: [ReactBench](third_party/ReactBench) for endpoint
  connectivity classification

Gaussian is proprietary software and is not distributed by this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export HORM_ROOT="$PWD/third_party/HORM"
export HORM_CHECKPOINT="$PWD/models/eqv2.ckpt"
export HORM_DEVICE=cpu          # or cuda
```

Install the additional HORM dependencies according to the upstream HORM
environment. The exact PyTorch/PyG build must match the local CUDA runtime.

## One-shot handoff

Prepare the checkpoint containing the ML Hessian:

```bash
mlip-gaussian-prepare \
  --input examples/gaussian/ts_guess.gjf \
  --output work/rxn_example \
  --method 'wB97X/6-31G(d)' \
  --nproc 24 \
  --run-sp
```

This creates `initial.gjf/.chk/.fchk`, `readfc.fchk/.chk`, a QM
`ts_freq.gjf`, and `handoff_metadata.json` with units, hashes, and timings.

Run the prepared QM refinement and audit it:

```bash
mlip-gaussian-run work/rxn_example/ts_freq.gjf
mlip-gaussian-audit work/rxn_example/ts_freq.log
```

After an accepted Opt+Freq calculation, generate and run IRC:

```bash
mlip-gaussian-write-irc \
  --checkpoint work/rxn_example/readfc.chk \
  --output work/rxn_example/irc.gjf \
  --method 'wB97X/6-31G(d)'

mlip-gaussian-run work/rxn_example/irc.gjf
```

See [docs/workflow.md](docs/workflow.md) for the scientific and file-level
contract and [docs/reproducibility.md](docs/reproducibility.md) for a release
checklist.

## Continuous-MLIP control

The `mlip-gaussian-external` command implements the Gaussian `.EIn`/`.EOu`
protocol. A route example is provided in
[`examples/gaussian/external_calcfc.gjf`](examples/gaussian/external_calcfc.gjf).
External-control timings must be reported with their hardware and interface
configuration; they are not automatically comparable with CPU QM wall times.

## Validation

```bash
python -m pytest
```

The tests cover Gaussian input parsing, fchk force-constant injection, unit
conversion, Hessian symmetrization, and Opt+Freq classification. They do not
replace a small end-to-end test using the licensed Gaussian installation and
the selected ML checkpoint.

## Data and provenance policy

The repository distributes the raw 960-reaction input archive and the prepared
TS guesses in `data/gsm/` and `data/react_ot/`, while the
single EquiformerV2 checkpoint used in the study is attached to the
`eqv2-model-v1` GitHub release. It excludes derived result
tables, Gaussian production trees, scheduler logs, credentials, and
machine-specific paths. See
[data/README.md](data/README.md), [models/README.md](models/README.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[docs/provenance.md](docs/provenance.md).

ReactBench, HORM, and Transition1x must be cited when their code, model, or
data are used. Ready-to-import entries are provided in
[CITATIONS.bib](CITATIONS.bib).

## Status

The upstream commits, model checksum, dataset checksum, and production
workflow entry points are frozen in this repository. Before assigning a DOI,
archive the GitHub release and record its commit identifier in the manuscript.
