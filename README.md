# MLIP–Gaussian TS curvature handoff

Code for a one-shot transition-state workflow in which a Hessian-supervised
MLIP supplies the Cartesian curvature only at the initial TS-guess geometry.
The curvature is written to a Gaussian checkpoint and imported with `ReadFC`;
all subsequent energies, gradients, TS optimization, frequency analysis, and
IRC calculations can then be performed on the selected QM potential-energy
surface.

This repository currently contains **code only**. Manuscript files, figures,
source data, model checkpoints, third-party datasets, and Gaussian production
outputs are intentionally excluded.

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
```

## External requirements

- Python 3.10+
- NumPy
- PyTorch and PyTorch Geometric versions compatible with the HORM checkout
- a local [HORM](https://github.com/yhong55/HORM) checkout and model checkpoint
- Gaussian 16 plus `formchk` and `unfchk`
- optional: [ReactBench](https://github.com/yueguanwen/ReactBench) for endpoint
  connectivity classification

Gaussian and the ML checkpoint are not distributed by this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

export HORM_ROOT=/path/to/HORM
export HORM_CHECKPOINT=/path/to/eqv2.ckpt
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

Do not commit checkpoints, third-party datasets, Gaussian production trees,
scheduler logs, credentials, or machine-specific absolute paths. See
[docs/provenance.md](docs/provenance.md).

## Status

This is an initial code-only release assembled from the research workflow.
Before assigning a DOI, freeze the HORM commit/checkpoint hash, dependency lock,
Gaussian revision, and one public minimal regression example.
