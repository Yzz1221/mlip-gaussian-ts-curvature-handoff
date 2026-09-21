# MLIP–Gaussian transition-state workflows

This repository runs the four workflows in the manuscript: Gaussian--CalcFC,
Gaussian--CalcAll, MLIP--OneShot--ReadFC, and External--CalcAll. The OneShot
workflow transfers one EquiformerV2 Hessian into Gaussian and then optimizes
on the QM potential-energy surface. External--CalcAll asks EquiformerV2 for
energy, gradients, and Hessians throughout Gaussian optimization.

## Install

Clone with the pinned [HORM](https://github.com/deepprinciple/HORM) and
[ReactBench](https://github.com/deepprinciple/ReactBench) source repositories:

```bash
git clone --recurse-submodules https://github.com/Yzz1221/mlip-gaussian-ts-curvature-handoff.git
cd mlip-gaussian-ts-curvature-handoff
```

Use Python 3.10 or newer and install PyTorch/PyTorch Geometric builds that
match your CPU or CUDA environment, following the pinned HORM dependencies.
Gaussian 16, `formchk`, and `unfchk` must be available on `PATH`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e third_party/HORM
python examples/setup.py
```

`examples/setup.py` installs the locally used EquiformerV2 network source into
the HORM checkout, downloads the exact `eqv2.ckpt` model from the
[`eqv2-model-v1` release](https://github.com/Yzz1221/mlip-gaussian-ts-curvature-handoff/releases/tag/eqv2-model-v1),
and verifies its SHA256. If you already have this checkpoint, run
`python examples/setup.py --model-source /path/to/eqv2.ckpt`.

## Run a workflow

Choose an initial TS guess from `data/react_ot/` or `data/gsm/`, then give its
charge and spin multiplicity explicitly. For example:

```bash
python examples/run.py --workflow oneshot \
  --xyz data/react_ot/rxn9.xyz --charge 0 --multiplicity 1 \
  --output runs/oneshot_rxn9
```

The other choices use the same command and input arguments:

```bash
python examples/run.py --workflow calcfc --xyz data/react_ot/rxn9.xyz --charge 0 --multiplicity 1 --output runs/calcfc_rxn9
python examples/run.py --workflow calcall --xyz data/react_ot/rxn9.xyz --charge 0 --multiplicity 1 --output runs/calcall_rxn9
python examples/run.py --workflow external_calcall --xyz data/react_ot/rxn9.xyz --charge 0 --multiplicity 1 --output runs/external_rxn9
```

Replace the `--xyz` path with your own initial TS guess. Each invocation
creates an independent output directory, generates the corresponding Gaussian
inputs, runs Opt+Freq, and runs IRC when Opt+Freq meets the paper's primary
criterion. Use `--dry-run` to inspect inputs without calculations, `--skip-irc`
to stop after Opt+Freq, or `--cores N` to change the Gaussian core count. The
paper used four CPU cores for Gaussian and an A100 GPU for External--CalcAll;
`--external-device cpu` is available for a CPU-only trial. Gaussian and the
model have substantial runtime requirements, so use your own job scheduler to
invoke `examples/run.py` on a compute node.

The four Gaussian Opt+Freq and IRC routes are also shown separately in
[`examples/gaussian/`](examples/gaussian/). No cluster submission scripts are
required.

## Code and model provenance

The numerical implementation used for the manuscript is in
[`production/oneshot/`](production/oneshot/) and
[`production/external/`](production/external/). `examples/run.py` connects
those implementations to input XYZ files and avoids the original cluster's
absolute paths. The OneShot path retains the production model prediction,
Cartesian alignment/rotation, and Hessian injection; External--CalcAll uses
the production Gaussian External adapter with one resident model process.
The source copies and their hashes are documented in
[`production/README.md`](production/README.md).

The full HORM model code is pinned under `third_party/HORM`; the two local
EquiformerV2 changes are preserved in [`model_architecture/`](model_architecture/).
The model checkpoint is available from the release linked above. HORM and
ReactBench authors, paper citations, and licenses are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`CITATIONS.bib`](CITATIONS.bib). Gaussian is proprietary and must be installed
separately.

## Validate the installation

```bash
python examples/run.py --workflow oneshot --xyz data/react_ot/rxn9.xyz \
  --charge 0 --multiplicity 1 --output runs/check_inputs --dry-run
python -m unittest discover -s tests
```

This dry run checks the chosen XYZ and writes the Gaussian input files. A
full OneShot or External calculation additionally needs Gaussian 16 and the
released model. The IRC output still requires endpoint connectivity analysis
to determine whether the intended reaction was recovered.

## Data

The `data/` directory contains three datasets:

| Path | Contents | Count |
|---|---|---:|
| [`data/transition1x_960.tar.gz`](data/transition1x_960.tar.gz) | Original reactant and product structures for the selected Transition1x reactions | 960 reaction pairs |
| [`data/gsm/`](data/gsm/) | Initial TS guesses from converged GSM paths | 871 structures |
| [`data/react_ot/`](data/react_ot/) | Initial TS guesses generated by React-OT | 960 structures |

The GSM and React-OT files are named `rxn<ID>.xyz`, so the same reaction can
be matched across datasets by its ID. They retain the atom ordering and
Cartesian coordinates used to start the Gaussian calculations. GSM did not
produce a converged path for 89 of the 960 reactions. These are initial
guesses; no optimization result or success criterion was used to select them.

The original Transition1x archive is unchanged from the ReactBench benchmark
distribution (SHA256:
`bf39f131988328a9368f410dd6e98045925f7cad62df70a13fdd3cfa131c88a3`).
Extract it with `tar -xzf data/transition1x_960.tar.gz -C data`.

Please cite ReactBench, HORM, and Transition1x when using their code, model,
or data; the ready-to-use entries are in [CITATIONS.bib](CITATIONS.bib).
