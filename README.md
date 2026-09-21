# MLIP–Gaussian transition-state workflows

This repository runs the four workflows in the manuscript: Gaussian--CalcFC,
Gaussian--CalcAll, MLIP--OneShot--ReadFC, and External--CalcAll. The OneShot
workflow transfers one EquiformerV2 Hessian into Gaussian and then optimizes
on the QM potential-energy surface. External--CalcAll asks EquiformerV2 for
energy, gradients, and Hessians throughout Gaussian optimization.

## Install

Clone the repository. The local HORM source used in the study is included in
`horm/`. The ReactBench submodule provides its separate upstream framework:

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
pip install -e horm
python examples/setup.py
```

`examples/setup.py` verifies the locally used EquiformerV2 network already
included in `horm/`, downloads the exact `eqv2.ckpt` model from the
[`eqv2-model-v1` release](https://github.com/Yzz1221/mlip-gaussian-ts-curvature-handoff/releases/tag/eqv2-model-v1),
and verifies its SHA256. If you already have this checkpoint, run
`python examples/setup.py --model-source /path/to/eqv2.ckpt`.

## Run a workflow

Choose a reaction from the packaged React-OT or GSM inputs. The selected
Gaussian input already specifies its geometry, charge and multiplicity:

```bash
python examples/run.py --workflow oneshot \
  --dataset react_ot --reaction rxn9 \
  --output runs/oneshot_rxn9
```

The other choices use the same command and input arguments:

```bash
python examples/run.py --workflow calcfc --dataset react_ot --reaction rxn9 --output runs/calcfc_rxn9
python examples/run.py --workflow calcall --dataset react_ot --reaction rxn9 --output runs/calcall_rxn9
python examples/run.py --workflow external_calcall --dataset react_ot --reaction rxn9 --output runs/external_rxn9
```

For GSM, set `--dataset gsm`. For your own initial TS guess, use `--xyz`
with explicit `--charge` and `--multiplicity`. Each invocation
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

The local HORM code is included in [`horm/`](horm/) with its upstream license.
The EquiformerV2 network file there already contains the two local changes.
The model checkpoint is available from the release linked above. HORM and
ReactBench authors, paper citations, and licenses are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`CITATIONS.bib`](CITATIONS.bib). Gaussian is proprietary and must be installed
separately.

## Validate the installation

```bash
python examples/run.py --workflow oneshot --dataset react_ot --reaction rxn9 \
  --output runs/check_inputs --dry-run
python -m unittest discover -s tests
```

This dry run checks the chosen XYZ and writes the Gaussian input files. A
full OneShot or External calculation additionally needs Gaussian 16 and the
released model. The IRC output still requires endpoint connectivity analysis
to determine whether the intended reaction was recovered.

## Data

The `data/` directory contains the original reaction pairs and two initial
guess datasets. The full Gaussian input trees follow the calculation layout:

```text
data/Gaussian_GSM_eqV2/<method>/rxn<ID>/TS+Freq/opt+freq.gjf
data/Gaussian_GSM_eqV2/<method>/rxn<ID>/IRC/irc.gjf
data/Gaussian_ReactOT_exact/<method>/rxn<ID>/TS+Freq/opt+freq.gjf
data/Gaussian_ReactOT_exact/<method>/rxn<ID>/IRC/irc.gjf
```

`data/Gaussian_GSM_eqV2/` contains 871 reaction directories per method;
`data/Gaussian_ReactOT_exact/` contains 960. Each dataset includes
`Gaussian_calcfc`, `Gaussian_calcall`,
`mlip_oneshot_SP_Readfc`, and `C_eqv2_EFH_gaussian_calcall` (External--CalcAll).
Each reaction and method has exactly these two `.gjf` files. The OneShot
Opt+Freq input reads its checkpoint; its initial SP input and ML Hessian
handoff are created by the running code from the matching CalcFC starting
geometry and charge. External inputs refer to the repository's
`production/external/horm.sh` by a relative path, and each IRC input refers to
the final checkpoint from its own Opt+Freq workflow.

The original 960 Transition1x reactant/product pairs remain available in
[`data/transition1x_960.tar.gz`](data/transition1x_960.tar.gz).

The four-core External--CalcAll source directories contain 100 Opt+Freq
inputs per dataset. The remaining 771 GSM and 860 React-OT External inputs
were generated from the matching initial guesses using the same
CalcAll/External route. For OneShot, the four-core source lacked 21 GSM and
one React-OT Opt+Freq input; those were generated with the same ReadFC route.
These generated inputs are calculation templates, not additional completed
calculations.

The matching `rxn<ID>` directory identifies the same benchmark reaction in
both datasets. The Opt+Freq input contains the unoptimized initial guess
geometry in CalcFC, CalcAll, and External--CalcAll; OneShot obtains that same
geometry from CalcFC before building its checkpoint. GSM did not produce a
converged path for 89 of the 960 reactions. These inputs were not selected
according to downstream optimization results.

The original Transition1x archive is unchanged from the ReactBench benchmark
distribution (SHA256:
`bf39f131988328a9368f410dd6e98045925f7cad62df70a13fdd3cfa131c88a3`).
Extract it with `tar -xzf data/transition1x_960.tar.gz -C data`.

Please cite ReactBench, HORM, and Transition1x when using their code, model,
or data; the ready-to-use entries are in [CITATIONS.bib](CITATIONS.bib).
