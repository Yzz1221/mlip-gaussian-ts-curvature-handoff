# Implementation used for the manuscript calculations

The files in `production/` are byte-for-byte copies of the local source files
used for the four-core campaign and the External--CalcAll EquiformerV2 adapter.
`SOURCE_SHA256SUMS` records their checksums. The paths to the original local
copies are listed below to distinguish the original experiment from the
later generic interface in `src/mlip_gaussian_handoff/`.

| Workflow | Production code | Original location |
|---|---|---|
| MLIP--OneShot--ReadFC | `oneshot/build_manifests.py`, `oneshot/predict_hessians.py`, `oneshot/run_local_phase.py` | `Gaussian_ex/full_core_scaling_20260904/scripts/` |
| OneShot helper implementation | `oneshot/horm_bridge/` | `Experiment/TS+IRC/G_EF_Mlps_H/horm/` and its `mlip_readfc_scheme_experiments/` subdirectory |
| External--CalcAll | `external/horm_external.py`, `external/horm.sh` | `Experiment/G_Mlps_EFH/horm/` |

The OneShot campaign first computed ML Hessians with one model load per shard.
For each reaction, Gaussian created a single-point checkpoint; the runner
formatted it, aligned and rotated the precomputed Hessian to its Cartesian
frame, inserted the Hessian, reconstructed the checkpoint, and ran Gaussian
`Opt=(TS,ReadFC,NoEigenTest,NoMicro,MaxCycles=150) Freq` on the target QM
PES. This alignment step is part of the production implementation and is not
present in the later generic `src/` package.

For External--CalcAll, Gaussian requested energies, gradients, and a complete
ML Hessian at every Berny cycle through `External`. The production adapter
supports a persistent EquiformerV2 model process so that the model is not
reloaded on every derivative request. The four-core External measurements
also used an A100 GPU. `horm.sh` is the Gaussian External client wrapper;
the model and numerical implementation are in `horm_external.py`.

These archived runners include paths specific to the original cluster and
expect the original Gaussian input tree. Before running elsewhere, set the
model and Gaussian environment, replace the hard-coded source and scratch
paths, and construct equivalent input directories from `data/`. The separate
`examples/gaussian/` inputs show the four calculation routes and the IRC
stage without imposing a particular cluster scheduler. Slurm submission
files are intentionally omitted.

Model construction requires HORM. The complete upstream architecture and
dependencies are pinned by `third_party/HORM`; the locally modified
EquiformerV2 network file used in the calculations is supplied separately
under `model_architecture/`. Copy it over the matching path in the HORM
checkout before loading the released `eqv2.ckpt` checkpoint. HORM authors,
paper, source URL, and CC BY-NC-SA 4.0 license are recorded in
`THIRD_PARTY_NOTICES.md` and `CITATIONS.bib`.
