# Gaussian inputs matching the four manuscript workflows

All templates use the same React-OT `rxn9` unoptimized TS guess, the
ωB97X/6-31G(d) target QM level where applicable, and four CPU cores
for Gaussian. The initial XYZ coordinates are copied from the four-core
CalcFC input. Use an independent working directory for each workflow because
the checkpoint names repeat.

| Workflow | Preparation | TS optimization and frequency | IRC PES |
|---|---|---|---|
| Gaussian--CalcFC | none | `gaussian_calcfc/opt_freq.gjf`: QM `CalcFC` | QM |
| Gaussian--CalcAll | none | `gaussian_calcall/opt_freq.gjf`: QM `CalcAll` | QM |
| MLIP--OneShot--ReadFC | `mlip_oneshot_readfc/initial_sp.gjf`, then inject the EquiformerV2 Hessian | `mlip_oneshot_readfc/opt_freq.gjf`: QM `ReadFC` | QM |
| External--CalcAll | set up `horm_external.sh` and the EquiformerV2 checkpoint | `external_calcall/opt_freq.gjf`: MLIP `CalcAll` through Gaussian `External` | EquiformerV2 |

Each workflow's `irc.gjf` implements the manuscript protocol
`IRC=(MaxPoints=30,RCFC,LQA,StepSize=15)` using the checkpoint produced by
that workflow's successful Opt+Freq job. Run IRC only after confirming normal
optimization and frequency termination and exactly one imaginary frequency
below -10 cm^-1. The final `Freq` and IRC stages are validation calculations;
they are excluded from the reported TS-search wall time.

For CalcFC and CalcAll, place `opt_freq.gjf` and `irc.gjf` in the same working
directory. Run Opt+Freq first, then IRC; `%oldchk=ts_freq.chk` imports the
optimized TS and frequency force constants. The `ReadFC` example starts from
`mlip_readfc.chk`, which must contain the injected ML Hessian. Its Opt+Freq run
updates that checkpoint; the IRC input then reads this *final* checkpoint,
not the initial SP checkpoint. Run
`mlip-gaussian-prepare --input examples/gaussian/mlip_oneshot_readfc/initial_sp.gjf --output WORK --nproc 4 --run-sp`
to exercise the later generic interface. To reproduce the paper's four-core
calculation code and timing, use the sources documented in
[`production/README.md`](../../production/README.md).

For External--CalcAll, put copies of both `production/external/horm.sh`
and `production/external/horm_external.py` in the same working directory as
each Gaussian input, and make `horm.sh` executable. Keep `External` on
both the Opt+Freq and IRC routes so energy, gradient, and requested Hessian
evaluations remain on the EquiformerV2 PES. Set `HORM_ROOT`,
`HORM_CHECKPOINT`, `HORM_DEVICE=cuda`, and `HORM_PYTHON_BIN` before running.
The study's External timing used a resident model process
(`HORM_USE_DAEMON=1`) and an A100 GPU.

The bundled `rxn9` coordinates illustrate the route and file handoff. They
do not assert that this reaction met the paper's success criteria in every
workflow. Both GSM and React-OT production calculations use the same route
definitions with their respective initial guesses.
