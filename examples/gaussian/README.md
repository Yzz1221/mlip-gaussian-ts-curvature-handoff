# Gaussian input templates

These four directories show the manuscript's Opt+Freq and IRC routes. The
templates start from the same React-OT `rxn9` guess and use four Gaussian CPU
cores. To run another structure, use `python examples/run.py` from the
repository root with `--xyz`, `--charge`, `--multiplicity`, and `--output`.
The runner inserts the chosen XYZ coordinates, prepares the model and
checkpoint handoff where needed, and executes the selected route.

| Directory | Initial curvature | Optimization, Freq and IRC surface |
|---|---|---|
| `gaussian_calcfc/` | Full QM Hessian at the starting geometry | QM |
| `gaussian_calcall/` | Full QM Hessian at each Berny cycle | QM |
| `mlip_oneshot_readfc/` | One EquiformerV2 Hessian injected through `ReadFC` | QM |
| `external_calcall/` | EquiformerV2 Hessian at each Berny cycle | EquiformerV2 |

OneShot first runs the fixed-geometry single-point input
`initial_sp.gjf`. The runner then obtains the ML Hessian, aligns it with the
Gaussian checkpoint coordinates, and creates `mlip_readfc.chk` for
`opt_freq.gjf`. External--CalcAll uses the production `horm_external.py` and
`horm.sh` adapter with a resident model process. Its `External` keyword is
present in both Opt+Freq and IRC inputs.

Each IRC input reads the final checkpoint from its own successful Opt+Freq
calculation. The route uses `RCFC`, `LQA`, `MaxPoints=30`, and `StepSize=15`.
The final Freq and IRC calculations validate the TS; their times are excluded
from the manuscript's TS-search wall time. An IRC trajectory by itself does
not establish intended-reaction recovery: both endpoints must be checked
against the reference reactant and product connectivities.
