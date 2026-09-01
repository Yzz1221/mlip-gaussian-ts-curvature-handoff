# End-to-end workflow contract

## 1. Initial geometry

The input is a Cartesian Gaussian input with an explicit charge,
multiplicity, atom order, and TS-guess geometry. The workflow does not generate
the TS guess. GSM, React-OT, or another generator can be used upstream, but
their code and data are separate dependencies.

## 2. Fixed-geometry QM checkpoint

A single-point calculation at the unoptimized geometry creates a Gaussian
checkpoint. The formatted checkpoint is authoritative for the coordinate
orientation used by `ReadFC`; the ML Hessian is evaluated at those checkpoint
coordinates rather than blindly at the text coordinates in the original gjf.

## 3. ML curvature

The local HORM checkout loads the selected checkpoint once and evaluates the
Cartesian Hessian. The raw HORM convention is eV/angstrom^2. Before injection,
the matrix is symmetrized,

```text
H_sym = 0.5 * (H + H.T)
```

and converted to hartree/bohr^2. Symmetrization removes small numerical or
automatic-differentiation asymmetry; it does not change the intended scalar-
energy Hessian, which is symmetric analytically.

## 4. Checkpoint injection

The complete lower triangle is written to the `Cartesian Force Constants`
array in a copy of the formatted checkpoint. `unfchk` reconstructs the binary
checkpoint. Atom count and atom order are asserted before injection.

## 5. QM TS refinement

The generated route uses `Opt=(TS,ReadFC,NoEigenTest,NoMicro,MaxCycles=150)`
and `Freq`. `ReadFC` imports the injected initial curvature. No `External`
keyword appears in this job, so subsequent energies and gradients are supplied
by the target QM method.

`NoMicro` is retained to reproduce the production route. It has no operative
role in a conventional single-layer QM calculation; it disables microiterations
used mainly by multilayer/ONIOM optimization.

## 6. Opt+Freq audit

The primary classifier requires normal Opt and final Freq termination, the
final completed optimization to have all four Gaussian convergence tests
marked `YES`, a subsequent `Stationary point found`, and exactly one frequency
below -10 cm^-1 in the last complete frequency block. Two sensitivity fields
are also returned: Gaussian-accepted without the four-YES gate, and a -50 cm^-1
imaginary-frequency cutoff.

## 7. IRC

The generated bidirectional IRC route uses `RCFC`, `LQA`, `MaxPoints=30`, and
`StepSize=15` by default. Endpoint identity is a separate analysis and must not
be inferred from normal termination alone.

## 8. Continuous-MLIP controls

For an External control, Gaussian still executes Berny optimization, frequency
analysis, and IRC integration, while the external adapter supplies energies,
gradients, and requested Hessians. Consequently, the stationary points and
paths belong to the MLIP PES, not the target QM PES.
