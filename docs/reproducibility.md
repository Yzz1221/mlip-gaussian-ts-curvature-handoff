# Reproducibility checklist

- [ ] Record the HORM Git commit.
- [ ] Record checkpoint filename, public source, and SHA256.
- [ ] Export a pinned Python/PyTorch/PyG environment.
- [ ] Record Gaussian revision and route section.
- [ ] Validate energy, gradient sign, Hessian units, and atom order on one case.
- [ ] Confirm `formchk -3` and `unfchk` round-trip successfully.
- [ ] Run a CalcFC reference at the same initial geometry.
- [ ] Confirm exactly one chemically relevant imaginary mode.
- [ ] Run bidirectional IRC and classify endpoint connectivity.
- [ ] Record CPU/GPU model, core count, model-loading mode, and timing scope.
- [ ] Archive a minimal non-proprietary regression example where licensing permits.
- [ ] Choose and add a software license before a formal public release.
