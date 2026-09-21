# EquiformerV2 network source used in the study

`nets/equiformer_v2/equiformer_v2_oc20.py` is a copy of the local HORM
EquiformerV2 source file used with `eqv2.ckpt`. Its SHA256 is
`62c7e11c4dfa15a74be816462049178c100c09ac77f90a9a9046c5de59f73a70`.
It is based on HORM commit
`b4c2a35a28985c72ca47261bad0a96b2bc2ba084`, pinned in
`third_party/HORM`. Two local changes are present:

1. Select a CPU device when CUDA is unavailable, so constructing the network
   does not require a GPU.
2. Skip an unused internal `grad_hess_ij` call inside `forward`; the Hessian
   used by the Gaussian workflows is calculated by HORM's
   `eval.compute_hessian` from the predicted forces.

To reproduce the local source tree after cloning with submodules, replace
`third_party/HORM/nets/equiformer_v2/equiformer_v2_oc20.py` with this file.
The upstream HORM source and this modified copy retain the CC BY-NC-SA 4.0
terms. See `THIRD_PARTY_NOTICES.md` at the repository root for author,
license, and publication attribution. The model checkpoint is distributed
separately in the `eqv2-model-v1` release.
