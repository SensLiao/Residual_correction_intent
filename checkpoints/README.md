# Checkpoints

| File | Contents |
|---|---|
| `sirb_psma.pt` | SIRB trained on the PSMA training set for 50,000 training iterations (31.7 MB); used by both predictor modes |

A checkpoint holds `{"model": state_dict, "config": dict, "iteration": int}`: the weights, the model configuration
and the number of completed training iterations. It loads with `torch.load(..., weights_only=True)`, and
`build_sirb(config, "checkpoints/sirb_psma.pt")` builds the model from it. `training/train.py` writes checkpoints in
this format; other `*.pt` files in this folder are not tracked by git.
