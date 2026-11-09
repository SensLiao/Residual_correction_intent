# Checkpoints

Model weights go in this folder. `*.pt` files are not tracked by git.

| File | Contents |
|---|---|
| `sirb_psma.pt` | SIRB trained on the PSMA training set for 50,000 training iterations |

A checkpoint holds `{"model": state_dict, "config": dict, "iteration": int}`: the weights, the merged configuration
of the training run and the number of completed training iterations. `sirb.utils.checkpoint.load_checkpoint` reads
it, and a file that holds a plain state dict is read as well. `training/train.py` writes checkpoints in this format.
