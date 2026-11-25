# Official autoPET V evaluation protocol

SIRB is evaluated with the interaction protocol of the autoPET V challenge:
an initial segmentation and five correction rounds, each round with one scribble from a simulated user, and every
round's mask scored with the official metrics. The two official files this needs are included here, so **nothing has
to be downloaded**. The full official repository is <https://github.com/lab-midas/autoPETV>.

The files come from [autoPET V revision `4a20268`](https://github.com/lab-midas/autoPETV/tree/4a20268)
and are distributed under the accompanying Apache License 2.0 (`LICENSE`).

| File here | Upstream path | Used for |
|---|---|---|
| `simulate_scribbles.py` | `interactive/simulate_scribbles.py` | the simulated scribble: drawn in the largest 2D connected component over all axial slices of a binary mask, in the centerline, random or boundary style |
| `metrics.py` | `metrics.py` | the official Dice, detection metric (DMM), false-positive volume and false-negative volume |

The `sirb` package uses them through `sirb/interaction/scribble_simulator.py` (which error to draw on, and the
random seed 42) and `sirb/evaluation/metrics.py` (scores of every round, averaged by patient). `tools/evaluate.py`
runs the whole protocol.

The vendored source matches the upstream content. The local copies use CRLF line endings, preserved by
`.gitattributes`. SIRB-specific behaviour is implemented in the surrounding package.

`metrics.py` imports `matplotlib` at module level for its plotting helpers, so `matplotlib` is a dependency of the
evaluation code.
