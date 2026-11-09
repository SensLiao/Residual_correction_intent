# Official autoPET V evaluation protocol

SIRB is evaluated with the interaction protocol of the [autoPET V challenge](https://www.autopet.org/autopetv.html):
an initial segmentation and five correction rounds, each round with one scribble from a simulated user, and every
round's mask scored with the official metrics. The two official files this needs are included here, so **nothing has
to be downloaded**. The full official repository is <https://github.com/lab-midas/autoPETV>.

The files are unmodified copies from that repository (upstream commit of 1 July 2026), distributed under the Apache
License 2.0 (`LICENSE`, copied from the same repository).

| File here | Upstream path | Used for |
|---|---|---|
| `simulate_scribbles.py` | `interactive/simulate_scribbles.py` | the simulated scribble: drawn in the largest 2D connected component over all axial slices of a binary mask, in the centerline, random or boundary style |
| `metrics.py` | `metrics.py` | the official Dice, detection metric (DMM), false-positive volume and false-negative volume |

The `sirb` package uses them through `sirb/interaction/scribble_simulator.py` (which error to draw on, and the
random seed 42) and `sirb/evaluation/metrics.py` (scores of every round, averaged by patient). `tools/evaluate.py`
runs the whole protocol.

The files are byte-for-byte identical to the upstream files, including their CRLF line endings; `.gitattributes`
keeps git from converting them. Do not edit them: code that needs a different behaviour wraps them in the `sirb`
package instead.

`metrics.py` imports `matplotlib` at module level for its plotting helpers, so `matplotlib` is a dependency of the
evaluation code.
