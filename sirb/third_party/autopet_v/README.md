# Official autoPET V files

Unmodified copies of two files from the official autoPET V challenge repository,
<https://github.com/lab-midas/autoPETV> (upstream commit of 1 July 2026), distributed under the Apache License 2.0
(`LICENSE`, copied from the same repository).

| File here | Upstream path | Used for |
|---|---|---|
| `simulate_scribbles.py` | `interactive/simulate_scribbles.py` | the simulated scribble: drawn in the largest 2D connected component over all axial slices of a binary mask, in the centerline, random or boundary style |
| `metrics.py` | `metrics.py` | the official Dice, detection metric (DMM), false-positive volume and false-negative volume |

The files are byte-for-byte identical to the upstream files, including their CRLF line endings; `.gitattributes`
keeps git from converting them. Do not edit them: code that needs a different behaviour wraps them in the `sirb`
package instead.

`metrics.py` imports `matplotlib` at module level for its plotting helpers, so `matplotlib` is a dependency of the
evaluation code.
