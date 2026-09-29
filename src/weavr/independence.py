"""How independent are the forecast sources, really?

Every argument for blending assumes the sources make *different* mistakes:
averaging correlated errors cancels nothing. This module measures that
assumption instead of asserting it, with three standard diagnostics computed
on forecast-minus-observation errors:

- `error_correlation_matrix` -- pairwise Pearson correlation of the errors.
- `effective_number_of_models` -- the classic
  `N / (1 + (N-1) * mean_off_diagonal)` reduction: how many *independent*
  sources the correlated set is worth. Equal to N when the errors are
  uncorrelated, and 1 when every source makes the same error.
- `participation_ratio` -- `(sum eigenvalues)^2 / sum(eigenvalues^2)`, the
  same question asked of the correlation matrix's spectrum, which (unlike
  the mean-off-diagonal formula) also notices when one *pair* is nearly
  identical while the rest are not.

Both summaries are reported, because they disagree in informative ways and
neither alone is a complete description of the dependence structure.

Errors must already be in matching units. WeatherBench 2 precipitation is in
**metres** while IMD's rain is in millimetres; comparing them unconverted
makes every source's error effectively `-obs`, and every pairwise
correlation comes out at exactly 1.0 -- a plausible-looking "the sources are
identical" result that is purely a unit bug. Convert with
`run_tier0_baseline.PRECIP_M_TO_MM` first (see
docs/single-source-and-independence-results.md).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _stacked_common_finite(
    errors: dict[str, xr.DataArray], dims: list[str] | tuple[str, ...]
) -> tuple[list[str], np.ndarray]:
    """Flatten each source's error over `dims`, keeping only the cells that
    are finite in *every* source.

    Restricting to the common finite set (rather than dropping NaNs per
    pair) means every entry of the resulting correlation matrix is computed
    over the same sample, so the matrix is internally consistent -- a
    per-pair mask can produce a matrix that isn't positive semi-definite,
    which would make `participation_ratio`'s eigenvalues meaningless.
    """
    names = sorted(errors)
    if len(names) < 2:
        raise ValueError(f"Need at least 2 sources to correlate, got {len(names)}: {names}")

    aligned = xr.align(*(errors[name] for name in names), join="inner")
    flat = np.stack([da.stack(_cell=list(dims)).values.ravel() for da in aligned])
    common_finite = np.isfinite(flat).all(axis=0)
    return names, flat[:, common_finite]


def error_correlation_matrix(
    errors: dict[str, xr.DataArray],
    dims: list[str] | tuple[str, ...],
) -> tuple[list[str], np.ndarray]:
    """Pairwise Pearson correlation of forecast-minus-obs errors.

    `errors` maps source name -> an error field; `dims` are the dimensions to
    pool over (e.g. `["sample", "latitude", "longitude"]`). Returns
    `(names, matrix)` with `names` sorted, so row/column `i` is `names[i]`.

    Raises `ValueError` if fewer than 2 cells are finite in every source --
    a correlation over one point is not a number worth returning silently.
    """
    names, flat = _stacked_common_finite(errors, dims)
    n_cells = flat.shape[1]
    if n_cells < 2:
        raise ValueError(
            f"Only {n_cells} cell(s) are finite in all of {names}; a correlation "
            "needs at least 2. Check the alignment and the NaN masks."
        )
    matrix = np.corrcoef(flat)
    return names, np.asarray(matrix, dtype=float)


def effective_number_of_models(corr: np.ndarray) -> float:
    """`N / (1 + (N-1) * mean_off_diagonal)`: how many independent sources
    a correlated set of N is worth.

    N for uncorrelated errors, 1 for identical errors. The mean is over the
    off-diagonal entries only; the matrix is assumed symmetric with a unit
    diagonal (what `error_correlation_matrix` returns).
    """
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be a square matrix, got shape {corr.shape}")
    if n < 2:
        raise ValueError(f"Need at least 2 sources, got {n}")

    off_diagonal = corr[~np.eye(n, dtype=bool)]
    mean_off_diagonal = float(np.mean(off_diagonal))
    denominator = 1.0 + (n - 1) * mean_off_diagonal
    if denominator <= 0.0:
        # Strongly anti-correlated errors: the formula's variance-reduction
        # interpretation breaks down (it would claim more independent models
        # than there are members). Report the ceiling rather than a negative
        # or explosive "effective N".
        return float(n)
    return float(n / denominator)


def participation_ratio(corr: np.ndarray) -> float:
    """`(sum eigenvalues)^2 / sum(eigenvalues^2)` of the correlation matrix.

    An eigenvalue-based effective dimensionality: N when the N sources are
    uncorrelated (every eigenvalue 1), and 1 when they collapse onto a
    single direction. Unlike `effective_number_of_models`, which sees only
    the mean off-diagonal, this responds to the *shape* of the dependence:
    one near-duplicate pair among otherwise independent sources and a set of
    uniformly mild correlations can share a mean off-diagonal exactly, and
    this ratio still separates them (tests/test_independence.py).
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be a square matrix, got shape {corr.shape}")
    # eigvalsh, not eigvals: a correlation matrix is symmetric, and the
    # symmetric solver cannot return the tiny imaginary parts the general
    # one does on near-degenerate input.
    eigenvalues = np.linalg.eigvalsh(corr)
    sum_of_squares = float(np.sum(eigenvalues**2))
    if sum_of_squares == 0.0:
        raise ValueError("Degenerate correlation matrix: all eigenvalues are zero.")
    return float(np.sum(eigenvalues) ** 2 / sum_of_squares)
