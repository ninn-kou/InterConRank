"""Simultaneous within-layer and cross-layer influence propagation."""

from numbers import Integral, Real

import numpy as np
from scipy import sparse


def _matrix(block):
    if np.iscomplexobj(block):
        raise ValueError("Connectivity weights must be real.")
    matrix = sparse.csr_matrix(block, dtype=float, copy=False)
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError("Connectivity weights must be finite and nonnegative.")
    return matrix


def _inverse_sqrt(degrees):
    degrees = np.asarray(degrees).ravel()
    if not np.all(np.isfinite(degrees)):
        raise ValueError("Weighted degrees exceed the floating-point range.")
    factors = np.zeros_like(degrees)
    positive = degrees > 0
    factors[positive] = 1.0 / np.sqrt(degrees[positive])
    return factors


def InterConRank(blocks, *, lambda_=0.51, delta=0.43, priors=None,
                 tol=1e-14, max_iter=100_000) -> tuple[np.ndarray, ...]:
    """Compute InterConRank scores, in input node order, for every layer.

    Parameters
    ----------
    blocks : square sequence of sequences of dense or SciPy sparse matrices
        Block [a][b] has shape (n_a, n_b). Entry [i, j] carries influence
        from node j in layer b to node i in layer a. Supply a square diagonal
        block for every layer. Use None for absent off-diagonal blocks.
        Weights must be finite and nonnegative.
    lambda_, delta : float
        Weights on within-layer and summed cross-layer influence. Both must
        be nonnegative and their sum must be at most one. The remaining
        weight multiplies the fixed prior.
    priors : sequence of one-dimensional arrays, optional
        One finite, nonnegative score per node. Each layer defaults to a
        uniform prior of 1 / n_a per node. Supplied priors retain their scale.
    tol : float
        Stop when the largest layerwise L2 change is at most this value.
    max_iter : int
        Maximum number of simultaneous updates, starting from all ones.

    Returns
    -------
    tuple of numpy.ndarray
        One score vector per layer. Larger values rank higher within a layer.

    Raises
    ------
    RuntimeError
        If scores become nonfinite or the iteration limit is reached before
        convergence. Convergence depends on the weighted propagation operator.
    """
    for name, value in (("lambda_", lambda_), ("delta", delta), ("tol", tol)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real number.")
    if lambda_ < 0 or delta < 0 or lambda_ + delta > 1:
        raise ValueError("lambda_ and delta must be nonnegative and sum to at most one.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")

    n_layers = len(blocks)
    if n_layers == 0 or any(len(row) != n_layers for row in blocks):
        raise ValueError("blocks must contain a nonempty square grid of matrices.")
    if any(row[a] is None for a, row in enumerate(blocks)):
        raise ValueError("Every layer needs a square diagonal block.")

    within = [_matrix(row[a]) for a, row in enumerate(blocks)]
    sizes = [matrix.shape[0] for matrix in within]
    if any(matrix.shape[1] != size for matrix, size in zip(within, sizes)):
        raise ValueError("Diagonal blocks must be square.")
    # Both sides of a within-layer block use its row degrees, even if directed.
    within_scale = [_inverse_sqrt(matrix.sum(axis=1)) for matrix in within]

    incoming = [[] for _ in sizes]
    for a, row in enumerate(blocks):
        for b, block in enumerate(row):
            if a == b or block is None:
                continue
            matrix = _matrix(block)
            if matrix.shape != (sizes[a], sizes[b]):
                raise ValueError(f"Block [{a}][{b}] must have shape {(sizes[a], sizes[b])}.")
            if np.any(matrix.data > 0):
                row_scale = _inverse_sqrt(matrix.sum(axis=1))
                col_scale = _inverse_sqrt(matrix.sum(axis=0))
                incoming[a].append((b, matrix, row_scale, col_scale))

    if priors is None:
        priors = [np.full(size, 1.0 / size if size else 0.0) for size in sizes]
    if len(priors) != n_layers:
        raise ValueError("Supply one prior vector per layer.")
    prior_weight = max(0.0, 1.0 - lambda_ - delta)
    fixed = []
    for size, prior in zip(sizes, priors):
        if np.iscomplexobj(prior):
            raise ValueError("Priors must be real.")
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (size,) or not np.all(np.isfinite(prior)) or np.any(prior < 0):
            raise ValueError("Each layer needs one finite, nonnegative prior per node.")
        fixed.append(prior_weight * prior)

    scores = [np.ones(size) for size in sizes]
    for _ in range(max_iter):
        updated = []
        change = 0.0
        for a, size in enumerate(sizes):
            cross = np.zeros(size)
            for b, matrix, row_scale, col_scale in incoming[a]:
                cross += row_scale * (matrix @ (col_scale * scores[b]))
            same = within_scale[a] * (within[a] @ (within_scale[a] * scores[a]))
            vector = lambda_ * same + delta * cross + fixed[a]
            if not np.all(np.isfinite(vector)):
                raise RuntimeError("InterConRank produced nonfinite scores.")
            updated.append(vector)
            change = max(change, float(np.linalg.norm(vector - scores[a])))
        scores = updated
        if change <= tol:
            return tuple(scores)
    raise RuntimeError(f"InterConRank did not converge in {max_iter} iterations (change={change:g}).")
