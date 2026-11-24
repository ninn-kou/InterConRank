"""Katz centrality on one layer, with receiving rows and sending columns.

Leo Katz (1953), A New Status Index Derived from Sociometric Analysis,
Psychometrika 18, 39--43. https://doi.org/10.1007/BF02289026
Primary text: https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Katz-1953-status.pdf
Weighted modern formulation:
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html
"""

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def _matrix(A):
    if np.iscomplexobj(A):
        raise ValueError('Connectivity weights must be real and nonnegative.')
    A = sparse.csr_matrix(A, dtype=float, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    if A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError('Katz requires one nonempty square adjacency matrix.')
    if not np.isfinite(A.data).all() or np.any(A.data < 0):
        raise ValueError('Connectivity weights must be finite and nonnegative.')
    if not np.isfinite(np.asarray(A.sum(axis=1))).all():
        raise ValueError('Connectivity row sums must be finite.')
    return A


def _spectral_upper_bound(A, tolerance, max_iter):
    """Collatz lower/upper Perron bounds, applied to every strongly connected block.

    Each nontrivial SCC is irreducible; its identity shift is primitive.
    For any positive v, min((A v)/v) <= rho(A) <= max((A v)/v).
    Shifted power iteration tightens the bracket without relying on unstable
    full-matrix eigenvalues for a reducible graph or a nilpotent citation DAG.
    Singleton SCCs have their diagonal entry as their exact eigenvalue.
    """
    count, labels = connected_components(A, directed=True, connection='strong')
    members = [[] for _ in range(count)]
    for node, label in enumerate(labels):
        members[label].append(node)
    upper = 0.
    for nodes in members:
        if len(nodes) == 1:
            root = float(A[nodes[0], nodes[0]])
            upper = max(upper, root)
            continue
        block = A[nodes, :][:, nodes].tocsr()
        x = np.ones(len(nodes))
        for _ in range(max_iter):
            Ax = block @ x
            ratios = Ax / x
            lo, hi = float(ratios.min()), float(ratios.max())
            if not np.isfinite([lo, hi]).all():
                raise RuntimeError('Katz spectral bounds became non-finite.')
            if hi - lo <= tolerance * max(1., hi):
                upper = max(upper, hi)
                break
            x = x + Ax
            x /= x.max()
            if np.any(x <= 0):
                raise RuntimeError('Katz spectral iteration lost positive support.')
        else:
            raise RuntimeError('Katz spectral bounds did not converge within max_iter.')
    # Use a conservative upward rounding margin. This numerical diagnostic is
    # not an interval-arithmetic certificate for every floating-point operation.
    upper = upper * (1 + 32 * np.finfo(float).eps)
    return upper


def katz(A, attenuation=.85, epsilon=1.0e-12, max_iter=100000):
    """Solve the incoming Katz equation with uniform forcing and return unit-L2 scores.

    Attenuation is the fraction of a tight spectral upper bound's reciprocal.
    For acyclic graphs use the maximum row sum. Restore ties for identical
    receiving rows and constant-degree components after the sparse direct solve.
    """
    for name, value in [('attenuation', attenuation), ('epsilon', epsilon)]:
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(name + ' must be a finite real number.')
    if not 0 <= attenuation < 1:
        raise ValueError('attenuation must lie in [0, 1).')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive.')
    if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0):
        raise ValueError('max_iter must be a positive integer.')
    A = _matrix(A)
    upper = _spectral_upper_bound(A, min(1e-10, epsilon * 100), max_iter)
    if upper > 0:
        scale = upper
    else:
        row_bound = float(np.asarray(A.sum(axis=1)).max())
        scale = row_bound if row_bound > 0 else 1.
    alpha = attenuation / scale
    n = A.shape[0]
    forcing = np.full(n, 1. / n)
    system = sparse.eye(n, format='csc') - alpha * A.tocsc()
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        x = np.asarray(spsolve(system, forcing), dtype=float)
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise RuntimeError('Katz direct solve did not return finite positive scores.')
    # Exact equation for a row with no incoming edge: x_i = forcing_i.
    # This follows from graph structure, without a score threshold or rounding.
    zero_rows = np.diff(A.indptr) == 0
    x[zero_rows] = forcing[zero_rows]
    # Canonical sorted CSR rows with identical data have the same equation.
    # One equation evaluation gives exact equality for those structural ties.
    x = alpha * (A @ x) + forcing
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise RuntimeError('Katz equation polishing produced invalid scores.')
    # Weak components have no edges between them. Constant row strength d
    # makes the uniform vector an eigenvector, hence this exact Katz solution.
    # Compare strengths exactly; no score threshold or approximate grouping.
    component_count, component_labels = connected_components(A, directed=True, connection='weak')
    degrees = np.asarray(A.sum(axis=1)).ravel()
    minimum = np.full(component_count, np.inf)
    maximum = np.zeros(component_count)
    np.minimum.at(minimum, component_labels, degrees)
    np.maximum.at(maximum, component_labels, degrees)
    regular = minimum == maximum
    regular_nodes = regular[component_labels]
    denominators = 1 - alpha * minimum[regular]
    if np.any(denominators <= 0):
        raise RuntimeError('Katz regular-component denominator is not positive.')
    component_scores = np.zeros(component_count)
    component_scores[regular] = (1. / n) / denominators
    x[regular_nodes] = component_scores[component_labels[regular_nodes]]
    propagated = alpha * (A @ x)
    residual = x - propagated - forcing
    absolute_residual = float(np.linalg.norm(residual))
    backward_residual = absolute_residual / (
        float(np.linalg.norm(x)) + float(np.linalg.norm(propagated)) + float(np.linalg.norm(forcing)))
    if backward_residual > epsilon:
        raise RuntimeError('Katz direct-solve backward residual exceeds epsilon.')
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError('Katz scores cannot be normalized safely.')
    return x / norm
