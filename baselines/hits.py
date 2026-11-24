"""HITS authority and hub scores on one supplied graph layer.

Kleinberg (1999), Authoritative Sources in a Hyperlinked Environment,
Journal of the ACM 46(5), 604--632. DOI: 10.1145/324133.324140.
Primary source: https://www.cs.cornell.edu/home/kleinber/auth.pdf
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _component_direction(W, epsilon, max_iter):
    """Positive hub direction of one nonzero bipartite component, without W.T W."""
    count = W.shape[1]
    WT = W.T.tocsr()
    ones = np.ones(count)
    row_strength = WT @ (W @ ones)
    # The all-ones hub start is already an exact eigenvector in this case.
    regular = bool(np.all(row_strength == row_strength[0]))
    if regular:
        root = float(row_strength[0])
        return ones / np.sqrt(count), root, root, regular
    h = ones / np.sqrt(count)
    local_epsilon = max(32 * np.finfo(float).eps, epsilon * .01)
    for _ in range(max_iter):
        product = WT @ (W @ h)
        norm = float(np.linalg.norm(product))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError('HITS component iteration lost finite positive mass.')
        new_h = product / norm
        change = float(np.linalg.norm(new_h - h))
        h = new_h
        product = WT @ (W @ h)
        if np.any(h <= 0):
            raise RuntimeError('HITS component iteration lost positive support.')
        ratios = product / h
        lower, upper = float(ratios.min()), float(ratios.max())
        root = float(h @ product)
        residual = float(np.linalg.norm(product - root * h))
        if (change <= local_epsilon and residual <= local_epsilon * root
                and upper-lower <= max(local_epsilon, 128*np.finfo(float).eps)*root):
            return h, lower, upper, regular
    raise RuntimeError('HITS did not converge within max_iter; no scores were returned.')


def hits_authorities(A, epsilon=1.0e-12, max_iter=100000):
    """Return incoming HITS authority scores from uniform starting hubs.

    Solve the hub Gram system on each source–receiver component. Components
    with maximal singular value retain their projection of the uniform start.
    Other components receive exact zero. Check both singular equations before
    returning authorities. An edgeless layer receives uniform zero scores.
    """
    if (isinstance(epsilon, (bool, np.bool_))
            or not isinstance(epsilon, (int, float, np.integer, np.floating))
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('epsilon must be a finite positive real number.')
    if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0):
        raise ValueError('max_iter must be a positive integer.')
    if len(getattr(A, 'shape', ())) != 2 or np.iscomplexobj(A):
        raise ValueError('HITS requires a real square adjacency matrix.')
    A = sparse.csr_matrix(A, dtype=float, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    n = A.shape[0]
    if n == 0 or A.shape[1] != n:
        raise ValueError('HITS requires a nonempty square adjacency matrix.')
    if not np.isfinite(A.data).all() or np.any(A.data < 0):
        raise ValueError('HITS link strengths must be finite and nonnegative.')
    if A.nnz == 0:
        return np.zeros(n)

    # Uniform positive scaling leaves HITS vectors and component order unchanged.
    scale = float(A.data.max())
    A.data /= scale
    AT = A.T.tocsr()
    lift = sparse.bmat([[None, A], [AT, None]], format='csr')
    count, labels = connected_components(lift, directed=False)
    del lift
    rows, columns = [[] for _ in range(count)], [[] for _ in range(count)]
    for node in range(n):
        rows[labels[node]].append(node)
        columns[labels[n+node]].append(node)
    components = []
    arithmetic_margin = 64 * np.finfo(float).eps
    for component in range(count):
        if not rows[component] or not columns[component]:
            continue
        W = A[rows[component], :][:, columns[component]].tocsr()
        h, lower, upper, regular = _component_direction(W, epsilon, max_iter)
        components.append({'columns': columns[component], 'h': h,
                           'lower': lower * (1-arithmetic_margin),
                           'upper': upper * (1+arithmetic_margin), 'regular': regular})
    largest_lower = max(item['lower'] for item in components)
    active = [item for item in components if item['upper'] >= largest_lower]
    h = np.zeros(n)
    for item in active:
        # P_top * ones is the exact limiting direction of the published hub start.
        # Preserve the known constant projection for regular Gram components.
        projection = np.ones(len(item['columns'])) if item['regular'] else item['h'] * item['h'].sum()
        h[item['columns']] = projection
    h /= np.linalg.norm(h)
    raw_authority = A @ h
    singular_value = float(np.linalg.norm(raw_authority))
    a = raw_authority / singular_value
    authority_residual = float(np.linalg.norm(A @ h - singular_value * a)) / singular_value
    hub_residual = float(np.linalg.norm(AT @ a - singular_value * h)) / singular_value
    if (not np.isfinite(h).all() or not np.isfinite(a).all()
            or max(authority_residual, hub_residual) > epsilon):
        raise RuntimeError('HITS final singular equations exceed epsilon; no scores were returned.')
    return a
