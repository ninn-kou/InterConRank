"""Weighted single-layer centralities. Rows receive edges from columns."""

import math

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import connected_components


def degree(matrix):
    """Return weighted coauthor degree or incoming citation count."""
    return np.asarray(matrix.sum(axis=1)).ravel()


def pagerank(matrix, epsilon, max_iter):
    """Weighted PageRank with uniform personalization and dangling redistribution."""
    n = matrix.shape[0]
    prior = np.ones(n) / n
    outgoing = np.asarray(matrix.sum(axis=0)).ravel()
    inverse = np.divide(1., outgoing, out=np.zeros(n), where=outgoing > 0)
    transition = matrix.multiply(inverse).tocsr()
    dangling = outgoing == 0
    damping = .85
    scores = prior.copy()
    for _ in range(max_iter):
        new = damping * (transition @ scores + prior * scores[dangling].sum()) + (1 - damping) * prior
        if np.linalg.norm(new - scores) <= epsilon:
            return new
        scores = new
    raise RuntimeError('PageRank did not converge within max_iter.')


def shortest_paths(matrix, directed, method):
    """All-source betweenness or incoming Wasserman–Faust closeness.

    For positive integer link strengths, scale reciprocal distances to integers
    so equally long paths remain tied during Dijkstra's search.
    """
    if not directed and (matrix != matrix.T).nnz:
        raise ValueError('An undirected layer must be symmetric.')
    if np.any(matrix.diagonal()):
        raise ValueError('Remove self-loops before computing shortest-path baselines.')
    if np.any(matrix.data != np.floor(matrix.data)):
        raise ValueError('Exact inverse-strength paths require integer weights.')
    scale = math.lcm(*(int(weight) for weight in np.unique(matrix.data)))
    graph = nx.from_scipy_sparse_array(matrix.T, create_using=nx.DiGraph if directed else nx.Graph,
                                       edge_attribute='strength')
    for _, _, edge in graph.edges(data=True):
        edge['distance'] = scale // int(edge['strength'])
    if method == 'Betweenness':
        values = nx.betweenness_centrality(graph, normalized=True, weight='distance', endpoints=False)
        return np.array([values[node] for node in range(matrix.shape[0])])
    values = nx.closeness_centrality(graph, distance='distance', wf_improved=True)
    return np.array([values[node] * scale for node in range(matrix.shape[0])])


def _power_eigenvector(matrix, epsilon, max_iter):
    # Adding I preserves eigenvectors and removes periodic sign oscillation.
    x = np.ones(matrix.shape[0], dtype=float)
    x /= np.linalg.norm(x)
    for _ in range(max_iter):
        y = x + matrix @ x
        y /= np.linalg.norm(y)
        change = float(np.linalg.norm(y - x, ord=1))
        x = y
        product = matrix @ x
        eigenvalue = float(x @ product)
        residual = float(np.linalg.norm(product - eigenvalue * x))
        if change <= epsilon and residual <= epsilon * max(1., abs(eigenvalue)):
            return x, eigenvalue
    raise RuntimeError('Eigenvector did not converge within max_iter.')


def eigenvector(matrix, epsilon, max_iter):
    """Incoming Perron centrality with exact zeros outside the dominant support."""
    count, labels = connected_components(matrix, directed=True, connection='strong')
    roots = np.zeros(count)
    component_nodes = [[] for _ in range(count)]
    for node, component in enumerate(labels):
        component_nodes[component].append(node)
    for component, nodes in enumerate(component_nodes):
        if len(nodes) == 1:
            roots[component] = matrix[nodes[0], nodes[0]]
        else:
            block = matrix[nodes, :][:, nodes]
            roots[component] = _power_eigenvector(block, epsilon * .01, max_iter)[1]

    dominant_root = float(roots.max())
    if dominant_root == 0 and matrix.nnz:
        raise ValueError('Eigenvector centrality is undefined on this acyclic graph.')
    tolerance = 20 * epsilon * max(1., dominant_root)
    critical = set(np.flatnonzero(np.abs(roots - dominant_root) <= tolerance))
    condensation = nx.DiGraph()
    condensation.add_nodes_from(range(count))
    receivers, senders = matrix.nonzero()
    for source, target in zip(labels[senders], labels[receivers]):
        if source != target:
            condensation.add_edge(source, target)
    support = set(critical)
    for component in critical:
        reachable = nx.descendants(condensation, component)
        if reachable & critical:
            raise ValueError('Dominant components are linked, so the Perron ranking is ambiguous.')
        support.update(reachable)

    x, _ = _power_eigenvector(matrix, epsilon, max_iter)
    x[~np.isin(labels, list(support))] = 0.
    x /= np.linalg.norm(x)
    eigenvalue = float(x @ (matrix @ x))
    if np.linalg.norm(matrix @ x - eigenvalue * x) > epsilon * max(1., abs(eigenvalue)):
        raise RuntimeError('The eigenvector residual exceeds tolerance.')
    return x
