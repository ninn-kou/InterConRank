"""The published comparison methods on the author–paper graphs."""

import numpy as np
from scipy import sparse

from .bipartite import birank, cohits
from .co_ranking import co_ranking
from .hits import hits_authorities
from .katz import katz
from .single_layer import degree, eigenvector, pagerank, shortest_paths

VISPUB_BASELINES = ('Betweenness', 'Closeness', 'Degree', 'Eigenvector', 'PageRank',
                    'Katz', 'HITS', 'BiRank', 'CoHITS', 'Co-Ranking')
LARGE_BASELINES = ('Degree', 'PageRank', 'BiRank', 'CoHITS', 'Co-Ranking')

__all__ = ['rank_baseline', 'VISPUB_BASELINES', 'LARGE_BASELINES']


def _matrix(values):
    if np.iscomplexobj(values):
        raise ValueError('Edge weights must be real.')
    matrix = sparse.csr_matrix(values, dtype=float)
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Edge weights must be finite and nonnegative.')
    if not matrix.has_canonical_format or np.any(matrix.data == 0):
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if not np.isfinite(matrix.data).all():
            raise ValueError('Combined edge weights must be finite.')
    return matrix


def rank_baseline(method, coauthors, citations, authorship, *, max_iter=10000):
    """Return (author_scores, paper_scores) in matrix row order.

    The two square blocks and the binary author-by-paper block all use
    receiving rows. The paper fixes iterative tolerance at 1e-12 and uses
    uniform within-layer priors. VisPubData uses a 100000 iteration limit.
    """
    if method not in VISPUB_BASELINES:
        raise ValueError(f'Unknown baseline {method!r}. Choose one of {VISPUB_BASELINES}.')
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer.')
    n_authors, n_papers = authorship.shape
    if (min(n_authors, n_papers) == 0 or coauthors.shape != (n_authors, n_authors)
            or citations.shape != (n_papers, n_papers)):
        raise ValueError('Expected two square layer matrices and an author-by-paper matrix.')

    epsilon = 1e-12
    if method in ('BiRank', 'CoHITS'):
        score = birank if method == 'BiRank' else cohits
        return score(_matrix(authorship), epsilon, max_iter)
    if method == 'Co-Ranking':
        return co_ranking(_matrix(coauthors), _matrix(citations), _matrix(authorship), epsilon, max_iter)

    def score_layer(values, directed):
        matrix = _matrix(values)
        if method == 'Degree':
            return degree(matrix)
        if method in ('Betweenness', 'Closeness'):
            return shortest_paths(matrix, directed, method)
        if method == 'Katz':
            return katz(matrix, epsilon=epsilon, max_iter=max_iter)
        scorers = {'Eigenvector': eigenvector, 'PageRank': pagerank, 'HITS': hits_authorities}
        return scorers[method](matrix, epsilon, max_iter)

    return score_layer(coauthors, False), score_layer(citations, True)
