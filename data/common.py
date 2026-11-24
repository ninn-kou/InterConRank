"""Shared sparse graph construction and corpus-based impact measures."""

import numpy as np
from scipy import sparse


def binary_matrix(rows, cols, shape):
    """Return a canonical float64 CSR with duplicate coordinates counted once."""
    weights = np.ones(len(rows), dtype=np.float64)
    matrix = sparse.coo_matrix((weights, (rows, cols)), shape=shape).tocsr()
    matrix.data.fill(1)
    matrix.sort_indices()
    return matrix


def coauthors_from_authorship(authorship):
    """Count shared papers between distinct authors."""
    coauthors = (authorship @ authorship.T).tocsr()
    coauthors.setdiag(0)
    coauthors.eliminate_zeros()
    coauthors.sort_indices()
    return coauthors


def author_measures(authorship, counts):
    """Corpus h-index and total citations; incomplete paper references yield NaN."""
    counts = np.asarray(counts, dtype=float)
    complete = np.asarray(authorship @ np.isnan(counts).astype(float)).ravel() == 0
    totals = np.asarray(authorship @ np.nan_to_num(counts)).ravel()
    totals[~complete] = np.nan
    h_index = np.full(authorship.shape[0], np.nan)
    for author in np.flatnonzero(complete):
        start, stop = authorship.indptr[author:author + 2]
        ranked = np.sort(counts[authorship.indices[start:stop]])[::-1]
        h_index[author] = np.count_nonzero(ranked >= np.arange(1, len(ranked) + 1))
    return h_index, totals
