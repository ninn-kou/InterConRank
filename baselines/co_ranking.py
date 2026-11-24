"""Co-Ranking of authors and documents (Zhou et al., ICDM 2007)."""

import numpy as np


def _transition(matrix):
    """Normalize outgoing columns, leaving the dangling correction implicit."""
    degrees = np.asarray(matrix.sum(axis=0)).ravel()
    if not np.isfinite(degrees).all():
        raise ValueError('Outgoing weight sums must be finite.')
    transition = matrix.copy()
    transition.data /= degrees[transition.indices]
    return transition, degrees == 0


def _step(walk, scores):
    transition, dangling = walk
    return transition @ scores + float(np.sum(scores[dangling])) / transition.shape[0]


def _within(walk, scores):
    # Two intra-layer hops, each allowing a uniform teleport with probability .1.
    for _ in range(2):
        scores = .9 * _step(walk, scores) + .1 * float(np.sum(scores)) / len(scores)
    return scores


def _across(forward, backward, scores):
    return _step(forward, _step(backward, _step(forward, scores)))


def co_ranking(coauthors, citations, authorship, epsilon, max_iter):
    """Two intra-layer hops or three alternating authorship hops per update.

    Each paper first assigns equal fractional credit to its authors. The
    resulting weights define both directed cross-layer random walks.
    """
    if np.any(authorship.data != 1):
        raise ValueError('Co-Ranking requires binary authorship.')
    paper_to_author = _transition(authorship)
    author_to_paper = _transition(paper_to_author[0].T.tocsr())
    author_walk, paper_walk = _transition(coauthors), _transition(citations)
    authors, papers = [np.full(n, 1. / n) for n in authorship.shape]
    for _ in range(max_iter):
        new_authors = (.8 * _within(author_walk, authors)
                       + .2 * _across(paper_to_author, author_to_paper, papers))
        new_papers = (.8 * _within(paper_walk, papers)
                      + .2 * _across(author_to_paper, paper_to_author, authors))
        error = max(np.linalg.norm(new_authors - authors), np.linalg.norm(new_papers - papers))
        authors, papers = new_authors, new_papers
        if error <= epsilon:
            return authors, papers
    raise RuntimeError('Co-Ranking did not converge within max_iter.')
