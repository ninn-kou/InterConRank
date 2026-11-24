"""BiRank (He et al., 2017) and CoHITS (Deng et al., 2009) on authorship."""

import numpy as np


def birank(authorship, epsilon, max_iter):
    """Alternate papers then authors on symmetrically normalized authorship."""
    damping = .85
    priors = [np.ones(n) / n for n in authorship.shape]
    inverse_roots = []
    for axis in (1, 0):
        strength = np.asarray(authorship.sum(axis=axis)).ravel()
        positive = strength > 0
        strength[positive] = 1. / np.sqrt(strength[positive])
        inverse_roots.append(strength)
    author_scale, paper_scale = inverse_roots
    authors, papers = [np.ones(n) for n in authorship.shape]
    transpose = authorship.T
    for _ in range(max_iter):
        new_papers = (damping * paper_scale * (transpose @ (author_scale * authors))
                      + (1 - damping) * priors[1])
        new_authors = (damping * author_scale * (authorship @ (paper_scale * new_papers))
                       + (1 - damping) * priors[0])
        error = max(np.linalg.norm(new_authors - authors), np.linalg.norm(new_papers - papers))
        authors, papers = new_authors, new_papers
        if error <= epsilon:
            return authors, papers
    raise RuntimeError('BiRank did not converge within max_iter.')


def cohits(authorship, epsilon, max_iter):
    """Alternate authors then papers, redistributing dangling mass uniformly."""
    damping = .85
    priors, inverse, dangling = [], [], []
    for n, axis in zip(authorship.shape, (1, 0)):
        uniform = np.ones(n) / n
        priors.append(uniform / uniform.sum())
        strength = np.asarray(authorship.sum(axis=axis)).ravel()
        inverse.append(np.divide(1., strength, out=np.zeros_like(strength), where=strength > 0))
        dangling.append(strength == 0)
    transpose = authorship.T
    authors, papers = priors
    for _ in range(max_iter):
        to_authors = authorship @ (inverse[1] * papers) + priors[0] * papers[dangling[1]].sum()
        new_authors = damping * to_authors + (1 - damping) * priors[0]
        to_papers = transpose @ (inverse[0] * new_authors) + priors[1] * new_authors[dangling[0]].sum()
        new_papers = damping * to_papers + (1 - damping) * priors[1]
        error = max(np.linalg.norm(new_authors - authors), np.linalg.norm(new_papers - papers))
        authors, papers = new_authors, new_papers
        if error <= epsilon:
            return authors, papers
    raise RuntimeError('CoHITS did not converge within max_iter.')
