"""Run the methods and fixed parameter settings used in the paper."""

from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from baselines import LARGE_BASELINES, VISPUB_BASELINES, rank_baseline
from interconrank import InterConRank


def run_rankings(dataset, blocks, *, show_progress=False):
    """Score both layers of one prepared dataset and measure fresh wall times.

    Pass the 2 × 2 block array returned by `data.prepare_dataset`. Each block
    uses receiver rows and sender columns. Scores are keyed by the method
    names used in the tables, with one (authors, papers) pair per method.
    Set show_progress=True to display the active method and completed count.
    Timing covers each scoring call, excluding progress updates, preparation and evaluation.
    """
    if dataset == "VisPubData":
        baselines, max_iter = VISPUB_BASELINES, 100_000
    elif dataset in ("SemanticScholar", "AMiner", "DBLP_V14"):
        baselines, max_iter = LARGE_BASELINES, 10_000
    else:
        raise ValueError(f"Unknown paper dataset: {dataset!r}")

    coauthors, authorship = blocks[0]
    citations = blocks[1][1]
    sizes = (coauthors.shape[0], citations.shape[0])
    scores = {}
    timings = []
    methods = (*baselines, "InterConRank")
    with tqdm(total=len(methods), desc=f"{dataset} rankings", unit="method",
              disable=not show_progress, leave=False) as bar:
        for method in methods:
            bar.set_postfix_str(method)
            start = perf_counter()
            if method == "InterConRank":
                pair = InterConRank(
                    blocks, lambda_=0.51, delta=0.43, tol=1e-14, max_iter=max_iter
                )
            else:
                pair = rank_baseline(
                    method, coauthors, citations, authorship, max_iter=max_iter
                )
            elapsed = perf_counter() - start
            for size, vector in zip(sizes, pair, strict=True):
                if np.shape(vector) != (size,) or not np.isfinite(vector).all():
                    raise RuntimeError(f"{method} returned invalid scores for {dataset}")
            scores[method] = pair
            timings.append({"Dataset": dataset, "Method": method, "Seconds": elapsed})
            bar.update(1)
    return scores, pd.DataFrame(timings)
