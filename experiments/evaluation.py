"""Ranking metrics for the paper's author–paper graphs."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from baselines import VISPUB_BASELINES

TARGETS = (
    ("Author h-index", 0, "h_index"),
    ("Author total citations", 0, "total_citations"),
    ("Paper citations", 1, "citations"),
)

# Method order in the pairwise correlation figure.
METHODS = (*VISPUB_BASELINES, "InterConRank")

__all__ = ["graph_counts", "evaluate", "agreement"]


def graph_counts(dataset, blocks):
    """Count each undirected coauthor pair and each authorship link once."""
    coauthors, authorship = blocks[0]
    citations = blocks[1][1]
    coauthor_pairs = coauthors.nnz // 2
    return {
        "Dataset": dataset,
        "Layers": len(blocks),
        "Authors": coauthors.shape[0],
        "Papers": citations.shape[0],
        "Nodes": coauthors.shape[0] + citations.shape[0],
        "Coauthors": coauthor_pairs,
        "Citations": citations.nnz,
        "Authorship": authorship.nnz,
        "Edges": coauthor_pairs + citations.nnz + authorship.nnz,
    }


def _top_quarter(values, source_ids, k):
    """Select exactly k nodes, breaking an exact cutoff tie by source ID."""
    cutoff = np.partition(values, len(values) - k)[len(values) - k]
    selection = values > cutoff
    tied = np.flatnonzero(values == cutoff)
    needed = k - np.count_nonzero(selection)
    if needed < len(tied):
        chosen = np.argpartition(source_ids[tied], needed - 1)[:needed]
        selection[tied[chosen]] = True
    else:
        selection[tied] = True
    return selection, cutoff, len(tied), needed


def _centred_ranks(values):
    ranks = rankdata(values, method="average")
    ranks -= ranks.mean()
    return ranks, np.linalg.norm(ranks)


def evaluate(dataset, author_refs, paper_refs, scores):
    """Compute target-level Spearman and Hits@25% from unrounded scores.

    Use reference frames from data.prepare_dataset and method score pairs from
    experiments.ranking.run_rankings, both in the same graph-node order.
    Source IDs keep their native order (numeric for VisPubData/AMiner,
    lexical for Semantic Scholar/DBLP V14). Missing references are excluded
    separately for each target, with known zeros retained.
    """
    refs = (author_refs, paper_refs)
    ids = []
    for frame in refs:
        if frame.source_id.hasnans or not frame.source_id.is_unique:
            raise ValueError("Each layer requires a unique, nonmissing source_id for each node.")
        ids.append(frame.source_id.to_numpy())
    for method, pair in scores.items():
        if len(pair) != 2:
            raise ValueError(f"{method} must supply one score vector per layer.")
        for vector, frame in zip(pair, refs):
            if np.shape(vector) != (len(frame),) or not np.isfinite(vector).all():
                raise ValueError(f"{method} needs one finite score per node in graph order.")

    rows = []
    for target, layer, column in TARGETS:
        reference = refs[layer][column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(reference).any():
            raise ValueError(f"{target} reference contains an infinite value.")
        total = len(reference)
        known = ~np.isnan(reference)
        n = int(np.count_nonzero(known))
        k = (n + 3) // 4
        source_ids = ids[layer]
        if n < total:
            reference = reference[known]
            source_ids = source_ids[known]
        cutoff = tie_size = tie_selected = np.nan
        if n:
            ref_top, cutoff, tie_size, tie_selected = _top_quarter(reference, source_ids, k)
            ref_ranks, ref_norm = _centred_ranks(reference)
        for method, pair in scores.items():
            hits, rho, rate = pd.NA, np.nan, np.nan
            if n:
                values = np.asarray(pair[layer])
                if n < total:
                    values = values[known]
                predicted_top = _top_quarter(values, source_ids, k)[0]
                hits = int(np.count_nonzero(predicted_top & ref_top))
                rate = 100.0 * hits / k
                ranks, norm = _centred_ranks(values)
                if norm and ref_norm:
                    # Reversing both rank orderings preserves their correlation.
                    rho = np.clip((ranks @ ref_ranks) / (norm * ref_norm), -1.0, 1.0)
            rows.append({
                "Dataset": dataset, "Method": method, "Target": target,
                "Nodes": total, "Evaluated": n, "K": k, "Hits": hits,
                "Spearman": rho, "Hits@25%": rate, "ReferenceCutoff": cutoff,
                "ReferenceTieSize": tie_size, "ReferenceTieSelected": tie_selected,
            })
    metrics = pd.DataFrame(rows)
    metrics["Hits"] = metrics["Hits"].astype("Int64")
    rates = metrics.pivot(index="Method", columns="Target", values="Hits@25%")
    mean_author_rate = (rates["Author h-index"] + rates["Author total citations"]) / 2
    # Give equal weight to layers with a paper-defined target.
    average = mean_author_rate if dataset == "AMiner" else (mean_author_rate + rates["Paper citations"]) / 2
    metrics["AvgHits@25%"] = metrics["Method"].map(average)
    return metrics


def agreement(scores):
    """Pairwise Spearman matrices over *all* nodes in each layer."""
    method_order = [method for method in METHODS if method in scores]
    matrices = []
    for layer in (0, 1):
        frame = pd.DataFrame({method: scores[method][layer] for method in method_order})
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Agreement requires finite scores for all graph nodes.")
        matrices.append(frame.corr(method="spearman"))
    return tuple(matrices)

