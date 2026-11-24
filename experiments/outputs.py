"""Export paper measurements and ground truth from the current dataset run."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .evaluation import agreement, evaluate, graph_counts
from .plotting import plot_agreement


def save_dataset_results(dataset, output_dir, blocks, author_refs, paper_refs,
                         scores, timings, *, display_figure=None):
    """Write CSVs from the supplied in-memory results and return per-target metrics.

    Each execution updates the same files in output_dir. Author and paper
    reference rows follow graph order and retain their source IDs. Pass
    IPython's display function to show the VisPubData agreement figure.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(dataset, author_refs, paper_refs, scores)
    counts = graph_counts(dataset, blocks)
    # Finish all exports before replacing the previous measurements.
    with TemporaryDirectory(dir=destination) as temporary:
        stage = Path(temporary)
        author_refs.to_csv(stage / "author_references.csv", index=False)
        paper_refs.to_csv(stage / "paper_references.csv", index=False)
        pd.DataFrame([counts]).to_csv(stage / "graph_counts.csv", index=False)
        metrics.to_csv(stage / "per_target_metrics.csv", index=False)
        timings.to_csv(stage / "running_times.csv", index=False)

        if dataset == "VisPubData":
            import matplotlib.pyplot as plt

            author_corr, paper_corr = agreement(scores)
            author_corr.to_csv(stage / "agreement_authors.csv", index_label="Method")
            paper_corr.to_csv(stage / "agreement_papers.csv", index_label="Method")
            figure = plot_agreement(author_corr, paper_corr, counts["Authors"], counts["Papers"])
            try:
                with plt.rc_context({"pdf.fonttype": 42}):
                    figure.savefig(stage / "pairwise_correlations.pdf", format="pdf", dpi=1200)
                if display_figure is not None:
                    display_figure(figure)
            finally:
                plt.close(figure)
        for file in stage.iterdir():
            file.replace(destination / file.name)
    return metrics
