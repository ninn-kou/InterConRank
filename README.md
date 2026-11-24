# InterConRank

InterConRank computes a centrality score for each node in an interconnected network. Each layer has its own node set and ranking. The standalone Python package supports any number of layers and depends on NumPy and SciPy.

## Use the Algorithm

From the extracted project directory, run `python -m pip install .` with Python 3.10 or newer. You can then import the module from any directory.

```python
from scipy.sparse import csr_matrix
from interconrank import InterConRank

coauthors = csr_matrix([[0, 1], [1, 0]])
citations = csr_matrix([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
authorship = csr_matrix([[1, 1, 0], [0, 1, 1]])

blocks = [[coauthors, authorship], [authorship.T, citations]]
author_scores, paper_scores = InterConRank(blocks, lambda_=0.51, delta=0.43)
```

In `blocks[target][source]`, rows receive influence and columns supply it. For example, `citations[i, j]` records a citation from paper *j* to paper *i*. Scores follow the input node order. Use `None` for an absent cross-layer block. The default prior is uniform within each layer. Supply `priors` for custom vectors or set `tol` and `max_iter` to control convergence.

## Run the Paper Experiments

Use Python 3.12 for the experiments. From the extracted project directory, run

```bash
python -m pip install -r requirements.txt
python -m jupyter lab notebooks/reproduce_paper.ipynb
```

Choose **Run All Cells** to score all 29 dataset–method combinations. To run one dataset, execute both setup cells and its experiment cell. The notebook looks for each source file in `data/<dataset>/raw/` and downloads only missing files. Starting with empty folders requires about 7.68 GB for all four datasets. Progress bars show downloads and the active ranking method. AMiner requires a system libarchive with RAR support. See [dataset sources](data/README.md).

Each experiment rebuilds its graph and evaluation references from the raw files. It writes the following outputs to `results/<dataset>/`, replacing the previous values for that dataset.

| File | Contents |
| --- | --- |
| `graph_counts.csv` | Author, paper, and edge counts underlying Table 1 |
| `author_references.csv`, `paper_references.csv` | Fresh evaluation targets in graph row order, with original source IDs |
| `per_target_metrics.csv` | Unrounded Spearman correlation, eligible-node count, cutoff, hit count, Hits@25% and weighted average. VisPubData supplies Table 2, and the other datasets supply Table 3. |
| `running_times.csv` | Wall-clock seconds to score both layers with each method on this run |

VisPubData also exports the author and paper agreement matrices and the correlation figure. AMiner supplies author references only, so its paper evaluation fields are empty.

The installable algorithm is in `interconrank/`, comparison methods in `baselines/`, and scoring and evaluation code in `experiments/`. The four dataset sections share a single [Jupyter Notebook](notebooks/Experiments.ipynb).
