"""Build the directed VisPubData network from the 9 September 2021 CSVs."""

import pandas as pd

from .common import author_measures, binary_matrix, coauthors_from_authorship


def build(raw):
    authors = pd.read_csv(raw / "authors-unique-ext.csv", usecols=["ID", "Name"]).sort_values("ID")
    papers = pd.read_csv(raw / "papers.csv", usecols=["pID", "pAuthors", "pCitXPlore"]).sort_values("pID")
    links = pd.read_csv(raw / "paper-to-paper-citations.csv", usecols=["cFromID", "cToID"])
    author_index = {name.strip(): i for i, name in enumerate(authors.Name)}
    paper_index = {pid: i for i, pid in enumerate(papers.pID)}
    rows, cols = [], []
    for paper, names in enumerate(papers.pAuthors):
        for name in set(name.strip() for name in names.split(";") if name.strip()):
            rows.append(author_index[name])
            cols.append(paper)
    authorship = binary_matrix(rows, cols, (len(authors), len(papers)))
    edges = links.loc[links.cFromID != links.cToID]
    citations = binary_matrix(edges.cToID.map(paper_index), edges.cFromID.map(paper_index),
                              (len(papers), len(papers)))
    counts = papers.pCitXPlore.to_numpy(dtype=float)
    h_index, totals = author_measures(authorship, counts)
    author_refs = pd.DataFrame({"source_id": authors.ID.to_numpy(), "h_index": h_index,
                                "total_citations": totals})
    paper_refs = pd.DataFrame({"source_id": papers.pID.to_numpy(), "citations": counts})
    return coauthors_from_authorship(authorship), authorship, citations, author_refs, paper_refs
