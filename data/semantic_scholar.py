"""Build the 2005 author–paper graph and its cumulative 2015 references."""

from array import array
import gzip
import json
from pathlib import Path

import pandas as pd

from .common import binary_matrix, coauthors_from_authorship


def _identifiers(path):
    # The file uses LF as the record boundary. Some names contain other
    # Unicode line separators and must remain single source identifiers.
    with gzip.open(path, "rb") as stream:
        return [line.removesuffix(b"\n").removesuffix(b"\r").decode("utf-8")
                for line in stream]


def _authorship(path, author_ids, paper_index):
    authors = {name: index for index, name in enumerate(author_ids)}
    rows, cols = array("i"), array("i")
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as stream:
        for line in stream:
            record = json.loads(line)
            paper = paper_index.get(record["id"])
            if paper is None:
                continue
            if record["year"] > 2005:
                raise ValueError("A paper in the 2005 cohort has a later publication year.")
            for name in set(record["authors"]):
                rows.append(authors[name])
                cols.append(paper)
    return binary_matrix(rows, cols, (len(authors), len(paper_index)))


def _citations(path, paper_index):
    rows, cols = array("i"), array("i")
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as stream:
        for line in stream:
            record = json.loads(line)
            citing = paper_index.get(record["id"])
            if citing is None:
                continue
            for identifier in set(record["cites"] or ()):
                cited = paper_index.get(identifier)
                if cited is not None and cited != citing:
                    rows.append(cited)
                    cols.append(citing)
    return binary_matrix(rows, cols, (len(paper_index), len(paper_index)))


def _references(path, ids, columns):
    # The release stores response rows in the order of its separate ID list.
    refs = pd.read_csv(path, sep="\t", usecols=list(columns)).rename(columns=columns)
    refs.insert(0, "source_id", ids)
    return refs[["source_id", *columns.values()]]


def build(raw: Path):
    """Read the six source archives for the June 2016 release.

    The 1975–2005 cohort files fix matrix rows and columns. Restrict citation
    edges to two cohort members. Reference column 9 is the last of ten
    cumulative observations (2006–2015).
    """
    cohort = "1975-2005-2015-2.tsv.gz"
    author_ids = _identifiers(raw / f"authors-{cohort}")
    paper_ids = _identifiers(raw / f"paperIds-{cohort}")
    papers = {identifier: index for index, identifier in enumerate(paper_ids)}
    authorship = _authorship(raw / "paperIdToPaperData-1975-2015.json.gz", author_ids, papers)
    coauthors = coauthors_from_authorship(authorship)
    citations = _citations(raw / "citationGraph-1975-2015.json.gz", papers)
    authors = _references(raw / f"authorResponses-{cohort}", author_ids,
                          {"hindex_in_9": "h_index", "total_citations_in_9": "total_citations"})
    refs = _references(raw / f"paperResponses-{cohort}", paper_ids,
                       {"total_citations_in_9": "citations"})
    return coauthors, authorship, citations, authors, refs
