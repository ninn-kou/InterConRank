"""Prepare the DBLP + Citation V14 release in its original node order."""

from array import array
import codecs
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from .common import author_measures, binary_matrix, coauthors_from_authorship


def _papers(stream):
    """Decode objects from the pinned array without loading the entire JSON member."""
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    buffer = stream.read(chunk_size).lstrip()
    if not buffer.startswith("["):
        raise ValueError("Invalid DBLP V14 array header")
    pos = 1
    while True:
        while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
            pos += 1
        if pos < len(buffer) and buffer[pos] == "]":
            return
        if pos < len(buffer):
            try:
                record, pos = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                pass  # The next chunk may complete the current object.
            else:
                yield record
                continue
        tail = buffer[pos:]
        chunk = stream.read(chunk_size)
        if not chunk or len(tail) > 64 * chunk_size:
            raise ValueError("Malformed or truncated DBLP V14 JSON member")
        buffer, pos = tail + chunk, 0


def _records(path):
    with tarfile.open(path, mode="r|gz", bufsize=256 * 1024) as archive:
        for member in archive:
            if member.isfile() and Path(member.name).name == "dblp_v14.json":
                with archive.extractfile(member) as binary:
                    yield from _papers(codecs.getreader("utf-8-sig")(binary))
                return
        raise ValueError("Missing dblp_v14.json in the DBLP V14 archive")


def build(raw: Path):
    """Build binary citation/authorship blocks, weighted coauthorship and references.

    Citations may point to papers that appear later in the archive. Temporary
    indices let us read the 20 GB JSON member once, then resolve its endpoints.
    Paper rows follow source order and author rows follow first appearance.
    """
    author_index = {}
    paper_index = {}
    paper_ids = []
    paper_to_temporary = array("i")
    counts = array("d")
    authors, publications = array("i"), array("i")
    cited, citing = array("i"), array("i")
    for record in _records(raw / "dblp_v14.tar.gz"):
        pid = record["id"]
        paper = len(paper_ids)
        paper_ids.append(pid)
        paper_to_temporary.append(paper_index.setdefault(pid, len(paper_index)))
        count = record.get("n_citation")
        counts.append(np.nan if count is None or count == -1 else count)

        # dict.fromkeys preserves the order of authors on the original paper.
        for aid in dict.fromkeys(person.get("id") for person in record.get("authors") or []):
            if aid is None or aid == "":
                continue
            authors.append(author_index.setdefault(aid, len(author_index)))
            publications.append(paper)
        for ref in set(record.get("references") or []):
            if ref != pid:
                cited.append(paper_index.setdefault(ref, len(paper_index)))
                citing.append(paper)

    n_papers = len(paper_ids)
    n_authors = len(author_index)
    paper_rows = np.asarray(paper_to_temporary, dtype=np.int32)
    original_rows = np.full(len(paper_index), -1, dtype=np.int32)
    original_rows[paper_rows] = np.arange(n_papers, dtype=np.int32)
    if np.count_nonzero(original_rows >= 0) != n_papers:
        raise ValueError("Duplicate DBLP paper IDs")
    del paper_index, paper_to_temporary, paper_rows
    receivers = original_rows[np.asarray(cited, dtype=np.int32)]
    known = receivers >= 0
    citations = binary_matrix(receivers[known], np.asarray(citing, dtype=np.int32)[known],
                              (n_papers, n_papers))
    del cited, citing, original_rows, receivers, known

    authorship = binary_matrix(authors, publications, (n_authors, n_papers))
    del authors, publications
    coauthors = coauthors_from_authorship(authorship)
    citation_counts = np.asarray(counts, dtype=np.float64)
    h_index, totals = author_measures(authorship, citation_counts)
    author_refs = pd.DataFrame({"source_id": list(author_index), "h_index": h_index,
                                "total_citations": totals})
    paper_refs = pd.DataFrame({"source_id": paper_ids, "citations": citation_counts})
    return coauthors, authorship, citations, author_refs, paper_refs
