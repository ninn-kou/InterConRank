"""Build the full AMiner Academic Social Network from its four coordinated archives."""

from array import array
from contextlib import contextmanager
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import sparse

from .common import binary_matrix


class _BlockReader(io.RawIOBase):
    """Adapt libarchive's decompressed byte chunks to a buffered text reader."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = iter(blocks)
        self.pending = memoryview(b"")

    def readable(self):
        return True

    def readinto(self, destination):
        if not self.pending:
            self.pending = memoryview(next(self.blocks, b""))
        size = min(len(self.pending), len(destination))
        destination[:size] = self.pending[:size]
        self.pending = self.pending[size:]
        return size


@contextmanager
def _member(path):
    name = path.stem + ".txt"
    if path.suffix == ".rar":
        import libarchive

        with libarchive.file_reader(str(path)) as archive:
            for entry in archive:
                if entry.isfile and Path(entry.pathname).name == name:
                    with io.BufferedReader(_BlockReader(entry.get_blocks())) as binary:
                        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="\n") as text:
                            yield text
                    return
            raise ValueError(f"{name} missing from {path.name}")
    with ZipFile(path) as archive:
        member, = [p for p in archive.namelist() if Path(p).name == name]
        with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig") as text:
            yield text


def _author_references(raw):
    ids, h_index, total = array("q"), array("d"), array("d")
    with _member(raw / "AMiner-Author.zip") as stream:
        for line in stream:
            if line.startswith("#index"):
                ids.append(int(line[6:]))
                h_index.append(np.nan)
                total.append(np.nan)
            elif line.startswith("#hi"):
                value = int(line[3:])
                h_index[-1] = np.nan if value < 0 else value
            elif line.startswith("#cn"):
                value = int(line[3:])
                total[-1] = np.nan if value < 0 else value
    order = np.argsort(ids)
    return pd.DataFrame({"source_id": np.asarray(ids)[order], "h_index": np.asarray(h_index)[order],
                         "total_citations": np.asarray(total)[order]})


def _paper_layer(raw):
    ids, citing_ids, cited_ids = array("q"), array("q"), array("q")
    with _member(raw / "AMiner-Paper.rar") as stream:
        for line in stream:
            if line.startswith("#index"):
                ids.append(int(line[6:]))
            elif line.startswith("#%"):
                citing_ids.append(ids[-1])
                cited_ids.append(int(line[2:]))
    ids = np.sort(ids)
    source, target = np.searchsorted(ids, citing_ids), np.searchsorted(ids, cited_ids)
    valid = target < len(ids)
    valid &= ids[np.minimum(target, len(ids) - 1)] == np.asarray(cited_ids)
    valid &= target != source
    citations = binary_matrix(target[valid], source[valid], (len(ids), len(ids)))
    return pd.DataFrame({"source_id": ids, "citations": np.nan}), citations


def build(raw):
    authors = _author_references(raw)
    papers, citations = _paper_layer(raw)
    author_ids = authors.source_id.to_numpy()
    author, paper = array("q"), array("q")
    with _member(raw / "AMiner-Author2Paper.zip") as stream:
        for line in stream:
            if line.strip():
                _, aid, pid, _ = line.split()
                author.append(int(aid))
                paper.append(int(pid))
    authorship = binary_matrix(np.searchsorted(author_ids, author),
                               np.searchsorted(papers.source_id.to_numpy(), paper),
                               (len(authors), len(papers)))
    del author, paper
    first, second, weights = array("q"), array("q"), array("d")
    with _member(raw / "AMiner-Coauthor.zip") as stream:
        for line in stream:
            if line.strip():
                left, right, count = line.strip().removeprefix("#").split()
                if left != right:
                    first.append(int(left))
                    second.append(int(right))
                    weights.append(float(count))
    # Each undirected pair appears exactly once in the pinned coauthor archive.
    upper = sparse.coo_matrix((weights, (np.searchsorted(author_ids, first),
                                        np.searchsorted(author_ids, second))),
                              shape=(len(authors), len(authors))).tocsr()
    coauthors = (upper + upper.T).tocsr()
    coauthors.sort_indices()
    return coauthors, authorship, citations, authors, papers
