"""Fetch missing source files and rebuild each evaluation dataset."""

import json
from pathlib import Path

from tqdm.auto import tqdm

from . import aminer, dblp, semantic_scholar, vispub
from .sources import download_source


SOURCE_MANIFEST = Path(__file__).with_name("sources.json")
BUILDERS = {"VisPubData": vispub.build, "SemanticScholar": semantic_scholar.build,
            "AMiner": aminer.build, "DBLP_V14": dblp.build}


def download_dataset(name, dataset_dir=None):
    """Return the raw source directory, downloading only missing files."""
    if name not in BUILDERS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {', '.join(BUILDERS)}.")
    sources = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))[name]
    directory = SOURCE_MANIFEST.parent / name if dataset_dir is None else Path(dataset_dir).expanduser()
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for filename, spec in sources.items():
        target = raw / filename
        if target.is_file():
            tqdm.write(f"{name} — using {filename}")
        else:
            tqdm.write(f"{name} — downloading {filename}")
            download_source(target, spec)
    return raw


def prepare_dataset(name, dataset_dir=None):
    """Build graph blocks and aligned reference frames from the raw sources."""
    raw = download_dataset(name, dataset_dir)
    tqdm.write(f"{name} — building the graph and evaluation references")
    coauthors, authorship, citations, author_refs, paper_refs = BUILDERS[name](raw)
    blocks = ((coauthors, authorship), (authorship.T.tocsr(), citations))
    return blocks, author_refs, paper_refs
