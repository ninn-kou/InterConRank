# Data Sources and Evaluation References

The notebook keeps each publisher file under `data/<dataset>/raw/`. It reuses files already in that folder and downloads any missing files using the URLs in [sources.json](sources.json).

| Dataset | Source files | Graph and evaluation references |
| --- | ---: | --- |
| VisPubData | 3 CSVs · 5 MB | The September 2021 export includes IEEE Xplore paper citation counts from February 2021. Each author's h-index and total citations are computed over their included papers. [Publisher's CSVs](https://hauser-net.net/download/VisPubDataEtc/index.html) |
| Semantic Scholar | 6 compressed files · 810 MB | The graph uses the 1975-2005 author and paper cohorts. The final observation in the response files supplies cumulative author h-index, author citations and paper citations through 2015. [Weihs and Etzioni release](https://github.com/Lucaweihs/impact-prediction) |
| AMiner | 4 archives · 787 MB | The full Academic Social Network provides collaboration weights, citation links, authorships and the author h-index and total citation targets. AMiner supplies references for the author layer only. [Publisher's release](https://www.aminer.cn/aboutus/en-US/articles/55af4228dabfae1ce3ed1253) |
| DBLP V14 | 1 compressed archive · 6.08 GB | Paper citation counts in the January 2023 release supply the paper target and the corpus-based author h-index and citation totals. The parser streams the JSON member from the archive. [Publisher's release](https://www.aminer.cn/aboutus/en-US/articles/655db2202ab17a072284bc0c) |

The complete set of raw files occupies approximately 7.68 GB. AMiner requires the Python package `libarchive-c` and a system libarchive with RAR support. The largest graph requires several gigabytes of working memory.

`prepare_dataset(name)` loads the raw files and rebuilds the graph blocks, author references and paper references. The dataset names used for both `data/<dataset>/raw/` and `results/<dataset>/` are `VisPubData`, `SemanticScholar`, `AMiner` and `DBLP_V14`.

Author and paper reference rows follow their matrix order. The `source_id` column identifies the node used to break cutoff ties. Missing values exclude that node from the corresponding target. Known zeros participate in evaluation. Each graph block stores receiving nodes in its rows and sending nodes in its columns. Coauthor edges carry collaboration counts, authorship links are binary and reciprocal, and citation edges point from citing to cited papers.
