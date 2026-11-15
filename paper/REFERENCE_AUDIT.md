# Reference audit — `paper/references.bib` (2026-11-15)

Every entry was looked up programmatically: Crossref REST API (title + author queries,
then the DOI record), the arXiv API (ID and title), and the Hugging Face model API.
"Verified" means the title, authors, venue, year, volume/pages and DOI in the `.bib`
match the registry record. Entries that no registry covers (NeurIPS/ICLR/ICML/JMLR/USENIX
papers without Crossref DOIs) are marked **check** — they match common citations but were not
confirmed by a registry in this audit; spot-check them against the venue page.

## Changes made
* Added DOIs to 20 entries.
* Fixed missing issue numbers (Pushpakom 18(1), Huang 30(12), Wu 27(5) article 60).
* Removed "Proceedings of the" from booktitles: the MDPI style already prints
  "In Proceedings of the …", which produced "Proceedings of the Proceedings of …".
* `naeini2015ece`: first author's surname is "Pakdaman Naeini" (Crossref).
* `es2024ragas`: title capitalization "RAGAs", pages 150–158 (Crossref).
* `kadavath2022know`: replaced "and others" with the first nine authors (arXiv).
* `openai2025gptoss`: arXiv:2508.10925 confirmed; TODO note removed.
* New entries for the model reports / cards now cited in Section 3.4 and for the judge:
  Gemma 4 (arXiv:2607.02770), Kimi K3 (arXiv:2607.24653), Nemotron 3 Super (arXiv:2604.12374),
  Qwen3.5-122B-A10B and Qwen3.8-27B (Hugging Face model cards; no technical report for these
  two text models was found on arXiv — only Qwen3.5-Omni and Qwen3.8-Omni reports).
* `sheneman2026mindrouter` copied from the RAGO/RAMP bibliographies (DOI confirmed on Crossref).

## Status per entry

| Key | Status | Source |
|---|---|---|
| pushpakom2019 | Verified | Crossref 10.1038/nrd.2018.168 |
| chandak2023primekg | Verified | Crossref 10.1038/s41597-023-01960-3 |
| huang2024txgnn | Verified | Crossref 10.1038/s41591-024-03233-x |
| himmelstein2017hetionet | Verified | Crossref 10.7554/eLife.26726 |
| zitnik2018decagon | Verified | Crossref 10.1093/bioinformatics/bty294 |
| bordes2013transe | **check** pages 2787–2795 | NeurIPS proceedings (no DOI) |
| sun2019rotate | **check** | ICLR 2019 (OpenReview, no DOI) |
| ali2021pykeen | **check** | JMLR 22(82) (no DOI) |
| wang2019kgat | Verified | Crossref 10.1145/3292500.3330989 |
| zhang2020explainable | Verified | Crossref 10.1561/1500000066 |
| wu2024llmrec | Verified | Crossref 10.1007/s11280-024-01291-2 |
| lewis2020rag | **check** pages 9459–9474 | NeurIPS proceedings (no DOI) |
| pan2024unifying | Verified | Crossref 10.1109/TKDE.2024.3352100 |
| ji2023hallucination | Verified | Crossref 10.1145/3571730 |
| es2024ragas | Verified | Crossref 10.18653/v1/2024.eacl-demo.16 |
| zheng2023judge | Verified | Crossref 10.52202/075280-2020 |
| guo2017calibration | **check** | PMLR vol. 70 (no DOI) |
| platt1999 | **check** year | Book *Advances in Large Margin Classifiers* (MIT Press); conventionally cited as 1999, the book's imprint may read 2000 |
| zadrozny2002 | Verified | Crossref 10.1145/775047.775151 |
| naeini2015ece | Verified | Crossref 10.1609/aaai.v29i1.9602 |
| kadavath2022know | Verified | arXiv 2207.05221 |
| xiong2024uncertainty | **check** | ICLR 2024 (OpenReview, no DOI) |
| cormack2009rrf | Verified | Crossref 10.1145/1571941.1572114 |
| zugner2018nettack | Verified | Crossref 10.1145/3219819.3220078 |
| zhang2019kgepoison | Verified | Crossref 10.24963/ijcai.2019/674 |
| carlini2021extracting | **check** pages 2633–2650 | USENIX Security 2021 (no DOI) |
| efron1993bootstrap | Verified | Crossref 10.1007/978-1-4899-4541-9 |
| cohen1960kappa | Verified | Crossref 10.1177/001316446002000104 |
| sheneman2026mindrouter | Verified | Crossref 10.1145/3785462.3815861 |
| openai2025gptoss | Verified | arXiv 2508.10925 |
| gemma2026gemma4 | Verified | arXiv 2607.02770 |
| kimi2026k3 | Verified | arXiv 2607.24653 |
| nvidia2026nemotron3super | Verified | arXiv 2604.12374 |
| qwen2026qwen35 | Verified (model card exists) | huggingface.co/Qwen/Qwen3.5-122B-A10B |
| qwen2026qwen38 | Verified (model card exists) | huggingface.co/Qwen/Qwen3.8-27B |

**Summary:** 35 entries; 28 verified against a registry, 7 marked *check* (venues without DOIs).
Whether each citation supports the sentence it is attached to was checked by reading the
manuscript context, not the cited papers' full text.
