# CLAUDE.md — Trustworthy Drug Repurposing Recommendation

## Paper
- **Title:** Trustworthy Drug Repurposing Recommendation via Knowledge-Graph-Grounded Large Language Models
- **Venue:** MDPI *Electronics*, special issue "Recommendation Systems: Recent Advances and Future Directions"
- **Deadline:** 2027-01-31

## Pipeline
Full original brief: `PROJECT_BRIEF.md` (source of truth for scope).

1. **Candidate generation** — train a link-prediction model (RotatE / TransE via PyKEEN, or a GNN) on PrimeKG to predict missing drug–disease `indication` edges. Output the top-k candidate drugs for each disease.
2. **Grounded explanation + re-ranking** — for each (disease, candidate drug), retrieve 2–3-hop KG paths from drug to disease. Send them to an LLM through MindRouter. The LLM must:
   - explain the mechanism using **only** the given paths (no outside knowledge),
   - give a confidence score in [0, 1],
   - re-rank the candidates.
3. **Evaluation**
   - Accuracy: Hits@1/3/10 and MRR on held-out indication edges, graph-only vs. graph+LLM re-ranking
   - Faithfulness: % of LLM claims supported by the retrieved paths
   - Calibration: ECE of the LLM confidence scores
   - Robustness: inject fake edges (KG poisoning), then measure the drop in accuracy and faithfulness
   - Model comparison: 5 MindRouter models (see below)
4. **Provenance** — every recommendation logs the data version (PrimeKG release + hash), the KG model and checkpoint, the LLM name, the prompt (text + template version), the retrieved paths, the raw response, and a timestamp.

## MindRouter
- OpenAI-compatible endpoint: `https://mindrouter.uidaho.edu/v1` (use the `openai` Python client with `base_url`).
- Credentials go in `.env` (git-ignored): `MINDROUTER_API_KEY`, `MINDROUTER_BASE_URL`, `MINDROUTER_MODEL`. See `.env.example`.
- Reasoning models (e.g. Qwen3) may return `<think>...</think>` blocks or `reasoning_content`. Strip them before parsing.
- **Models for the paper** (exact IDs, never aliases like `default-llm`, which can be remapped):
  - `openai/gpt-oss-120b`: OpenAI family, MoE (117B total, 5.1B active), reasoning, 128k context
  - `qwen/qwen3.5-122b`: Qwen family, MoE (122B total, ~10B active), reasoning, FP8, 256k context
  - `google/gemma-4-31b`: Google family, dense 31B, 256k context
  - `moonshotai/kimi-k3`: Moonshot family, MoE (2.8T total, ~104B active), reasoning, 1M context (largest)
  - `qwen/qwen3.8-27b`: Qwen family, dense 27B, reasoning, 256k context (the within-family comparison with qwen3.5-122b)
  - Optional non-generative baseline: `Qwen/Qwen3-Reranker-8B` (scores relevance only, no explanation or confidence)
- All of the above are served by vLLM and answered the smoke test in under 1 s (2026-11-13). `llama3.3:70b` (Ollama, Q4) took 9 s.

## Layout
```
data/raw/          PrimeKG downloads (git-ignored)
data/processed/    splits, filtered triples
data/poisoned/     KGs with injected fake edges
src/kg/            loading PrimeKG, splits
src/candidates/    link-prediction training/inference
src/retrieval/     KG path retrieval
src/llm/           MindRouter client, prompts, response parsing
src/eval/          ranking metrics, faithfulness, ECE, robustness
src/provenance/    per-recommendation logging
experiments/       run scripts + configs/
results/           runs/ (JSONL logs), tables/, figures/, logs/
paper/             manuscript, sections/, figures/
scripts/           one-off utilities (e.g. test_mindrouter.py)
```

## Conventions
- Python venv in `.venv/` (Python 3.11). `source .venv/bin/activate`; deps in `requirements.txt`.
- Never commit `.env`, `.venv/`, or `data/raw/`.
- Fix random seeds; record them in provenance logs.
- Use temperature 0 for LLM calls unless an experiment is studying sampling.

## Status
- [x] Scaffold, venv, git
- [x] MindRouter connection test (`scripts/test_mindrouter.py`)
- [x] PrimeKG v2.1 downloaded, md5 verified; stats in `results/tables/primekg_stats.md` (`src/kg/explore.py`)
- [x] Splits (`src/kg/split.py`, seed 42): 7,510 / 939 / 939 indication edges (train/valid/test);
      off-label/contraindication edges for held-out pairs removed from train (50 edges).
      Outputs in `data/processed/primekg_v2.1_seed42/` (large `.tsv.gz` files are git-ignored; rerun the script to rebuild them)
- [x] Link prediction: RotatE (seeds 42/1/2) + TransE (`src/candidates/`); TxGNN baseline in `.venv-txgnn` (`src/baselines/`)
- [x] Path retrieval, grounded/no-paths LLM runs for 5 models, faithfulness judge, recalibration (valid split), prompt v2, poisoning
- [x] All tables: `python -m experiments.make_results`; figures: `python -m experiments.make_figures`
- [x] Manuscript draft `paper/main.tex` (MDPI template); build locally with `cd paper && tectonic -X compile main.tex`
      (no Ghostscript locally: EPS logos replaced by `paper/build_stub/` under XeTeX only; Overleaf/pdflatex uses real logos)
- [x] Disease-held-out split (`python -m src.kg.split --by disease`; set `KG_SPLIT=data/processed/primekg_v2.1_disease_seed42` for all scripts): RotatE, TxGNN, LLM runs -> paper Section 4.5
- [ ] Human labels for 126 faithfulness statements (`results/human_eval/faithfulness_sheet.csv`) -> Cohen's kappa
- [ ] Remaining `\TODO{}` items in `paper/main.tex` (affiliation, model-card citations, GenAI disclosure, repo DOI)
