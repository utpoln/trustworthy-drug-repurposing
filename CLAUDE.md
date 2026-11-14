# CLAUDE.md — Trustworthy Drug Repurposing Recommendation

## Paper
- **Title:** Trustworthy Drug Repurposing Recommendation via Knowledge-Graph-Grounded Large Language Models
- **Venue:** MDPI *Electronics*, special issue "Recommendation Systems: Recent Advances and Future Directions"
- **Deadline:** 2027-01-31

## Pipeline
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
   - Model comparison: 2–3 MindRouter models
4. **Provenance** — every recommendation logs the data version (PrimeKG release + hash), the KG model and checkpoint, the LLM name, the prompt (text + template version), the retrieved paths, the raw response, and a timestamp.

## MindRouter
- OpenAI-compatible endpoint: `https://mindrouter.uidaho.edu/v1` (use the `openai` Python client with `base_url`).
- Credentials go in `.env` (git-ignored): `MINDROUTER_API_KEY`, `MINDROUTER_BASE_URL`, `MINDROUTER_MODEL`. See `.env.example`.
- Reasoning models (e.g. Qwen3) may return `<think>...</think>` blocks or `reasoning_content`. Strip them before parsing.
- Models used in sibling projects: `llama3.3:70b`, `qwen2.5:72b`, `mistral-large:123b`, `Qwen/Qwen3-32B`, `google/gemma-4-31b`.

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
- [ ] Everything else: not started (on purpose)
