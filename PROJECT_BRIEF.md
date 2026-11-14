# Project brief (original, verbatim)

I'm writing a paper for the MDPI Electronics special issue "Recommendation
Systems: Recent Advances and Future Directions" (deadline Jan 31, 2027).

Title: "Trustworthy Drug Repurposing Recommendation via Knowledge-Graph-
Grounded Large Language Models"

Idea:
1. Candidate generation: train a link-prediction model (e.g. RotatE/TransE
   via PyKEEN, or a GNN) on PrimeKG to predict missing drug–disease
   "indication" edges. Output top-k candidate drugs per disease.
2. Grounded explanation + re-ranking: for each candidate, retrieve KG paths
   (2–3 hops) linking drug to disease. Send paths to an LLM via MindRouter
   (credentials in .env, OpenAI-compatible if possible). LLM must explain
   the mechanism using ONLY the given paths, give a confidence score 0–1,
   and re-rank candidates.
3. Evaluation:
   - Accuracy: Hits@1/3/10, MRR on held-out indication edges
     (graph-only vs graph+LLM re-ranking)
   - Faithfulness: % of LLM claims supported by the retrieved paths
   - Calibration: ECE of LLM confidence scores
   - Robustness: inject fake edges into the KG (poisoning) and measure the
     drop in accuracy and faithfulness
   - Compare 2–3 MindRouter models
4. Provenance: log data version, model, prompt, and paths for every
   recommendation.
