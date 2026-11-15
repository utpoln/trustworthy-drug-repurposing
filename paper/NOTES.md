# Paper notes (points that must appear in the manuscript)

## Limitations (to state openly)
- **Memorisation / contamination.** Held-out "indications" are mostly approved uses that the
  LLMs may have seen during pre-training, and the LLM sees real drug and disease names. The
  no-paths baseline measures this effect (+0.02 to +0.04 MRR over RotatE with no KG input),
  and we report the grounding gain *relative to no-paths* (+0.03 to +0.05). Future work:
  time-split evaluation on indications approved after the models' training cut-offs, or
  anonymised entity names.
- Faithfulness is judged by an LLM (Nemotron-3-Super-120b) and validated on 126
  human-labelled statements (stratified by model x statement type x judge label).
- Re-ranking only reorders the KGE top-20: the ceiling is the 37.7% of test pairs whose gold drug is
  in the top-20.
- Single KG (PrimeKG v2.1), single split seed (42), single KGE seed.
- The v2 prompt judges mechanism sentences only against their cited paths (stricter than v1,
  where uncited sentences were judged against all paths).

## RotatE seed variance (seeds 42, 1, 2; same split and hyper-parameters)
- Test MRR 0.176 ± 0.009, Hits@10 0.305 ± 0.021. The LLM gains (+0.07–0.09 MRR over RotatE) are ~8–10× the seed std.
- Seeds 1 and 2 reached their best validation MRR at epoch 50, the fixed budget, so they may be slightly under-trained; same budget for all seeds, state it.
- LLM re-ranking was run on the seed-42 candidates only.

## Recalibration / prompt v2 (see results/tables/recalibration.md, prompt_v2_*.md)
- Platt fitted on 100 validation diseases: test ECE 0.24–0.29 -> 0.005–0.013; temperature alone cannot fix the base-rate offset.
- After Platt nearly all mass is in the [0, 0.1] bin; also report equal-mass-bin ECE.
- Prompt v2: mechanism support 24–28% -> 88–91% (judge), MRR -0.017 to -0.023; Gemma v2 parse rate 97.1%.
