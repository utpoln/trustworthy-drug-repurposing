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
