# Prompt v1 vs v2 on the 100 sampled test diseases (RotatE top-20; dMRR v2-v1, disease bootstrap)

| Model | Prompt | MRR | H@1 | H@10 | ECE | AUROC | dMRR |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | v1 | 0.260 | 0.198 | 0.353 | 0.276 | 0.850 | – |
| Qwen3.8-27B | v2 | 0.238 | 0.171 | 0.332 | 0.248 | 0.843 | -0.023 [-0.044, +0.000] |
| Gemma-4-31B | v1 | 0.272 | 0.225 | 0.326 | 0.288 | 0.862 | – |
| Gemma-4-31B | v2 | 0.255 | 0.198 | 0.326 | 0.204 | 0.842 | -0.017 [-0.041, +0.009] |
