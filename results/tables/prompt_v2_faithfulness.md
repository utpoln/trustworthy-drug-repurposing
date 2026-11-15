# Faithfulness, prompt v1 vs v2 (v2: every mechanism sentence must cite paths; judge Nemotron-3-Super-120b)

| Model | Pairs | Stmts | Claim S | Claim U | Mech S | Mech P | Mech U | All S | Pair all-S | Cite valid |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B v1 | 2000 | 15390 | 0.975 | 0.002 | 0.278 | 0.189 | 0.532 | 0.697 | 0.155 | 1.000 |
| Qwen3.8-27B v2 | 2000 | 13899 | 0.976 | 0.003 | 0.879 | 0.088 | 0.033 | 0.957 | 0.777 | 1.000 |
| Gemma-4-31B v1 | 1999 | 15432 | 0.973 | 0.003 | 0.235 | 0.183 | 0.578 | 0.755 | 0.178 | 1.000 |
| Gemma-4-31B v2 | 1942 | 7710 | 0.971 | 0.005 | 0.912 | 0.038 | 0.049 | 0.952 | 0.843 | 1.000 |
