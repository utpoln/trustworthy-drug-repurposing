# Test-set calibration of grounded LLM confidence after maps fitted on the validation split (100 validation diseases, top-20 candidates)

| Model | Map | Params | MeanConf | ECE | Brier | AUROC |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | none | – | 0.303 | 0.268 | 0.145 | 0.860 |
| Qwen3.8-27B | temperature | T=0.86 | 0.290 | 0.255 | 0.147 | 0.860 |
| Qwen3.8-27B | platt | a=0.99, b=-3.26 | 0.034 | 0.005 | 0.029 | 0.860 |
| Qwen3.8-27B | isotonic | n_steps=11 | 0.035 | 0.002 | 0.029 | 0.860 |
| Gemma-4-31B | none | – | 0.328 | 0.293 | 0.152 | 0.868 |
| Gemma-4-31B | temperature | T=0.78 | 0.310 | 0.275 | 0.152 | 0.868 |
| Gemma-4-31B | platt | a=1.34, b=-3.44 | 0.033 | 0.006 | 0.030 | 0.868 |
| Gemma-4-31B | isotonic | n_steps=11 | 0.033 | 0.004 | 0.029 | 0.865 |
| Qwen3.5-122B | none | – | 0.329 | 0.294 | 0.146 | 0.857 |
| Qwen3.5-122B | temperature | T=0.63 | 0.282 | 0.247 | 0.138 | 0.857 |
| Qwen3.5-122B | platt | a=1.26, b=-3.50 | 0.037 | 0.013 | 0.031 | 0.857 |
| Qwen3.5-122B | isotonic | n_steps=10 | 0.036 | 0.012 | 0.031 | 0.856 |
| gpt-oss-120B | none | – | 0.291 | 0.257 | 0.110 | 0.802 |
| gpt-oss-120B | temperature | T=0.39 | 0.168 | 0.133 | 0.091 | 0.802 |
| gpt-oss-120B | platt | a=1.33, b=-2.73 | 0.030 | 0.005 | 0.031 | 0.802 |
| gpt-oss-120B | isotonic | n_steps=17 | 0.031 | 0.008 | 0.033 | 0.802 |
| Kimi-K3 | none | – | 0.271 | 0.236 | 0.113 | 0.902 |
| Kimi-K3 | temperature | T=0.59 | 0.223 | 0.188 | 0.110 | 0.902 |
| Kimi-K3 | platt | a=1.48, b=-3.08 | 0.033 | 0.006 | 0.029 | 0.902 |
| Kimi-K3 | isotonic | n_steps=15 | 0.033 | 0.007 | 0.029 | 0.897 |
