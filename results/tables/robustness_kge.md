# Training-time poisoning: RotatE retrained on the poisoned graph

| Graph | MRR | H@1 | H@10 | Decoy@1 | Decoy med |
|---|---|---|---|---|---|
| clean | 0.179 | 0.114 | 0.296 | – | – |
| targeted (n=3) | 0.183 | 0.119 | 0.299 | 0.050 -> 0.103 | 10 -> 11 |
| random 5% | 0.192 | 0.121 | 0.331 | – | – |
