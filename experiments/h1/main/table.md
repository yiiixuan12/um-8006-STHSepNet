# H1: full test-set results

Mean ± sample standard deviation across three adapter seeds. Lower is better.

| Dataset | Method | MAE | RMSE | MSE |
|---|---|---:|---:|---:|
| PEMS03 | Fixed corrected baseline (seed 2021) | 21.6194 | 35.8744 | 1286.9726 |
| PEMS03 | H1 (three adapter seeds) | 18.2208 ± 0.1031 | 32.0173 ± 0.1739 | 1025.1255 ± 11.1542 |
| METR-LA | Fixed corrected baseline (seed 2021) | 10.3596 | 17.8723 | 319.4191 |
| METR-LA | H1 (three adapter seeds) | 8.9335 ± 0.1238 | 17.9998 ± 0.1236 | 324.0029 ± 4.4506 |
| BJ500 | Fixed corrected baseline (seed 2021) | 6.0906 | 10.4954 | 110.1538 |
| BJ500 | H1 (three adapter seeds) | 5.2374 ± 0.0205 | 9.6827 ± 0.0435 | 93.7553 ± 0.8437 |
| BIKE-Inflow | Fixed corrected baseline (seed 2021) | 5.4349 | 15.1656 | 229.9940 |
| BIKE-Inflow | H1 (three adapter seeds) | 4.4859 ± 0.0509 | 12.5628 ± 0.1608 | 157.8417 ± 4.0548 |
| BIKE-Outflow | Fixed corrected baseline (seed 2021) | 5.4764 | 13.9943 | 195.8417 |
| BIKE-Outflow | H1 (three adapter seeds) | 4.4453 ± 0.0579 | 11.5271 ± 0.1172 | 132.8834 ± 2.6966 |

Each dataset uses one shared frozen baseline. The three seeds apply to adapter initialization and training order, not independent end-to-end repetitions.
H1 reads additional history (2064 steps for five-minute data; 240 hourly steps for BIKE); baseline input is 48 steps. Therefore gains are not a matched-information test of the representation alone.
All adapter runs reached the five-epoch cap; convergence was not confirmed. Validation raw MSE selects the best checkpoint. These are local corrected-baseline results, not the paper's official numbers.
METR-LA improves mean MAE but worsens mean RMSE/MSE. Negative outcomes are retained.
