# BIKE-Inflow ablations

Whole test-set 48-hour forecasts; mean ± sample standard deviation over three adapter seeds. Lower is better.

| Arm | MAE | RMSE | MSE |
|---|---:|---:|---:|
| Full H1 | 4.4859 ± 0.0509 | 12.5628 ± 0.1608 | 157.8417 ± 4.0548 |
| Global-mean older history | 4.8124 ± 0.0461 | 13.3780 ± 0.1246 | 178.9821 ± 3.3272 |
| No auxiliary utility supervision | 4.5014 ± 0.0372 | 12.6832 ± 0.1468 | 160.8784 ± 3.7238 |

![Ablation bars](ablation_bars.png)

Error bars are sample standard deviations, not confidence intervals. All arms share one frozen baseline and have a five-epoch adapter cap.
`global_long` replaces only the older 192 hours of the 240-hour history with the all-station raw mean; each target's own recent 48 hours are retained. This affects neural memory, candidate/teacher histories and the separately refitted numeric route. It is not a 'no history' or 'no retrieval' control.
`no_utility` removes only the auxiliary utility cross-entropy. Trainable retrieval, candidate values, abstention and forecast-error gradients remain. It is not a 'no retrieval' control.
Node-specific older history improves all metrics in all three seeds. Utility supervision gives a small average gain, but seed 2023 reverses the direction on all metrics: its independent benefit is not stable. This study does not establish language-pretraining superiority.
