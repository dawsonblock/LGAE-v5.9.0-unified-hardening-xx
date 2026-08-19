# v6.0-exp4.2 — Held-Out Structural Prediction Study

**Experiment ID:** v6.0-exp4.2
**Generated at:** 2026-08-19T03:29:36Z

## 1. Question

Can LGAE predict which structural changes will actually improve
an unseen graph?

## 2. Hypotheses

**H1:** There exists f(S, a) that predicts relative intervention
quality on graph families not used during training/model selection,
materially outperforming simple baselines.

**H0:** Available state/action information provides no useful
generalizable predictive signal beyond simple statistical baselines.

## 3. Experimental Protocol

1. Freeze exp2 dataset hashes.
2. Freeze exp3 encoder configs.
3. Freeze exp4 predictor configs.
4. Train only on TRAIN.
5. Select models/hyperparameters only on VALIDATION.
6. Lock winning configurations.
7. Open HELD_OUT once.
8. Generate final scientific report.

## 4. Dataset

The exp2 structural transition dataset with frozen splits:
train, validation, held_out. The held-out partition was not
accessed during any selection decision.

## 5. Models and Encoders

Total combinations evaluated: 12

| Encoder | Predictor | Target | Val Score | Held Spearman |
|---------|-----------|--------|-----------|---------------|
| global-local | linear | realized_delta | 0.1325 | 0.7099 |
| global-local | ridge | realized_delta | 0.1325 | 0.7099 |
| geometric | linear | realized_delta | 0.1325 | 0.7099 |
| minimal-control | tree | sign_delta | 0.1142 | 0.5062 |
| global | tree | sign_delta | 0.1142 | 0.5062 |
| learned-graph | tree | sign_delta | 0.1142 | 0.5062 |
| minimal-control | global_mean | realized_delta | 0.0241 | -0.3044 |
| minimal-control | mutation_type_mean | realized_delta | 0.0241 | -0.3044 |
| minimal-control | nearest_experience | realized_delta | 0.1113 | 0.6013 |
| minimal-control | global_mean | sign_delta | 0.0948 | -0.0226 |
| minimal-control | mutation_type_mean | sign_delta | 0.0000 | -0.0226 |
| minimal-control | nearest_experience | sign_delta | 0.0000 | 0.5627 |

## 6. Held-Out Results

- Best held-out Spearman: 0.7099
- Best held-out regret: 0.0052
- Best encoder: global-local
- Best predictor: linear

## 7. Counterfactual Transfer

- CF→Real transfer OK: True

## 8. Calibration

- Uncertainty useful: False

## 9. Conclusion

**Status:** PRELIMINARY_SIGNAL_DETECTED

**exp5 authorized:** True

## 10. Limitations

- Uncertainty does not correlate with error — trust signal is weak.
- Dataset is synthetic with limited mutation type diversity.
- Multi-step rollout quality is not yet validated for MPC use.
- Risk target is near-constant — risk prediction is not scientifically tested.

## 11. Decision on exp5

Proceed to exp5 with architecture: lightweight_latent_dynamics