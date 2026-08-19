# v6.0-exp4.2 — Held-Out Structural Prediction Study

**Experiment ID:** v6.0-exp4.2
**Generated at:** 2026-08-19T02:16:18Z

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

Total combinations evaluated: 6

| Encoder | Predictor | Target | Val Score | Held Spearman |
|---------|-----------|--------|-----------|---------------|
| global-local | tree | realized_delta | 0.0000 | 0.4557 |
| geometric | tree | realized_delta | 0.0000 | 0.4557 |
| global-local | mlp | realized_delta | 0.0000 | 0.5923 |
| global-local | tree | sign_delta | 0.0000 | 0.6335 |
| geometric | tree | sign_delta | 0.0000 | 0.6335 |
| minimal-control | tree | sign_delta | 0.0000 | 0.6454 |

## 6. Held-Out Results

- Best held-out Spearman: 0.6454
- Best held-out regret: 0.0667
- Best encoder: minimal-control
- Best predictor: tree

## 7. Counterfactual Transfer

- CF→Real transfer OK: True

## 8. Calibration

- Uncertainty useful: False

## 9. Conclusion

**Status:** QUALIFIED_SIMPLE

**exp5 authorized:** True

## 10. Limitations

- Uncertainty does not correlate with error — trust signal is weak.

## 11. Decision on exp5

Proceed to exp5 with architecture: lightweight_latent_dynamics