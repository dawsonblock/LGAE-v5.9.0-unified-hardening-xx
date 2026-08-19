# v6.0-exp4.2 — Executive Summary

**Experiment ID:** v6.0-exp4.2
**Generated at:** 2026-08-19T03:29:36Z

## Scientific Question

> Can LGAE predict which structural mutation will work best on an unseen graph?

## Conclusion

**Status:** `PRELIMINARY_SIGNAL_DETECTED`

- Structural signal detected: **True**
- Generalizes to held-out: **True**
- Best encoder: **global-local**
- Best predictor: **linear**
- Best held-out Spearman: **0.7099**
- Best held-out regret: **0.0052**
- CF→Real transfer OK: **True**
- Uncertainty useful: **False**
- **exp5 authorized: True**

**Recommended exp5 architecture:** `lightweight_latent_dynamics`

## Limitations

- Uncertainty does not correlate with error — trust signal is weak.
- Dataset is synthetic with limited mutation type diversity.
- Multi-step rollout quality is not yet validated for MPC use.
- Risk target is near-constant — risk prediction is not scientifically tested.

## Decision

The experiment provides evidence that structural prediction
generalizes to unseen graph families. exp5 is authorized.
