# v6.0-exp4.2 — Executive Summary

**Experiment ID:** v6.0-exp4.2
**Generated at:** 2026-08-19T02:16:18Z

## Scientific Question

> Can LGAE predict which structural mutation will work best on an unseen graph?

## Conclusion

**Status:** `QUALIFIED_SIMPLE`

- Structural signal detected: **True**
- Generalizes to held-out: **True**
- Best encoder: **minimal-control**
- Best predictor: **tree**
- Best held-out Spearman: **0.6454**
- Best held-out regret: **0.0667**
- CF→Real transfer OK: **True**
- Uncertainty useful: **False**
- **exp5 authorized: True**

**Recommended exp5 architecture:** `lightweight_latent_dynamics`

## Limitations

- Uncertainty does not correlate with error — trust signal is weak.

## Decision

The experiment provides evidence that structural prediction
generalizes to unseen graph families. exp5 is authorized.
