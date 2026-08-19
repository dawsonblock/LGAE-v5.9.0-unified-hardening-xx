# EXPERIMENT FREEZE — v6.0-exp4.1

**Immutable experiment identifier:** `LGAE_V6_EXP4_2_STRUCTURAL_PREDICTION_STUDY_001`

**Frozen at:** 2026-08-19T01:30:00Z

## Purpose

This document freezes the v6.0-exp4.1 baseline so that v6.0-exp4.2
(held-out structural prediction study) runs against a fixed, auditable
substrate. No silent model/encoder/dataset changes are permitted after
this freeze.

## Frozen identities

| Item | Value |
|------|-------|
| LGAE authority version | 5.11.0 |
| Schema | LGAE_CANONICAL_CONVERGENCE_V5_11_0 |
| Source tree (git HEAD) | dbcdd66f14edfd7903300b21c1e182e8f482eb30 |
| Manifest | 735 files |
| Manifest SHA-256 | d16146bc343b7e5876e18a7c705129005c9542a44e60779bf77bf525bfe517ad |
| Python version | 3.12.0 |
| Qualification | 2008 collected, 2008 passed, 0 failed, 0 errors |
| Qualification status | QUALIFIED |
| Qualification elapsed | 419.66s |

## Experimental lineage frozen

| Phase | Status |
|-------|--------|
| exp1 — foundation & world-model interfaces | frozen |
| exp2 — structural transition dataset | frozen |
| exp3 — structural encoders (9 encoders) | frozen |
| exp4 — predictive models (baselines, linear, tree, MLP, ranking) | frozen |
| exp4.1 — competition harness + model persistence | frozen |

## exp2 dataset schema

- Schema version: `LGAE_STRUCTURAL_DATASET_V6_0_EXP2`
- Generator version: `6.0-exp2`
- Splits: train, validation, held_out
- Provenance types: REALIZED, COUNTERFACTUAL, SHADOW
- Frozen graph-family registry with fixed split assignment

## exp3 encoder versions

| Encoder ID | Class |
|------------|-------|
| minimal-control | MinimalControlEncoder |
| global | GlobalStateEncoder |
| global-local | LocalActionEncoder |
| semantic-action | SemanticActionEncoder |
| local-subgraph | LocalSubgraphEncoder |
| geometric | GeometricEncoder |
| spectral | SpectralEncoder |
| learned-graph | SmallLearnedGraphEncoder |
| hybrid | HybridEncoder |

## exp4 predictor versions

| Predictor ID | Class |
|--------------|-------|
| global_mean | GlobalMeanPredictor |
| mutation_type_mean | MutationTypeMeanPredictor |
| nearest_experience | NearestExperiencePredictor |
| linear | LinearRegressionPredictor |
| ridge | RidgeRegressionPredictor |
| logistic | LogisticRegressionPredictor |
| tree | GradientBoostedTreePredictor |
| mlp | MLPRegressor |
| mlp_clf | MLPClassifier |
| pointwise_rank | PointwiseRankingModel |
| pairwise_rank | PairwiseRankingModel |

## Prohibition

After this freeze, no changes to:
- encoder implementations or configurations
- predictor implementations or hyperparameter defaults
- dataset generation logic
- graph-family registry

are permitted within the exp4.2 study. Any required fix must be
documented as an explicit architectural exception.

## Next step

v6.0-exp4.2-heldout-structural-prediction-study may now begin.
Its purpose is to generate scientific evidence, not infrastructure.
