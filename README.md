# HybridPMLX

HybridPMLX is a configuration-driven Python package for hybrid multilayer-perceptron–pharmacometric models. A user supplies a dataset, structural equation, global initial values, personalized parameters, and clinically meaningful constraints. The engine performs ID-level splitting, fold-specific normalization, cross-validated joint optimization, pooled out-of-fold prediction, held-out ensemble evaluation, and parameter-specific SHAP analysis. Minibatches may be formed by observation or by individual; the latter preserves longitudinal records and subject-level ETA anchoring.

## Workflow

`Define -> Split -> Fit -> Evaluate -> Explain`

Predictive diagnostics do not block SHAP. They are reported beside SHAP results so users can judge the generalizability of model attribution.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

On Linux or macOS, activate with `source .venv/bin/activate`.

## Commands

```bash
hybridpmlx validate --config examples/gh_gompertz.yaml
hybridpmlx run --config examples/gh_gompertz.yaml
```

Every completed run automatically creates `hybridpmlx_summary.lst`. A report can
also be regenerated from an existing result directory without refitting:

```bash
hybridpmlx report --results results/gh
```

During fitting, the command line reports setup, split sizes, fold and epoch
progress, validation scores, ensemble evaluation, SHAP targets, and completion.
Set `training.progress_every` to control the epoch reporting interval.

```bash
hybridpmlx validate --config examples/dementia_weibull.yaml
hybridpmlx run --config examples/dementia_weibull.yaml
```

Update `data.file` and the column names before running an example.

## User-defined equations

Continuous models define one differentiable expression:

```yaml
model:
  equation: E0 + Emax * IDV / (EC50 + IDV)
```

Categorical models define one expression per category:

```yaml
model:
  probabilities:
    normal: exp(-(IDV / lambda_mci) ** shape)
    mci: exp(-(IDV / lambda_dem) ** shape) - exp(-(IDV / lambda_mci) ** shape)
    dementia: 1 - exp(-(IDV / lambda_dem) ** shape)
```

Allowed functions are `exp`, `log`, `sqrt`, `sigmoid`, `tanh`, `abs`, `pow`, and `clamp`. Expressions are parsed through a restricted syntax tree and are never passed to Python `eval`.

## User-friendly constraints

Users state what must be constrained; HybridPMLX selects the differentiable transformation.

| Constraint | Meaning |
|---|---|
| `unconstrained` | No range restriction |
| `positive` | Parameter remains positive |
| `bounded` | Fixed lower and upper bounds |
| `dynamic_bounded` | Upper bound depends on an earlier parameter |
| `relative_log` | Bounded log-relative individual change |
| `absolute_change` | Bounded absolute individual change |
| `derived` | Parameter computed from an expression |

Example:

```yaml
parameters:
  lambda_mci:
    initial: 0.826
    personalized: true
    constraint: {type: positive}
  difference:
    initial: 0.064
    personalized: true
    constraint: {type: positive}
  lambda_dem:
    derived: lambda_mci + difference
```

## Outputs

- `configuration_used.yaml`
- `data_split.csv`
- `fold_performance.csv`
- fold checkpoints and training histories
- `pooled_oof_predictions.csv`
- `test_ensemble_predictions.csv`
- `performance.json`
- development OOF and test ensemble SHAP values
- pooled parameter-specific SHAP importance
- separate development-OOF, held-out ensemble, and combined-cohort SHAP importance files
- `hybridpmlx_summary.lst`, combining specifications, fold validation, OOF/test
  performance, parameter summaries, largest individual errors, and SHAP rankings

## Current scope

Version 0.1 supports one ID column, one independent variable, one dependent variable, baseline features that are constant within individual, continuous regression with Smooth L1/MSE/MAE, categorical probability models with NLL, Softplus MLPs, ID-level five-fold cross-validation, and Deep SHAP. The GH Gompertz and dementia ordinal-Weibull configurations are included as examples.
