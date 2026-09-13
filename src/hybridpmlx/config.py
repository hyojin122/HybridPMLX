from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .expressions import SafeExpression


DEFAULTS: dict[str, Any] = {
    "split": {"method": "random", "test_ratio": 0.2, "seed": 42},
    "network": {"hidden_layers": [32, 32], "activation": "softplus", "dropout": 0.0},
    "training": {
        "folds": 5, "seed": 42, "batch_size": 32, "batch_unit": "observation",
        "ids_per_batch": 1, "learning_rate": 1e-3,
        "weight_decay": 1e-4, "max_epochs": 1000, "gradient_clip": 1.0,
        "patience": 50, "min_delta": 1e-4, "minimum_epochs": 0,
        "progress_every": 25,
    },
    "objective": {"loss": "smooth_l1", "beta": 0.1, "eta_penalty": {"type": "none", "weight": 0.0}},
    "shap": {"enabled": True, "method": "deep", "background_size": 100, "targets": "all_eta"},
    "output": {"directory": "results"},
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    cfg = _merge(DEFAULTS, supplied)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    errors: list[str] = []
    data = cfg.get("data", {})
    for field in ("file", "id_column", "idv_column", "dv_column", "feature_columns"):
        if field not in data:
            errors.append(f"data.{field} is required")
    if not data.get("feature_columns"):
        errors.append("data.feature_columns must contain at least one feature")
    if cfg.get("training", {}).get("batch_unit") not in {"observation", "individual"}:
        errors.append("training.batch_unit must be observation or individual")

    model = cfg.get("model", {})
    if not model.get("equation") and not model.get("probabilities"):
        errors.append("model.equation or model.probabilities is required")
    params = model.get("parameters", {})
    if not params:
        errors.append("model.parameters is required")
    for name, spec in params.items():
        if "derived" not in spec and "initial" not in spec:
            errors.append(f"model.parameters.{name}.initial is required")
    known: set[str] = set()
    for name, spec in params.items():
        if "derived" in spec:
            try:
                expr = SafeExpression(spec["derived"])
                unknown = expr.names - known
                if unknown: errors.append(f"derived parameter {name} references unavailable symbols: {sorted(unknown)}")
            except ValueError as exc:
                errors.append(f"invalid derived expression for {name}: {exc}")
        else:
            c = spec.get("constraint", {})
            if c.get("type") == "dynamic_bounded" and c.get("upper_parameter") not in known:
                errors.append(f"dynamic bound for {name} must reference an earlier parameter")
        known.add(name)

    expressions = []
    if model.get("equation"): expressions.append(("model.equation", model["equation"]))
    expressions.extend((f"model.probabilities.{k}", v) for k, v in model.get("probabilities", {}).items())
    for label, source in expressions:
        try:
            expr = SafeExpression(source)
            unknown = expr.names - (set(params) | {"IDV"})
            if unknown: errors.append(f"{label} contains unknown symbols: {sorted(unknown)}")
        except ValueError as exc:
            errors.append(f"invalid {label}: {exc}")

    loss = cfg.get("objective", {}).get("loss")
    categorical = bool(model.get("probabilities"))
    if categorical and loss != "categorical_nll":
        errors.append("probability models require objective.loss: categorical_nll")
    if not categorical and loss not in {"smooth_l1", "mse", "mae"}:
        errors.append("continuous equation models require smooth_l1, mse, or mae")

    activation = cfg.get("network", {}).get("activation")
    if activation != "softplus":
        errors.append("version 0.1 supports network.activation: softplus only")
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def constraint_summary(cfg: dict[str, Any]) -> str:
    lines = ["PARAMETER CONSTRAINT SUMMARY"]
    for name, spec in cfg["model"]["parameters"].items():
        if "derived" in spec:
            lines += [f"\n{name}", f"  Derived: {spec['derived']}"]
            continue
        c = spec.get("constraint", {"type": "unconstrained"})
        kind = c.get("type", "unconstrained")
        lines += [f"\n{name}", f"  Initial value: {spec['initial']}", f"  Personalized: {'Yes' if spec.get('personalized') else 'No'}"]
        descriptions = {
            "unconstrained": "Unconstrained",
            "positive": "Positive",
            "bounded": f"Range: {c.get('lower')} to {c.get('upper')}",
            "dynamic_bounded": f"Dynamic range: {c.get('lower', 0)} to {c.get('upper_constant', 100)} {c.get('upper_multiplier', -1):+g} × {c.get('upper_parameter')}",
            "relative_log": f"Log-relative range: exp(±{c.get('log_scale')}) × population value",
            "absolute_change": f"Absolute change: ±{c.get('value')}",
        }
        lines.append("  Constraint: " + descriptions.get(kind, kind))
    return "\n".join(lines)
