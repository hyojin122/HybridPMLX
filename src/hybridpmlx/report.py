from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


LINE = "=" * 79
THIN = "-" * 79


def _value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def _mapping(lines: list[str], values: dict, indent: str = "  ") -> None:
    width = max((len(str(k)) for k in values), default=0)
    for key, value in values.items():
        lines.append(f"{indent}{str(key):<{width}} : {_value(value)}")


def _individual_errors(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    if "prediction" not in frame or cfg["objective"]["loss"] == "categorical_nll":
        return pd.DataFrame()
    tmp = frame.assign(
        absolute_error=(frame["prediction"] - frame[d["dv_column"]]).abs(),
        error=frame["prediction"] - frame[d["dv_column"]],
    )
    return (tmp.groupby(d["id_column"], as_index=False)
            .agg(observations=(d["dv_column"], "size"), MAE=("absolute_error", "mean"),
                 bias=("error", "mean"))
            .sort_values("MAE", ascending=False))


def write_summary(results_directory: str | Path) -> Path:
    """Create a NONMEM-like plain-text summary from a completed analysis."""
    out = Path(results_directory).resolve()
    with (out / "configuration_used.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with (out / "performance.json").open(encoding="utf-8") as handle:
        performance = json.load(handle)
    folds = pd.read_csv(out / "fold_performance.csv")
    parameters = pd.read_csv(out / "fold_global_parameters.csv")
    split = pd.read_csv(out / "data_split.csv")
    oof = pd.read_csv(out / "pooled_oof_predictions.csv")
    test = pd.read_csv(out / "test_ensemble_predictions.csv")

    lines = [LINE, "HYBRIDPMLX ANALYSIS REPORT", LINE,
             f"Generated (UTC) : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"Project         : {cfg.get('project', 'unnamed')}",
             f"Results         : {out}", ""]

    d, tr, net, obj = cfg["data"], cfg["training"], cfg["network"], cfg["objective"]
    lines += ["1. DATA AND MODEL SPECIFICATION", THIN]
    _mapping(lines, {
        "Data file": d["file"], "ID / IDV / DV": f"{d['id_column']} / {d['idv_column']} / {d['dv_column']}",
        "Features": len(d["feature_columns"]), "Development individuals": int((split["split"] == "development").sum()),
        "Held-out individuals": int((split["split"] == "test").sum()),
        "Development observations": len(oof), "Held-out observations": len(test),
        "Structural equation": cfg["model"].get("equation", "categorical probability model"),
        "Loss": obj["loss"], "Hidden layers": net["hidden_layers"], "Activation": net["activation"],
        "Dropout": net["dropout"], "Folds": tr["folds"], "Batch unit": tr.get("batch_unit", "observation"),
        "ETA penalty": obj.get("eta_penalty", {}), "Monotonicity": obj.get("monotonicity", {}),
    })
    lines += ["", "Personalized parameters:"]
    for name, spec in cfg["model"]["parameters"].items():
        if spec.get("personalized", False):
            lines.append(f"  {name}: initial={spec.get('initial')}, constraint={spec.get('constraint', {})}")

    lines += ["", "2. CROSS-VALIDATION", THIN,
              f"  {'Fold':>4} {'Best epoch':>12} {'Epochs run':>12} {'Validation metric':>20}"]
    for row in folds.itertuples(index=False):
        history = pd.read_csv(out / f"fold_{int(row.fold)}_history.csv")
        lines.append(f"  {int(row.fold):>4} {int(row.best_epoch):>12} {len(history):>12} {row.validation_metric:>20.6f}")

    lines += ["", "3. PREDICTIVE PERFORMANCE", THIN, "Pooled out-of-fold:"]
    _mapping(lines, performance["pooled_oof"], "  ")
    lines.append("Held-out ensemble:")
    _mapping(lines, performance["test"], "  ")

    worst = _individual_errors(test, cfg)
    if not worst.empty:
        lines += ["", "Largest held-out individual errors:",
                  f"  {d['id_column']:>14} {'N':>6} {'MAE':>12} {'Bias':>12}"]
        for row in worst.head(10).itertuples(index=False):
            lines.append(f"  {str(row[0]):>14} {int(row.observations):>6} {row.MAE:>12.4f} {row.bias:>12.4f}")

    lines += ["", "4. FOLD-SPECIFIC PARAMETER SUMMARY", THIN,
              f"  {'Parameter':<22} {'Mean':>12} {'SD':>12} {'Minimum':>12} {'Maximum':>12}"]
    summary = parameters.groupby("parameter")["value_at_zero_features"].agg(["mean", "std", "min", "max"])
    for name, row in summary.iterrows():
        lines.append(f"  {name:<22} {row['mean']:>12.6g} {row['std']:>12.6g} {row['min']:>12.6g} {row['max']:>12.6g}")

    lines += ["", "5. PARAMETER-SPECIFIC SHAP IMPORTANCE", THIN]

    def source_importance(path: Path, raw_path: Path) -> Path:
        if path.exists() or not raw_path.exists():
            return path
        raw = pd.read_csv(raw_path)
        summary = (raw.assign(abs_shap=lambda z: z.shap_value.abs())
                   .groupby(["target", "feature"], as_index=False).abs_shap.mean()
                   .rename(columns={"abs_shap": "mean_absolute_shap"}))
        summary["normalized_importance"] = summary.groupby("target")["mean_absolute_shap"].transform(
            lambda x: x / x.max() if x.max() > 0 else x)
        summary["rank"] = summary.groupby("target")["mean_absolute_shap"].rank(
            method="min", ascending=False).astype(int)
        summary.to_csv(path, index=False)
        return path

    def append_importance(title: str, shap_path: Path) -> None:
        lines.append(title)
        if not shap_path.exists():
            lines.append("  No source-specific SHAP importance file was found.\n")
            return
        importance = pd.read_csv(shap_path).sort_values(["target", "rank"])
        for target, group in importance.groupby("target", sort=False):
            lines.extend([f"Target: {target}", f"  {'Rank':>4} {'Feature':<24} {'Mean |SHAP|':>14} {'Scaled':>10}"])
            for row in group.itertuples(index=False):
                lines.append(f"  {int(row.rank):>4} {row.feature:<24} {row.mean_absolute_shap:>14.6g} {row.normalized_importance:>10.4f}")
            lines.append("")

    development_path = source_importance(
        out / "shap_importance_development_oof.csv", out / "development_oof_shap.csv")
    external_path = source_importance(
        out / "shap_importance_external_ensemble.csv", out / "test_ensemble_shap.csv")
    append_importance("5.1 Development out-of-fold SHAP", development_path)
    append_importance("5.2 Held-out external five-model ensemble SHAP", external_path)

    lines += ["6. OUTPUT FILES", THIN]
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "hybridpmlx_summary.lst":
            lines.append(f"  {path.name}")
    lines += ["", LINE, "END OF HYBRIDPMLX REPORT", LINE, ""]
    destination = out / "hybridpmlx_summary.lst"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
