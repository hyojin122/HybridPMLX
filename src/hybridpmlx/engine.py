from __future__ import annotations

import copy
import json
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from .data import fit_id_scaler, load_and_split
from .model import build_model_class
from .report import write_summary


class IndividualBatchSampler:
    """Yield all observations from one or more randomly ordered individuals."""

    def __init__(self, ids, ids_per_batch: int, seed: int):
        self.groups = {}
        for index, subject in enumerate(ids):
            self.groups.setdefault(subject, []).append(index)
        self.subjects = list(self.groups)
        self.ids_per_batch = max(1, int(ids_per_batch))
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        subjects = self.subjects.copy()
        rng.shuffle(subjects)
        self.epoch += 1
        for start in range(0, len(subjects), self.ids_per_batch):
            batch = []
            for subject in subjects[start:start + self.ids_per_batch]:
                batch.extend(self.groups[subject])
            yield batch

    def __len__(self):
        return math.ceil(len(self.subjects) / self.ids_per_batch)


def _imports():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install the package with: pip install -e .") from exc
    return torch, F, DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch, _, _, _ = _imports()
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _folds(df, ids, cfg):
    data, train = cfg["data"], cfg["training"]
    unique = df[df[data["id_column"]].isin(ids)].drop_duplicates(data["id_column"])
    uid = unique[data["id_column"]].to_numpy()
    n = int(train["folds"])
    strat = train.get("stratify_by")
    if strat:
        splitter = StratifiedKFold(n, shuffle=True, random_state=int(train["seed"]))
        iterator = splitter.split(uid, unique[strat].to_numpy())
    else:
        splitter = KFold(n, shuffle=True, random_state=int(train["seed"]))
        iterator = splitter.split(uid)
    for fold, (a, b) in enumerate(iterator, 1):
        yield fold, uid[a], uid[b]


def _arrays(df, ids, scaler, cfg):
    d = cfg["data"]
    part = df[df[d["id_column"]].isin(ids)].copy()
    x = scaler.transform(part[d["feature_columns"]]).astype("float32")
    t = part[d["idv_column"]].to_numpy(dtype="float32").reshape(-1, 1)
    categorical = bool(cfg["model"].get("probabilities"))
    dtype = "int64" if categorical else "float32"
    y = part[d["dv_column"]].to_numpy(dtype=dtype)
    if not categorical: y = y.reshape(-1, 1)
    return part, x, t, y


def _loss(model, xb, tb, yb, cfg):
    torch, F, _, _ = _imports()
    pred, eta, _ = model(xb, tb, return_eta=True)
    name = cfg["objective"]["loss"]
    if name == "smooth_l1": base = F.smooth_l1_loss(pred, yb, beta=float(cfg["objective"].get("beta", .1)))
    elif name == "mse": base = F.mse_loss(pred, yb)
    elif name == "mae": base = F.l1_loss(pred, yb)
    elif name == "categorical_nll": base = -torch.log(pred[torch.arange(len(yb), device=yb.device), yb]).mean()
    else: raise ValueError(f"Unknown loss: {name}")

    penalty = cfg["objective"].get("eta_penalty", {})
    weight, ptype = float(penalty.get("weight", 0)), penalty.get("type", "none")
    if weight and eta:
        if ptype == "mean_anchor": reg = sum(value.mean().square() for value in eta.values())
        elif ptype == "l2": reg = sum(value.square().mean() for value in eta.values())
        else: raise ValueError(f"Unknown ETA penalty: {ptype}")
        base = base + weight * reg

    mono = cfg["objective"].get("monotonicity", {})
    if mono.get("enabled", False):
        step = float(mono.get("idv_step", .1))
        future = model(xb, tb + step)
        reg = F.relu(pred - future).mean() if mono.get("direction", "increasing") == "increasing" else F.relu(future - pred).mean()
        base = base + float(mono.get("weight", 0)) * reg
    return base


def _metric(y, pred, cfg):
    if cfg["objective"]["loss"] == "categorical_nll":
        return float(log_loss(y, pred, labels=list(range(pred.shape[1]))))
    return float(mean_absolute_error(y, pred))


def _train_fold(train_arrays, val_arrays, cfg, device, fold):
    torch, _, DataLoader, TensorDataset = _imports()
    Model = build_model_class()
    train_frame, xtr, ttr, ytr = train_arrays; _, xva, tva, yva = val_arrays
    model = Model(cfg, xtr.shape[1]).to(device)
    categorical = cfg["objective"]["loss"] == "categorical_nll"
    tensor_y = torch.long if categorical else torch.float32
    ds = TensorDataset(torch.tensor(xtr), torch.tensor(ttr), torch.tensor(ytr, dtype=tensor_y))
    generator = torch.Generator().manual_seed(int(cfg["training"]["seed"]))
    if cfg["training"].get("batch_unit", "observation") == "individual":
        sampler = IndividualBatchSampler(
            train_frame[cfg["data"]["id_column"]].to_numpy(),
            cfg["training"].get("ids_per_batch", 1),
            cfg["training"]["seed"],
        )
        loader = DataLoader(ds, batch_sampler=sampler)
    else:
        loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, generator=generator)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best, state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    progress_every = max(1, int(cfg["training"].get("progress_every", 25)))
    xv, tv = torch.tensor(xva, device=device), torch.tensor(tva, device=device)
    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        model.train(); total = 0.0; seen = 0
        for xb, tb, yb in loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True); loss = _loss(model, xb, tb, yb, cfg); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip"]))
            opt.step(); total += float(loss.item()) * len(xb); seen += len(xb)
        model.eval()
        with torch.no_grad(): vp = model(xv, tv).cpu().numpy()
        score = _metric(yva, vp, cfg); history.append({"epoch": epoch, "train_loss": total/max(seen,1), "validation_metric": score})
        if best - score > float(cfg["training"]["min_delta"]):
            best, state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else: stale += 1
        if epoch == 1 or epoch % progress_every == 0:
            print(f"[Fold {fold}/{cfg['training']['folds']}] epoch {epoch}/{cfg['training']['max_epochs']} | train loss={total/max(seen,1):.6f} | validation={score:.6f} | best={best:.6f}", flush=True)
        if epoch >= int(cfg["training"].get("minimum_epochs", 0)) and stale >= int(cfg["training"]["patience"]): break
    model.load_state_dict(state)
    return model, best, best_epoch, pd.DataFrame(history)


def _predict(model, arrays, device):
    torch, _, _, _ = _imports(); frame, x, t, y = arrays
    model.eval()
    with torch.no_grad(): pred = model(torch.tensor(x, device=device), torch.tensor(t, device=device)).cpu().numpy()
    return frame, y, pred


def _metrics(y, pred, cfg):
    if cfg["objective"]["loss"] == "categorical_nll":
        cls = pred.argmax(1)
        return {"N": len(y), "accuracy": accuracy_score(y, cls), "macro_f1": f1_score(y, cls, average="macro", zero_division=0), "negative_log_likelihood": log_loss(y, pred, labels=list(range(pred.shape[1])))}
    y, pred = np.asarray(y).ravel(), np.asarray(pred).ravel()
    return {"N": len(y), "MAE": mean_absolute_error(y, pred), "RMSE": mean_squared_error(y, pred) ** .5, "bias": float(np.mean(pred-y)), "R2": float(np.corrcoef(y,pred)[0,1]**2)}


def run(cfg: dict) -> Path:
    torch, _, _, _ = _imports(); set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = load_and_split(cfg); df = prepared.frame; d = cfg["data"]
    print(f"[Setup] device={device} | observations={len(df)} | individuals={df[d['id_column']].nunique()}", flush=True)
    print(f"[Split] development={len(prepared.development_ids)} IDs | held-out test={len(prepared.test_ids)} IDs", flush=True)
    out = Path(cfg["output"]["directory"])
    if not out.is_absolute(): out = Path(cfg["_config_dir"]) / out
    out.mkdir(parents=True, exist_ok=True)
    with (out/"configuration_used.yaml").open("w", encoding="utf-8") as f: yaml.safe_dump({k:v for k,v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)
    split_df = pd.DataFrame({d["id_column"]: np.r_[prepared.development_ids, prepared.test_ids], "split": ["development"]*len(prepared.development_ids)+["test"]*len(prepared.test_ids)})
    split_df.to_csv(out/"data_split.csv", index=False)

    fold_models, oof_frames, fold_metrics, parameter_rows, scaler_rows = [], [], [], [], []
    for fold, train_ids, val_ids in _folds(df, prepared.development_ids, cfg):
        print(f"\n[Fold {fold}/{cfg['training']['folds']}] starting | train={len(train_ids)} IDs | validation={len(val_ids)} IDs", flush=True)
        scaler = fit_id_scaler(df, train_ids, d["id_column"], d["feature_columns"])
        tr, va = _arrays(df, train_ids, scaler, cfg), _arrays(df, val_ids, scaler, cfg)
        model, best, epoch, history = _train_fold(tr, va, cfg, device, fold)
        print(f"[Fold {fold}/{cfg['training']['folds']}] complete | best epoch={epoch} | validation={best:.6f}", flush=True)
        torch.save(model.state_dict(), out/f"fold_{fold}.pt"); history.assign(fold=fold).to_csv(out/f"fold_{fold}_history.csv", index=False)
        frame, y, pred = _predict(model, va, device)
        record = frame[[d["id_column"], d["idv_column"], d["dv_column"]]].copy(); record["fold"] = fold
        if pred.shape[1] == 1: record["prediction"] = pred.ravel()
        else:
            for j, name in enumerate(cfg["model"]["probabilities"]): record[f"prob_{name}"] = pred[:,j]
            record["prediction"] = pred.argmax(1)
        oof_frames.append(record); fold_metrics.append({"fold": fold, "best_epoch": epoch, "validation_metric": best})
        for feature, mean, scale in zip(d["feature_columns"], scaler.mean_, scaler.scale_):
            scaler_rows.append({"fold": fold, "feature": feature, "mean": mean, "scale": scale})
        model.eval()
        with torch.no_grad():
            zero = torch.zeros((1, len(d["feature_columns"])), device=device)
            values = model.individual_parameters(zero)
        for name, value in values.items():
            parameter_rows.append({"fold": fold, "parameter": name, "value_at_zero_features": float(value.reshape(-1)[0].cpu())})
        fold_models.append((model, scaler, train_ids))
    oof = pd.concat(oof_frames, ignore_index=True); oof.to_csv(out/"pooled_oof_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out/"fold_performance.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(out/"fold_global_parameters.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(out/"fold_scalers.csv", index=False)
    if cfg["objective"]["loss"] == "categorical_nll":
        oof_probs = oof[[f"prob_{name}" for name in cfg["model"]["probabilities"]]].to_numpy()
        oof_y = oof[d["dv_column"]].to_numpy(dtype=int)
    else:
        oof_probs = oof["prediction"].to_numpy().reshape(-1, 1)
        oof_y = oof[d["dv_column"]].to_numpy()
    pooled_metrics = _metrics(oof_y, oof_probs, cfg)
    print(f"\n[OOF] pooled performance: {pooled_metrics}", flush=True)

    # Test ensemble: continuous predictions or category probabilities are averaged.
    test_predictions = []
    test_frame = None; test_y = None
    for model, scaler, _ in fold_models:
        arrays = _arrays(df, prepared.test_ids, scaler, cfg); test_frame, test_y, pred = _predict(model, arrays, device); test_predictions.append(pred)
    ensemble = np.mean(test_predictions, axis=0)
    result = test_frame[[d["id_column"], d["idv_column"], d["dv_column"]]].copy()
    if ensemble.shape[1] == 1: result["prediction"] = ensemble.ravel()
    else:
        for j, name in enumerate(cfg["model"]["probabilities"]): result[f"prob_{name}"] = ensemble[:,j]
        result["prediction"] = ensemble.argmax(1)
    result.to_csv(out/"test_ensemble_predictions.csv", index=False)
    test_metrics = _metrics(test_y, ensemble, cfg)
    print(f"[Test ensemble] performance: {test_metrics}", flush=True)
    with (out/"performance.json").open("w") as f: json.dump({"pooled_oof": pooled_metrics, "test": test_metrics, "folds": fold_metrics}, f, indent=2)
    print("[SHAP] starting development OOF and held-out five-model ensemble explanations", flush=True)
    _run_shap(cfg, df, prepared, fold_models, out, device)
    print("[Report] writing hybridpmlx_summary.lst", flush=True)
    write_summary(out)
    print(f"[Complete] all outputs saved to {out}", flush=True)
    return out


def _run_shap(cfg, df, prepared, fold_models, out, device):
    if not cfg["shap"].get("enabled", True): return
    try:
        import shap
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("SHAP analysis requested but shap is not installed") from exc
    d = cfg["data"]; background_n = int(cfg["shap"].get("background_size", 100))
    targets = cfg["shap"].get("targets", "all_eta")
    if targets == "all_eta": targets = fold_models[0][0].personalized
    rows, test_accumulator = [], {target: [] for target in targets}
    class Wrapper(nn.Module):
        def __init__(self, model, target): super().__init__(); self.model=model; self.target=target
        def forward(self, x): return self.model.eta_values(x)[self.target]
    for fold, (model, scaler, train_ids) in enumerate(fold_models, 1):
        print(f"[SHAP fold {fold}/{len(fold_models)}] preparing background and held-out individuals", flush=True)
        # Explain the held-out fold in development data.
        assignment = list(_folds(df, prepared.development_ids, cfg))[fold-1]
        val_ids = assignment[2]
        train_unique = df[df[d["id_column"]].isin(train_ids)].drop_duplicates(d["id_column"])
        val_unique = df[df[d["id_column"]].isin(val_ids)].drop_duplicates(d["id_column"])
        bg = scaler.transform(train_unique[d["feature_columns"]]).astype("float32")
        if len(bg)>background_n: bg=bg[np.random.default_rng(int(cfg["training"]["seed"])+fold).choice(len(bg),background_n,replace=False)]
        xv = scaler.transform(val_unique[d["feature_columns"]]).astype("float32")
        for target in targets:
            print(f"[SHAP fold {fold}/{len(fold_models)}] target={target}", flush=True)
            explainer=shap.DeepExplainer(Wrapper(model,target),torch.tensor(bg,device=device))
            values=np.asarray(explainer.shap_values(torch.tensor(xv,device=device),check_additivity=False)).squeeze()
            for i, ident in enumerate(val_unique[d["id_column"]]):
                for j, feature in enumerate(d["feature_columns"]): rows.append({"ID":ident,"source":"development_oof","fold":fold,"target":target,"feature":feature,"shap_value":float(values[i,j])})
            test_unique = df[df[d["id_column"]].isin(prepared.test_ids)].drop_duplicates(d["id_column"])
            xt = scaler.transform(test_unique[d["feature_columns"]]).astype("float32")
            test_values = np.asarray(explainer.shap_values(torch.tensor(xt,device=device),check_additivity=False)).squeeze()
            test_accumulator[target].append((test_unique[d["id_column"]].to_numpy(), test_values))
    pd.DataFrame(rows).to_csv(out/"development_oof_shap.csv",index=False)
    test_rows=[]
    for target, items in test_accumulator.items():
        ids=items[0][0]; mean_values=np.mean([item[1] for item in items],axis=0)
        for i,ident in enumerate(ids):
            for j,feature in enumerate(d["feature_columns"]): test_rows.append({"ID":ident,"source":"test_ensemble","target":target,"feature":feature,"shap_value":float(mean_values[i,j])})
    test_shap=pd.DataFrame(test_rows); test_shap.to_csv(out/"test_ensemble_shap.csv",index=False)
    development_shap = pd.DataFrame(rows).drop(columns="fold")
    combined=pd.concat([development_shap,test_shap],ignore_index=True)

    def summarize(values):
        importance=(values.assign(abs_shap=lambda z:z.shap_value.abs())
                    .groupby(["target","feature"],as_index=False).abs_shap.mean()
                    .rename(columns={"abs_shap":"mean_absolute_shap"}))
        importance["normalized_importance"]=importance.groupby("target")["mean_absolute_shap"].transform(lambda x:x/x.max() if x.max()>0 else x)
        importance["rank"]=importance.groupby("target")["mean_absolute_shap"].rank(method="min",ascending=False).astype(int)
        return importance

    development_importance = summarize(development_shap)
    external_importance = summarize(test_shap)
    combined_importance = summarize(combined)
    combined.to_csv(out/"combined_individual_shap.csv",index=False)
    development_importance.to_csv(out/"shap_importance_development_oof.csv",index=False)
    external_importance.to_csv(out/"shap_importance_external_ensemble.csv",index=False)
    combined_importance.to_csv(out/"shap_importance_combined.csv",index=False)
    combined_importance.to_csv(out/"shap_importance.csv",index=False)  # backward-compatible alias
    print(f"[SHAP] complete | development={development_shap['ID'].nunique()} IDs | external={test_shap['ID'].nunique()} IDs | combined={combined['ID'].nunique()} IDs", flush=True)
