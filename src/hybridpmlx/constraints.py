from __future__ import annotations

import math
from typing import Any


def logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def initial_raw(spec: dict[str, Any], namespace: dict[str, float]) -> float:
    initial = float(spec["initial"])
    c = spec.get("constraint", {})
    kind = c.get("type", "unconstrained")
    if kind in {"positive", "relative_log"}:
        if initial <= 0: raise ValueError("Positive parameter initial value must be > 0")
        return math.log(initial)
    if kind in {"bounded", "dynamic_bounded"}:
        lower = float(c.get("lower", 0.0))
        upper = float(c["upper"]) if kind == "bounded" else float(namespace[c["upper_parameter"]] * c.get("upper_multiplier", -1.0) + c.get("upper_constant", 100))
        if not lower < initial < upper:
            raise ValueError(f"Initial value {initial} must lie inside ({lower}, {upper})")
        return logit((initial - lower) / (upper - lower))
    return initial


def apply_constraint(raw_global, eta, spec, namespace, torch):
    c = spec.get("constraint", {})
    kind = c.get("type", "unconstrained")
    eta_scale = float(spec.get("eta_scale", 1.0))
    if eta is None:
        eta = 0.0
    if kind == "unconstrained":
        return raw_global + eta_scale * eta
    if kind == "positive":
        return torch.exp(raw_global + eta_scale * eta)
    if kind == "bounded":
        lo, hi = float(c["lower"]), float(c["upper"])
        return lo + (hi - lo) * torch.sigmoid(raw_global + eta_scale * eta)
    if kind == "dynamic_bounded":
        lo = float(c.get("lower", 0.0))
        upper = float(c.get("upper_constant", 100.0)) + float(c.get("upper_multiplier", -1.0)) * namespace[c["upper_parameter"]]
        return lo + (upper - lo) * torch.sigmoid(raw_global + eta_scale * eta)
    if kind == "relative_log":
        scale = float(c["log_scale"])
        base = torch.exp(raw_global)
        return base * torch.exp(scale * torch.tanh(eta))
    if kind == "absolute_change":
        base = raw_global
        value = float(c["value"])
        out = base + value * torch.tanh(eta)
        if "lower" in c:
            out = torch.clamp(out, min=float(c["lower"]))
        return out
    raise ValueError(f"Unknown constraint type: {kind}")
