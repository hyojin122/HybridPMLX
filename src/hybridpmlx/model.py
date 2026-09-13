from __future__ import annotations

from collections import OrderedDict

from .constraints import apply_constraint, initial_raw
from .expressions import SafeExpression


def torch_ops(torch):
    return {
        "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt,
        "sigmoid": torch.sigmoid, "tanh": torch.tanh, "abs": torch.abs,
        "pow": torch.pow, "clamp": torch.clamp,
    }


def build_model_class():
    import torch
    import torch.nn as nn

    class HybridStructuralModel(nn.Module):
        def __init__(self, cfg: dict, input_dim: int):
            super().__init__()
            self.cfg = cfg
            self.param_specs = OrderedDict(cfg["model"]["parameters"])
            self.personalized = [n for n, s in self.param_specs.items() if s.get("personalized", False) and "derived" not in s]
            layers, previous = [], input_dim
            for units in cfg["network"]["hidden_layers"]:
                layers += [nn.Linear(previous, int(units)), nn.Softplus()]
                if float(cfg["network"].get("dropout", 0)) > 0:
                    layers.append(nn.Dropout(float(cfg["network"]["dropout"])))
                previous = int(units)
            self.features = nn.Sequential(*layers)
            self.eta_head = nn.Linear(previous, len(self.personalized)) if self.personalized else None

            numeric_ns: dict[str, float] = {}
            self.raw_globals = nn.ParameterDict()
            for name, spec in self.param_specs.items():
                if "derived" in spec:
                    continue
                raw = initial_raw(spec, numeric_ns)
                self.raw_globals[name] = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
                numeric_ns[name] = float(spec["initial"])

            self.derived = {n: SafeExpression(s["derived"]) for n, s in self.param_specs.items() if "derived" in s}
            self.equation = SafeExpression(cfg["model"]["equation"]) if cfg["model"].get("equation") else None
            self.probabilities = OrderedDict((name, SafeExpression(expr)) for name, expr in cfg["model"].get("probabilities", {}).items())

        def eta_values(self, x):
            if self.eta_head is None:
                return {}
            raw = self.eta_head(self.features(x))
            return {name: raw[:, j:j+1] for j, name in enumerate(self.personalized)}

        def individual_parameters(self, x, return_eta=False):
            eta = self.eta_values(x)
            values = {}
            for name, spec in self.param_specs.items():
                if "derived" in spec:
                    values[name] = self.derived[name](values, torch_ops(torch))
                else:
                    values[name] = apply_constraint(self.raw_globals[name], eta.get(name), spec, values, torch)
            return (values, eta) if return_eta else values

        def forward(self, x, idv, return_eta=False):
            params, eta = self.individual_parameters(x, return_eta=True)
            namespace = {**params, "IDV": idv}
            if self.equation is not None:
                output = self.equation(namespace, torch_ops(torch))
            else:
                columns = [expr(namespace, torch_ops(torch)) for expr in self.probabilities.values()]
                output = torch.cat(columns, dim=1)
                eps = float(self.cfg["model"].get("probability_epsilon", 1e-10))
                output = torch.clamp(output, min=eps)
                output = output / output.sum(dim=1, keepdim=True)
            return (output, eta, params) if return_eta else output

    return HybridStructuralModel
