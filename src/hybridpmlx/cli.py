from __future__ import annotations

import argparse

from .config import constraint_summary, load_config
from .engine import run
from .report import write_summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hybridpmlx", description="Hybrid MLP-pharmacometric modeling")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate a YAML configuration")
    check.add_argument("--config", required=True)
    execute = sub.add_parser("run", help="fit, evaluate, and explain a model")
    execute.add_argument("--config", required=True)
    report = sub.add_parser("report", help="create a NONMEM-like summary for completed results")
    report.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        output = write_summary(args.results)
        print(f"Report created: {output}")
        return
    cfg = load_config(args.config)
    if args.command == "validate":
        print("Configuration is valid.\n")
        print(constraint_summary(cfg))
    else:
        output = run(cfg); print(f"Analysis complete: {output}")


if __name__ == "__main__":
    main()
