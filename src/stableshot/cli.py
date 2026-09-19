from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from stableshot.audit import explain_events, plot_events, read_jsonl, verify_events
from stableshot.demo import self_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StableShots demo and audit utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="start the interactive Streamlit demo")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8501)

    subparsers.add_parser("self-check", help="run deterministic artifact checks")

    audit = subparsers.add_parser("audit", help="inspect StableShots audit JSONL")
    audit_subparsers = audit.add_subparsers(dest="audit_command", required=True)
    for command in ("verify", "explain"):
        child = audit_subparsers.add_parser(command)
        child.add_argument("audit_jsonl")
    plot = audit_subparsers.add_parser("plot")
    plot.add_argument("audit_jsonl")
    plot.add_argument("output_png")
    return parser


def _run_demo(host: str, port: int) -> int:
    app = Path(__file__).with_name("webapp.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.run(command, check=False).returncode


def _run_self_check() -> int:
    rows = self_check()
    print("StableShots artifact self-check")
    print()
    for row in rows:
        print(f"[{row['status']}] {row['check']}: {row['detail']}")
    failed = [row for row in rows if row["status"] != "PASS"]
    print()
    print(f"{len(rows) - len(failed)}/{len(rows)} checks passed.")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return _run_demo(args.host, args.port)
    if args.command == "self-check":
        return _run_self_check()

    events = read_jsonl(args.audit_jsonl)
    if args.audit_command == "verify":
        print(json.dumps(verify_events(events), indent=2, sort_keys=True))
    elif args.audit_command == "explain":
        print(json.dumps(explain_events(events), indent=2, sort_keys=True))
    else:
        output = plot_events(events, args.output_png)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
