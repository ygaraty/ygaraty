"""Compare saved ConvNet scaling reports without a hosted application."""

import argparse
import json
from pathlib import Path


def report(args):
    """Read saved reports without importing coordinator or worker code."""
    from ygaraty.analysis.convnet_scaling import comparison_lines

    cases = [json.loads((p / "report.json").read_text()) for p in args.cases]
    print("\n".join(comparison_lines(cases)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("report", help="Compare saved case reports")
    compare.add_argument("cases", nargs="+", type=Path)
    return report(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
