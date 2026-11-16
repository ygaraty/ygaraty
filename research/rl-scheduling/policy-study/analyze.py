"""Score runs/policy-study/summary.json against POLICY-STUDY-PROTOCOL.md's
criteria, per hop.

python public/research/rl-scheduling/policy-study/analyze.py
runs/policy-study/summary.json
"""

import json
import statistics as S
import sys
from collections import defaultdict

rows = json.load(open(sys.argv[1]))
for hop in ("l5", "published"):
    rs = [r for r in rows if r["hop"] == hop]
    print(f"\n######## hop {hop}: {len(rs)} cases")

    def name(r):
        return f"{r['workload']}/{r['scenario']}/{r['budget_mib'] if r['budget_mib'] < 1024 else 'nocap'}"  # noqa: E501 -- preserve fixture or report layout

    print(
        "\n#3 guards (gain over compute/none, tuned depth; vs "
        "hindsight-best fixed)"
    )
    for r in rs:
        if (
            r["scenario"] == "heterogeneous_8"
            or abs(r["compute/transfer vs compute/none"]) >= 2
            or abs(r["compute/v1 vs compute/none"]) >= 2
        ):
            print(
                f"  {name(r):40} v1 {r['compute/v1 vs compute/none']:+7.2f}  transfer {r['compute/transfer vs compute/none']:+7.2f}"  # noqa: E501 -- preserve fixture or report layout
                f"  | hindsight v1 {r['compute/v1 vs hindsight']:+7.2f} transfer {r['compute/transfer vs hindsight']:+7.2f}"  # noqa: E501 -- preserve fixture or report layout
                f"  | same depth v1 {r['compute/v1 same depth']:+6.2f} transfer {r['compute/transfer same depth']:+6.2f}"  # noqa: E501 -- preserve fixture or report layout
            )
    het_f = [
        r
        for r in rs
        if r["workload"] == "tiled_1024" and r["scenario"] == "heterogeneous_8"
    ]
    het_m = [
        r
        for r in rs
        if r["workload"].startswith("mlp")
        and r["scenario"] == "heterogeneous_8"
    ]
    a = all(
        r["compute/transfer vs compute/none"] > r["compute/v1 vs compute/none"]
        for r in het_f
    )
    b = min(r["compute/transfer vs compute/none"] for r in het_m) >= -1
    c = min(r["compute/transfer vs compute/none"] for r in rs) >= -2
    worst = min(rs, key=lambda r: r["compute/transfer vs compute/none"])
    print(
        f"  criteria: beats v1 on het. factorization {a}; het. MLP >= -1% {b}; no case < -2% {c} (worst {name(worst)} {worst['compute/transfer vs compute/none']:+.2f})"  # noqa: E501 -- preserve fixture or report layout
        f"  mean transfer {S.mean(r['compute/transfer vs compute/none'] for r in rs):+.2f}, mean v1 {S.mean(r['compute/v1 vs compute/none'] for r in rs):+.2f}"  # noqa: E501 -- preserve fixture or report layout
    )
    print(
        "\n#4 comm vs compute priority, per scenario (mean over "
        "workloads); compute vs fifo"
    )
    by = defaultdict(list)
    for r in rs:
        by[r["scenario"]].append(r)
    for sc, xs in sorted(by.items()):
        print(
            f"  {sc:16} comm vs compute mean {S.mean(r['comm vs compute'] for r in xs):+6.2f} [{min(r['comm vs compute'] for r in xs):+6.2f}, {max(r['comm vs compute'] for r in xs):+6.2f}]"  # noqa: E501 -- preserve fixture or report layout
            f"   compute vs fifo mean {S.mean(r['compute vs fifo'] for r in xs):+6.2f} worst {min(r['compute vs fifo'] for r in xs):+6.2f}"  # noqa: E501 -- preserve fixture or report layout
        )
    target = min(rs, key=lambda r: r["compute vs fifo"])
    print(
        f"  overall mean {S.mean(r['comm vs compute'] for r in rs):+.2f}; worst scenario mean {min(S.mean(r['comm vs compute'] for r in xs) for xs in by.values()):+.2f};"  # noqa: E501 -- preserve fixture or report layout
        f" worst case {min(r['comm vs compute'] for r in rs):+.2f}"
    )
    print(
        f"  target (worst compute vs fifo): {name(target)} compute vs fifo {target['compute vs fifo']:+.2f}; comm vs compute there {target['comm vs compute']:+.2f};"  # noqa: E501 -- preserve fixture or report layout
        f" comm vs fifo there {100 * (1 - target['test_mean_s']['comm/none'] / target['test_mean_s']['fifo/none']):+.2f}"  # noqa: E501 -- preserve fixture or report layout
    )
    print(
        "\n#5 depth and placement apart (compute priority): mean "
        "over cases [min, max]"
    )
    for k in (
        "compute/none depth gain",
        "compute/v1 same depth",
        "compute/transfer same depth",
        "compute joint vs depth 1",
    ):
        v = [r[k] for r in rs]
        print(f"  {k:30} {S.mean(v):+6.2f} [{min(v):+7.2f}, {max(v):+7.2f}]")
    for sc, xs in sorted(by.items()):
        print(
            f"  {sc:16} depth {S.mean(r['compute/none depth gain'] for r in xs):+6.2f}  v1 same depth {S.mean(r['compute/v1 same depth'] for r in xs):+6.2f}"  # noqa: E501 -- preserve fixture or report layout
            f"  transfer same depth {S.mean(r['compute/transfer same depth'] for r in xs):+6.2f}  joint {S.mean(r['compute joint vs depth 1'] for r in xs):+6.2f}"  # noqa: E501 -- preserve fixture or report layout
        )
    print("\n#6 selectors vs fifo at its tuned depth (test seeds)")
    for sel in ("seed0", "conservative", "critical path"):
        m = [r[f"select {sel}"]["mean_gain"] for r in rs]
        w = [r[f"select {sel}"]["worst_gain"] for r in rs]
        print(
            f"  {sel:14} mean gain {S.mean(m):+6.2f}  worst per-seed {min(w):+7.2f}  cases with worst < -2%: {sum(x < -2 for x in w)}/{len(w)}"  # noqa: E501 -- preserve fixture or report layout
            f"  kept fifo: {sum(r[f'select {sel}']['config'].startswith('fifo') for r in rs)}"  # noqa: E501 -- preserve fixture or report layout
        )
    bad = sorted(rs, key=lambda r: r["select conservative"]["worst_gain"])[:6]
    for r in bad:
        print(
            f"    conservative worst {name(r):40} "
            f"{r['select conservative']}  seed0 {r['select seed0']}"
        )
