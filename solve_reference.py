#!/usr/bin/env python
"""
solve_reference.py -- compute reference solutions for SteinerMineBench.
======================================================================

SPDX-License-Identifier: MIT

Runs the MineOptimizer WP3 reference solver on each instance and writes
``reference.json`` and ``reference_paths.npz`` into the instance bundle.

For every family the solver labels EXACT, an independent recomputation
(``steinerbench.verify_exact``) rebuilds the graph, the Dijkstra fields and the
argmin from scratch, sharing no code with the solver, and asserts agreement.
A failure aborts the run rather than silently downgrading the label -- that
assertion is what earns an instance its ``exact`` badge.

You only need this script to REGENERATE references.  Loading instances and
scoring your own solver need neither it nor a MineOptimizer checkout.

Usage
-----
    python solve_reference.py --all                  # every solvable instance
    python solve_reference.py --only zones-04
    python solve_reference.py --group A --resume     # skip ones already done
    python solve_reference.py --all --dual-ascent    # add the Wong bound too

Each instance runs in its own subprocess: ``config.CELL_SIZE`` is fixed when the
solver's configuration module is first imported, so a single process cannot
serve instances at different resolutions.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steinerbench import spec                                    # noqa: E402
from steinerbench.loader import instance_path, load_instance     # noqa: E402
from steinerbench.lower_bound import compute_lower_bound         # noqa: E402
from steinerbench.spec import EXACT_FAMILIES                     # noqa: E402
from steinerbench.verify_exact import (                          # noqa: E402
    ExactnessFailure, verify_instance,
)

OBJECTIVE = (
    "Total raw voxel cost of a portal-rooted, monotonically descending network "
    "reaching every production zone, with edge weight "
    "w(u,v) = (cost_grid[v] + excavation_rate_per_m) * effective_length_m(u,v) "
    "over the 26-neighbourhood. Excludes buildability post-processing, "
    "operating cost and portal establishment."
)


def run_solver_subprocess(instance_id: str, mineoptimizer: str | None,
                          keep_log: Path | None) -> dict:
    """Run the adapter worker for one instance and return its harvested dict."""
    scratch = Path(tempfile.mkdtemp(prefix=f"sbref-{instance_id}-"))
    out = scratch / "result.pkl"
    cmd = [sys.executable, "-m", "steinerbench.mineopt_adapter",
           "--instance", instance_id, "--scratch", str(scratch),
           "--out", str(out)]
    if mineoptimizer:
        cmd += ["--mineoptimizer", mineoptimizer]

    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        if keep_log:
            keep_log.parent.mkdir(parents=True, exist_ok=True)
            keep_log.write_text(proc.stdout + "\n" + proc.stderr,
                                encoding="utf-8")
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
            raise RuntimeError(
                f"reference solver failed for {instance_id} "
                f"(exit {proc.returncode}). Last output:\n{tail}")
        with open(out, "rb") as fh:
            return pickle.load(fh)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build_reference(instance_id: str, harvested: dict, checks: dict,
                    lb: dict) -> tuple[dict, dict]:
    """Assemble ``reference.json`` and the path archive for one instance."""
    inst = load_instance(instance_id)
    applicable = set(inst["metadata"]["topology_families"]["applicable"])
    search_kind = {name: kind for name, _, kind in spec.FAMILIES}

    families = [f for f in harvested["families"] if f["family"] in applicable]
    if not families:
        raise RuntimeError(f"{instance_id}: solver returned no applicable family")

    missing = applicable - {f["family"] for f in families}
    if missing:
        raise RuntimeError(
            f"{instance_id}: solver did not return applicable families "
            f"{sorted(missing)}; the reference would understate the best-known "
            f"cost")

    best = min(families, key=lambda f: f["cost"])
    buildable = [f for f in families if f.get("cost_buildable") is not None]
    best_build = min(buildable, key=lambda f: f["cost_buildable"]) if buildable else None

    # An instance is 'exact' only when its cheapest family is one of the
    # exact-search families AND that family survived independent verification.
    check = checks.get(best["family"])
    is_exact = (best["family"] in EXACT_FAMILIES
                and check is not None
                and check.get("independent_argmin_matches") is True)

    if lb["value"] > best["cost"] * (1 + 1e-9):
        raise RuntimeError(
            f"{instance_id}: lower bound {lb['value']:,.2f} exceeds the "
            f"reference cost {best['cost']:,.2f}. One of them is wrong; "
            f"refusing to ship an inconsistent bundle.")

    per_family = []
    paths_archive: dict[str, np.ndarray] = {}
    for f in sorted(families, key=lambda x: x["cost"]):
        rec = {
            "family": f["family"],
            "search": search_kind[f["family"]],
            "cost": round(f["cost"], 4),
            "cost_buildable": (round(f["cost_buildable"], 4)
                               if f["cost_buildable"] is not None else None),
            "total_length_m": f["total_length_m"],
            "junctions_voxel": f["junctions_voxel"],
            "junctions_world_m": [[round(c, 3) for c in w]
                                  for w in f["junctions_world_m"]],
            "support_class_length_m": f["support_class_length_m"],
            "buildable_summary": f["buildable_summary"],
            "topology_label": f["topology_label"],
        }
        if f["family"] in checks:
            rec["exactness_check"] = checks[f["family"]]
        elif f["family"] in EXACT_FAMILIES:
            rec["exactness_check"] = {
                "performed": False,
                "note": "Exact-search family, but no independent check ran.",
            }
        per_family.append(rec)

        for si, path in enumerate(f["_paths"]):
            if path:
                paths_archive[f"{f['family']}/{si}"] = np.asarray(
                    path, dtype=np.int32)

    reference = {
        "instance_id": instance_id,
        "schema_version": spec.SCHEMA_VERSION,
        "benchmark_version": spec.BENCHMARK_VERSION,
        "reference_type": "exact" if is_exact else "best_known",
        "objective": OBJECTIVE,
        "reference_cost": round(best["cost"], 4),
        "reference_cost_buildable": (round(best_build["cost_buildable"], 4)
                                     if best_build else None),
        "best_topology": best["family"],
        "best_topology_buildable": best_build["family"] if best_build else None,
        "lower_bound": lb,
        "gap_to_lower_bound": (round((best["cost"] - lb["value"]) / lb["value"], 6)
                               if lb["value"] > 0 else None),
        "exactness_note": (
            f"The cheapest family ({best['family']}) uses a closed-form argmin "
            f"over all passable voxels and was independently recomputed from "
            f"scratch; reference_cost is optimal for it."
            if is_exact else
            f"The cheapest family ({best['family']}) uses a "
            f"{search_kind[best['family']]} search, so reference_cost is a "
            f"best-known bound and may be improvable. Exact-search families "
            f"on this instance were still independently verified; see "
            f"per_family[].exactness_check."),
        "per_family": per_family,
        "environment": harvested["environment"],
        "solver_runtime_s": harvested["total_runtime_s"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return reference, paths_archive


def unsolved_reference(instance_id: str) -> dict:
    """The stub reference shipped for instances with no computed solution."""
    inst = load_instance(instance_id)
    g = inst["metadata"]["grid"]
    return {
        "instance_id": instance_id,
        "schema_version": spec.SCHEMA_VERSION,
        "benchmark_version": spec.BENCHMARK_VERSION,
        "reference_type": "unsolved",
        "objective": OBJECTIVE,
        "reference_cost": None,
        "reference_cost_buildable": None,
        "best_topology": None,
        "note": (
            f"No reference solution. At {g['n_voxels']:,} voxels "
            f"({g['n_passable']:,} passable) a multi-source Dijkstra over the "
            f"26-neighbour graph needs roughly 22 GB for the CSR structure "
            f"alone, beyond the machine used to build this suite. The grid, "
            f"terminals and cost model are fully specified, so this instance "
            f"stands as an open scaling frontier: the first valid solution "
            f"submitted becomes its best-known reference. score.py reports "
            f"submissions for this instance without a gap."),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def solve_one(instance_id: str, args) -> str:
    """Solve one instance and write its reference. Returns a status word."""
    dest = instance_path(instance_id)
    ref_path = dest / "reference.json"

    if args.resume and ref_path.exists():
        existing = json.loads(ref_path.read_text(encoding="utf-8"))
        if existing.get("reference_type") != "unsolved" or \
                instance_id in spec.UNSOLVED_INSTANCES:
            return "skipped"

    if instance_id in spec.UNSOLVED_INSTANCES:
        ref_path.write_text(
            json.dumps(unsolved_reference(instance_id), indent=2) + "\n",
            encoding="utf-8")
        print("    reference_type = unsolved (open scaling frontier)")
        return "unsolved"

    t0 = time.time()
    print("    running reference solver ...", flush=True)
    log = (Path(args.logs) / f"{instance_id}.log") if args.logs else None
    harvested = run_solver_subprocess(instance_id, args.mineoptimizer, log)
    print(f"    solver: {len(harvested['families'])} families in "
          f"{harvested['total_runtime_s']:.1f} s")

    inst = load_instance(instance_id)

    print("    independent exactness verification ...")
    try:
        checks = verify_instance(inst, harvested["families"], verbose=True)
    except ExactnessFailure as exc:
        print(f"    EXACTNESS FAILURE: {exc}")
        raise

    print("    lower bound ...")
    lb = compute_lower_bound(inst, dual_ascent=args.dual_ascent, verbose=True)

    reference, paths = build_reference(instance_id, harvested, checks, lb)

    ref_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(dest / "reference_paths.npz", **paths)

    print(f"    -> {reference['reference_type']}  "
          f"cost ${reference['reference_cost']:,.0f} "
          f"({reference['best_topology']})  "
          f"LB ${lb['value']:,.0f}  "
          f"gap {reference['gap_to_lower_bound']*100:.1f}%  "
          f"[{time.time() - t0:.1f} s]")
    return "solved"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true")
    sel.add_argument("--only", nargs="+", metavar="ID")
    sel.add_argument("--group", metavar="A|B|C|D")

    p.add_argument("--resume", action="store_true",
                   help="skip instances that already have a reference")
    p.add_argument("--dual-ascent", action="store_true",
                   help="also run Wong dual ascent for the lower bound "
                        "(slow and rarely competitive on lattice graphs)")
    p.add_argument("--mineoptimizer", default=None,
                   help="path to the MineOptimizer checkout providing WP3")
    p.add_argument("--logs", default=None, metavar="DIR",
                   help="keep the full solver log per instance in DIR")
    p.add_argument("--continue-on-error", action="store_true",
                   help="report and move on instead of stopping at the first "
                        "failure (exactness failures always stop the run)")
    args = p.parse_args(argv)

    if args.only:
        ids = spec.select(only=args.only)
    elif args.group:
        ids = spec.select(group=args.group)
    elif args.all:
        ids = spec.select()
    else:
        p.print_help()
        return 2

    tally: dict[str, list[str]] = {"solved": [], "skipped": [], "unsolved": [],
                                   "failed": []}
    t_start = time.time()

    for n, iid in enumerate(ids, 1):
        print(f"\n[{n}/{len(ids)}] {iid}")
        try:
            tally[solve_one(iid, args)].append(iid)
        except ExactnessFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, then re-raised or skipped
            tally["failed"].append(iid)
            print(f"    FAILED: {exc}")
            if not args.continue_on_error:
                return 1

    print(f"\n{'=' * 62}")
    print(f"  solved   {len(tally['solved']):>3}")
    print(f"  unsolved {len(tally['unsolved']):>3}"
          + (f"  ({', '.join(tally['unsolved'])})" if tally["unsolved"] else ""))
    print(f"  skipped  {len(tally['skipped']):>3}")
    print(f"  failed   {len(tally['failed']):>3}"
          + (f"  ({', '.join(tally['failed'])})" if tally["failed"] else ""))
    print(f"  elapsed  {time.time() - t_start:.1f} s")
    print("=" * 62)
    return 1 if tally["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
