#!/usr/bin/env python
"""
validate.py -- check every SteinerMineBench bundle is complete and consistent.
=============================================================================

SPDX-License-Identifier: MIT

Per instance this verifies:

  files       cost_grid.npz, metadata.json, README.md all present
  schema      metadata.json and reference.json validate (needs jsonschema)
  checksum    cost_grid.npz SHA-256 matches metadata.checksums
  grid        dims, cell size, voxel and passable counts are self-consistent,
              tier indices are in range, the tier histogram matches the array
  terminals   every portal and zone voxel is in bounds and passable
  reference   lower_bound <= reference_cost; the per-family costs are
              reproduced by independently re-costing the stored voxel paths
              straight off the grid, with no reference to the solver
  exact       for instances labelled 'exact', the winning family carries a
              passing independent-argmin check

Usage
-----
    python validate.py                # everything
    python validate.py --fast         # skip the two largest instances
    python validate.py --only zones-04
    python validate.py --publication  # additionally require DOI/author fields
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from steinerbench import spec, tiers                            # noqa: E402
from steinerbench.grid import sha256_file                       # noqa: E402
from steinerbench.loader import (                               # noqa: E402
    instance_path, load_instance, load_reference_paths,
)

#: Instances skipped by --fast (slowest to load and re-cost).
SLOW = {"scale-8m", "scale-129m"}

#: Relative tolerance when re-costing float32 solver output in float64.
COST_RTOL = 1e-6

PLACEHOLDER_MARKERS = ("<DOI-PLACEHOLDER>", "<AUTHOR", "<AFFILIATION>",
                       "<ORCID>", "<PAPER-TITLE>", "<VENUE>", "<GITHUB-OWNER>",
                       "<GITHUB-REPO>", "10.5281/zenodo.0000000")


class Report:
    """Collects pass/fail results for one validation run."""

    def __init__(self, verbose: bool = True):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks = 0
        self.verbose = verbose

    def check(self, ok: bool, message: str) -> bool:
        self.checks += 1
        if not ok:
            self.errors.append(message)
        return ok

    def warn(self, ok: bool, message: str) -> None:
        self.checks += 1
        if not ok:
            self.warnings.append(message)


def _validate_schema(doc: dict, schema_name: str, label: str, rep: Report) -> None:
    try:
        import jsonschema
    except ImportError:
        return
    schema_path = ROOT / "schemas" / schema_name
    if not schema_path.exists():
        rep.check(False, f"{label}: missing schema {schema_name}")
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(doc, schema)
    except jsonschema.ValidationError as exc:
        loc = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        rep.check(False, f"{label}: schema violation at {loc}: {exc.message}")


def _recost(path: np.ndarray, cost_grid: np.ndarray, cell: float,
            exc: float) -> float:
    """
    Re-cost one stored voxel polyline straight off the grid.

    Uses the published edge weight, charging the voxel being entered:
    ``w(u,v) = (cost[v] + excavation_rate) * ||v - u|| * cell_size``.
    Stored paths are oriented source -> junction, matching the direction the
    solver's Dijkstra charged.
    """
    p = np.asarray(path, dtype=np.int64)
    if len(p) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(p, axis=0).astype(np.float64), axis=1) * cell
    c = cost_grid[p[1:, 0], p[1:, 1], p[1:, 2]].astype(np.float64)
    return float(((c + exc) * d).sum())


def validate_instance(instance_id: str, rep: Report) -> None:
    """Run every check for one instance bundle."""
    label = instance_id
    d = instance_path(instance_id)

    for fname in ("cost_grid.npz", "metadata.json", "README.md"):
        if not rep.check((d / fname).exists(), f"{label}: missing {fname}"):
            return

    metadata = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    _validate_schema(metadata, "metadata.schema.json", f"{label}/metadata", rep)

    # ── provenance: the synthetic claim must be present and explicit ─────────
    rep.check(metadata.get("synthetic") is True,
              f"{label}: metadata.synthetic must be true")
    rep.check("not derived from" in metadata.get("provenance_statement", "").lower()
              or "no real" in metadata.get("provenance_statement", "").lower(),
              f"{label}: provenance_statement must state the data is not "
              f"derived from a real mine")

    # ── checksum ─────────────────────────────────────────────────────────────
    recorded = metadata.get("checksums", {}).get("cost_grid.npz", "")
    actual = f"sha256:{sha256_file(d / 'cost_grid.npz')}"
    rep.check(recorded == actual,
              f"{label}: cost_grid.npz checksum mismatch\n"
              f"      recorded {recorded}\n      actual   {actual}")

    inst = load_instance(instance_id)
    g = metadata["grid"]
    cost_grid = inst["cost_grid"]
    passable = inst["passable"]
    dims = tuple(g["dims"])
    cell = float(g["cell_size_m"])
    exc = metadata["cost_model"]["excavation_rate_per_m"]

    # ── grid self-consistency ────────────────────────────────────────────────
    rep.check(cost_grid.shape == dims,
              f"{label}: grid shape {cost_grid.shape} != metadata dims {dims}")
    rep.check(g["n_voxels"] == int(np.prod(dims)),
              f"{label}: n_voxels {g['n_voxels']} != prod(dims)")
    rep.check(g["n_passable"] == int(passable.sum()),
              f"{label}: n_passable {g['n_passable']} != "
              f"{int(passable.sum())} in the array")
    rep.check(dims == spec.dims_for(cell),
              f"{label}: dims {dims} are not the fixed domain at {cell} m")

    ti = inst["tier_index"]
    bad = ti[(ti >= tiers.N_TIERS) & (ti != tiers.SENTINEL_TIER)]
    rep.check(bad.size == 0,
              f"{label}: {bad.size} voxels carry an out-of-range tier index")

    hist = np.bincount(ti[passable].ravel(), minlength=tiers.N_TIERS)[:tiers.N_TIERS]
    rep.check(list(hist) == metadata["cost_model"]["tier_histogram_passable"],
              f"{label}: tier histogram in metadata disagrees with the array")

    finite = cost_grid[passable]
    rep.check(bool(np.isin(finite, tiers.TIER_COSTS.astype(np.float32)).all()),
              f"{label}: passable voxels carry costs outside the tier schedule")

    # ── terminals ────────────────────────────────────────────────────────────
    for name, voxset in (("portal", [inst["portal_voxels"]]),
                         ("zone", inst["zone_voxels"])):
        for n, voxels in enumerate(voxset):
            rep.check(len(voxels) > 0, f"{label}: {name} {n} has no voxels")
            arr = np.asarray(voxels, dtype=np.int64)
            in_bounds = ((arr >= 0).all(axis=1)
                         & (arr < np.array(dims)).all(axis=1)).all()
            rep.check(bool(in_bounds),
                      f"{label}: {name} {n} has out-of-bounds voxels")
            if in_bounds:
                rep.check(bool(passable[arr[:, 0], arr[:, 1], arr[:, 2]].all()),
                          f"{label}: {name} {n} has impassable voxels")

    n_lev = metadata["topology_families"]["n_sublevels"]
    rep.check(metadata["topology_families"]["applicable"]
              == spec.applicable_families(n_lev),
              f"{label}: applicable families disagree with the gating rule at "
              f"{n_lev} sublevels")

    # ── reference ────────────────────────────────────────────────────────────
    ref_path = d / "reference.json"
    if not ref_path.exists():
        rep.warn(False, f"{label}: no reference.json (run solve_reference.py)")
        return

    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    _validate_schema(reference, "reference.schema.json", f"{label}/reference", rep)
    rep.check(reference["instance_id"] == instance_id,
              f"{label}: reference.json instance_id mismatch")

    if reference["reference_type"] == "unsolved":
        rep.check(instance_id in spec.UNSOLVED_INSTANCES,
                  f"{label}: marked unsolved but is not in spec.UNSOLVED_INSTANCES")
        return

    ref_cost = reference["reference_cost"]
    lb = (reference.get("lower_bound") or {}).get("value")
    if lb is not None:
        rep.check(lb <= ref_cost * (1 + 1e-9),
                  f"{label}: lower bound {lb:,.2f} exceeds reference cost "
                  f"{ref_cost:,.2f}")
        rep.check((reference.get("lower_bound") or {}).get("valid") is True,
                  f"{label}: lower_bound.valid must be true; an unsound bound "
                  f"must not ship")

    fams = {f["family"] for f in reference["per_family"]}
    rep.check(fams == set(metadata["topology_families"]["applicable"]),
              f"{label}: per_family {sorted(fams)} != applicable "
              f"{metadata['topology_families']['applicable']}")

    best = min(reference["per_family"], key=lambda f: f["cost"])
    rep.check(abs(best["cost"] - ref_cost) < 1e-6,
              f"{label}: reference_cost is not the cheapest per_family cost")
    rep.check(best["family"] == reference["best_topology"],
              f"{label}: best_topology is not the cheapest family")

    # ── independent re-costing of the stored paths ───────────────────────────
    if (d / "reference_paths.npz").exists():
        paths = load_reference_paths(instance_id)
        for f in reference["per_family"]:
            segs = [p for k, p in paths.items() if k.startswith(f["family"] + "/")]
            if not segs:
                rep.warn(False, f"{label}: no stored paths for {f['family']}")
                continue
            total = sum(_recost(p, cost_grid, cell, exc) for p in segs)
            rel = abs(total - f["cost"]) / max(f["cost"], 1.0)
            rep.check(rel <= COST_RTOL,
                      f"{label}/{f['family']}: re-costing the stored paths "
                      f"gives {total:,.2f} but reference.json says "
                      f"{f['cost']:,.2f} (relative delta {rel:.2e})")

            for j in f["junctions_voxel"]:
                rep.check(bool(passable[j[0], j[1], j[2]]),
                          f"{label}/{f['family']}: junction {j} is impassable")
    else:
        rep.warn(False, f"{label}: no reference_paths.npz to re-cost")

    # ── the 'exact' label must be earned ─────────────────────────────────────
    if reference["reference_type"] == "exact":
        chk = best.get("exactness_check") or {}
        rep.check(best["family"] in spec.EXACT_FAMILIES,
                  f"{label}: labelled exact but the winning family "
                  f"{best['family']} uses a heuristic search")
        rep.check(chk.get("performed") is True
                  and chk.get("independent_argmin_matches") is True,
                  f"{label}: labelled exact but the winning family has no "
                  f"passing independent-argmin check")


def validate_publication(rep: Report) -> None:
    """Refuse to certify a release while citation placeholders remain."""
    for fname in ("CITATION.cff", ".zenodo.json", "README.md"):
        path = ROOT / fname
        if not rep.check(path.exists(), f"publication: missing {fname}"):
            continue
        text = path.read_text(encoding="utf-8")
        found = [m for m in PLACEHOLDER_MARKERS if m in text]
        rep.check(not found,
                  f"publication: {fname} still contains placeholder(s) "
                  f"{', '.join(sorted(set(found)))} -- fill these in before "
                  f"minting a DOI")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="ID")
    p.add_argument("--group", metavar="A|B|C|D")
    p.add_argument("--fast", action="store_true",
                   help=f"skip the largest instances ({', '.join(sorted(SLOW))})")
    p.add_argument("--publication", action="store_true",
                   help="also require citation metadata to be filled in")
    args = p.parse_args(argv)

    if args.only:
        ids = spec.select(only=args.only)
    elif args.group:
        ids = spec.select(group=args.group)
    else:
        ids = spec.select()
    if args.fast:
        ids = [i for i in ids if i not in SLOW]

    ids = [i for i in ids if (instance_path(i) / "metadata.json").exists()]
    if not ids:
        print("No instance bundles found. Run:  python generate.py --all")
        return 1

    rep = Report()
    print(f"Validating {len(ids)} instance bundle(s)\n")
    for n, iid in enumerate(ids, 1):
        before = len(rep.errors)
        validate_instance(iid, rep)
        status = "OK" if len(rep.errors) == before else "FAIL"
        print(f"  [{n}/{len(ids)}] {iid:<18} {status}")

    if args.publication:
        print("\nChecking publication metadata ...")
        validate_publication(rep)

    print(f"\n{'=' * 62}")
    print(f"  {rep.checks} checks over {len(ids)} instance(s)")

    if rep.warnings:
        print(f"\n  {len(rep.warnings)} warning(s):")
        for w in rep.warnings:
            print(f"    - {w}")

    if rep.errors:
        print(f"\n  {len(rep.errors)} ERROR(S):")
        for e in rep.errors:
            print(f"    - {e}")
        print("=" * 62)
        return 1

    print("  all checks passed")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
