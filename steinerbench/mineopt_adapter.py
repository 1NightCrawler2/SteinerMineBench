"""
Adapter that drives the MineOptimizer WP3 solver on a SteinerMineBench instance.
================================================================================

SPDX-License-Identifier: MIT

This module does NOT reimplement the solver.  It synthesises the three artefacts
WP3 expects, points the solver's configuration at a scratch directory, and calls
``wp3_steiner.run_steiner_poly.main()``.  WP1 (kriging) and WP2 (A*) are skipped
entirely, so a reference run never touches the upstream project's real data and
never introduces geostatistical variability.

Why it has to be done this way
------------------------------
``run_steiner_poly.py`` reads its configuration with ``from config import ...``
at module import time and freezes several paths as module constants
(``POLY_PKL``, ``STEINER_POLY_PKL``, line 110-111).  It also creates
``OUTPUT_REPORTS`` as an import side effect.  Every override therefore has to
land on the ``config`` module **before** ``run_steiner_poly`` is first imported.

Determinism controls applied to every reference run
---------------------------------------------------
``MINEOPT_SKIP_LOCAL=1``
    Ignore the user's gitignored ``config_local.py``.  Without this a reference
    run silently depends on one developer's local calibration.
``MINEOPT_FLOOD_ENGINE=scipy``
    Use true Dijkstra.  The default ``stencil`` engine is an iterative
    Bellman-Ford relaxation with a 1e-3 tolerance and a sweep cap, which cannot
    support an ``exact`` label.
``MINEOPT_FORCE_CPU=1``
    Never take the CuPy path; GPU floating-point reduction order is not
    guaranteed reproducible.
``MINEOPT_FORCE_CELL_SIZE=<cell>``
    Match ``config.CELL_SIZE`` to the instance.
``config.OPEX_IN_EDGE_WEIGHTS = False``
    Otherwise ``main()`` folds a depth-dependent ventilation and pumping term
    into the cost grid and the objective stops being pure geotechnical support
    cost.
``--no-lattice``
    ``refine_with_lattice`` is wall-clock budgeted (``LATTICE_TIME_BUDGET_S``)
    and therefore irreproducible.  ``annotate_buildable`` -- which supplies the
    secondary buildable cost track -- has no RNG and no timing dependence, so it
    is kept.

Because ``config`` is a module singleton whose ``CELL_SIZE`` is fixed at import,
each instance must run in its own process.  ``solve_reference.py`` invokes this
module as a subprocess entry point, one per instance.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parent.parent


def _force_utf8_streams() -> None:
    """
    Make stdout/stderr UTF-8 tolerant.

    The reference solver prints arrows and box characters in its progress
    output.  On Windows, redirecting that to a file gives stdout the cp1252
    codec, and the first arrow raises UnicodeEncodeError from deep inside a
    print() -- killing an otherwise successful solve.  Reconfiguring here fixes
    it however this module is invoked.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def find_mineoptimizer(explicit: str | None = None) -> Path:
    """
    Locate the MineOptimizer checkout that provides the reference solver.

    Search order: the ``--mineoptimizer`` argument, the ``MINEOPTIMIZER_ROOT``
    environment variable, then the benchmark repo's parent directory (the usual
    layout, with ``steinermine-bench/`` sitting inside the project).
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MINEOPTIMIZER_ROOT"):
        candidates.append(Path(os.environ["MINEOPTIMIZER_ROOT"]))
    candidates.append(BENCH_ROOT.parent)

    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "config.py").exists() and (c / "wp3_steiner").is_dir():
            return c
    raise FileNotFoundError(
        "Could not locate a MineOptimizer checkout (needs config.py and "
        "wp3_steiner/).\nTried:\n  " + "\n  ".join(str(c) for c in candidates) +
        "\nSet MINEOPTIMIZER_ROOT or pass --mineoptimizer.\n\n"
        "Note: the reference solver is only needed to REGENERATE references. "
        "Loading instances and scoring your own solver need neither it nor "
        "this module.")


# ---------------------------------------------------------------------------
# Building the artefacts WP3 expects
# ---------------------------------------------------------------------------
def write_wp3_inputs(instance: dict, scratch: Path) -> dict:
    """
    Write the four files WP3 reads, into ``scratch``.

    ``cost_grid.npy``      float32 support cost per metre
    ``fault_grid.npy``     int32 distinct-fault count (viewer + portal filter)
    ``surface_rl.npy``     float32 ground surface RL per plan column
    ``grid_metadata.npy``  pickled dict, schema per wp1_voxel/pipeline.py:1205
    ``wp2_poly_results.pkl`` the terminal groups, standing in for a WP2 run

    ``path_matrix`` is deliberately absent: WP3 only reads it under
    ``--corridor`` (run_steiner_poly.py:2289), which reference runs never use.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    md = instance["metadata"]
    g = md["grid"]

    np.save(scratch / "cost_grid.npy", instance["cost_grid"])
    np.save(scratch / "fault_grid.npy",
            instance["fault_count"].astype(np.int32))
    np.save(scratch / "surface_rl.npy", instance["surface_rl"])

    meta = {
        "min_coords": np.asarray(g["min_coords_m"], dtype=np.float64),
        "cell_size": float(g["cell_size_m"]),
        "dims": np.asarray(g["dims"], dtype=np.int64),
        "sentinel": float(g["sentinel_cost_per_m"]),
        "surface_source": "steinerminebench_synthetic",
        "estimation": "steinerminebench_analytic_q_field",
    }
    np.save(scratch / "grid_metadata.npy", meta, allow_pickle=True)

    groups = [{
        "label": md["portals"][0]["label"],
        "voxels": [tuple(int(c) for c in v) for v in instance["portal_voxels"]],
        "type": "portal",
    }]
    for zmeta, zvox in zip(md["zones"], instance["zone_voxels"]):
        groups.append({
            "label": zmeta["label"],
            "voxels": [tuple(int(c) for c in v) for v in zvox],
            "type": "zone",
            "tonnage_mt": zmeta.get("tonnage_mt"),
            "mean_grade": zmeta.get("mean_grade_g_t"),
            "sublevel_rl": zmeta.get("sublevel_rl_m"),
        })

    with open(scratch / "wp2_poly_results.pkl", "wb") as fh:
        pickle.dump({"groups": groups, "meta": meta, "no_portal_zones": []}, fh)

    return meta


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------
#: Maps a solver topology label prefix to its canonical benchmark family name.
#:
#: The solver decorates labels with level counts, e.g. ``"sublevel_fan (4
#: levels)"``.  Note that the hybrid family reports itself as
#: ``"hybrid_chained_fan (2L chain + 2L branch)"``
#: (run_steiner_poly.py:1330) -- WITHOUT the trailing ``_branch`` that its
#: evaluator's name carries.  Prefixes are tested longest-first so
#: ``hybrid_chained_fan`` can never be shadowed by ``chained_fan``.
_FAMILY_PREFIXES = [
    ("hybrid_chained_fan", "hybrid_chained_fan_branch"),
    ("sequential_ramp", "sequential_ramp"),
    ("single_junction", "single_junction"),
    ("sublevel_fan", "sublevel_fan"),
    ("three_branch", "three_branch"),
    ("chained_fan", "chained_fan"),
    ("two_branch", "two_branch"),
]
_FAMILY_PREFIXES.sort(key=lambda kv: -len(kv[0]))


def canonical_family(topology_string: str) -> str | None:
    """Recover the canonical family name from a solver topology label."""
    s = topology_string.strip()
    for prefix, family in _FAMILY_PREFIXES:
        if s.startswith(prefix):
            return family
    return None


def support_class_lengths(paths, cost_grid, cell_size_m, tier_costs,
                          support_classes) -> tuple[dict, float]:
    """
    Metres of drive in each Barton Q support class, plus the total length.

    Length is accumulated per step as the Euclidean distance between voxel
    centres, and attributed to the support class of the voxel being entered --
    the same convention the edge weight uses.
    """
    lengths = {cls: 0.0 for cls in support_classes}
    total = 0.0
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            d = np.sqrt(sum((b[ax] - a[ax]) ** 2 for ax in range(3))) * cell_size_m
            c = float(cost_grid[b[0], b[1], b[2]])
            idx = int(np.argmin(np.abs(np.asarray(tier_costs) - c)))
            lengths[support_classes[idx]] += d
            total += d
    return lengths, total


def extract_results(all_topologies, meta, cost_grid, tier_costs,
                    support_classes) -> list[dict]:
    """Reduce the solver's topology dicts to compact, JSON-friendly records."""
    mn = np.asarray(meta["min_coords"], dtype=np.float64)
    cs = float(meta["cell_size"])
    out = []

    for t in all_topologies:
        if t is None:
            continue
        family = canonical_family(t.get("topology", ""))
        if family is None:
            # Never drop a family silently: a reference that quietly omits a
            # topology would understate the benchmark's best-known cost.
            raise RuntimeError(
                f"cannot map solver topology label {t.get('topology')!r} to a "
                f"benchmark family. The upstream solver has added or renamed a "
                f"topology; update _FAMILY_PREFIXES in mineopt_adapter.py.")

        sps = [tuple(int(c) for c in sp) for sp in t.get("steiner_points", [])]
        paths = [[tuple(int(c) for c in v) for v in p]
                 for p in t.get("paths", []) if p]
        cls_len, total_len = support_class_lengths(
            paths, cost_grid, cs, tier_costs, support_classes)

        bd = t.get("buildable") or {}
        out.append({
            "family": family,
            "topology_label": t.get("topology"),
            "cost": float(t["cost"]),
            "cost_buildable": (float(t["cost_buildable"])
                               if t.get("cost_buildable") is not None else None),
            "junctions_voxel": [list(sp) for sp in sps],
            "junctions_world_m": [
                [float(mn[ax] + (sp[ax] + 0.5) * cs) for ax in range(3)]
                for sp in sps],
            "support_class_length_m": {k: round(v, 3)
                                       for k, v in cls_len.items() if v > 0},
            "total_length_m": round(total_len, 3),
            "buildable_summary": {
                "ramp_length_m": bd.get("ramp_length"),
                "worst_grade": bd.get("worst_grade"),
                "worst_min_radius_m": bd.get("worst_min_radius"),
                "n_spiral_ramps": bd.get("n_spiral_ramps"),
                "method": bd.get("method"),
            } if bd else None,
            "segment_labels": list(t.get("segment_labels", [])),
            "_paths": paths,
        })
    return out


# ---------------------------------------------------------------------------
# The worker: one instance, one process
# ---------------------------------------------------------------------------
def run_instance(instance_id: str, scratch: Path, mineopt_root: Path) -> dict:
    """
    Run the reference solver on one instance.  Must be called in a fresh
    process: it mutates ``os.environ`` and the ``config`` module singleton.
    """
    sys.path.insert(0, str(BENCH_ROOT))
    from steinerbench import tiers
    from steinerbench.loader import load_instance

    instance = load_instance(instance_id)
    md = instance["metadata"]
    cell = float(md["grid"]["cell_size_m"])

    # ── 1. Environment, before `config` is imported ──────────────────────────
    os.environ["MINEOPT_SKIP_LOCAL"] = "1"
    os.environ["MINEOPT_FLOOD_ENGINE"] = "scipy"
    os.environ["MINEOPT_FORCE_CPU"] = "1"
    os.environ["MINEOPT_FORCE_CELL_SIZE"] = repr(cell)

    sys.path.insert(0, str(mineopt_root))
    import config  # noqa: E402

    tiers.assert_matches_mineoptimizer(config)

    # ── 2. Redirect every path, before `run_steiner_poly` is imported ────────
    scratch = scratch.resolve()
    reports = scratch / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    config.DATA_PROCESSED = scratch
    config.COST_GRID_NPY = scratch / "cost_grid.npy"
    config.GRID_META_NPY = scratch / "grid_metadata.npy"
    config.FAULT_GRID_NPY = scratch / "fault_grid.npy"
    config.SURFACE_RL_NPY = scratch / "surface_rl.npy"
    config.OUTPUT_REPORTS = reports
    config.OPEX_IN_EDGE_WEIGHTS = False

    if abs(config.CELL_SIZE - cell) > 1e-9:
        raise RuntimeError(
            f"config.CELL_SIZE is {config.CELL_SIZE} but instance {instance_id} "
            f"needs {cell}; MINEOPT_FORCE_CELL_SIZE did not take effect")

    write_wp3_inputs(instance, scratch)

    # ── 3. Import and drive the solver ───────────────────────────────────────
    from wp3_steiner import run_steiner_poly as wp3  # noqa: E402

    if wp3.POLY_PKL.parent != scratch:
        raise RuntimeError(
            f"WP3 resolved its input path to {wp3.POLY_PKL}, not {scratch}; "
            f"config was mutated too late")

    argv = sys.argv
    t0 = time.time()
    try:
        sys.argv = ["run_steiner_poly.py", "--auto", "--no-lattice"]
        wp3.main()
    finally:
        sys.argv = argv
    runtime = time.time() - t0

    # ── 4. Harvest ───────────────────────────────────────────────────────────
    results_pkl = scratch / "wp3_poly_steiner_results.pkl"
    if not results_pkl.exists():
        raise RuntimeError(
            f"solver produced no results pickle at {results_pkl}; "
            f"see the solver output above")
    with open(results_pkl, "rb") as fh:
        res = pickle.load(fh)

    families = extract_results(
        res["all_topologies"], res["meta"], instance["cost_grid"],
        tiers.TIER_COSTS, tiers.SUPPORT_CLASSES)

    import platform
    import scipy

    return {
        "instance_id": instance_id,
        "families": families,
        "total_runtime_s": round(runtime, 2),
        "environment": {
            "solver": "MineOptimizer WP3 (wp3_steiner/run_steiner_poly.py)",
            "solver_commit": _git_commit(mineopt_root),
            "flood_engine": config.MINEOPT_FLOOD_ENGINE,
            "opex_in_edge_weights": config.OPEX_IN_EDGE_WEIGHTS,
            "lattice_refinement": False,
            "buildable_method": "geometric (deterministic)",
            "cell_size_m": cell,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def _git_commit(root: Path) -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Worker: run the MineOptimizer reference solver on one "
                    "SteinerMineBench instance. Normally invoked by "
                    "solve_reference.py, not directly.")
    p.add_argument("--instance", required=True)
    p.add_argument("--scratch", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="pickle path for the harvested result")
    p.add_argument("--mineoptimizer", default=None,
                   help="path to the MineOptimizer checkout")
    args = p.parse_args(argv)

    _force_utf8_streams()
    root = find_mineoptimizer(args.mineoptimizer)
    result = run_instance(args.instance, args.scratch, root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\n[adapter] harvested {len(result['families'])} families -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
