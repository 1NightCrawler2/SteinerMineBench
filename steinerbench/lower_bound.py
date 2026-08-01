"""
Valid lower bounds for the geotechnically-weighted directed Steiner problem.
============================================================================

SPDX-License-Identifier: MIT

What is being bounded
---------------------
A ramp network is a portal-rooted, monotonically descending subgraph that
reaches every production zone.  Formally that is a **directed Steiner
arborescence**: root ``r`` = the portal terminal set, terminals ``T`` = the
production zones, arc set ``A`` = the descent arcs (dz <= 0), which already
contains the horizontal crosscut arcs (dz == 0) as a subset.

Why the bound is valid for all seven families
---------------------------------------------
Every one of the seven topology families produces a feasible portal-rooted
subgraph over exactly that arc set.  Therefore

    reported_cost(family)  >=  subgraph_cost(family)  >=  OPT_arborescence
                           >=  LP_relaxation  >=  dual_ascent_bound

The first inequality is an equality for ``chained_fan``, which counts shared
ramp voxels once, and is strict for ``sublevel_fan``, ``two_branch`` and
``three_branch``, which charge a shared upper ramp once per branch.  The bound
therefore holds a fortiori for the double-counting families.

Direction-safe arc cost
-----------------------
The reference solver charges a traversal ``u -> v`` as
``(cost[v] + excavation_rate) * length``, but it obtains junction-to-zone drive
costs from an *ascent* search run in the opposite direction, which charges
``(cost[u] + excavation_rate) * length`` for the same physical segment.  To
stay valid against whichever direction was charged, this module prices every
arc at

    c_LB(u, v) = (min(cost[u], cost[v]) + excavation_rate) * length

which is a pointwise lower bound on both conventions.  Any path's true reported
cost is therefore at least its ``c_LB`` cost, and the bound survives.

The bounds implemented here
---------------------------
Three valid bounds are available; the reported value is the strongest of those
computed, and all three are sound, differing only in tightness.

1. ``max_terminal_shortest_path`` -- ``max_k d(r, z_k)``.  Every arborescence
   contains a root-to-terminal path.  Free, very weak.

2. ``pairwise_divergence`` (**default**) -- for any two terminals ``i`` and
   ``j``, the paths ``r -> z_i`` and ``r -> z_j`` inside an arborescence share a
   prefix and separate at some node ``v``; after ``v`` the two branches are
   arc-disjoint, because a tree has no other way to reach both.  Hence

       OPT >= min_v [ d(r, v) + d(v, z_i) + d(v, z_j) ]

   for *every* pair, so the maximum over all pairs is valid.  It costs one
   descent Dijkstra from the portal plus one reverse Dijkstra per zone, then a
   vectorised argmin per pair.  It strictly dominates bound 1, since taking the
   best pair already implies ``max(d(r,z_i), d(r,z_j))``.

   Note the analogous all-terminals expression ``min_v [d(r,v) + sum_k
   d(v,z_k)]`` is **not** a lower bound -- it forces every branch to diverge at
   one node, which is the single-junction *feasible solution* and therefore an
   upper bound.  Pairs are the largest subset for which the divergence argument
   is forced.

3. ``wong_dual_ascent`` -- Wong (1984), "A dual ascent approach for Steiner tree
   problems on a directed graph", Math. Programming 28:271-287.  The dual of the
   cut LP is ``max sum_W u_W`` subject to ``sum_{W: a in delta^-(W)} u_W <=
   c_a``; the ascent repeatedly picks a terminal whose zero-reduced-cost
   in-component ``W`` excludes the root and raises ``u_W`` by
   ``eps = min{cbar_a : a in delta^-(W)}``.  The dual stays feasible after every
   iteration, so a truncated run still yields a valid bound -- which is what
   makes the deterministic ``max_iterations`` cap safe.

   **Empirically it is not competitive on these instances** and is therefore
   opt-in rather than default.  Dual ascent grows its root component one
   zero-reduced-cost arc at a time; on a fine 26-connected voxel lattice with
   millions of near-parallel arcs each ``eps`` is tiny.  On ``scale-130k``
   (114k passable voxels, 1.9 M arcs) 3,000 iterations took 55 s and reached
   1.47e6, still below the free trivial bound of 1.94e6, with no terminal yet
   connected.  Dual ascent is designed for sparse networks, not dense grid
   graphs.  It is retained here because it is valid and useful for comparison,
   and because it may converge usefully on coarser instances.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

_SENTINEL_TEST = 0.9e9

#: Deterministic iteration cap.  Reached only on the largest instances; the
#: bound remains valid, just weaker.  Never made a wall-clock budget, because a
#: shipped reference must be reproducible.
DEFAULT_MAX_ITERATIONS = 30_000

#: Above this many passable voxels, dual ascent is skipped in favour of the
#: trivial bound.  Each iteration costs O(|arcs|) and the arc count grows ~17x
#: the voxel count.
DEFAULT_MAX_NODES = 1_500_000


def _descent_arcs(cost_grid: np.ndarray, cell_size_m: float,
                  excavation_rate: float):
    """
    Build the direction-safe descent arc list over passable voxels.

    Returns ``(tail, head, cost, n_nodes, flat_to_node, node_flat)`` where arcs
    point downhill (dz <= 0) and are priced with the min-endpoint rule above.
    """
    nx, ny, nz = cost_grid.shape
    flat_cost = cost_grid.ravel().astype(np.float64)
    passable = flat_cost < _SENTINEL_TEST

    node_flat = np.flatnonzero(passable).astype(np.int64)
    n_nodes = node_flat.size
    flat_to_node = np.full(nx * ny * nz, -1, dtype=np.int64)
    flat_to_node[node_flat] = np.arange(n_nodes)

    ijk = np.empty((n_nodes, 3), dtype=np.int64)
    ijk[:, 0] = node_flat // (ny * nz)
    ijk[:, 1] = (node_flat // nz) % ny
    ijk[:, 2] = node_flat % nz

    tails, heads, costs = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0):          # descent only; dz == 0 covers crosscuts
                if dx == dy == dz == 0:
                    continue
                length = math.sqrt(dx * dx + dy * dy + dz * dz) * cell_size_m
                nc = ijk + (dx, dy, dz)
                ok = ((nc[:, 0] >= 0) & (nc[:, 0] < nx)
                      & (nc[:, 1] >= 0) & (nc[:, 1] < ny)
                      & (nc[:, 2] >= 0) & (nc[:, 2] < nz))
                nc_safe = np.clip(nc, 0, (nx - 1, ny - 1, nz - 1))
                n_flat = (nc_safe[:, 0] * ny + nc_safe[:, 1]) * nz + nc_safe[:, 2]
                dst = flat_to_node[n_flat]
                valid = ok & (dst >= 0)
                if not valid.any():
                    continue
                src = np.flatnonzero(valid)
                cu = flat_cost[node_flat[src]]
                cv = flat_cost[n_flat[valid]]
                tails.append(src)
                heads.append(dst[valid])
                costs.append((np.minimum(cu, cv) + excavation_rate) * length)

    return (np.concatenate(tails), np.concatenate(heads),
            np.concatenate(costs), n_nodes, flat_to_node, node_flat)


def _add_virtual_terminals(tail, head, cost, n_nodes, portal_nodes,
                           zone_node_sets):
    """
    Add a virtual root with zero-cost arcs to the portal, and one virtual sink
    per zone with zero-cost arcs from that zone's voxels.

    Node ``n_nodes`` is the root; ``n_nodes + 1 + k`` is zone ``k``'s sink.
    """
    root = n_nodes
    sinks = [n_nodes + 1 + k for k in range(len(zone_node_sets))]
    total = n_nodes + 1 + len(zone_node_sets)

    extra_tail = [np.full(portal_nodes.size, root, dtype=np.int64)]
    extra_head = [portal_nodes]
    extra_cost = [np.zeros(portal_nodes.size)]

    for sink, zn in zip(sinks, zone_node_sets):
        extra_tail.append(zn)
        extra_head.append(np.full(zn.size, sink, dtype=np.int64))
        extra_cost.append(np.zeros(zn.size))

    return (np.concatenate([tail] + extra_tail),
            np.concatenate([head] + extra_head),
            np.concatenate([cost] + extra_cost),
            total, root, sinks)


def wong_dual_ascent(tail, head, cost, n_total, root, sinks,
                     max_iterations: int = DEFAULT_MAX_ITERATIONS,
                     verbose: bool = False) -> dict:
    """
    Wong's dual ascent lower bound for the directed Steiner arborescence.

    Deterministic: terminals are processed smallest-root-component first, with
    ties broken by terminal index, so the result depends only on the input.

    Returns a dict with ``value``, ``iterations``, ``converged`` and
    ``runtime_s``.  ``value`` is a valid lower bound whether or not the ascent
    converged.
    """
    t0 = time.time()
    cbar = cost.astype(np.float64).copy()
    lb = 0.0
    iterations = 0
    n_arcs = cbar.size

    def in_component(terminal: int) -> np.ndarray:
        """Nodes that reach ``terminal`` using only zero-reduced-cost arcs."""
        z = cbar <= 0.0
        # Reverse graph: an arc u->v becomes v->u, so a forward BFS from the
        # terminal enumerates exactly the nodes that can reach it.
        rev = csr_matrix(
            (np.ones(int(z.sum()), dtype=np.int8), (head[z], tail[z])),
            shape=(n_total, n_total))
        order = breadth_first_order(rev, terminal, directed=True,
                                    return_predecessors=False)
        mask = np.zeros(n_total, dtype=bool)
        mask[order] = True
        return mask

    open_sinks = list(sinks)

    while open_sinks and iterations < max_iterations:
        # Wong's rule: grow the smallest root component first.
        comps = []
        for s in open_sinks:
            w = in_component(s)
            if w[root]:
                continue                      # this terminal is now connected
            comps.append((int(w.sum()), s, w))

        if not comps:
            open_sinks = []
            break

        comps.sort(key=lambda c: (c[0], c[1]))
        _, chosen, W = comps[0]
        open_sinks = [s for _, s, _ in comps]

        entering = W[head] & ~W[tail]
        if not entering.any():
            # No arc enters W: the terminal is unreachable from the root under
            # the descent constraint. The instance is infeasible for it.
            raise ValueError(
                f"terminal node {chosen} is unreachable from the root over the "
                f"descent arc set; the instance is infeasible")

        eps = float(cbar[entering].min())
        if eps <= 0.0:
            raise RuntimeError(
                "dual ascent stalled: an entering arc already has zero reduced "
                "cost, which should have grown the component")

        cbar[entering] -= eps
        lb += eps
        iterations += 1

        if verbose and iterations % 250 == 0:
            print(f"        dual ascent it {iterations:>6}  LB {lb:>16,.0f}  "
                  f"open terminals {len(open_sinks)}  "
                  f"zero arcs {int((cbar <= 0).sum()):>10,}/{n_arcs:,}")

    return {
        "value": lb,
        "iterations": iterations,
        "converged": not open_sinks,
        "runtime_s": round(time.time() - t0, 2),
    }


def trivial_bound(tail, head, cost, n_total, root, sinks) -> float:
    """
    ``max_k d(root, sink_k)``: every arborescence contains a root-to-terminal
    path, so the longest such shortest path is a valid (if weak) lower bound.

    Free to compute; used as a floor under the other two bounds.
    """
    from scipy.sparse.csgraph import dijkstra
    g = csr_matrix((cost, (tail, head)), shape=(n_total, n_total))
    d = dijkstra(g, directed=True, indices=root)
    finite = [d[s] for s in sinks if np.isfinite(d[s])]
    if len(finite) != len(sinks):
        raise ValueError("some terminal is unreachable from the portal over "
                         "the descent arc set; the instance is infeasible")
    return float(max(finite))


def pairwise_divergence_bound(tail, head, cost, n_total, root, sinks,
                              verbose: bool = False) -> dict:
    """
    The pairwise-divergence lower bound (see the module docstring).

    In any arborescence the paths to two terminals ``z_i`` and ``z_j`` share a
    prefix and then separate at a node ``v``, beyond which the two branches are
    arc-disjoint.  So for every pair

        OPT >= min_v [ d(r, v) + d(v, z_i) + d(v, z_j) ]

    and the maximum over pairs is valid.  ``v`` ranges over all nodes, including
    ``r`` itself and the terminals, so no divergence structure is excluded.

    Costs one forward Dijkstra from the root plus one reverse Dijkstra per
    terminal, then an O(n) vectorised minimum per pair.
    """
    from itertools import combinations

    from scipy.sparse.csgraph import dijkstra

    t0 = time.time()
    fwd = csr_matrix((cost, (tail, head)), shape=(n_total, n_total))
    rev = csr_matrix((cost, (head, tail)), shape=(n_total, n_total))

    d_root = dijkstra(fwd, directed=True, indices=root)
    # dijkstra on the reversed graph from sink k gives d(v, sink_k) for all v.
    d_to = [dijkstra(rev, directed=True, indices=s) for s in sinks]

    for k, s in enumerate(sinks):
        if not np.isfinite(d_root[s]):
            raise ValueError(
                f"terminal {k} is unreachable from the portal over the descent "
                f"arc set; the instance is infeasible")

    best = 0.0
    best_pair = None
    for i, j in combinations(range(len(sinks)), 2):
        total = d_root + d_to[i] + d_to[j]
        v = float(np.nanmin(np.where(np.isfinite(total), total, np.inf)))
        if v > best:
            best, best_pair = v, (i, j)

    if len(sinks) == 1:
        best = float(d_root[sinks[0]])
        best_pair = (0, 0)

    if verbose:
        print(f"        pairwise divergence: best pair {best_pair} -> "
              f"{best:,.0f}  ({time.time() - t0:.1f} s)")

    return {
        "value": best,
        "best_pair": list(best_pair) if best_pair else None,
        "runtime_s": round(time.time() - t0, 2),
    }


def compute_lower_bound(instance: dict,
                        max_nodes: int = DEFAULT_MAX_NODES,
                        max_iterations: int = DEFAULT_MAX_ITERATIONS,
                        dual_ascent: bool = False,
                        verbose: bool = True) -> dict:
    """
    Compute the strongest valid lower bound available for one instance.

    Runs the trivial and pairwise-divergence bounds always, and Wong dual
    ascent only when ``dual_ascent=True`` (it is not competitive here -- see the
    module docstring).  Reports the largest, since the maximum of valid bounds
    is valid.

    Returns a ``lower_bound`` record ready to embed in ``reference.json``:
    ``value``, ``method``, ``valid``, ``note``, ``runtime_s``, plus the
    individual component values under ``components``.

    Every method reported is provably valid against all seven topology
    families; they differ in tightness, not soundness.
    """
    md = instance["metadata"]
    g = md["grid"]
    exc = md["cost_model"]["excavation_rate_per_m"]

    t0 = time.time()
    tail, head, cost, n_nodes, flat_to_node, _ = _descent_arcs(
        instance["cost_grid"], float(g["cell_size_m"]), exc)

    nx, ny, nz = instance["cost_grid"].shape

    def to_nodes(voxels):
        flat = np.array([(v[0] * ny + v[1]) * nz + v[2] for v in voxels],
                        dtype=np.int64)
        n = flat_to_node[flat]
        return np.unique(n[n >= 0])

    portal_nodes = to_nodes(instance["portal_voxels"])
    zone_nodes = [to_nodes(zv) for zv in instance["zone_voxels"]]

    tail, head, cost, n_total, root, sinks = _add_virtual_terminals(
        tail, head, cost, n_nodes, portal_nodes, zone_nodes)

    if verbose:
        print(f"      lower bound: {n_nodes:,} passable voxels, "
              f"{cost.size:,} descent arcs, {len(sinks)} terminals")

    components: dict[str, float] = {}

    components["max_terminal_shortest_path"] = trivial_bound(
        tail, head, cost, n_total, root, sinks)

    pair = pairwise_divergence_bound(tail, head, cost, n_total, root, sinks,
                                     verbose=verbose)
    components["pairwise_divergence"] = pair["value"]

    if dual_ascent:
        if n_nodes > max_nodes:
            if verbose:
                print(f"        dual ascent skipped: {n_nodes:,} passable "
                      f"voxels exceeds the {max_nodes:,} threshold")
        else:
            da = wong_dual_ascent(tail, head, cost, n_total, root, sinks,
                                  max_iterations=max_iterations,
                                  verbose=verbose)
            components["wong_dual_ascent"] = da["value"]

    method = max(components, key=components.get)
    value = components[method]

    notes = {
        "pairwise_divergence": (
            "Pairwise-divergence bound: for the binding terminal pair "
            f"{pair['best_pair']}, min over all nodes v of "
            "d(portal,v) + d(v,zone_i) + d(v,zone_j). Valid because the two "
            "root-to-terminal paths in any arborescence separate at some v and "
            "are arc-disjoint thereafter."),
        "max_terminal_shortest_path": (
            "Max over terminals of the shortest descending path from the "
            "portal. Valid but weak; reported because it exceeded the other "
            "bounds computed."),
        "wong_dual_ascent": (
            "Wong dual ascent on the LP relaxation of the directed Steiner cut "
            "formulation over the full descent arc set."),
    }
    note = notes[method] + (
        " Arcs are priced at (min(cost[u], cost[v]) + excavation_rate) * length "
        "so the bound stays valid regardless of which traversal direction the "
        "reference solver charged for a segment. The bound applies to all seven "
        "topology families: each produces a feasible portal-rooted subgraph over "
        "this arc set, and the families that charge a shared ramp once per "
        "branch report at least their subgraph cost.")

    return {
        "value": value,
        "method": method,
        "valid": True,
        "note": note,
        "components": {k: round(v, 4) for k, v in components.items()},
        "runtime_s": round(time.time() - t0, 2),
    }
