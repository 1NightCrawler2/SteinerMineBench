# SteinerMineBench

A citable benchmark suite for **geotechnically-weighted Steiner trees on 3-D voxel grids** — the
underground mine ramp network design problem.

24 frozen instances. Each ships a cost grid, complete metadata, a reference solution, and a
**provably valid lower bound**, so you can run your own solver on identical inputs and report an
optimality gap that means something.

> ### All data here is synthetic
>
> Every instance is **procedurally generated** from a recorded seed. The rock mass quality fields,
> fault geometry, topography, portal locations and production-zone geometry were constructed to be
> *geotechnically representative* of a hard-rock underground operation, using the published NGI
> Barton Q-system support classes.
>
> **They are not derived from, measured at, or descriptive of any real or operating mine.** The
> coordinate frame is a neutral local origin at (0, 0, 0) with no geographic meaning. This statement
> is repeated in every `metadata.json` and every per-instance `README.md`.

---

## Quick start (under 10 minutes)

```bash
git clone https://github.com/1NightCrawler2/SteinerMineBench.git
cd SteinerMineBench
pip install -r requirements.txt        # numpy, scipy, jsonschema

python score.py                        # see the frozen references
python examples/minimal_solver.py --only scale-130k zones-04 > sub.json
python score.py --submission sub.json  # score it
```

The whole repository is about 10 MB. You do **not** need the reference solver, a GPU, or any
mining software — only numpy and scipy.

### Loading an instance

```python
from loader import load_instance

inst = load_instance("xgrid-f2x-qpoor")

inst["cost_grid"]      # float32 (126, 82, 100), support cost in $/m; 1e9 = not excavatable
inst["passable"]       # bool, True where the voxel can be excavated
inst["fault_count"]    # uint8, distinct faults intersecting each voxel
inst["portal_voxels"]  # list of (i, j, k) — the root terminal
inst["zone_voxels"]    # list of lists of (i, j, k) — one per production zone
inst["metadata"]       # grid geometry, cost model, faults, zones, generation seed
inst["reference"]      # reference_type, reference_cost, lower_bound, per-family results
```

`list_instances()` enumerates them; `world_from_voxel(ijk, metadata)` converts to metres.

---

## The problem

Find a minimum-cost network that connects a surface **portal** to every underground **production
zone**, on a 26-connected voxel graph, subject to **monotone descent** — a ramp may never go up and
back down.

**Objective.** Total cost of the network, with edge weight

```
w(u, v) = (cost_grid[v] + excavation_rate_per_m) · ‖v − u‖ · cell_size_m
```

`excavation_rate_per_m` is $1,000/m and is deliberately **not** baked into the grid, so the shipped
array is pure geotechnical support cost. The exact formula is repeated in
`metadata["cost_model"]` on every instance — read it from there rather than hard-coding.

**Cost model.** Support cost is a step function of the Barton Q rock mass quality index, using the
six-tier NGI schedule:

| Q | $/m | Support class |
|---|---|---|
| ≤ 0.04 | 7,585.4 | Exceptionally poor — steel sets + shotcrete + spiling |
| ≤ 0.10 | 5,679.5 | Extremely poor — heavy systematic bolt + mesh + shotcrete |
| ≤ 0.34 | 4,460.9 | Very poor — systematic bolt + mesh + shotcrete |
| ≤ 1.181 | 3,059.0 | Poor — systematic bolting + shotcrete |
| ≤ 6.0 | 1,806.9 | Fair — spot bolting |
| > 6.0 | 1,111.9 | Good — unsupported / spot bolting |

Fault intercepts impose a **floor**, not a multiplier, because a fault destabilises the ground
independently of the host rock: `cost = max(tier_cost, floor)`, with the floor $3,059.0/m for one
distinct fault and $4,460.9/m for two or more.

**Conventions.** Axis order is `(EAST, NORTH, RL)`; **RL is positive upward**;
`world_m = min_coords_m + (ijk + 0.5) · cell_size_m`; impassable voxels (above ground surface) carry
`1e9`; test passability with `cost < 0.9e9` or just use `inst["passable"]`.

---

## The 24 instances

Every group holds all other parameters fixed and varies **exactly one axis**, so a difference in
solver behaviour is attributable. This is enforced structurally in `steinerbench/spec.py`: each
instance is built as `{**group_base, axis_key: value}`.

| Group | n | Varied axis | Instances |
|---|---|---|---|
| **A** Crossing grid | 12 | fault system × Q regime | `xgrid-{f0,f1,f2x,fcj}-{qcomp,qmix,qpoor}` |
| **B** Portal sweep | 4 | portal location | `portal-{south,north,east,west}` |
| **C** Zone count | 4 | number of production zones | `zones-{02,04,06,08}` |
| **D** Scale ladder | 4 | voxel resolution | `scale-{130k,1m,8m,129m}` |

Fault systems are `none`, one through-going fault, two crossing faults (producing a multi-intercept
core where the higher floor applies), and a conjugate pair converging into a wedge with depth.
Q regimes are competent (median Q ≈ 8), mixed (bimodal), and poor-dominated (median Q ≈ 0.15).

Shared geometry across all instances: a 630 × 410 × 500 m domain and an analytic ridge topography,
with two structures that make routing non-trivial.

**A breached barrier.** A low-Q slab spans the whole plan at RL 340–400, between the surface and
the orebody, so every decline must cross it. Two **competent windows** breach it — at EAST
235–275 m and 350–395 m, confined to the slab's RL band. Driving through the intact lid costs
2.0–3.1× more per metre than going through a window (measured on the shipped grids), so a solver
must weigh a lateral detour against a more expensive descent. The windows straddle the north/south
portal at roughly equal offset, so which one wins is decided by the orebody layout and the faults;
the east and west portals sit close enough to one window to settle it outright.

> **Not the solver's `--corridor`.** These windows are a geological feature of the synthetic rock
> mass. They are unrelated to the MineOptimizer solver's `--corridor` search mask — a tube of
> `--corridor-radius` metres around WP2 A\* paths, used to shrink the Dijkstra graph. The benchmark
> never enables that mask; reference runs search the full grid.

**Fault damage.** Fault halos raise the cost floor to $3,059/m for one intercept and $4,461/m where
two cross. This is the strongest single driver of route shape: on the two-crossing instances,
14.3% of the passable domain is faulted but only ~1% of network length runs on a fault.

### The scale ladder is a genuine controlled experiment

The Q field is drawn once on a **fixed 10 m world-space lattice** and trilinearly interpolated to
voxel centres; all structural features are closed-form predicates with metre-valued parameters. So
all four rungs sample *the same geology* and differ only in discretisation.

| Instance | Cell | Dims | Voxels |
|---|---|---|---|
| `scale-130k` | 10 m | 63 × 41 × 50 | 129,150 |
| `scale-1m` | 5 m | 126 × 82 × 100 | 1,033,200 |
| `scale-8m` | 2.5 m | 252 × 164 × 200 | 8,265,600 |
| `scale-129m` | 1 m | 630 × 410 × 500 | 129,150,000 |

Two consequences worth knowing:

- A per-voxel RNG, or a fault halo specified in *voxels* rather than metres, would silently change
  the underlying problem between rungs. This benchmark therefore **departs from the reference
  solver's production pipeline**, which widens the fault cost floor with a 3×3×3 voxel dilation
  (30 m wide at a 10 m cell, 3 m at a 1 m cell). Here the damage halo is specified in metres and no
  voxel dilation is applied. `metadata["cost_model"]["fault_floor_rule"]` states this verbatim.
- **`scale-129m` ships with no reference solution.** A multi-source Dijkstra over 129 M voxels needs
  roughly 22 GB for the CSR graph alone, beyond the machine that built this suite. The grid,
  terminals and cost model are fully specified, so it stands as an open scaling frontier — the first
  valid solution submitted becomes its best-known bound.

### Topology family gating

The reference solver ranks seven topology families, four of which need a minimum number of
sublevels: `two_branch` and `chained_fan` need ≥ 2, `three_branch` and `hybrid_chained_fan_branch`
need ≥ 3. `zones-02` therefore exercises only 5 of the 7. This is a consequence of the varied axis,
not a defect; `metadata["topology_families"]["applicable"]` records it per instance, and `score.py`
only compares families present in both reference and submission.

---

## Reference solutions and what `exact` means

Each `reference.json` carries a `reference_type`:

**`exact`** — the cheapest family on this instance uses a closed-form argmin over *all* passable
voxels, and that argmin was **independently recomputed from scratch**. The verification
(`steinerbench/verify_exact.py`) shares no code and no distance fields with the reference solver: it
builds its own CSR graph, its own direction filters, its own `scipy.sparse.csgraph` Dijkstra and its
own argmin in float64, then asserts the junction voxels and the family cost agree. A mismatch aborts
the run rather than quietly downgrading the label. Re-running the solver's own argmin would prove
nothing — it would just agree with itself — which is why the check is built this way.

**`best_known`** — the cheapest family uses a greedy or coordinate-descent search
(`sequential_ramp`, `chained_fan`, `hybrid_chained_fan_branch`), so the cost is an upper bound that
may be improvable. Exact-search families on the same instance are still verified; see
`per_family[].exactness_check`.

**`unsolved`** — no reference computed (`scale-129m` only).

### The lower bound

Every solved instance carries a **provably valid** lower bound in `reference.lower_bound`, so
`gap_to_lower_bound` is a genuine upper bound on how far the reference could be from optimal.

The default method is the **pairwise-divergence bound**. In any arborescence the paths to two
terminals `z_i` and `z_j` share a prefix and separate at some node `v`, beyond which the two
branches are arc-disjoint. Hence for *every* pair

```
OPT  ≥  min_v [ d(portal, v) + d(v, z_i) + d(v, z_j) ]
```

and the maximum over pairs is valid. It costs one Dijkstra from the portal plus one reverse Dijkstra
per zone.

Three points that matter if you are reading the associated paper:

- The analogous all-terminals expression `min_v [d(portal,v) + Σ_k d(v,z_k)]` is **not** a lower
  bound. It forces every branch to diverge at a single node, which is the `single_junction`
  *feasible solution* and therefore an **upper** bound. Pairs are the largest subset for which the
  divergence argument is forced. Likewise, the sum of shortest descending paths is an upper bound
  (the union of K independent paths is feasible), not a lower one.
- **Validity across all seven families.** Each family produces a feasible portal-rooted subgraph over
  the descent arc set, so `reported_cost ≥ subgraph_cost ≥ OPT_arborescence ≥ bound`. The first step
  is an equality for `chained_fan`, which counts shared ramp voxels once, and strict for
  `sublevel_fan`/`two_branch`/`three_branch`, which charge a shared upper ramp once per branch — so
  the bound holds a fortiori for the double-counting families.
- Arcs are priced at `(min(cost[u], cost[v]) + excavation_rate) · length`, a pointwise lower bound on
  both traversal conventions the reference solver uses, so the bound survives regardless of which
  direction a segment was charged in.

`steinerbench/lower_bound.py` also implements **Wong dual ascent** (the classical LP-dual bound for
directed Steiner arborescence), available via `solve_reference.py --dual-ascent`. It is not the
default because it is not competitive here: dual ascent grows its root component one
zero-reduced-cost arc at a time, and on a fine 26-connected lattice with millions of near-parallel
arcs each step is tiny. On `scale-130k` it reached 1.47e6 after 3,000 iterations and 55 s — still
below the *free* trivial bound of 1.94e6, with no terminal yet connected. It is designed for sparse
networks, not dense grid graphs. The pairwise bound gets 2.4e6 in 0.2 s.

---

## Scoring your solver

```
optimality_gap = (solver_cost − reference_cost) / reference_cost
```

Write a submission JSON (schema: `schemas/submission.schema.json`):

```json
{
  "solver": "MyDirectedSteiner v0.3",
  "authors": "A. Researcher",
  "url": "https://github.com/…",
  "hardware": "32-core EPYC 7543, 256 GB RAM",
  "results": {
    "xgrid-f2x-qpoor": {
      "cost": 2874112.5,
      "topology": "chained_fan",
      "junctions_voxel": [[61, 34, 63]],
      "runtime_s": 42.1
    }
  }
}
```

then

```bash
python score.py --submission mysolver.json              # raw track (normative)
python score.py --submission mysolver.json --track buildable
```

Omitted instances are reported as *unattempted*, not counted as failures — partial submissions are
fine.

### Two tracks

**`raw`** (default, normative) is the pure voxel Steiner cost defined above. It is solver-agnostic
and exactly reproducible from the bundle.

**`buildable`** is the cost after buildability post-processing — grade limited to 20%, turn radius
≥ 25 m, spirals inserted where a drive is too steep. The reference values come from the reference
solver's deterministic geometric post-processor (its time-budgeted lattice refinement is disabled,
because a wall-clock budget is not reproducible). This track exists so you can compare constructible
designs, but it depends on each solver's own post-processor, so treat it as informative rather than
definitive.

### Negative gaps

- On a **`best_known`** instance, a negative gap is a **new best-known bound** — exactly what this
  benchmark is for. See below.
- On an **`exact`** instance, a negative gap is an **error**: the reference is provably optimal for
  its family, so either the submission is infeasible (check monotone descent, 26-connectivity, and
  that every zone is actually reached) or the reference is wrong. `score.py` flags both loudly and
  never averages them silently into the summary.

---

## Submitting an improved bound

Improvements to `best_known` instances are welcome and expected — that is why the label exists.

Open a pull request containing your submission JSON with **`paths_voxel` included** for each
improved instance, so the network can be re-costed straight off the grid and verified. Please also
give the solver name/version and a link to source or a preprint. Verified improvements are merged
into the frozen references with attribution, and the benchmark version is bumped.

If you believe you have beaten an `exact` reference, please open an issue rather than a pull
request — a genuine counterexample is a benchmark bug and we want to fix it.

---

## Repository layout

```
instances/<id>/  cost_grid.npz  metadata.json  reference.json
                 reference_paths.npz  README.md
loader.py                what you import
score.py                 gap, leaderboard, results.csv
validate.py              bundle completeness and self-consistency
generate.py              rebuilds every instance from its seed
solve_reference.py       recomputes references (needs the reference solver)
make_tables.py           paper-ready CSV + LaTeX tables
schemas/                 JSON Schema for metadata, reference, submission
tables/instance_manifest.csv   machine-readable manifest of all 24 instances
examples/minimal_solver.py
steinerbench/            the package (tiers, spec, geology, grid, loader,
                         verify_exact, lower_bound, mineopt_adapter)
```

### Scope

The MineOptimizer pipeline has three stages, and this benchmark
**exercises stage 3 only**: it synthesises the stage-1 and stage-2 artefacts
from each bundle and calls the stage-3 optimiser, bypassing the ordinary
kriging of stage 1 and the A\* pathfinding of stage 2. That is deliberate — it
removes geostatistical and search variability so a reference run is exactly
reproducible — but it means the suite validates the **network optimiser**, not
the grade-shell extraction or the interpolation.

Note that stage 3 does **not** consume stage 2's A\* paths: those are read only
under the solver's `--corridor` search mask, which reference runs never enable.
What stage 3 takes from stage 2 is the terminal geometry alone — the portal and
production-zone voxel sets — and that is fully specified in every
`metadata.json` and reconstructed by the loader.

### `cost_grid.npz`

Stores the **post-fault-floor tier index** as uint8 rather than float costs, plus the tier schedule
and the surface RL. This is lossless — both fault floors are themselves tier costs, so applying the
floor maps a tier index onto another tier index — and it compresses a 129 M-voxel grid to about
5 MB, which is why no Git LFS or out-of-band download is needed anywhere in the suite. The loader
reconstructs float32 $/m for you.

---

## Reproducing and verifying

```bash
python validate.py            # completeness, checksums, schemas, re-costing, exactness
python validate.py --fast     # skips the two largest instances
python generate.py --check    # regenerate every grid, diff SHA-256 against the shipped set
python make_tables.py         # rebuild tables/
```

`validate.py` re-costs every stored reference path straight off the grid using the published edge
weight and asserts it matches `reference.json` — an independent check that the shipped numbers
correspond to real, connected networks on the shipped grids.

### Regenerating the references

Only needed if you want to rebuild the reference solutions themselves. Requires a checkout of the
MineOptimizer WP3 solver:

```bash
export MINEOPTIMIZER_ROOT=/path/to/MineOptimizer
python solve_reference.py --all --resume
```

Each instance runs in its own subprocess with `MINEOPT_SKIP_LOCAL=1`,
`MINEOPT_FLOOD_ENGINE=scipy` (true Dijkstra, not the iterative stencil with its 1e-3 tolerance),
`MINEOPT_FORCE_CPU=1` and `OPEX_IN_EDGE_WEIGHTS=False`, so the run is reproducible and the objective
stays pure geotechnical support cost. WP1 (kriging) and WP2 (A*) are bypassed entirely.

---

## Citing

Please cite the dataset:

> Hasozdemir, K. (2026). *SteinerMineBench: a benchmark suite for
> geotechnically-weighted Steiner trees on voxel grids* (Version 1.0.0)
> [Data set]. Zenodo.

Machine-readable metadata is in `CITATION.cff`, so GitHub's **Cite this
repository** button and most reference managers will pick it up directly. The
DOI is added there once the first release is archived.

The accompanying paper — *Geotechnically Weighted Steiner Trees on Voxel Grids:
Rock-Mass-Constrained Optimisation of Underground Ramp Networks for Minimum
Excavation and Support Cost* — is in preparation. Once it is published, please
cite both.

## Licence

- **Data** (`instances/`, `tables/`) — CC-BY-4.0, see `LICENSE`
- **Code** (everything else) — MIT, see `LICENSE-CODE`
