"""Parameter optimisation: find the reconstruction settings that fit the
reference (ground-truth) mesh best.

Objective (``OBJECTIVES``) - always two-sided, so the optimiser cannot
"win" by dropping the difficult parts of the surface:

* ``fscore``   maximise the F-score @ 1 % of the diagonal (precision and
               recall together) - the default;
* ``chamfer``  minimise the Chamfer distance (accuracy + completeness);
* ``accuracy`` minimise the RMS distance reconstruction -> reference
               (the one-sided error of the Compare tab), only meaningful
               with the minimum-recall constraint below.

Constraint: runs whose recall (share of the reference covered within τ)
is below ``min_recall`` are not eligible as "best" (they are kept in the
table, marked).

Search (``optimize_algorithm``) - coordinate descent, no extra library:

1. a coarse pass: each parameter in turn, over a short list of sensible
   values (the most influential parameter first), the others held at the
   best values so far;
2. refinement passes around the best value (integers ±1, real values
   ×0.8 / ×1.25 and midpoints to the nearest tried neighbours) until
   nothing improves or the run budget is used.

Every configuration is built and evaluated once (cached); preprocessing
is shared through ``cached_preprocess``.  Optionally the best and the
starting settings are re-run on a noisy copy of the input, to show
whether the gain carries over to other data (a guard against tuning to
one data set).
"""

import time

from modules import isolated
from modules.degradation import Degradation
from modules.poisson_reconstruction import (
    ALPHA_SHAPE,
    BALL_PIVOTING,
    DEFAULT_FACTORS,
    POISSON,
    RIMLS,
    SIGNED_DISTANCE,
    build_mesh,
)

OBJECTIVES = {
    "fscore": "F-score (higher is better)",
    "chamfer": "Chamfer distance (lower is better)",
    "accuracy": "Accuracy RMS, reconstruction → reference (lower is better)",
}

# (parameter, kind, coarse values, (min, max)) - most influential first
SMOOTH = ("smooth", "int", (0, 3, 8), (0, 30))
SPACES = {
    POISSON: (
        ("depth", "int", (6, 7, 8, 9, 10), (5, 11)),
        ("density_q", "float", (0.0, 0.01, 0.03, 0.08), (0.0, 0.3)),
        SMOOTH,
        ("scale", "float", (1.05, 1.1, 1.3), (1.0, 2.0)),
    ),
    BALL_PIVOTING: (("factor", "float", (1.0, 1.5, 2.0, 3.0, 4.0), (0.5, 8.0)),
                    SMOOTH),
    ALPHA_SHAPE: (("factor", "float", (5.0, 8.0, 12.0, 18.0), (2.0, 40.0)),
                  SMOOTH),
    RIMLS: (("factor", "float", (3.0, 4.5, 6.0, 8.0), (2.0, 14.0)), SMOOTH),
    SIGNED_DISTANCE: (("factor", "float", (3.0, 4.0, 5.0, 7.0), (2.0, 14.0)),
                      SMOOTH),
}
MAX_PASSES = 4
VALIDATION_NOISE_PCT = 0.25
# A run is stopped when it takes far longer than the slowest one so far
# (Ball Pivoting with a large radius on noisy data ran for > 30 min - the
# window froze and looked crashed); the first run of each algorithm gets
# FIRST_RUN_LIMIT.
FIRST_RUN_LIMIT = 600.0
RUN_LIMIT_FACTOR = 6.0
MIN_RUN_LIMIT = 90.0


def run_limit(trials):
    """Time limit (s) for the next run, from the runs that worked."""
    times = [t["time"] for t in trials if "metrics" in t]
    if not times:
        return FIRST_RUN_LIMIT
    return max(MIN_RUN_LIMIT, RUN_LIMIT_FACTOR * max(times))


def objective_value(metrics, objective):
    """Bigger is better."""
    if objective == "chamfer":
        return -metrics["chamfer_pct"]
    if objective == "accuracy":
        return -metrics["accuracy"]["rms"]
    return metrics["f_score"]


def score(trial, objective, min_recall):
    """Sort key: eligible runs first, then the objective, then Chamfer."""
    if "metrics" not in trial:
        return (False, float("-inf"), float("-inf"))
    m = trial["metrics"]
    return (m["recall"] >= min_recall, objective_value(m, objective),
            -m["chamfer_pct"])


def summarise(metrics):
    """The numbers a trial keeps (the mesh itself is dropped)."""
    return {
        "f_score": metrics["f_score"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "chamfer_pct": metrics["chamfer_pct"],
        "hausdorff_95_pct": metrics["hausdorff_95_pct"],
        "accuracy": {"rms": metrics["accuracy"]["rms"],
                     "mean": metrics["accuracy"]["mean"]},
        "completeness": {"rms": metrics["completeness"]["rms"],
                         "mean": metrics["completeness"]["mean"]},
        "components": metrics.get("components"),
        "triangles": metrics.get("triangles"),
        "reference": metrics.get("reference"),
    }


def describe_values(algorithm, values):
    parts = []
    for name, value in values.items():
        if name == "density_q":
            parts.append(f"density {100 * value:g}%")
        elif name == "factor":
            parts.append(f"×{value:g}")
        else:
            parts.append(f"{name} {value:g}")
    return ", ".join(parts)


def _neighbours(spec, best, tried):
    """Refinement candidates around ``best``."""
    name, kind, _, (low, high) = spec
    if kind == "int":
        values = {best - 1, best + 1}
    else:
        values = {round(best * 0.8, 4), round(best * 1.25, 4)} if best else {0.005}
        lower = [v for v in tried if v < best]
        upper = [v for v in tried if v > best]
        if lower:
            values.add(round((best + max(lower)) / 2, 4))
        if upper:
            values.add(round((best + min(upper)) / 2, 4))
    return sorted(v for v in values if low <= v <= high and v not in tried)


def optimize_algorithm(points, normals, reference, base_params, algorithm,
                       budget, objective, min_recall, cancel=None,
                       progress=None, counter=None):
    """Coordinate descent for one algorithm -> {"trials", "best", "initial"}.

    ``counter`` ([done, total]) is shared by several algorithms so the
    progress covers the whole optimisation.
    """
    space = SPACES[algorithm]
    params = dict(base_params, algorithm=algorithm, width=0.0)
    start = {}
    for name, kind, coarse, (low, high) in space:
        if name == "factor":
            value = base_params.get("factor") or DEFAULT_FACTORS[algorithm]
        else:
            value = base_params.get(name, coarse[len(coarse) // 2])
        value = min(max(value, low), high)
        start[name] = int(round(value)) if kind == "int" else float(value)

    trials, cache = [], {}
    counter = counter if counter is not None else [0, budget]

    def run(values):
        key = tuple(sorted(values.items()))
        if key in cache:
            return cache[key]
        if len(trials) >= budget or (cancel and cancel.is_set()):
            return None
        started = time.perf_counter()
        trial = {"algorithm": algorithm, "values": dict(values)}
        try:
            result = build_mesh(dict(params, **values), points, normals,
                                reference, run_evaluation=True,
                                timeout=run_limit(trials), cancel=cancel,
                                keep_mesh=False)
            trial["metrics"] = summarise(result["metrics"])
            trial["setting"] = result["setting"]
        except isolated.Stopped:                    # cancelled: not a result
            return None
        except Exception as e:                      # e.g. an empty mesh, too slow
            trial["error"] = str(e)
            trial["setting"] = describe_values(algorithm, values)
        trial["time"] = time.perf_counter() - started
        trial["eligible"] = ("metrics" in trial
                             and trial["metrics"]["recall"] >= min_recall)
        trials.append(trial)
        cache[key] = trial
        counter[0] += 1
        trial["run"] = counter[0]
        if progress is not None:
            best_now = max(trials, key=lambda t: score(t, objective, min_recall))
            progress({"kind": "optimize", "trial": trial, "best": best_now,
                      "done": counter[0], "total": counter[1]})
        return trial

    best_values = dict(start)
    initial = run(best_values)
    best = initial

    def better(trial):
        return trial is not None and score(trial, objective, min_recall) > \
            score(best, objective, min_recall)

    for number in range(MAX_PASSES):
        improved = False
        for spec in space:
            name = spec[0]
            tried = {t["values"][name] for t in trials
                     if all(t["values"][k] == v for k, v in best_values.items()
                            if k != name)}
            candidates = (spec[2] if number == 0
                          else _neighbours(spec, best_values[name], tried))
            for value in candidates:
                values = dict(best_values, **{name: value})
                trial = run(values)
                if better(trial):
                    best, best_values, improved = trial, values, True
        if not improved or len(trials) >= budget or (cancel and cancel.is_set()):
            break
    return {"algorithm": algorithm, "trials": trials, "best": best,
            "initial": initial}


def validate(points, normals, reference, base_params, results, noise_pct,
             cancel=None):
    """Re-run the starting and the best settings on a noisy copy."""
    noisy_points, noisy_normals = Degradation(
        noise_pct=noise_pct, seed=1).apply(points, normals)
    checks = {}
    for result in results:
        algorithm = result["algorithm"]
        row = {}
        for label in ("initial", "best"):
            trial = result[label]
            if trial is None or "metrics" not in trial:
                continue
            params = dict(base_params, algorithm=algorithm, width=0.0,
                          **trial["values"])
            if cancel is not None and cancel.is_set():
                break
            try:
                built = build_mesh(params, noisy_points, noisy_normals,
                                   reference, run_evaluation=True,
                                   timeout=run_limit(result["trials"]),
                                   cancel=cancel, keep_mesh=False)
                row[label] = summarise(built["metrics"])
            except Exception as e:
                row[label] = {"error": str(e)}
        checks[algorithm] = row
    return {"noise_pct": noise_pct, "results": checks}


def run_optimization(points, normals, reference, base_params, algorithms,
                     budget, objective="fscore", min_recall=0.0,
                     validate_noise=0.0, cancel=None, progress=None):
    """Optimise each algorithm (``budget`` runs each) and pick the overall
    best.  Returns {"results": [...], "best": trial, "validation": ...}."""
    counter = [0, budget * len(algorithms)]
    results = []
    for algorithm in algorithms:
        if cancel and cancel.is_set():
            break
        results.append(optimize_algorithm(
            points, normals, reference, base_params, algorithm, budget,
            objective, min_recall, cancel, progress, counter))
    bests = [r["best"] for r in results if r["best"] is not None]
    best = max(bests, key=lambda t: score(t, objective, min_recall)) if bests else None
    validation = None
    if validate_noise > 0 and results and not (cancel and cancel.is_set()):
        validation = validate(points, normals, reference, base_params,
                              results, validate_noise, cancel)
    return {"results": results, "best": best, "objective": objective,
            "min_recall": min_recall, "validation": validation,
            "cancelled": bool(cancel and cancel.is_set())}
