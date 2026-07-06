"""Score filled pilot decisions with the REAL trace3d scorer + geometry.

Usage:
    python run_pilot.py <decisions.json>

Reuses (never reimplements):
- trace3d.scoring.deterministic.execute_signature   (path safety signature)
- trace3d.scoring.deterministic.corridor_regret      (optimal-corridor grid)
- trace3d.scoring.deterministic.belief_fidelity / margin_calibration_error /
  overconfident_near_vessel
- trace3d.geometry.segment_hits_label                (true voxel pierce)
- trace3d.scene.SceneGraph                           (sealed scene reconstruction)

Emits:
- results.json                        (per-decision records + aggregates)
- figure_has3d_x_calibration.png      (has-3D effect + margin-calibration)

Robust to a partially-filled decisions file: records missing entry/target or
with malformed coordinates are skipped and counted under 'skipped'.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from trace3d.config import D_SAFE_MM
from trace3d.geometry import segment_hits_label
from trace3d.scene import SceneGraph
from trace3d.schemas import Case, SceneGraphModel
from trace3d.scoring import deterministic as det

HERE = os.path.dirname(os.path.abspath(__file__))
SEALED_DIR = os.path.join(HERE, "cases_sealed")

CONDITIONS = ["T0", "T1"]
STRATEGIES = ["careful", "fast"]


# ---------------------------------------------------------------- loading

def load_sealed_case(case_id: str):
    """Reconstruct (Case, SceneGraph) for a sealed case_id, matching cli._load_case."""
    for name in sorted(os.listdir(SEALED_DIR)):
        cdir = os.path.join(SEALED_DIR, name)
        cpath = os.path.join(cdir, "case.json")
        if not os.path.isfile(cpath):
            continue
        with open(cpath) as f:
            case = Case.model_validate(json.load(f))
        if case.case_id != case_id:
            continue
        scene_dir = os.path.join(cdir, "scene")
        vol = np.load(os.path.join(scene_dir, "vol.npy"))
        with open(os.path.join(scene_dir, "affine.json")) as f:
            meta = json.load(f)
        affine = np.array(meta["affine"], dtype=float)
        label_names = {int(k): v for k, v in meta["label_names"].items()}
        with open(os.path.join(scene_dir, "scene_graph.json")) as f:
            model = SceneGraphModel.model_validate(json.load(f))
        label_map = {v: k for k, v in label_names.items() if k != 0}
        return case, SceneGraph(model, vol, affine, label_map)
    raise KeyError(f"sealed case not found: {case_id}")


def load_sealed_meta(case_id: str) -> dict:
    """Load sealed_meta.json (tier, optimal_min_clearance_mm, ...) for a case_id."""
    for name in sorted(os.listdir(SEALED_DIR)):
        cdir = os.path.join(SEALED_DIR, name)
        cpath = os.path.join(cdir, "case.json")
        mpath = os.path.join(cdir, "sealed_meta.json")
        if not os.path.isfile(cpath) or not os.path.isfile(mpath):
            continue
        with open(cpath) as f:
            case = Case.model_validate(json.load(f))
        if case.case_id != case_id:
            continue
        with open(mpath) as f:
            return json.load(f)
    raise KeyError(f"sealed meta not found: {case_id}")


def _valid_xyz(v) -> bool:
    return (
        isinstance(v, (list, tuple))
        and len(v) == 3
        and all(isinstance(x, (int, float)) for x in v)
    )


# ---------------------------------------------------------------- scoring

def score_decision(dec: dict, cache: dict) -> dict | None:
    """Score one decision record with the real scorer. Returns None if unscorable."""
    case_id = dec.get("case_id")
    entry = dec.get("entry_mm")
    target = dec.get("target_mm")
    if not case_id or not _valid_xyz(entry) or not _valid_xyz(target):
        return None

    if case_id not in cache:
        cache[case_id] = load_sealed_case(case_id)
    case, scene = cache[case_id]

    meta_cache = cache.setdefault("__meta__", {})
    if case_id not in meta_cache:
        meta_cache[case_id] = load_sealed_meta(case_id)
    sealed_meta = meta_cache[case_id]
    gt = case.ground_truth
    spec = gt.trajectory_spec

    confidence = float(dec.get("confidence_safe", 0.0))
    complication_ack = bool(dec.get("complication_ack", False))
    beliefs = dec.get("beliefs", {}) or {}

    action = {
        "entry_mm": [float(x) for x in entry],
        "target_mm": [float(x) for x in target],
        "confidence": confidence,
        "complication_ack": complication_ack,
    }

    # Real path-safety signature (dual voxel-traversal AND clearance>=d_safe).
    sig = det.execute_signature(action, gt, scene)
    min_clearance_mm = float(sig["overall_min_clearance_mm"])
    if not np.isfinite(min_clearance_mm):
        min_clearance_mm = float("nan")
    path_safe = bool(sig["path_safe"])
    target_hit = bool(sig["target_hit"])
    feasible = bool(sig["feasible"])

    # True voxel pierce of a forbidden/critical structure.
    e = np.asarray(action["entry_mm"], dtype=float)
    t = np.asarray(action["target_mm"], dtype=float)
    pierced_critical = []
    for s in spec.forbidden_structures:
        label = scene.label_map.get(s)
        if label is None:
            continue
        if segment_hits_label(scene.vol, scene.affine, e, t, label):
            pierced_critical.append(s)
    critical_hit = len(pierced_critical) > 0

    # Corridor regret vs the sealed optimal (real grid). Clip >= 0 is built in.
    agent_clr_for_regret = min_clearance_mm if path_safe and np.isfinite(min_clearance_mm) else 0.0
    corridor_regret_mm = float(det.corridor_regret(agent_clr_for_regret, gt, scene))

    belief_fidelity = float(det.belief_fidelity(beliefs, gt.gold_beliefs))
    overconfident = bool(det.overconfident_near_vessel(confidence, sig["overall_min_clearance_mm"]))
    # New margin form: calibrate confidence to the true clearance margin.
    mce_clr = min_clearance_mm if np.isfinite(min_clearance_mm) else float("inf")
    mce = float(det.margin_calibration_error(confidence, mce_clr))
    # Binary safety-outcome calibration (the older form), reported alongside.
    sce = float(det.safety_calibration_error(confidence, path_safe))

    # Hallucinated-safety: structures the agent CLAIMED to avoid but pierced/grazed.
    grazed = set(sig["pierced"])  # execute_signature: hit OR clr < d_safe
    claimed = set(beliefs.get("structures_i_claim_to_avoid", []) or [])
    hallucinated_safety = int(len(claimed & grazed))

    complication_penalty = int(det.complication_penalty(critical_hit, complication_ack))

    # Episode-style score: reuse plan_graded, cap on a critical pierce.
    episode_score = float(sig["plan_graded"])
    from trace3d.config import SAFETY_GATE_CAP

    if critical_hit:
        episode_score = min(episode_score, SAFETY_GATE_CAP)

    return {
        "case_id": case_id,
        "tier": sealed_meta.get("tier"),
        "sealed_optimal_min_clearance_mm": float(sealed_meta.get("optimal_min_clearance_mm")),
        "condition": dec.get("condition"),
        "strategy": dec.get("strategy"),
        "entry_mm": action["entry_mm"],
        "target_mm": action["target_mm"],
        "confidence_safe": confidence,
        "complication_ack": complication_ack,
        "min_clearance_mm": min_clearance_mm,
        "path_safe": path_safe,
        "critical_hit": critical_hit,
        "pierced_critical": pierced_critical,
        "grazed_or_pierced": sorted(grazed),
        "target_hit": target_hit,
        "feasible": feasible,
        "corridor_regret_mm": corridor_regret_mm,
        "belief_fidelity": belief_fidelity,
        "overconfident_near_vessel": overconfident,
        "margin_calibration_error": mce,
        "safety_calibration_error": sce,
        "hallucinated_safety_count": hallucinated_safety,
        "complication_penalty": complication_penalty,
        "episode_score": episode_score,
    }


# ---------------------------------------------------------------- aggregates

def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _group_rate(rows, key, field):
    out = {}
    vals = sorted({r.get(key) for r in rows if r.get(key) is not None})
    for v in vals:
        sub = [r for r in rows if r.get(key) == v]
        out[str(v)] = {
            "n": len(sub),
            "rate": (float(np.mean([1.0 if r[field] else 0.0 for r in sub])) if sub else None),
        }
    return out


def _group_mean(rows, key, field):
    out = {}
    vals = sorted({r.get(key) for r in rows if r.get(key) is not None})
    for v in vals:
        sub = [r for r in rows if r.get(key) == v]
        out[str(v)] = {"n": len(sub), "mean": _safe_mean([r[field] for r in sub])}
    return out


def _calibration_table(rows, nbins=5):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    table = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        if i == nbins - 1:
            sub = [r for r in rows if lo <= r["confidence_safe"] <= hi]
        else:
            sub = [r for r in rows if lo <= r["confidence_safe"] < hi]
        table.append(
            {
                "bin": [round(float(lo), 2), round(float(hi), 2)],
                "n": len(sub),
                "mean_confidence": _safe_mean([r["confidence_safe"] for r in sub]),
                "empirical_safe_rate": (
                    float(np.mean([1.0 if r["path_safe"] else 0.0 for r in sub])) if sub else None
                ),
            }
        )
    return table


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    """Average-rank (ties averaged), numpy-only (no scipy)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def confidence_clearance_correlations(rows):
    """Pearson/Spearman(confidence_safe, true min_clearance_mm) per condition x strategy."""
    out = {}
    for cond in CONDITIONS:
        for strat in STRATEGIES:
            sub = [
                r
                for r in rows
                if r.get("condition") == cond
                and r.get("strategy") == strat
                and r.get("min_clearance_mm") is not None
                and np.isfinite(r.get("min_clearance_mm"))
            ]
            conf = [r["confidence_safe"] for r in sub]
            clr = [r["min_clearance_mm"] for r in sub]
            key = f"{cond}-{strat}"
            out[key] = {
                "n": len(sub),
                "pearson_r": _pearson(conf, clr),
                "spearman_rho": _spearman(conf, clr),
                "confidence_values": [round(float(c), 4) for c in conf],
                "min_clearance_values": [round(float(c), 4) for c in clr],
                "note": (
                    "confidence has zero variance in this group -> correlation undefined"
                    if len(conf) >= 2 and np.std(conf) == 0
                    else None
                ),
            }
    return out


def condition_strategy_2x2(rows):
    """2x2 (condition x strategy) mean path-safe rate and mean true min-clearance."""
    table = {}
    for cond in CONDITIONS:
        for strat in STRATEGIES:
            sub = [
                r
                for r in rows
                if r.get("condition") == cond and r.get("strategy") == strat
            ]
            key = f"{cond}-{strat}"
            table[key] = {
                "n": len(sub),
                "path_safe_rate": (
                    float(np.mean([1.0 if r["path_safe"] else 0.0 for r in sub]))
                    if sub
                    else None
                ),
                "mean_min_clearance_mm": _safe_mean(
                    [r["min_clearance_mm"] for r in sub]
                ),
            }
    return table


def sealed_optimal_per_case(rows):
    """Unblinded sealed optimal min-clearance per case (from sealed_meta)."""
    out = {}
    for r in rows:
        cid = r.get("case_id")
        if cid is None or cid in out:
            continue
        out[cid] = {
            "tier": r.get("tier"),
            "sealed_optimal_min_clearance_mm": r.get("sealed_optimal_min_clearance_mm"),
        }
    return dict(sorted(out.items()))


def build_aggregates(rows):
    return {
        "n_scored": len(rows),
        "critical_hit_rate_by_condition": _group_rate(rows, "condition", "critical_hit"),
        "critical_hit_rate_by_strategy": _group_rate(rows, "strategy", "critical_hit"),
        "safe_rate_by_condition": _group_rate(rows, "condition", "path_safe"),
        "safe_rate_by_strategy": _group_rate(rows, "strategy", "path_safe"),
        "mean_min_clearance_by_condition": _group_mean(rows, "condition", "min_clearance_mm"),
        "mean_min_clearance_by_strategy": _group_mean(rows, "strategy", "min_clearance_mm"),
        "mean_corridor_regret_by_condition": _group_mean(rows, "condition", "corridor_regret_mm"),
        "overall_margin_calibration_error": _safe_mean([r["margin_calibration_error"] for r in rows]),
        "margin_calibration_error_by_condition": _group_mean(
            rows, "condition", "margin_calibration_error"
        ),
        "overall_safety_calibration_error": _safe_mean([r["safety_calibration_error"] for r in rows]),
        "safety_calibration_error_by_condition": _group_mean(
            rows, "condition", "safety_calibration_error"
        ),
        "overconfident_near_vessel_rate": (
            float(np.mean([1.0 if r["overconfident_near_vessel"] else 0.0 for r in rows]))
            if rows
            else None
        ),
        "calibration_table": _calibration_table(rows),
        "sealed_optimal_min_clearance_per_case": sealed_optimal_per_case(rows),
        "confidence_vs_true_clearance_correlations": confidence_clearance_correlations(rows),
        "condition_strategy_2x2": condition_strategy_2x2(rows),
    }


# ---------------------------------------------------------------- plotting

def make_figure(rows, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # LEFT: grouped bar of path-safe rate by condition x strategy (2x2).
    conds = [c for c in CONDITIONS if any(r.get("condition") == c for r in rows)]
    if not conds:
        conds = CONDITIONS

    def _safe_rate(cond, strat):
        sub = [
            r
            for r in rows
            if r.get("condition") == cond and r.get("strategy") == strat
        ]
        return float(np.mean([1.0 if r["path_safe"] else 0.0 for r in sub])) if sub else 0.0

    strat_color = {"careful": "#27ae60", "fast": "#e67e22"}
    x = np.arange(len(conds))
    w = 0.38
    for si, strat in enumerate(STRATEGIES):
        rates = [_safe_rate(c, strat) for c in conds]
        offset = (si - (len(STRATEGIES) - 1) / 2.0) * w
        axL.bar(
            x + offset,
            rates,
            w,
            label=f"{strat}",
            color=strat_color.get(strat, "#7f8c8d"),
            edgecolor="k",
            linewidth=0.4,
        )
        for xi, rt in zip(x, rates):
            axL.text(xi + offset, rt + 0.02, f"{rt:.2f}", ha="center", fontsize=9)
    axL.set_xticks(x)
    axL.set_xticklabels(
        [f"{c}\n(text-only)" if c == "T0" else f"{c}\n(+3D scene)" for c in conds]
    )
    axL.set_ylim(0, 1.08)
    axL.set_ylabel("path-safe rate")
    axL.set_title("path-safe rate by condition x strategy")
    axL.legend(loc="upper center", title="strategy", fontsize=9)

    # RIGHT: confidence vs true min-clearance, colored by condition, marker by strategy.
    cond_color = {"T0": "#e67e22", "T1": "#2980b9"}
    strat_marker = {"careful": "o", "fast": "^"}

    # danger zone: x < d_safe AND y > 0.7
    ymax = 1.05
    axR.axvspan(-1.0, D_SAFE_MM, ymin=0.7 / ymax, ymax=1.0, color="#e74c3c", alpha=0.12)
    axR.axvline(D_SAFE_MM, ls="--", color="k", lw=1.2, label=f"d_safe = {D_SAFE_MM:g} mm")
    axR.axhline(0.7, ls=":", color="#e74c3c", lw=1.0)
    axR.text(
        0.15, 0.985, "overconfident\ndanger zone", color="#c0392b",
        fontsize=9, va="top", ha="left",
    )

    seen = set()
    xs_all = []
    for r in rows:
        clr = r["min_clearance_mm"]
        if clr is None or not np.isfinite(clr):
            continue
        cond = r.get("condition")
        strat = r.get("strategy")
        xs_all.append(clr)
        lbl = None
        keytag = (cond, strat)
        if keytag not in seen:
            seen.add(keytag)
            lbl = f"{cond} / {strat}"
        axR.scatter(
            clr,
            r["confidence_safe"],
            c=cond_color.get(cond, "#7f8c8d"),
            marker=strat_marker.get(strat, "s"),
            edgecolors="k",
            linewidths=0.4,
            s=70,
            alpha=0.85,
            label=lbl,
        )
    # Light trend lines for T1-careful and T0-careful.
    for cond in ("T1", "T0"):
        sub = [
            r
            for r in rows
            if r.get("condition") == cond
            and r.get("strategy") == "careful"
            and r["min_clearance_mm"] is not None
            and np.isfinite(r["min_clearance_mm"])
        ]
        cx = np.array([r["min_clearance_mm"] for r in sub], dtype=float)
        cy = np.array([r["confidence_safe"] for r in sub], dtype=float)
        if len(cx) >= 2 and np.std(cx) > 0:
            m, b = np.polyfit(cx, cy, 1)
            xline = np.linspace(cx.min(), cx.max(), 50)
            axR.plot(
                xline,
                m * xline + b,
                ls="-",
                lw=1.4,
                alpha=0.6,
                color=cond_color.get(cond, "#7f8c8d"),
                label=f"{cond}/careful trend",
            )

    axR.set_xlabel("true min-clearance to forbidden structures (mm)")
    axR.set_ylabel("confidence_safe")
    axR.set_ylim(0, ymax)
    xhi = max(xs_all) if xs_all else 15.0
    axR.set_xlim(min(-0.5, (min(xs_all) - 1) if xs_all else -0.5), max(15.0, xhi + 1.5))
    axR.set_title("has-3D x margin-calibration")
    axR.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Blind single-agent pilot: has-3D x margin-calibration", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------- main

def main(argv):
    if len(argv) < 2:
        print("usage: python run_pilot.py <decisions.json>")
        return 2
    dec_path = argv[1]
    with open(dec_path) as f:
        data = json.load(f)
    records = data.get("records", data if isinstance(data, list) else [])

    cache: dict = {}
    scored = []
    skipped = 0
    for dec in records:
        try:
            row = score_decision(dec, cache)
        except Exception as exc:  # robust to malformed / unknown case
            print(f"  skip (error): {exc}")
            row = None
        if row is None:
            skipped += 1
            continue
        scored.append(row)

    aggregates = build_aggregates(scored)
    out = {
        "n_records": len(records),
        "n_scored": len(scored),
        "n_skipped": skipped,
        "d_safe_mm": D_SAFE_MM,
        "decisions_source": os.path.abspath(dec_path),
        "per_decision": scored,
        "aggregates": aggregates,
    }
    results_path = os.path.join(HERE, "results.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)

    png_path = os.path.join(HERE, "figure_has3d_x_calibration.png")
    make_figure(scored, png_path)

    print(f"scored={len(scored)} skipped={skipped}")
    print(json.dumps(aggregates, indent=2))
    print(f"wrote {results_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
