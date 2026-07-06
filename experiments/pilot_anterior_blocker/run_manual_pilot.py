"""Score the REAL agent-under-test pierce decisions with the REAL trace3d scorer.

Loads the 6 sealed anterior-blocker pierce cases (reusing run_agents._load_sealed_case)
and scores the 24 manual decisions in ``decisions_claude_pierce.json`` (6 cases x
{T0,T1} x {careful,fast}) with the real geometry + scoring stack. Nothing here
reimplements geometry or scoring; every number comes from:

- trace3d.scoring.deterministic.execute_signature      (path safety signature)
- trace3d.scoring.deterministic.corridor_regret         (optimal-corridor grid)
- trace3d.scoring.deterministic.belief_fidelity
- trace3d.scoring.deterministic.margin_calibration_error(confidence, clearance)
- trace3d.scoring.deterministic.safety_calibration_error(confidence, path_safe)
- trace3d.scoring.deterministic.overconfident_near_vessel
- trace3d.geometry.segment_hits_label                   (true voxel pierce)
- trace3d.scoring.aggregate.margin_confidence_correlation
- run_agents._load_sealed_case                          (sealed scene reconstruction)

The `fast` (naive straight anterior) decisions pierce the forbidden+critical
bowel_loop -> critical_hit, clearance 0, safety gate fires. The `careful` angled
decisions route around the anterior blocker and stay safe.

Emits:
- results_manual.json                (per-record rows + aggregates)
- figure_pierce_has3d.png            (crit-hit/safe-rate bars + confidence-vs-clearance)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_agents import _load_sealed_case  # reuse the sealed loader run_agents uses

from trace3d.config import D_SAFE_MM, SAFETY_GATE_CAP
from trace3d.geometry import segment_hits_label
from trace3d.scoring import deterministic as det
from trace3d.scoring.aggregate import margin_confidence_correlation

SEALED_DIR = os.path.join(HERE, "cases_sealed")
DECISIONS = os.path.join(HERE, "decisions_claude_pierce.json")

CONDITIONS = ["T0", "T1"]
STRATEGIES = ["careful", "fast"]


# ---------------------------------------------------------------- loading

def _build_case_index():
    """Map case_id -> (Case, SceneGraph, sealed_meta) for every sealed pierce case."""
    index = {}
    for name in sorted(os.listdir(SEALED_DIR)):
        cdir = os.path.join(SEALED_DIR, name)
        if not os.path.isfile(os.path.join(cdir, "case.json")):
            continue
        case, scene = _load_sealed_case(cdir)
        meta = {}
        mpath = os.path.join(cdir, "sealed_meta.json")
        if os.path.isfile(mpath):
            with open(mpath) as f:
                meta = json.load(f)
        index[case.case_id] = (case, scene, meta)
    return index


# ---------------------------------------------------------------- scoring

def score_decision(dec: dict, index: dict) -> dict:
    """Score one real decision with the real scorer + geometry."""
    case_id = dec["case_id"]
    case, scene, sealed_meta = index[case_id]
    gt = case.ground_truth
    spec = gt.trajectory_spec

    confidence = float(dec.get("confidence_safe", 0.0))
    complication_ack = bool(dec.get("complication_ack", False))
    beliefs = dec.get("beliefs", {}) or {}

    action = {
        "entry_mm": [float(x) for x in dec["entry_mm"]],
        "target_mm": [float(x) for x in dec["target_mm"]],
        "confidence": confidence,
        "complication_ack": complication_ack,
    }

    # Real path-safety signature (dual voxel-traversal AND clearance>=d_safe).
    sig = det.execute_signature(action, gt, scene)
    clr = sig["overall_min_clearance_mm"]
    min_clearance_mm = float(clr) if np.isfinite(clr) else float("nan")
    path_safe = bool(sig["path_safe"])
    target_hit = bool(sig["target_hit"])
    feasible = bool(sig["feasible"])

    # True voxel pierce of a forbidden AND critical structure.
    e = np.asarray(action["entry_mm"], dtype=float)
    t = np.asarray(action["target_mm"], dtype=float)
    critical_set = set(getattr(scene.model, "critical_structures", []) or [])
    pierced_critical = []
    for s in spec.forbidden_structures:
        if s not in critical_set:
            continue
        label = scene.label_map.get(s)
        if label is None:
            continue
        if segment_hits_label(scene.vol, scene.affine, e, t, label):
            pierced_critical.append(s)
    critical_hit = len(pierced_critical) > 0

    # Corridor regret vs the sealed optimal (real grid).
    agent_clr_for_regret = (
        min_clearance_mm if path_safe and np.isfinite(min_clearance_mm) else 0.0
    )
    corridor_regret_mm = float(det.corridor_regret(agent_clr_for_regret, gt, scene))

    belief_fidelity = float(det.belief_fidelity(beliefs, gt.gold_beliefs))
    overconfident = bool(det.overconfident_near_vessel(confidence, clr))
    mce_clr = min_clearance_mm if np.isfinite(min_clearance_mm) else float("inf")
    margin_cal_err = float(det.margin_calibration_error(confidence, mce_clr))
    safety_cal_err = float(det.safety_calibration_error(confidence, path_safe))

    # Hallucinated-safety: structures the agent CLAIMED to avoid but pierced/grazed.
    grazed = set(sig["pierced"])
    claimed = set(beliefs.get("structures_i_claim_to_avoid", []) or [])
    hallucinated_safety = int(len(claimed & grazed))

    # Episode score via the real stager path (plan_graded) with the safety gate.
    episode_score = float(sig["plan_graded"])
    if critical_hit:
        episode_score = min(episode_score, SAFETY_GATE_CAP)

    return {
        "case_id": case_id,
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
        "margin_calibration_error": margin_cal_err,
        "safety_calibration_error": safety_cal_err,
        "hallucinated_safety_count": hallucinated_safety,
        "episode_score": episode_score,
        "safety_violation": critical_hit,
        "sealed_optimal_min_clearance_mm": (
            float(sealed_meta.get("optimal_min_clearance_mm"))
            if sealed_meta.get("optimal_min_clearance_mm") is not None
            else None
        ),
    }


# ---------------------------------------------------------------- aggregates

def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _cell_rows(rows, cond, strat):
    return [
        r for r in rows if r.get("condition") == cond and r.get("strategy") == strat
    ]


def two_by_two_tables(rows):
    """condition x strategy 2x2 tables: crit-hit rate, safe rate, mean clr, mean score."""
    crit = {}
    safe = {}
    clr = {}
    score = {}
    for cond in CONDITIONS:
        for strat in STRATEGIES:
            sub = _cell_rows(rows, cond, strat)
            key = f"{cond}-{strat}"
            n = len(sub)
            crit[key] = {
                "n": n,
                "rate": (
                    float(np.mean([1.0 if r["critical_hit"] else 0.0 for r in sub]))
                    if n
                    else None
                ),
            }
            safe[key] = {
                "n": n,
                "rate": (
                    float(np.mean([1.0 if r["path_safe"] else 0.0 for r in sub]))
                    if n
                    else None
                ),
            }
            clr[key] = {
                "n": n,
                "mean": _safe_mean([r["min_clearance_mm"] for r in sub]),
            }
            score[key] = {
                "n": n,
                "mean": _safe_mean([r["episode_score"] for r in sub]),
            }
    return {
        "critical_hit_rate": crit,
        "path_safe_rate": safe,
        "mean_min_clearance_mm": clr,
        "mean_episode_score": score,
    }


def margin_confidence_correlations(rows):
    """margin_confidence_correlation over (confidence, min_clearance) per cond x strat."""
    out = {}
    for cond in CONDITIONS:
        for strat in STRATEGIES:
            sub = _cell_rows(rows, cond, strat)
            pairs = [
                (r["confidence_safe"], r["min_clearance_mm"])
                for r in sub
                if r["min_clearance_mm"] is not None
                and np.isfinite(r["min_clearance_mm"])
            ]
            key = f"{cond}-{strat}"
            out[key] = {
                "n": len(pairs),
                "margin_confidence_correlation": margin_confidence_correlation(pairs),
                "confidence_values": [round(float(c), 4) for c, _ in pairs],
                "min_clearance_values": [round(float(m), 4) for _, m in pairs],
            }
    return out


def sealed_optimal_per_case(rows):
    out = {}
    for r in rows:
        cid = r["case_id"]
        if cid not in out:
            out[cid] = r.get("sealed_optimal_min_clearance_mm")
    return dict(sorted(out.items()))


def build_aggregates(rows):
    return {
        "n_scored": len(rows),
        "two_by_two": two_by_two_tables(rows),
        "margin_confidence_correlations": margin_confidence_correlations(rows),
        "sealed_optimal_min_clearance_per_case": sealed_optimal_per_case(rows),
    }


# ---------------------------------------------------------------- plotting

def make_figure(rows, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- LEFT: grouped bars of critical-hit rate (primary) + path-safe rate.
    groups = [f"{c}\n{s}" for c in CONDITIONS for s in STRATEGIES]
    crit_rates = []
    safe_rates = []
    for c in CONDITIONS:
        for s in STRATEGIES:
            sub = _cell_rows(rows, c, s)
            crit_rates.append(
                float(np.mean([1.0 if r["critical_hit"] else 0.0 for r in sub]))
                if sub
                else 0.0
            )
            safe_rates.append(
                float(np.mean([1.0 if r["path_safe"] else 0.0 for r in sub]))
                if sub
                else 0.0
            )
    x = np.arange(len(groups))
    w = 0.38
    b1 = axL.bar(
        x - w / 2, crit_rates, w, label="critical-hit rate",
        color="#c0392b", edgecolor="k", linewidth=0.5,
    )
    b2 = axL.bar(
        x + w / 2, safe_rates, w, label="path-safe rate",
        color="#27ae60", edgecolor="k", linewidth=0.5,
    )
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            axL.text(
                rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
    axL.set_xticks(x)
    axL.set_xticklabels(groups)
    axL.set_ylim(0, 1.12)
    axL.set_ylabel("rate")
    axL.set_title("Real voxel-pierce evidence:\ncritical-hit vs path-safe rate by condition x strategy")
    axL.legend(loc="upper center", fontsize=9)
    axL.grid(axis="y", ls=":", alpha=0.4)

    # ---- RIGHT: confidence vs true min-clearance, color=condition, marker=strategy.
    cond_color = {"T0": "#e67e22", "T1": "#2980b9"}
    strat_marker = {"careful": "o", "fast": "^"}
    ymax = 1.05

    # shaded overconfident danger zone: x < d_safe AND y > 0.7
    axR.axvspan(-1.5, D_SAFE_MM, ymin=0.7 / ymax, ymax=1.0, color="#e74c3c", alpha=0.13)
    axR.axvline(D_SAFE_MM, ls="--", color="k", lw=1.3, label=f"d_safe = {D_SAFE_MM:g} mm")
    axR.axhline(0.7, ls=":", color="#e74c3c", lw=1.0)
    axR.text(
        -1.2, 1.02, "overconfident\ndanger zone", color="#c0392b",
        fontsize=9, va="top", ha="left", fontweight="bold",
    )

    # jitter identical points slightly so overlapping records stay visible.
    rng = np.random.default_rng(7)
    seen = set()
    xs_all = []
    for r in rows:
        c = r["min_clearance_mm"]
        if c is None or not np.isfinite(c):
            continue
        cond = r.get("condition")
        strat = r.get("strategy")
        xs_all.append(c)
        jx = c + rng.uniform(-0.12, 0.12)
        jy = r["confidence_safe"] + rng.uniform(-0.008, 0.008)
        lbl = None
        tag = (cond, strat)
        if tag not in seen:
            seen.add(tag)
            lbl = f"{cond} / {strat}"
        axR.scatter(
            jx, jy,
            c=cond_color.get(cond, "#7f8c8d"),
            marker=strat_marker.get(strat, "s"),
            edgecolors="k", linewidths=0.5, s=90, alpha=0.85, label=lbl,
        )

    axR.set_xlabel("true min-clearance to forbidden structures (mm)")
    axR.set_ylabel("confidence_safe")
    axR.set_ylim(0, ymax)
    xhi = max(xs_all) if xs_all else 12.0
    axR.set_xlim(-1.5, max(12.0, xhi + 1.5))
    axR.set_title("has-3D x margin-calibration\n(fast -> pierced clr~0 & conf 0.9 = danger zone)")
    axR.legend(loc="center right", fontsize=8, framealpha=0.9)
    axR.grid(ls=":", alpha=0.4)

    fig.suptitle(
        "Anterior-blocker pierce pilot (real agent-under-test, real scorer)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------- reporting

def _fmt(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, float) and not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def print_report(rows, aggregates):
    print("\n=== 24-row per-record table ===")
    hdr = (
        f"{'case':<18}{'cond':<5}{'strat':<8}{'min_clr':>8}{'safe':>6}"
        f"{'crit':>6}{'tgt':>5}{'feas':>6}{'conf':>6}{'overc':>7}{'score':>7}{'viol':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    order = {("T0", "careful"): 0, ("T0", "fast"): 1, ("T1", "careful"): 2, ("T1", "fast"): 3}
    for r in sorted(rows, key=lambda r: (order.get((r["condition"], r["strategy"]), 9), r["case_id"])):
        print(
            f"{r['case_id']:<18}{r['condition']:<5}{r['strategy']:<8}"
            f"{_fmt(r['min_clearance_mm']):>8}{str(r['path_safe']):>6}"
            f"{str(r['critical_hit']):>6}{str(r['target_hit']):>5}{str(r['feasible']):>6}"
            f"{_fmt(r['confidence_safe']):>6}{str(r['overconfident_near_vessel']):>7}"
            f"{_fmt(r['episode_score'],3):>7}{str(r['safety_violation']):>6}"
        )

    tb = aggregates["two_by_two"]

    def _print_2x2(title, table, nd=2):
        print(f"\n=== 2x2 {title} (condition x strategy) ===")
        print(f"{'':<10}{'careful':>12}{'fast':>12}")
        for cond in CONDITIONS:
            vals = []
            for strat in STRATEGIES:
                cell = table[f"{cond}-{strat}"]
                v = cell.get("rate", cell.get("mean"))
                vals.append(_fmt(v, nd))
            print(f"{cond:<10}{vals[0]:>12}{vals[1]:>12}")

    _print_2x2("critical-hit rate", tb["critical_hit_rate"])
    _print_2x2("path-safe rate", tb["path_safe_rate"])
    _print_2x2("mean min-clearance (mm)", tb["mean_min_clearance_mm"])
    _print_2x2("mean episode_score", tb["mean_episode_score"], nd=3)

    print("\n=== margin_confidence_correlation (confidence, min_clearance) per cond x strat ===")
    for key, v in aggregates["margin_confidence_correlations"].items():
        corr = v["margin_confidence_correlation"]
        corr_s = "None (undefined: zero variance / <2 pairs)" if corr is None else f"{corr:.4f}"
        print(f"  {key:<12} n={v['n']}  r = {corr_s}")

    print("\n=== sealed optimal min-clearance per case (unblinded) ===")
    for cid, v in aggregates["sealed_optimal_min_clearance_per_case"].items():
        print(f"  {cid:<20} {_fmt(v, 4)} mm")


# ---------------------------------------------------------------- main

def main():
    with open(DECISIONS) as f:
        records = json.load(f)["records"]

    index = _build_case_index()
    rows = [score_decision(dec, index) for dec in records]

    aggregates = build_aggregates(rows)
    out = {
        "n_records": len(records),
        "n_scored": len(rows),
        "d_safe_mm": D_SAFE_MM,
        "safety_gate_cap": SAFETY_GATE_CAP,
        "decisions_source": os.path.abspath(DECISIONS),
        "per_record": rows,
        "aggregates": aggregates,
    }
    results_path = os.path.join(HERE, "results_manual.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)

    png_path = os.path.join(HERE, "figure_pierce_has3d.png")
    make_figure(rows, png_path)

    print_report(rows, aggregates)
    print(f"\nwrote {results_path}")
    print(f"wrote {png_path}")

    # report figure pixel size
    try:
        from PIL import Image

        with Image.open(png_path) as im:
            print(f"figure size: {im.size[0]} x {im.size[1]} px")
    except Exception:
        import matplotlib.image as mpimg

        arr = mpimg.imread(png_path)
        print(f"figure size: {arr.shape[1]} x {arr.shape[0]} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
