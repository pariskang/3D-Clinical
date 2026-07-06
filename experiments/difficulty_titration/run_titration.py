"""Run graded agents across the difficulty continuum and fit a competence curve.

DELIVERABLE A (analysis). For every sealed titration scene and every graded
agent (``agents_graded.build_agents``) this:

1. loads the sealed scene + GroundTruth and rebuilds the real ``SceneGraph``;
2. VALIDATES the fast SDF clearance agrees with the exact scorer on a few paths;
3. scores ``path_safe`` with the REAL scorer ``deterministic.execute_signature``;
4. bins scenes by the sealed OPTIMAL corridor width ``w`` and computes
   P(safe | w-bin) per agent;
5. fits a 2-parameter logistic  P(safe | w) = 1 / (1 + exp(-b (w - w50)))  per
   agent by numpy MLE (ridge-stabilised IRLS), reporting the 50%-safe corridor
   width ``w50`` and slope ``b`` with bootstrap CIs (resampling scenes via
   ``trace3d.analysis.stats.bootstrap_ci``);
6. writes ``results_titration.json`` and ``figure_titration.png`` and prints the
   per-agent w50 table.

Everything is deterministic (all RNG seeded).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import agents_graded as ag  # noqa: E402

from trace3d import geometry_sdf as gsdf  # noqa: E402
from trace3d.analysis import bootstrap_ci  # noqa: E402
from trace3d.config import D_SAFE_MM  # noqa: E402
from trace3d.scene import SceneGraph  # noqa: E402
from trace3d.schemas import GroundTruth, SceneGraphModel  # noqa: E402
from trace3d.scoring import deterministic as det  # noqa: E402

SEALED_DIR = os.path.join(HERE, "titration_sealed")
BOOT_SEED = 20260705
N_BOOT = 2000


# --------------------------------------------------------------- loading

def load_manifest() -> dict:
    with open(os.path.join(SEALED_DIR, "manifest.json")) as f:
        return json.load(f)


def load_scene(dir_name: str):
    d = os.path.join(SEALED_DIR, dir_name)
    vol = np.load(os.path.join(d, "scene.npz"))["vol"]
    with open(os.path.join(d, "affine.json")) as f:
        meta = json.load(f)
    affine = np.array(meta["affine"], dtype=float)
    label_names = {int(k): v for k, v in meta["label_names"].items()}
    label_map = {v: k for k, v in label_names.items() if k != 0}
    with open(os.path.join(d, "scene_graph.json")) as f:
        model = SceneGraphModel.model_validate(json.load(f))
    with open(os.path.join(d, "gt.json")) as f:
        gt = GroundTruth.model_validate(json.load(f))
    with open(os.path.join(d, "sealed_meta.json")) as f:
        sealed = json.load(f)
    scene = SceneGraph(model, vol, affine, label_map)
    return scene, gt, sealed


def forbidden_field(scene, gt):
    mask = np.zeros(scene.vol.shape, dtype=bool)
    for s in gt.trajectory_spec.forbidden_structures:
        lab = scene.label_map.get(s)
        if lab is not None:
            mask |= scene.vol == lab
    return gsdf.distance_field_mm(mask, (1.0, 1.0, 1.0))


# --------------------------------------------------------- logistic MLE

def fit_logistic(w, y, lam: float = 1e-3, iters: int = 100):
    """Ridge-stabilised IRLS for  P(y=1|w) = sigmoid(beta0 + beta1 * w).

    Returns (beta0, beta1). A tiny L2 penalty ``lam`` keeps beta finite when the
    data are perfectly separable (e.g. an all-safe or all-unsafe agent), so the
    fit never diverges; with informative data its effect is negligible.
    """
    w = np.asarray(w, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    X = np.column_stack([np.ones_like(w), w])
    beta = np.zeros(2)
    ridge = lam * np.eye(2)
    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
        Wd = np.clip(p * (1.0 - p), 1e-9, None)
        grad = X.T @ (y - p) - lam * beta
        H = X.T @ (Wd[:, None] * X) + ridge
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def w50_slope(w, y):
    """(w50, slope b) from a logistic fit; w50 = -beta0/beta1, b = beta1."""
    b0, b1 = fit_logistic(w, y)
    if abs(b1) < 1e-9:
        return float("nan"), float(b1)
    return float(-b0 / b1), float(b1)


# --------------------------------------------------------------- scoring

def main() -> None:
    manifest = load_manifest()
    bin_edges = np.asarray(manifest["bin_edges_mm"], dtype=float)
    n_bins = manifest["n_bins"]

    scenes = []
    for s in manifest["scenes"]:
        scene, gt, sealed = load_scene(s["dir"])
        scenes.append({"scene_id": s["scene_id"], "scene": scene, "gt": gt, "sealed": sealed,
                       "field": None})

    # ---- SDF-vs-exact validation on a few real paths --------------------
    val = []
    for rec in scenes[:6]:
        scene, gt = rec["scene"], rec["gt"]
        field = forbidden_field(scene, gt)
        rec["field"] = field
        lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
        entry, _ = ag.oracle_entry(gt, scene, field=field)
        if entry is None:
            continue
        action = {"entry_mm": [float(v) for v in entry], "target_mm": [float(v) for v in lesion]}
        sig = det.execute_signature(action, gt, scene)
        exact = float(sig["overall_min_clearance_mm"])
        sdf = float(gsdf.clearance_along_segment(field, scene.affine, entry, lesion))
        val.append({"scene_id": rec["scene_id"], "exact_mm": exact, "sdf_mm": sdf,
                    "abs_diff_mm": abs(exact - sdf)})
    max_val_diff = max((v["abs_diff_mm"] for v in val), default=float("nan"))

    # Build remaining fields (reused by the fast oracle).
    for rec in scenes:
        if rec["field"] is None:
            rec["field"] = forbidden_field(rec["scene"], rec["gt"])

    # ---- run each agent on each scene -----------------------------------
    agents = ag.build_agents()
    w_arr = np.array([float(r["sealed"]["w_optimal_min_clearance_mm"]) for r in scenes])
    # Re-derive bins from the canonical sealed exact w (consistent with figure).
    bin_idx = np.clip(np.digitize(w_arr, bin_edges) - 1, 0, n_bins - 1)

    per_agent = {}
    for agent in agents:
        # ``safe`` is the per-scene safe-PROBABILITY: 0/1 for deterministic
        # agents, and the mean over n_mc seeded perturbations for the stochastic
        # (noisy) agents (Monte-Carlo reliability of the policy). ``successes`` /
        # ``attempts`` back a pass^k reading of the same draws.
        safe = np.zeros(len(scenes), dtype=float)
        successes = np.zeros(len(scenes), dtype=int)
        attempts = np.zeros(len(scenes), dtype=int)
        records = []
        stochastic = bool(getattr(agent, "stochastic", False))
        for i, rec in enumerate(scenes):
            scene, gt = rec["scene"], rec["gt"]
            if stochastic:
                actions = agent.act_ensemble(scene, gt, scene_index=i, field=rec["field"])
            else:
                actions = [agent.act(scene, gt, scene_index=i, field=rec["field"])]
            outcomes = [det.execute_signature(a, gt, scene) for a in actions]
            n_safe = int(sum(1 for s in outcomes if s["path_safe"]))
            safe[i] = n_safe / len(outcomes)
            successes[i] = n_safe
            attempts[i] = len(outcomes)
            sig0 = outcomes[0]
            records.append({
                "scene_id": rec["scene_id"],
                "w_mm": float(w_arr[i]),
                "n_attempts": len(outcomes),
                "n_safe": n_safe,
                "safe_prob": safe[i],
                "entry_mm_first": actions[0]["entry_mm"],
                "target_mm": actions[0]["target_mm"],
                "overall_min_clearance_mm_first": (
                    float(sig0["overall_min_clearance_mm"])
                    if np.isfinite(sig0["overall_min_clearance_mm"]) else None),
                "target_hit_first": bool(sig0["target_hit"]),
            })

        # Per-bin safe rate.
        bin_rates = []
        for b in range(n_bins):
            sel = bin_idx == b
            n = int(sel.sum())
            bin_rates.append({
                "bin": b,
                "w_range_mm": [round(float(bin_edges[b]), 3), round(float(bin_edges[b + 1]), 3)],
                "w_mean_mm": (float(w_arr[sel].mean()) if n else None),
                "n": n,
                "safe_rate": (float(safe[sel].mean()) if n else None),
            })

        overall_safe = float(safe.mean())
        degenerate = overall_safe >= 0.999 or overall_safe <= 0.001
        w50, b = w50_slope(w_arr, safe)

        # Bootstrap CIs by resampling scenes (index trick over bootstrap_ci).
        idx_vals = np.arange(len(scenes), dtype=float)

        def _w50_stat(sample_idx, _w=w_arr, _s=safe):
            ii = np.asarray(sample_idx, dtype=int)
            return w50_slope(_w[ii], _s[ii])[0]

        def _b_stat(sample_idx, _w=w_arr, _s=safe):
            ii = np.asarray(sample_idx, dtype=int)
            return w50_slope(_w[ii], _s[ii])[1]

        w50_ci = bootstrap_ci(idx_vals, statistic=_w50_stat, n_boot=N_BOOT, seed=BOOT_SEED)
        b_ci = bootstrap_ci(idx_vals, statistic=_b_stat, n_boot=N_BOOT, seed=BOOT_SEED)

        def _clean(d):
            return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in d.items()}

        per_agent[agent.name] = {
            "overall_safe_rate": overall_safe,
            "degenerate": bool(degenerate),
            "w50_mm": (None if not np.isfinite(w50) else float(w50)),
            "w50_ci": _clean({"lo": w50_ci["lo"], "hi": w50_ci["hi"]}),
            "slope_b": (None if not np.isfinite(b) else float(b)),
            "slope_ci": _clean({"lo": b_ci["lo"], "hi": b_ci["hi"]}),
            "bin_safe_rates": bin_rates,
            "records": records,
        }

    out = {
        "n_scenes": len(scenes),
        "n_bins": n_bins,
        "bin_edges_mm": [round(float(e), 4) for e in bin_edges],
        "d_safe_mm": float(D_SAFE_MM),
        "private_seed": manifest["private_seed"],
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "sdf_vs_exact_validation": {
            "paths": val,
            "max_abs_diff_mm": (None if not np.isfinite(max_val_diff) else float(max_val_diff)),
            "note": "fast SDF clearance vs exact execute_signature clearance on oracle paths",
        },
        "logistic_model": "P(safe|w) = 1 / (1 + exp(-b*(w - w50)))",
        "per_agent": per_agent,
    }
    results_path = os.path.join(HERE, "results_titration.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)

    make_figure(out, os.path.join(HERE, "figure_titration.png"))

    # ---- console report -------------------------------------------------
    print(f"scenes={len(scenes)} bins={n_bins}  SDF-vs-exact max diff={max_val_diff:.3e} mm\n")
    print(f"{'agent':16s} {'safe_rate':>9s} {'w50(mm)':>9s} {'w50 95% CI':>20s} "
          f"{'slope_b':>9s} {'degenerate':>10s}")
    for name, a in per_agent.items():
        w50 = a["w50_mm"]
        lo, hi = a["w50_ci"]["lo"], a["w50_ci"]["hi"]
        w50s = "n/a" if w50 is None else f"{w50:6.2f}"
        cis = ("n/a" if lo is None or hi is None else f"[{lo:6.2f}, {hi:6.2f}]")
        bs = "n/a" if a["slope_b"] is None else f"{a['slope_b']:6.2f}"
        print(f"{name:16s} {a['overall_safe_rate']:9.3f} {w50s:>9s} {cis:>20s} "
              f"{bs:>9s} {str(a['degenerate']):>10s}")
    print(f"\nwrote {results_path}")
    print(f"wrote {os.path.join(HERE, 'figure_titration.png')}")


# --------------------------------------------------------------- plotting

def make_figure(out, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(out["bin_edges_mm"], dtype=float)
    w_lo, w_hi = float(edges[0]), float(edges[-1])
    wgrid = np.linspace(w_lo, w_hi, 200)

    colors = {"oracle": "#1b7837", "noisy_s3": "#2166ac", "noisy_s6": "#d6604d",
              "naive_straight": "#762a83"}
    markers = {"oracle": "o", "noisy_s3": "s", "noisy_s6": "^", "naive_straight": "D"}

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.axvline(out["d_safe_mm"], ls="--", color="k", lw=1.2,
               label=f"d_safe = {out['d_safe_mm']:g} mm")

    for name, a in out["per_agent"].items():
        col = colors.get(name, "#555555")
        mk = markers.get(name, "o")
        # binned scatter
        xs = [br["w_mean_mm"] for br in a["bin_safe_rates"] if br["safe_rate"] is not None]
        ys = [br["safe_rate"] for br in a["bin_safe_rates"] if br["safe_rate"] is not None]
        ax.scatter(xs, ys, s=55, color=col, marker=mk, edgecolors="k", linewidths=0.4,
                   alpha=0.9, zorder=3)
        # fitted logistic (if non-degenerate slope available)
        w50 = a["w50_mm"]
        b = a["slope_b"]
        lab = name
        if w50 is not None and b is not None:
            p = 1.0 / (1.0 + np.exp(-np.clip(b * (wgrid - w50), -50, 50)))
            ax.plot(wgrid, p, color=col, lw=2.0, alpha=0.9)
            lab = f"{name}  (w50={w50:.2f} mm, b={b:.2f})"
        elif a["degenerate"]:
            ax.plot(wgrid, np.full_like(wgrid, a["overall_safe_rate"]), color=col,
                    lw=2.0, ls=":", alpha=0.9)
            lab = f"{name}  (flat, safe_rate={a['overall_safe_rate']:.2f})"
        ax.plot([], [], color=col, marker=mk, lw=2.0, label=lab)

    ax.set_xlim(w_lo - 0.3, w_hi + 0.3)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("optimal corridor width  w  (mm)   [small w = harder]")
    ax.set_ylabel("P(path_safe)")
    ax.set_title("Difficulty titration: competence vs corridor width (private seed 20260705)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
