"""Powered difficulty-titration run: rank graded-agent competence via w50.

Regenerates the SCALED in-memory scene family (``gen_titration_scaled``), scores
six graded agents with the SDF fast backend, fits a per-agent 2-parameter
logistic ``P(safe | w) = 1 / (1 + exp(-b (w - w50)))``, and asks the scientific
question: can the benchmark RANK competence with statistical power? i.e. do the
sigma tiers get ordered 50%-safe corridor-width thresholds ``w50`` with SEPARATED
(non-overlapping) bootstrap CIs?

Pipeline
--------
1. Regenerate ~450-500 scenes in memory (private seed 20260706); each carries its
   cropped forbidden distance field + shifted affine and its sealed optimal
   corridor width ``w`` (difficulty label).
2. VALIDATE the SDF-fast ``path_safe`` used for scoring against the EXACT scorer
   (``deterministic.execute_signature``) on ~50 paths spanning the safety
   boundary; report the max abs clearance diff and any safety flips.
3. Score each agent's per-scene safe probability with the SDF field:
   - oracle: argmax-clearance entry (construction ceiling; ~always safe),
   - noisy(sigma=2/4/6/8): mean over 25 seeded Gaussian(sigma) entry
     perturbations of ``clearance >= d_safe`` (Monte-Carlo reliability),
   - naive straight: anterior probe (construction floor; pierces the blocker).
4. Fit the logistic per agent (ridge-stabilised IRLS); bootstrap over SCENES
   (1000 resamples, shared indices across agents) for 95% CIs on ``w50`` and the
   slope ``b``.
5. PAIRWISE competence separation for adjacent sigma pairs (2v4, 4v6, 6v8):
   ``Delta w50`` with a paired (joint-scene) bootstrap CI and p-value, Holm-
   corrected across the family.
6. Write ``results_titration_scaled.json`` + ``figure_titration_scaled.png`` and
   print the w50 table, pairwise verdicts and the SDF-vs-exact validation.

Everything deterministic (all RNG seeded).
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
import gen_titration_scaled as gen  # noqa: E402

from trace3d import geometry_sdf as gsdf  # noqa: E402
from trace3d.analysis import holm_bonferroni  # noqa: E402
from trace3d.config import D_SAFE_MM  # noqa: E402
from trace3d.scoring import deterministic as det  # noqa: E402

BOOT_SEED = 20260706
N_BOOT = 1000
N_VALIDATE = 50


# ----------------------------------------------------------- logistic MLE

def fit_logistic(w, y, lam: float = 1e-3, iters: int = 200):
    """Ridge-stabilised IRLS for P(y=1|w) = sigmoid(beta0 + beta1*w)."""
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
        beta = beta + np.linalg.solve(H, grad)
    return float(beta[0]), float(beta[1])


def w50_slope(w, y):
    b0, b1 = fit_logistic(w, y)
    if abs(b1) < 1e-9:
        return float("nan"), float(b1)
    return float(-b0 / b1), float(b1)


# ----------------------------------------------------------------- scoring

def _clr(field, affine_c, entry, target):
    return float(gsdf.clearance_along_segment(field, affine_c, np.asarray(entry, float),
                                              np.asarray(target, float)))


def agent_actions(agent, rec):
    """Return the list of {entry_mm, target_mm} the agent submits for a scene."""
    lesion = rec["lesion_mm"]
    idx = rec["scene_index"]
    name = agent.name
    if name == "oracle":
        entry = rec["oracle_entry"]
        return [{"entry_mm": [float(v) for v in entry], "target_mm": [float(v) for v in lesion]}]
    if name == "naive_straight":
        entry = np.array([lesion[0], lesion[1] + gen.ENTRY_DY_MM, lesion[2]])
        return [{"entry_mm": [float(v) for v in entry], "target_mm": [float(v) for v in lesion]}]
    # NoisyAgent: seeded Gaussian perturbations of the precomputed oracle entry.
    return ag.perturbed_entries(rec["oracle_entry"], lesion, agent.sigma, idx, agent.n_mc)


def score_agent(agent, records):
    """Per-scene safe probability via the SDF fast backend (clearance >= d_safe)."""
    safe = np.zeros(len(records), dtype=float)
    successes = np.zeros(len(records), dtype=int)
    attempts = np.zeros(len(records), dtype=int)
    for i, rec in enumerate(records):
        actions = agent_actions(agent, rec)
        clrs = [_clr(rec["field"], rec["affine_c"], a["entry_mm"], a["target_mm"]) for a in actions]
        n_safe = int(sum(1 for c in clrs if c >= D_SAFE_MM))
        safe[i] = n_safe / len(clrs)
        successes[i] = n_safe
        attempts[i] = len(clrs)
    return safe, successes, attempts


def validate_sdf_vs_exact(records, n=N_VALIDATE, seed=BOOT_SEED):
    """Compare SDF-fast clearance vs EXACT execute_signature on boundary paths.

    Samples paths biased toward the d_safe boundary (a spread of noisy tiers) so
    the check exercises exactly the region where an SDF error could flip safety.
    """
    rng = np.random.default_rng(seed)
    paths = []
    order = rng.permutation(len(records))
    sigmas = [0.0, 2.0, 4.0, 6.0, 8.0]
    for j in order:
        rec = records[int(j)]
        sig = float(rng.choice(sigmas))
        if sig == 0.0:
            entry = rec["oracle_entry"]
        else:
            entry = np.asarray(ag.perturbed_entries(rec["oracle_entry"], rec["lesion_mm"],
                                                    sig, rec["scene_index"], 1)[0]["entry_mm"])
        paths.append((rec, np.asarray(entry, float)))
        if len(paths) >= n:
            break
    out = []
    max_diff = 0.0
    n_flip = 0
    for rec, entry in paths:
        lesion = rec["lesion_mm"]
        action = {"entry_mm": [float(v) for v in entry], "target_mm": [float(v) for v in lesion]}
        sig = det.execute_signature(action, rec["gt"], rec["scene"])
        exact = float(sig["overall_min_clearance_mm"])
        sdf = _clr(rec["field"], rec["affine_c"], entry, lesion)
        d = abs(exact - sdf)
        max_diff = max(max_diff, d)
        safe_exact = exact >= D_SAFE_MM
        safe_sdf = sdf >= D_SAFE_MM
        flip = bool(safe_exact != safe_sdf)
        n_flip += int(flip)
        out.append({"scene_index": rec["scene_index"], "exact_mm": exact, "sdf_mm": sdf,
                    "abs_diff_mm": d, "safety_flip": flip})
    return {"paths": out, "max_abs_diff_mm": float(max_diff), "n_paths": len(out),
            "n_safety_flips": int(n_flip)}


# --------------------------------------------------------------------- main

def main() -> None:
    print("regenerating scaled scene family (private seed, in memory) ...")
    records, manifest = gen.generate_scenes(seed=gen.PRIVATE_SEED, verbose=True)
    n = len(records)
    w_arr = np.array([r["w"] for r in records], dtype=float)
    bin_edges = np.asarray(manifest["bin_edges_mm"], dtype=float)
    n_bins = manifest["n_bins"]
    bin_idx = np.clip(np.digitize(w_arr, bin_edges) - 1, 0, n_bins - 1)
    print(f"scenes regenerated: {n}\n")

    # ---- SDF-vs-exact validation ---------------------------------------
    validation = validate_sdf_vs_exact(records)
    print(f"SDF-vs-exact validation: {validation['n_paths']} paths, "
          f"max |diff| = {validation['max_abs_diff_mm']:.3e} mm, "
          f"safety flips = {validation['n_safety_flips']}\n")

    # ---- score each agent ----------------------------------------------
    agents = ag.build_agents_scaled()
    names = [a.name for a in agents]
    safe_by = {}
    per_agent = {}
    for agent in agents:
        safe, succ, att = score_agent(agent, records)
        safe_by[agent.name] = safe
        overall = float(safe.mean())
        degenerate = overall >= 0.999 or overall <= 0.001
        w50, b = w50_slope(w_arr, safe)
        bin_rates = []
        for bb in range(n_bins):
            sel = bin_idx == bb
            n_sel = int(sel.sum())
            bin_rates.append({
                "bin": bb,
                "w_range_mm": [round(float(bin_edges[bb]), 3), round(float(bin_edges[bb + 1]), 3)],
                "w_mean_mm": (float(w_arr[sel].mean()) if n_sel else None),
                "n": n_sel,
                "safe_rate": (float(safe[sel].mean()) if n_sel else None),
            })
        per_agent[agent.name] = {
            "sigma_mm": (float(agent.sigma) if hasattr(agent, "sigma") else None),
            "n_mc": (int(agent.n_mc) if hasattr(agent, "n_mc") else 1),
            "overall_safe_rate": overall,
            "degenerate": bool(degenerate),
            "w50_mm": (None if not np.isfinite(w50) else float(w50)),
            "slope_b": (None if not np.isfinite(b) else float(b)),
            "bin_safe_rates": bin_rates,
        }

    # ---- bootstrap over scenes (shared resample indices across agents) --
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))
    boot_w50 = {name: np.empty(N_BOOT) for name in names}
    boot_slope = {name: np.empty(N_BOOT) for name in names}
    for t in range(N_BOOT):
        ii = boot_idx[t]
        wt = w_arr[ii]
        for name in names:
            w50, b = w50_slope(wt, safe_by[name][ii])
            boot_w50[name][t] = w50
            boot_slope[name][t] = b

    def _ci(arr):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return {"lo": None, "hi": None}
        return {"lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}

    for name in names:
        per_agent[name]["w50_ci"] = _ci(boot_w50[name])
        per_agent[name]["slope_ci"] = _ci(boot_slope[name])

    # ---- pairwise sigma separation (adjacent tiers), Holm-corrected -----
    sigma_names = [a.name for a in agents if hasattr(a, "sigma")]
    sigma_names = sorted(sigma_names, key=lambda nm: per_agent[nm]["sigma_mm"])
    pairs = list(zip(sigma_names[:-1], sigma_names[1:]))  # (2,4),(4,6),(6,8)
    pairwise = []
    raw_p = []
    for lo_name, hi_name in pairs:
        d = boot_w50[hi_name] - boot_w50[lo_name]  # expect > 0 (harder tier -> larger w50)
        d = d[np.isfinite(d)]
        point = float(per_agent[hi_name]["w50_mm"] - per_agent[lo_name]["w50_mm"]) \
            if (per_agent[hi_name]["w50_mm"] is not None and per_agent[lo_name]["w50_mm"] is not None) \
            else None
        ci_lo = float(np.percentile(d, 2.5))
        ci_hi = float(np.percentile(d, 97.5))
        # two-sided bootstrap p-value that Delta w50 != 0
        p = 2.0 * min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0)))
        p = min(1.0, p)
        raw_p.append(p)
        ci_excludes_0 = bool(ci_lo > 0.0 or ci_hi < 0.0)
        w50_ci_disjoint = bool(
            per_agent[lo_name]["w50_ci"]["hi"] is not None
            and per_agent[hi_name]["w50_ci"]["lo"] is not None
            and per_agent[lo_name]["w50_ci"]["hi"] < per_agent[hi_name]["w50_ci"]["lo"]
        )
        pairwise.append({
            "pair": f"{lo_name}_vs_{hi_name}",
            "sigma_lo": per_agent[lo_name]["sigma_mm"],
            "sigma_hi": per_agent[hi_name]["sigma_mm"],
            "delta_w50_mm": point,
            "delta_w50_ci": {"lo": ci_lo, "hi": ci_hi},
            "delta_ci_excludes_0": ci_excludes_0,
            "w50_ci_nonoverlapping": w50_ci_disjoint,
            "boot_p_raw": p,
        })
    holm = holm_bonferroni(raw_p)
    for k, pw in enumerate(pairwise):
        pw["boot_p_holm"] = float(holm["adjusted"][k])
        pw["separable_holm"] = bool(holm["reject"][k])

    # ---- assemble + write ----------------------------------------------
    out = {
        "n_scenes": n,
        "n_bins": n_bins,
        "bin_edges_mm": [round(float(e), 4) for e in bin_edges],
        "d_safe_mm": float(D_SAFE_MM),
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "logistic_model": "P(safe|w) = 1 / (1 + exp(-b*(w - w50)))",
        "repro_manifest": manifest,
        "sdf_vs_exact_validation": validation,
        "per_agent": per_agent,
        "pairwise_sigma_separation": pairwise,
    }
    results_path = os.path.join(HERE, "results_titration_scaled.json")
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    make_figure(out, os.path.join(HERE, "figure_titration_scaled.png"))

    # ---- console report -------------------------------------------------
    print(f"{'agent':16s} {'sigma':>5s} {'safe':>6s} {'w50(mm)':>8s} "
          f"{'w50 95% CI':>18s} {'slope_b':>8s} {'degen':>6s}")
    for name in names:
        a = per_agent[name]
        w50 = a["w50_mm"]
        lo, hi = a["w50_ci"]["lo"], a["w50_ci"]["hi"]
        w50s = "n/a" if w50 is None else f"{w50:6.2f}"
        cis = ("n/a" if lo is None or hi is None else f"[{lo:6.2f},{hi:6.2f}]")
        bs = "n/a" if a["slope_b"] is None else f"{a['slope_b']:6.2f}"
        sg = "-" if a["sigma_mm"] is None else f"{a['sigma_mm']:.0f}"
        print(f"{name:16s} {sg:>5s} {a['overall_safe_rate']:6.3f} {w50s:>8s} "
              f"{cis:>18s} {bs:>8s} {str(a['degenerate']):>6s}")

    print("\npairwise sigma separation (Delta w50 = w50[higher sigma] - w50[lower]):")
    for pw in pairwise:
        verdict = "SEPARABLE" if pw["separable_holm"] else "not separable"
        print(f"  {pw['pair']:26s} Dw50={pw['delta_w50_mm']:5.2f} mm  "
              f"CI=[{pw['delta_w50_ci']['lo']:5.2f},{pw['delta_w50_ci']['hi']:5.2f}]  "
              f"p_holm={pw['boot_p_holm']:.4f}  w50CIs disjoint={pw['w50_ci_nonoverlapping']}  "
              f"-> {verdict}")
    print(f"\nwrote {results_path}")
    print(f"wrote {os.path.join(HERE, 'figure_titration_scaled.png')}")


# --------------------------------------------------------------- plotting

def make_figure(out, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(out["bin_edges_mm"], dtype=float)
    w_lo, w_hi = float(edges[0]), float(edges[-1])
    wgrid = np.linspace(w_lo, w_hi, 300)
    colors = {"oracle": "#1b7837", "noisy_s2": "#2166ac", "noisy_s4": "#4393c3",
              "noisy_s6": "#f4a582", "noisy_s8": "#d6604d", "naive_straight": "#762a83"}
    markers = {"oracle": "o", "noisy_s2": "s", "noisy_s4": "P", "noisy_s6": "^",
               "noisy_s8": "v", "naive_straight": "D"}

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.axvline(out["d_safe_mm"], ls="--", color="k", lw=1.2, label=f"d_safe = {out['d_safe_mm']:g} mm")
    for name, a in out["per_agent"].items():
        col = colors.get(name, "#555555")
        mk = markers.get(name, "o")
        xs = [br["w_mean_mm"] for br in a["bin_safe_rates"] if br["safe_rate"] is not None]
        ys = [br["safe_rate"] for br in a["bin_safe_rates"] if br["safe_rate"] is not None]
        ax.scatter(xs, ys, s=42, color=col, marker=mk, edgecolors="k", linewidths=0.35,
                   alpha=0.9, zorder=3)
        w50, b = a["w50_mm"], a["slope_b"]
        lab = name
        if w50 is not None and b is not None and not a["degenerate"]:
            p = 1.0 / (1.0 + np.exp(-np.clip(b * (wgrid - w50), -50, 50)))
            ax.plot(wgrid, p, color=col, lw=2.0, alpha=0.95)
            lo, hi = a["w50_ci"]["lo"], a["w50_ci"]["hi"]
            ci = "" if lo is None else f", CI[{lo:.1f},{hi:.1f}]"
            lab = f"{name} (w50={w50:.2f}{ci})"
        else:
            ax.plot(wgrid, np.full_like(wgrid, a["overall_safe_rate"]), color=col, lw=2.0,
                    ls=":", alpha=0.9)
            lab = f"{name} (flat={a['overall_safe_rate']:.2f})"
        ax.plot([], [], color=col, marker=mk, lw=2.0, label=lab)

    ax.set_xlim(w_lo - 0.3, w_hi + 0.3)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("optimal corridor width  w  (mm)   [small w = harder]")
    ax.set_ylabel("P(path_safe)")
    ax.set_title(f"Powered titration: competence vs corridor width "
                 f"(N={out['n_scenes']} scenes, seed {out['repro_manifest']['private_seed']})")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
