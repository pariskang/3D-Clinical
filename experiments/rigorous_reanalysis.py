"""Rigorous re-analysis of the two existing single-agent TRACE-3D pilots.

DELIVERABLE B. Loads BOTH pilots' per-decision records and, using ONLY the
pure-numpy :mod:`trace3d.analysis` toolkit, computes proper eval-reporting
statistics:

- per condition x strategy safe-rate with a percentile bootstrap CI, plus a
  cluster-robust CI (cases nest in a scene family -> cluster by case_id);
- **DeltaSafety_3D**: the causal 3D-grounding contribution = paired bootstrap of
  path_safe(T1) - path_safe(T0) per strategy, 95% CI (CausalFlow
  arXiv:2605.25338);
- calibration: Spearman(confidence, min_clearance), risk-coverage + AURC,
  Brier, adaptive-mass ECE -- compared with the legacy L1 margin number;
- paired test: McNemar (T0 vs T1) per strategy, Cohen's h effect size, and
  Holm-Bonferroni across the whole family of contrasts;
- pass^k note: each case has exactly 1 attempt, so pass^1 == pass@1 (stated as a
  limitation; pass^k needs repeated attempts).

Writes ``experiments/rigorous_reanalysis.json`` and a short, HONEST
``experiments/RIGOR_SUMMARY.md``. Everything deterministic (all bootstrap RNG
seeded).
"""

from __future__ import annotations

import json
import os

import numpy as np

from trace3d.analysis import (
    adaptive_ece,
    aurc,
    bootstrap_ci,
    brier_score,
    cluster_bootstrap_ci,
    cohens_h,
    holm_bonferroni,
    margin_target,
    mcnemar_test,
    paired_bootstrap_diff,
    risk_coverage_curve,
    spearman_rho,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260705
N_BOOT = 10000

PILOTS = [
    {
        "name": "portal_vein",
        "path": os.path.join(HERE, "pilot_claude_manual", "results.json"),
        "records_key": "per_decision",
    },
    {
        "name": "anterior_blocker",
        "path": os.path.join(HERE, "pilot_anterior_blocker", "results_manual.json"),
        "records_key": "per_record",
    },
]

CONDITIONS = ["T0", "T1"]
STRATEGIES = ["careful", "fast"]


def load_rows(pilot) -> list[dict]:
    with open(pilot["path"]) as f:
        data = json.load(f)
    return data[pilot["records_key"]]


def _subset(rows, condition=None, strategy=None):
    out = []
    for r in rows:
        if condition is not None and r.get("condition") != condition:
            continue
        if strategy is not None and r.get("strategy") != strategy:
            continue
        out.append(r)
    return out


def _safe_arr(rows):
    return np.array([1.0 if r["path_safe"] else 0.0 for r in rows], dtype=float)


def _paired_by_case(rows, strategy, condition):
    """path_safe per case_id (sorted) for a (strategy, condition) cell."""
    sub = _subset(rows, condition=condition, strategy=strategy)
    by_case = {r["case_id"]: (1.0 if r["path_safe"] else 0.0) for r in sub}
    cases = sorted(by_case)
    return cases, np.array([by_case[c] for c in cases], dtype=float)


def analyze_pilot(pilot) -> dict:
    rows = load_rows(pilot)
    res: dict = {"name": pilot["name"], "n_records": len(rows)}

    # ---- (1) safe-rate per condition x strategy: bootstrap CI --------------
    cell = {}
    for cond in CONDITIONS:
        for strat in STRATEGIES:
            sub = _subset(rows, condition=cond, strategy=strat)
            y = _safe_arr(sub)
            ci = bootstrap_ci(y, n_boot=N_BOOT, seed=SEED)
            cell[f"{cond}-{strat}"] = {
                "n": len(sub),
                "safe_rate": ci["point"],
                "boot_ci": {"lo": ci["lo"], "hi": ci["hi"]},
            }
    res["safe_rate_cells"] = cell

    # ---- per condition (pooled strategies): cluster-robust CI --------------
    # Cases nest in a scene family: each case_id appears under both strategies,
    # so cluster the bootstrap on case_id to avoid understating uncertainty.
    cond_pooled = {}
    for cond in CONDITIONS:
        sub = _subset(rows, condition=cond)
        y = _safe_arr(sub)
        clusters = [r["case_id"] for r in sub]
        iid = bootstrap_ci(y, n_boot=N_BOOT, seed=SEED)
        clu = cluster_bootstrap_ci(y, clusters, n_boot=N_BOOT, seed=SEED)
        cond_pooled[cond] = {
            "n": len(sub),
            "safe_rate": iid["point"],
            "iid_boot_ci": {"lo": iid["lo"], "hi": iid["hi"]},
            "cluster_boot_ci": {"lo": clu["lo"], "hi": clu["hi"], "n_clusters": clu["n_clusters"]},
        }
    res["safe_rate_by_condition_pooled"] = cond_pooled

    # ---- (2) DeltaSafety_3D: paired T1 - T0 per strategy -------------------
    delta = {}
    mcnemar = {}
    cohens = {}
    contrast_pvals = []
    contrast_labels = []
    for strat in STRATEGIES:
        cases0, y0 = _paired_by_case(rows, strat, "T0")
        cases1, y1 = _paired_by_case(rows, strat, "T1")
        # align on common cases
        common = sorted(set(cases0) & set(cases1))
        m0 = {c: v for c, v in zip(cases0, y0)}
        m1 = {c: v for c, v in zip(cases1, y1)}
        a = np.array([m1[c] for c in common])  # T1
        b = np.array([m0[c] for c in common])  # T0
        pd = paired_bootstrap_diff(a, b, n_boot=N_BOOT, seed=SEED)
        delta[strat] = {
            "n_pairs": len(common),
            "safe_T1": float(a.mean()),
            "safe_T0": float(b.mean()),
            "delta_safety_3d": pd["diff"],
            "boot_ci": {"lo": pd["lo"], "hi": pd["hi"]},
        }
        # McNemar discordant cells: b_disc = T1 safe & T0 unsafe; c_disc = reverse
        b_disc = int(np.sum((a == 1.0) & (b == 0.0)))
        c_disc = int(np.sum((a == 0.0) & (b == 1.0)))
        mc = mcnemar_test(b_disc, c_disc)
        mcnemar[strat] = {"b_T1safe_T0unsafe": b_disc, "c_T1unsafe_T0safe": c_disc, **mc}
        cohens[strat] = cohens_h(float(a.mean()), float(b.mean()))
        contrast_pvals.append(mc["p_value"])
        contrast_labels.append(f"{pilot['name']}:{strat}:T1_vs_T0")

    res["delta_safety_3d"] = delta
    res["mcnemar_T1_vs_T0"] = mcnemar
    res["cohens_h_T1_vs_T0"] = cohens
    res["_contrasts"] = {"labels": contrast_labels, "pvalues": contrast_pvals}

    # ---- (3) calibration ---------------------------------------------------
    def _calib(sub):
        conf = np.array([r["confidence_safe"] for r in sub], dtype=float)
        clr = np.array(
            [r["min_clearance_mm"] if r["min_clearance_mm"] is not None else np.nan for r in sub],
            dtype=float,
        )
        safe = _safe_arr(sub)
        finite = np.isfinite(clr)
        rho = spearman_rho(conf[finite], clr[finite]) if finite.sum() >= 2 else None
        rc = risk_coverage_curve(conf, safe)
        legacy = float(np.mean([abs(c - margin_target(x)) for c, x in zip(conf, clr) if np.isfinite(x)]))
        return {
            "n": len(sub),
            "spearman_conf_vs_clearance": rho,
            "aurc_conf_vs_safe": aurc(conf, safe),
            "brier_conf_vs_safe": brier_score(conf, safe),
            "adaptive_ece": adaptive_ece(conf, safe, n_bins=5),
            "legacy_L1_margin_error": legacy,
            "risk_at_full_coverage": float(rc["risk"][-1]) if rc["risk"].size else None,
        }

    calib = {"overall": _calib(rows)}
    for cond in CONDITIONS:
        calib[cond] = _calib(_subset(rows, condition=cond))
    res["calibration"] = calib

    # ---- (5) pass^k note ---------------------------------------------------
    res["pass_hat_k_note"] = (
        "Each case has exactly 1 attempt, so pass^1 == pass@1 == the safe-rate "
        "reported above. pass^k for k>1 requires repeated independent attempts "
        "per case and is flagged as future work (see docs/METHODOLOGY.md)."
    )
    return res


def main() -> None:
    pilots = [analyze_pilot(p) for p in PILOTS]

    # ---- Holm-Bonferroni across the whole family of T1-vs-T0 contrasts -----
    all_labels, all_p = [], []
    for pr in pilots:
        all_labels += pr["_contrasts"]["labels"]
        all_p += pr["_contrasts"]["pvalues"]
    holm = holm_bonferroni(all_p)
    family = {
        "labels": all_labels,
        "raw_pvalues": [float(x) for x in all_p],
        "holm_adjusted": [float(x) for x in holm["adjusted"]],
        "reject_at_0.05": [bool(x) for x in holm["reject"]],
    }
    for pr in pilots:
        pr.pop("_contrasts", None)

    out = {
        "seed": SEED,
        "n_boot": N_BOOT,
        "citations": {
            "error_bars_evals": "arXiv:2411.00640 (Miller, Adding Error Bars to Evals)",
            "cluster_bootstrap": "arXiv:2411.00640",
            "pass_hat_k_reliability": "arXiv:2406.12045 (tau-bench); arXiv:2603.29231",
            "holm_reporting": "arXiv:2511.21140",
            "selective_prediction_aurc": "arXiv:2603.02719",
            "causal_ablation_delta_safety": "arXiv:2605.25338 (CausalFlow)",
        },
        "pilots": {pr["name"]: pr for pr in pilots},
        "holm_family_T1_vs_T0": family,
    }
    out_path = os.path.join(HERE, "rigorous_reanalysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    write_summary(out, os.path.join(HERE, "RIGOR_SUMMARY.md"))

    # ---- console report ----------------------------------------------------
    for name, pr in out["pilots"].items():
        print(f"\n=== {name} (n={pr['n_records']}) ===")
        print("DeltaSafety_3D (path_safe T1 - T0), paired bootstrap 95% CI:")
        for strat, d in pr["delta_safety_3d"].items():
            ci = d["boot_ci"]
            print(f"  {strat:8s}: delta={d['delta_safety_3d']:+.3f}  "
                  f"CI[{ci['lo']:+.3f}, {ci['hi']:+.3f}]  (T1={d['safe_T1']:.2f} T0={d['safe_T0']:.2f})")
        print("Calibration (overall): "
              f"spearman={pr['calibration']['overall']['spearman_conf_vs_clearance']}, "
              f"AURC={pr['calibration']['overall']['aurc_conf_vs_safe']:.3f}, "
              f"Brier={pr['calibration']['overall']['brier_conf_vs_safe']:.3f}, "
              f"aECE={pr['calibration']['overall']['adaptive_ece']:.3f}, "
              f"legacyL1={pr['calibration']['overall']['legacy_L1_margin_error']:.3f}")
    print("\nHolm-Bonferroni across T1-vs-T0 family:")
    for lab, raw, adj, rej in zip(family["labels"], family["raw_pvalues"],
                                  family["holm_adjusted"], family["reject_at_0.05"]):
        print(f"  {lab:34s} raw_p={raw:.3f} holm_p={adj:.3f} reject={rej}")
    print(f"\nwrote {out_path}")
    print(f"wrote {os.path.join(HERE, 'RIGOR_SUMMARY.md')}")


def write_summary(out, path):
    lines = []
    L = lines.append
    L("# Rigorous re-analysis of the two single-agent pilots — honest summary")
    L("")
    L("This re-analyses the two existing **single-model, single-attempt** TRACE-3D "
      "pilots (portal-vein and anterior-blocker) with proper eval-reporting "
      "statistics from `trace3d.analysis`. It is an **instrument-validation** "
      "exercise: the goal is to demonstrate correct reporting standards, **not** "
      "to claim a significant scientific result.")
    L("")
    L("## Hard limitations (read first)")
    L("")
    L("- **N = 1 model** (one manually-driven agent). No between-model variance is "
      "estimated; nothing here generalises to other models.")
    L("- **Small n** (6 cases per condition x strategy cell; 24 records/pilot). "
      "Bootstrap CIs are correspondingly **wide**.")
    L("- **1 attempt per case**, so `pass^k` reduces to `pass^1 = pass@1`. True "
      "reliability (`pass^k`, k>1) needs repeated attempts and is future work.")
    L("- Confidence is frequently **constant within a cell** (e.g. the `fast` "
      "strategy uses a fixed 0.9), so Spearman(conf, clearance) is undefined or "
      "degenerate there.")
    L("- Most contrasts are **NOT significant after Holm-Bonferroni** (see below). "
      "That is the expected and honest outcome at this sample size.")
    L("")
    L("## DeltaSafety_3D — the causal 3D-grounding contribution")
    L("")
    L("Paired bootstrap of `path_safe(T1) - path_safe(T0)` per strategy "
      "(pairing on case_id). Positive = the +3D scene-graph condition improved "
      "safety. Cite CausalFlow (arXiv:2605.25338).")
    L("")
    L("| pilot | strategy | T0 safe | T1 safe | DeltaSafety_3D | 95% bootstrap CI |")
    L("|---|---|---|---|---|---|")
    for name, pr in out["pilots"].items():
        for strat, d in pr["delta_safety_3d"].items():
            ci = d["boot_ci"]
            L(f"| {name} | {strat} | {d['safe_T0']:.2f} | {d['safe_T1']:.2f} | "
              f"{d['delta_safety_3d']:+.3f} | [{ci['lo']:+.3f}, {ci['hi']:+.3f}] |")
    L("")
    L("CIs that include 0 mean the pilot cannot establish a 3D-grounding effect "
      "for that cell at this n. Where a whole cell is all-safe or all-unsafe the "
      "paired difference and its CI collapse to a point.")
    L("")
    L("## Calibration & selective prediction")
    L("")
    L("Deprecating the raw L1 margin error in favour of proper scores: "
      "Spearman(confidence, true min-clearance), risk-coverage AURC, Brier, and "
      "adaptive-mass ECE (arXiv:2603.02719).")
    L("")
    L("| pilot | scope | Spearman | AURC | Brier | adaptive-ECE | legacy L1 |")
    L("|---|---|---|---|---|---|---|")
    for name, pr in out["pilots"].items():
        c = pr["calibration"]["overall"]
        rho = "n/a" if c["spearman_conf_vs_clearance"] is None else f"{c['spearman_conf_vs_clearance']:.3f}"
        L(f"| {name} | overall | {rho} | {c['aurc_conf_vs_safe']:.3f} | "
          f"{c['brier_conf_vs_safe']:.3f} | {c['adaptive_ece']:.3f} | "
          f"{c['legacy_L1_margin_error']:.3f} |")
    L("")
    L("## Paired significance testing (McNemar) + Holm-Bonferroni")
    L("")
    L("McNemar (continuity-corrected) on the T0-vs-T1 safe/unsafe discordant "
      "pairs per strategy, Cohen's h effect size, and Holm step-down correction "
      "across the whole family of contrasts (arXiv:2511.21140).")
    L("")
    L("| contrast | raw p | Holm-adjusted p | reject @0.05 |")
    L("|---|---|---|---|")
    fam = out["holm_family_T1_vs_T0"]
    for lab, raw, adj, rej in zip(fam["labels"], fam["raw_pvalues"],
                                  fam["holm_adjusted"], fam["reject_at_0.05"]):
        L(f"| {lab} | {raw:.3f} | {adj:.3f} | {rej} |")
    L("")
    L("## Bottom line")
    L("")
    L("The pilots **validate the measurement instrument** (the geometry scorer, "
      "the sealed corridor oracle, and the analysis toolkit all produce coherent "
      "numbers) and show a **direction** (DeltaSafety_3D is >= 0 wherever it is "
      "estimable, driven by the anterior-blocker `careful` cell), but with N=1 "
      "model and n=6/cell **no contrast survives multiple-comparison correction**. "
      "The scientific claim awaits the multi-model study (pending API access) and "
      "the difficulty-titration curves in `experiments/difficulty_titration/`.")
    L("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
