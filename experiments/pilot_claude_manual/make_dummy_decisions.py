"""Generate a trivial dummy decisions file to DRY-RUN run_pilot.py end-to-end.

Uses only agent-visible info from the T1 packets (lesion centroid is legitimately
provided there). For every (case, condition, strategy) it proposes a naive
straight anterior->lesion path with confidence 0.9. This is ONLY to exercise the
scoring/plotting pipeline; it is not a real agent's decisions.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PACKETS = os.path.join(HERE, "packets")

CONDITIONS = ["T0", "T1"]
STRATEGIES = ["careful", "fast"]


def main():
    records = []
    for name in sorted(os.listdir(PACKETS)):
        pdir = os.path.join(PACKETS, name)
        t1_path = os.path.join(pdir, "obs_T1_scene.json")
        if not os.path.isfile(t1_path):
            continue
        with open(t1_path) as f:
            t1 = json.load(f)
        case_id = t1["case_id"]
        lesion = t1["lesion_target"]["centroid_mm"]
        plane_y = t1["allowed_entry_surface"]["plane_y_mm"]
        # Naive straight anterior entry directly in front of the lesion.
        entry = [round(lesion[0], 2), round(float(plane_y), 2), round(lesion[2], 2)]
        target = [round(lesion[0], 2), round(lesion[1], 2), round(lesion[2], 2)]
        for cond in CONDITIONS:
            for strat in STRATEGIES:
                records.append(
                    {
                        "case_id": case_id,
                        "condition": cond,
                        "strategy": strat,
                        "entry_mm": entry,
                        "target_mm": target,
                        "beliefs": {
                            "lesion_side": t1["lesion_target"]["side"],
                            "nearest_critical": "portal_vein",
                            "structures_i_claim_to_avoid": ["portal_vein", "aorta", "colon"],
                        },
                        "confidence_safe": 0.9,
                        "complication_ack": True,
                    }
                )
    out = {"records": records}
    out_path = os.path.join(HERE, "dummy_decisions.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path} with {len(records)} records")


if __name__ == "__main__":
    main()
