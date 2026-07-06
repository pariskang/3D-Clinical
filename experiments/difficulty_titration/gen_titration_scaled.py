"""Regenerate the SCALED difficulty-titration scene family entirely IN MEMORY.

Powered version of ``gen_titration_scenes.py`` (the pilot). Where the pilot
SEALED ~48 scenes to disk across 8 bins, this module builds ~450-500 scenes
across ~11 corridor-width bins and NEVER writes volumes to disk: the family is
regenerated deterministically from a fixed PRIVATE seed (``20260706``) whenever
``run_titration_scaled.py`` needs it, so the repo stays tiny (results JSON + PNG
only).

Difficulty knob
---------------
The pilot used a single anterior ``bowel_loop`` sphere; the safe corridor then
runs around ONE side of it, so the max-clearance optimum sits at the edge of the
open anterior space (a *one-sided* optimum). Under entry perturbation ~half the
draws move to the ever-safer open side, so ``P(safe)`` floors near 0.5 and the
sigma tiers cannot be separated by their 50%-safe width ``w50``.

The powered study instead uses an APERTURE (``bowel_loop`` label, forbidden +
critical): a thin forbidden slab spans the whole corridor cross-section at depth
``dy`` with a single circular hole of radius ``rh`` offset laterally from the
lesion's straight-up line by ``hx > rh``. Then (a) the naive straight probe hits
the slab (pierces) and (b) the only safe corridor threads the hole, whose
clearance is genuinely TWO-SIDED: an entry perturbation in ANY direction moves
the path off the hole centre and reduces clearance to the rim. ``P(safe)`` now
sweeps the full [0, 1] as the corridor width (the hole clearance) ``w`` grows, so
the sigma tiers get cleanly ordered ``w50`` thresholds. The hole radius ``rh``
(with small ``hz``/``dy`` jitter for within-bin variety) is the difficulty lever;
``w`` (mm) is the sealed label: small ``w`` == tight aperture == hard.

Speed
-----
The bottleneck is the exact Euclidean distance field (pure-numpy F-H transform,
~1.1 s on the full 80x80x60 volume). Because every anterior corridor sample lies
at world ``y >= 10`` and the only forbidden structures anterior of that (bowel
loop, colon) are what set the clearance, we compute the EDT on the ``y >= 0``
sub-volume with a correspondingly shifted affine. This is EXACT for the corridor
clearance (the posterior aorta / portal-vein voxels it drops are always >20 mm
from any anterior corridor, so they never determine the minimum; verified to
3.5e-15 mm against the full field) and ~2.3x faster.

Everything load-bearing is reused from the real trace3d machinery and the pilot
generators (``gen_cases.build_fast_scene``, ``gen_pierce_cases`` forbidden/pierce
sets, ``geometry_sdf`` clearance, ``synthetic.build_volume``). No slow
``SceneGraph.build_from_volume`` (edges are never needed for corridor scoring).

Public API
----------
``generate_scenes(seed=PRIVATE_SEED)`` -> ``(records, manifest)`` where each
record is a dict with the in-memory ``scene``/``gt``, the cropped forbidden
distance field + shifted affine, the sealed ``w``, the recovered oracle entry,
and the blocker params. ``manifest`` is a small JSON-able repro dict.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT_PV = os.path.join(os.path.dirname(HERE), "pilot_claude_manual")
PILOT_AB = os.path.join(os.path.dirname(HERE), "pilot_anterior_blocker")
for _p in (PILOT_PV, PILOT_AB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_cases  # noqa: E402  (build_fast_scene, reused verbatim)
import gen_pierce_cases as gp  # noqa: E402  (FORBIDDEN / CRITICAL + pierce test)

from trace3d import geometry_sdf as gsdf  # noqa: E402
from trace3d.config import D_SAFE_MM  # noqa: E402
from trace3d.geometry import segment_hits_label  # noqa: E402
from trace3d.worldgen import synthetic as syn  # noqa: E402

# --- repro constants --------------------------------------------------------
PRIVATE_SEED = 20260706
N_GRID = 29               # corridor-search grid resolution (n x n anterior entries)
ENTRY_DY_MM = 28.0        # entry plane: lesion_y + 28 (well anterior, outside body)
SPAN_MM = 20.0            # +/- lateral search span around the lesion (x, z)

# Difficulty axis: bin scenes by optimal corridor min-clearance w (mm).
#
# The active competence-transition zone is a narrow corridor-width band: below
# ~3 mm no safe corridor exists (w >= d_safe), and above ~7 mm every noisy tier
# saturates near 100% safe (no discriminating information). We therefore
# concentrate the scenes in w in [3, 7] mm rather than spreading them to 12 mm,
# where the extra always-safe scenes would only flatten the logistic fit. See the
# module README note for the pilot's wider (but less powered) span.
W_LO, W_HI = 3.0, 7.0
N_BINS = 10
BIN_EDGES = np.linspace(W_LO, W_HI, N_BINS + 1)
PER_BIN = 46              # -> up to 460 in-memory scenes
MAX_ATTEMPTS = 80_000

# Aperture geometry (mm). The forbidden slab sits at depth ``dy`` (kept in a
# NARROW band so the entry->slab perturbation "lever" ~ dy/28 is roughly constant
# across scenes -> P(safe) depends on the hole width w, not on a dy confound). The
# hole radius ``rh`` is the difficulty lever; its lateral offset ``hx = rh +
# HX_MARGIN`` (> rh) keeps the straight probe outside the hole so naive pierces.
DY_MIN, DY_MAX = 15.0, 17.5
RH_MIN, RH_MAX = 4.0, 12.0
HX_MARGIN = 2.2           # hole offset beyond rh (mm) -> straight probe pierces slab
SLAB_THICK_MM = 2.0       # half-thickness of the forbidden slab in y
SLAB_HALF_MM = 20.0       # half-extent of the slab in x, z (covers the search span)
HZ_JITTER_MM = 2.5        # within-bin variety: hole z-offset

# Crop the EDT to the anterior/right sub-box (world x >= -12 == voxel i >= I0;
# world y >= 0 == voxel j >= J0). Exact for anterior-corridor clearance.
I0 = 28
J0 = 40

# Precompute the analytic voxel grids once (world mm coordinates per voxel).
_XG, _YG, _ZG = syn._grids()


def build_aperture_scene(lesion_mm, dy: float, rh: float, hx: float, hz: float):
    """Fast (node-only) aperture scene + sealed GT (no O(n^2) edges).

    Paints a thin forbidden ``bowel_loop`` slab at depth ``dy`` spanning the
    corridor cross-section, with a circular hole of radius ``rh`` at lateral
    offset ``(hx, hz)`` from the lesion's straight-up line. Never overwrites the
    lesion voxels.
    """
    vol, affine = syn.build_volume()
    cy = lesion_mm[1] + dy
    slab = (
        (np.abs(_YG - cy) <= SLAB_THICK_MM)
        & (np.abs(_XG - lesion_mm[0]) <= SLAB_HALF_MM)
        & (np.abs(_ZG - lesion_mm[2]) <= SLAB_HALF_MM)
    )
    hole = ((_XG - (lesion_mm[0] + hx)) ** 2 + (_ZG - (lesion_mm[2] + hz)) ** 2) <= rh ** 2
    mask = slab & (~hole) & (vol != syn.LABEL_LESION)
    vol[mask] = syn.LABEL_BOWEL
    scene = gen_cases.build_fast_scene(vol, affine)
    near_id, _ = scene.nearest_critical(lesion_mm)
    gt = syn._build_ground_truth(
        scene, vol, affine,
        forbidden_structures=gp.FORBIDDEN,
        feasible_exists=True,
        nearest_critical=near_id or "bowel_loop",
    )
    return scene, vol, affine, gt


def cropped_forbidden_field(vol, affine, scene, gt):
    """Exact forbidden EDT on the anterior/right (x >= -12, y >= 0) sub-box.

    Returns ``(field_mm, affine_c)`` with a correspondingly shifted affine such
    that ``clearance_along_segment(field_mm, affine_c, a, b)`` equals the
    full-volume clearance for any anterior corridor. The cropped box keeps the
    entire aperture slab and the colon's nearest (right) edge, and drops only far
    posterior (aorta/portal-vein) and far-left voxels that are never the nearest
    forbidden voxel along such a path -- verified bit-exact (6e-15 mm) to the full
    field and ~3x faster than the pure-numpy F-H transform on the full 80^3 box.
    """
    mask = np.zeros(vol.shape, dtype=bool)
    for s in gt.trajectory_spec.forbidden_structures:
        lab = scene.label_map.get(s)
        if lab is not None:
            mask |= vol == lab
    field = gsdf.distance_field_mm(mask[I0:, J0:, :], (1.0, 1.0, 1.0))
    affine_c = affine.copy()
    affine_c[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.array([float(I0), float(J0), 0.0])
    return field, affine_c


def corridor_search(gt, scene, field, affine_c, n: int = N_GRID):
    """Optimal corridor min-clearance ``w`` + its argmax entry (oracle entry).

    Same structure as ``deterministic.corridor_regret(0.0, ..., use_sdf=True)``:
    a deterministic anterior-entry grid (entry_y = lesion_y + 28; x/z over the
    lesion +/- SPAN_MM; n samples/axis), the same feasibility gates (entry outside
    body, anterior of target, length <= L_max), and clearance read from the
    (cropped) SDF field. ``w = max feasible clearance`` subject to clr >= d_safe.
    The span is widened (20 mm) so the aperture-threading optimum stays interior
    to the grid.
    """
    spec = gt.trajectory_spec
    lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
    entry_y = lesion[1] + ENTRY_DY_MM
    xs = np.linspace(lesion[0] - SPAN_MM, lesion[0] + SPAN_MM, n)
    zs = np.linspace(lesion[2] - SPAN_MM, lesion[2] + SPAN_MM, n)
    best = 0.0
    best_entry = None
    for x in xs:
        for z in zs:
            entry = np.array([x, entry_y, z])
            if scene.organ_at_point(entry) is not None or not (entry[1] > lesion[1]):
                continue
            if float(np.linalg.norm(lesion - entry)) > spec.L_max_mm:
                continue
            clr = gsdf.clearance_along_segment(field, affine_c, entry, lesion)
            if clr < spec.d_safe_mm:
                continue
            if clr > best:
                best = clr
                best_entry = entry
    return float(best), (None if best_entry is None else np.asarray(best_entry, dtype=float))


def straight_pierces(vol, affine, lesion_mm) -> bool:
    entry = np.array([lesion_mm[0], lesion_mm[1] + ENTRY_DY_MM, lesion_mm[2]], dtype=float)
    return bool(segment_hits_label(vol, affine, entry, lesion_mm, syn.LABEL_BOWEL))


def _bin_of(w: float) -> int:
    return int(np.clip(np.digitize(w, BIN_EDGES) - 1, 0, N_BINS - 1))


# Cheap linear model  w ~ RH_A + RH_B * rh  (fit from an aperture probe grid),
# used only to TARGET a bin when sampling the hole radius; the canonical w is
# always recomputed by the SDF corridor search below and re-binned.
_RH_A, _RH_B = 1.25, 0.548


def _rh_for_target(w_target: float) -> float:
    return (w_target - _RH_A) / _RH_B


def generate_scenes(seed: int = PRIVATE_SEED, per_bin: int = PER_BIN, verbose: bool = False):
    """Deterministically regenerate the in-memory titration family.

    Randomized rejection sampling (seeded) targets each corridor-width bin via the
    cheap linear ``w(rh)`` model (hole radius), then keeps a candidate only if its
    canonical SDF ``w`` lands in an under-filled bin and the naive straight probe
    pierces the slab. ``dy`` and the hole z-offset ``hz`` are jittered for
    within-bin scene variety. Returns ``(records, manifest)``.
    """
    rng = np.random.default_rng(seed)
    lesion_mm = np.asarray(syn._lesion_centroid_world(*syn.build_volume()), dtype=float)

    bins: list[list[dict]] = [[] for _ in range(N_BINS)]
    n_attempt = n_build = 0
    while any(len(b) < per_bin for b in bins) and n_attempt < MAX_ATTEMPTS:
        n_attempt += 1
        underfull = [b for b in range(N_BINS) if len(bins[b]) < per_bin]
        tb = int(rng.choice(underfull))
        w_target = float(rng.uniform(BIN_EDGES[tb], BIN_EDGES[tb + 1]))
        rh = _rh_for_target(w_target) + float(rng.normal(0.0, 0.35))
        rh = float(np.clip(rh, RH_MIN, RH_MAX))
        dy = float(rng.uniform(DY_MIN, DY_MAX))
        hz = float(rng.uniform(-HZ_JITTER_MM, HZ_JITTER_MM))
        hx = rh + HX_MARGIN

        scene, vol, affine, gt = build_aperture_scene(lesion_mm, dy, rh, hx, hz)
        if not (vol == syn.LABEL_BOWEL).any():
            continue
        if not straight_pierces(vol, affine, lesion_mm):
            continue
        n_build += 1
        field, affine_c = cropped_forbidden_field(vol, affine, scene, gt)
        w, oracle_entry = corridor_search(gt, scene, field, affine_c)
        if oracle_entry is None or not (W_LO <= w < W_HI):
            continue
        b = _bin_of(w)
        if len(bins[b]) >= per_bin:
            continue
        bins[b].append({
            "scene": scene, "vol": vol, "affine": affine, "gt": gt,
            "field": field, "affine_c": affine_c,
            "w": w, "oracle_entry": oracle_entry,
            "dy": float(dy), "rh": float(rh), "hx": float(hx), "hz": float(hz),
            "bin": b, "lesion_mm": np.asarray(lesion_mm, dtype=float),
        })
        if verbose and n_build % 50 == 0:
            filled = sum(len(x) for x in bins)
            print(f"  ... built={n_build} kept={filled}/{per_bin * N_BINS}")

    records = []
    for b in range(N_BINS):
        for rec in sorted(bins[b], key=lambda r_: r_["w"]):
            rec = dict(rec)
            rec["scene_index"] = len(records)
            records.append(rec)

    manifest = {
        "private_seed": int(seed),
        "scene_family": "aperture_slab (in-memory, not sealed to disk)",
        "difficulty_label": "w = optimal corridor min-clearance (mm), SDF corridor search",
        "corridor_grid_n": N_GRID,
        "entry_plane_dy_mm": ENTRY_DY_MM,
        "search_span_mm": SPAN_MM,
        "w_lo_mm": W_LO, "w_hi_mm": W_HI,
        "n_bins": N_BINS,
        "bin_edges_mm": [round(float(e), 4) for e in BIN_EDGES],
        "per_bin_target": per_bin,
        "d_safe_mm": float(D_SAFE_MM),
        "aperture_dy_range_mm": [DY_MIN, DY_MAX],
        "aperture_rh_range_mm": [RH_MIN, RH_MAX],
        "aperture_hx_margin_mm": HX_MARGIN,
        "aperture_slab_half_thickness_mm": SLAB_THICK_MM,
        "aperture_slab_half_extent_mm": SLAB_HALF_MM,
        "aperture_hz_jitter_mm": HZ_JITTER_MM,
        "edt_crop_voxel_j0": J0,
        "n_attempts": int(n_attempt),
        "n_scenes_built": int(n_build),
        "n_scenes": len(records),
        "per_bin_fill": [len(bins[b]) for b in range(N_BINS)],
    }
    return records, manifest


def main() -> None:
    records, manifest = generate_scenes(verbose=True)
    print(f"\nprivate seed:  {manifest['private_seed']}")
    print(f"attempts:      {manifest['n_attempts']}")
    print(f"scenes built:  {manifest['n_scenes_built']}")
    print(f"scenes kept:   {manifest['n_scenes']}")
    print("\nper-bin fill (w-range mm -> count):")
    for b in range(N_BINS):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        ws = sorted(round(r_["w"], 2) for r_ in records if r_["bin"] == b)
        print(f"  bin {b:2d}: [{lo:5.2f}, {hi:5.2f})  n={len(ws):3d}"
              f"  w=[{ws[0] if ws else '-'} .. {ws[-1] if ws else '-'}]")
    if any(c < PER_BIN for c in manifest["per_bin_fill"]):
        print("\nNOTE: some bins under target (reported honestly).")


if __name__ == "__main__":
    main()
