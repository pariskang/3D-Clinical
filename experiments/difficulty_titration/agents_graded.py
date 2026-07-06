"""Deterministic graded agents for the difficulty-titration study.

No API / no LLM: these are analytic reference agents of *graded competence* used
to validate that the benchmark yields a difficulty-vs-competence curve that
discriminates skill. Each agent maps a (scene, ground-truth) to a needle action
``{"entry_mm", "target_mm"}``; ``run_titration.py`` scores ``path_safe`` with the
REAL scorer ``trace3d.scoring.deterministic.execute_signature``.

Agents (increasing failure with a tighter corridor):

- ``OracleAgent``       : picks the max-clearance feasible entry recovered from
                          the SAME grid ``corridor_regret`` searches -> near
                          optimal, safe down to the feasibility floor.
- ``NoisyAgent(sigma)`` : oracle entry + a fixed pseudo-random Gaussian
                          perturbation on the entry (x, z), seeded per
                          (scene, agent) so it is exactly reproducible. Two tiers
                          (sigma = 3 mm, 6 mm) trade competence for jitter.
- ``NaiveStraightAgent``: straight anterior probe entry=(lx, ly+28, lz) aimed at
                          the lesion centroid -> ignores the corridor geometry.

Determinism: the only randomness is ``NoisyAgent``'s perturbation, drawn from a
``numpy.random.default_rng`` seeded from a fixed salt, the agent's sigma, and the
scene index -> identical on every run.
"""

from __future__ import annotations

import numpy as np

from trace3d import geometry_sdf as gsdf
from trace3d.config import ENTRY_GRID_N

# Fixed salt so NoisyAgent perturbations are reproducible yet scene/agent unique.
_NOISE_SALT = 770405


def oracle_entry(gt, scene, field=None, n: int = ENTRY_GRID_N):
    """Recover the argmax-clearance feasible entry over ``corridor_regret``'s grid.

    Replays the deterministic anterior-entry grid used by
    ``deterministic.corridor_regret`` (entry_y = lesion_y + 28; x/z over the
    lesion +/- 12 mm; ``n`` samples per axis) and returns the entry whose
    straight corridor to the lesion has the greatest min-clearance to forbidden
    structures (subject to clr >= d_safe, length <= L_max, entry outside body,
    anterior of target).

    If ``field`` (a forbidden-mask distance field, mm) is supplied the clearance
    is evaluated with the fast SDF sampler; otherwise the exact scorer geometry
    is used. Returns ``(best_entry_mm | None, best_clearance_mm)``.
    """
    spec = gt.trajectory_spec
    lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
    entry_y = lesion[1] + 28.0
    span = 12.0
    xs = np.linspace(lesion[0] - span, lesion[0] + span, n)
    zs = np.linspace(lesion[2] - span, lesion[2] + span, n)

    if field is None:
        from trace3d.geometry import clearance_to_labels, segment_hits_label

        forb = {}
        for s in spec.forbidden_structures:
            lab = scene.label_map.get(s)
            if lab is not None:
                forb[s] = (lab, np.argwhere(scene.vol == lab))

    best = 0.0
    best_entry = None
    for x in xs:
        for z in zs:
            entry = np.array([x, entry_y, z])
            if scene.organ_at_point(entry) is not None or not (entry[1] > lesion[1]):
                continue
            if float(np.linalg.norm(lesion - entry)) > spec.L_max_mm:
                continue
            if field is not None:
                clr = gsdf.clearance_along_segment(field, scene.affine, entry, lesion)
            else:
                min_clr = float("inf")
                pierced = False
                for s, (lab, coords) in forb.items():
                    if segment_hits_label(scene.vol, scene.affine, entry, lesion, lab):
                        pierced = True
                        break
                    min_clr = min(min_clr, clearance_to_labels(coords, scene.affine, entry, lesion))
                clr = -1.0 if pierced else min_clr
            if clr < spec.d_safe_mm:
                continue
            if clr > best:
                best = clr
                best_entry = entry
    return best_entry, float(best)


class OracleAgent:
    name = "oracle"
    stochastic = False

    def act(self, scene, gt, scene_index: int, field=None) -> dict:
        lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
        entry, _ = oracle_entry(gt, scene, field=field)
        if entry is None:
            # No feasible safe corridor recovered; fall back to straight probe.
            entry = np.array([lesion[0], lesion[1] + 28.0, lesion[2]])
        return {"entry_mm": [float(v) for v in entry], "target_mm": [float(v) for v in lesion]}


class NoisyAgent:
    """Oracle entry + a pseudo-random Gaussian perturbation on the entry (x, z).

    The agent is *stochastic*: its execution scatters the entry by ``sigma`` mm.
    A single fixed draw (``act``) is exactly reproducible via a seed derived from
    (salt, sigma, scene). Because one draw per scene is a high-variance estimate
    of the policy's safety, we characterise the agent by its safe-PROBABILITY
    over ``n_mc`` seeded draws per scene (``act_ensemble``) -- still fully
    deterministic, and the natural quantity for a reliability / pass^k reading.
    ``n_mc = 1`` recovers the literal single-draw agent.
    """

    stochastic = True

    def __init__(self, sigma_mm: float, n_mc: int = 24):
        self.sigma = float(sigma_mm)
        self.n_mc = int(n_mc)
        self.name = f"noisy_s{int(round(sigma_mm))}"

    def _base_entry(self, scene, gt, field):
        lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
        entry, _ = oracle_entry(gt, scene, field=field)
        if entry is None:
            entry = np.array([lesion[0], lesion[1] + 28.0, lesion[2]])
        return np.asarray(entry, dtype=float), lesion

    def _rng(self, scene_index: int):
        seed = np.array([_NOISE_SALT, int(round(self.sigma * 1000)), int(scene_index)], dtype=np.int64)
        return np.random.default_rng(seed)

    def act(self, scene, gt, scene_index: int, field=None) -> dict:
        base, lesion = self._base_entry(scene, gt, field)
        rng = self._rng(scene_index)
        dx, dz = rng.normal(0.0, self.sigma, size=2)
        e = base.copy()
        e[0] += dx
        e[2] += dz
        return {"entry_mm": [float(v) for v in e], "target_mm": [float(v) for v in lesion]}

    def act_ensemble(self, scene, gt, scene_index: int, field=None) -> list[dict]:
        base, lesion = self._base_entry(scene, gt, field)
        rng = self._rng(scene_index)
        out = []
        for _ in range(self.n_mc):
            dx, dz = rng.normal(0.0, self.sigma, size=2)
            e = base.copy()
            e[0] += dx
            e[2] += dz
            out.append({"entry_mm": [float(v) for v in e], "target_mm": [float(v) for v in lesion]})
        return out


class NaiveStraightAgent:
    name = "naive_straight"
    stochastic = False

    def act(self, scene, gt, scene_index: int, field=None) -> dict:
        lesion = np.asarray(gt.lesion_true_centroid_mm, dtype=float)
        entry = np.array([lesion[0], lesion[1] + 28.0, lesion[2]])
        return {"entry_mm": [float(v) for v in entry], "target_mm": [float(v) for v in lesion]}


def build_agents():
    """The graded agent roster, ordered from most to least competent."""
    return [OracleAgent(), NoisyAgent(3.0), NoisyAgent(6.0), NaiveStraightAgent()]


# ---------------------------------------------------------------------------
# Scaled (statistically-powered) study helpers
# ---------------------------------------------------------------------------
#
# ``run_titration_scaled.py`` precomputes the oracle (argmax-clearance) entry per
# scene from the corridor search and scores every path with the SDF fast backend,
# so it needs a perturbation generator DECOUPLED from the (expensive) oracle
# recovery. ``perturbed_entries`` reproduces ``NoisyAgent``'s exact seeding scheme
# (salt, sigma, scene index) around a supplied base entry, and
# ``build_agents_scaled`` is the denser sigma ladder used to titrate competence.


def perturbed_entries(base_entry, lesion, sigma_mm, scene_index, n_mc, salt=_NOISE_SALT):
    """Seeded Gaussian(sigma) perturbations of ``base_entry`` on the (x, z) axes.

    Bit-identical to ``NoisyAgent(sigma, n_mc).act_ensemble`` when ``base_entry``
    is the oracle entry: the RNG is ``default_rng([salt, round(sigma*1000),
    scene_index])`` and each draw shifts (x, z) by ``N(0, sigma)``. Returns a list
    of ``{"entry_mm", "target_mm"}`` actions (target fixed at the lesion centroid).
    """
    base = np.asarray(base_entry, dtype=float)
    lesion = np.asarray(lesion, dtype=float)
    seed = np.array([int(salt), int(round(float(sigma_mm) * 1000)), int(scene_index)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_mc)):
        dx, dz = rng.normal(0.0, float(sigma_mm), size=2)
        e = base.copy()
        e[0] += dx
        e[2] += dz
        out.append({"entry_mm": [float(v) for v in e], "target_mm": [float(v) for v in lesion]})
    return out


# The powered study's sigma ladder (mm) and Monte-Carlo draw count per scene.
SCALED_SIGMAS = (2.0, 4.0, 6.0, 8.0)
SCALED_N_MC = 25


def build_agents_scaled(n_mc: int = SCALED_N_MC):
    """Graded roster for the powered titration: oracle, sigma ladder, naive.

    Oracle and naive are the construction ceiling / floor; the scientific content
    is the four ``NoisyAgent`` tiers (sigma = 2, 4, 6, 8 mm), whose 50%-safe
    corridor width ``w50`` should be monotone in sigma.
    """
    agents = [OracleAgent()]
    agents += [NoisyAgent(s, n_mc=n_mc) for s in SCALED_SIGMAS]
    agents += [NaiveStraightAgent()]
    return agents
