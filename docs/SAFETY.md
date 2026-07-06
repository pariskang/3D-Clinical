# TRACE-3D Safety, Ethics & Limitations

> # NOT FOR CLINICAL USE
>
> TRACE-3D is a **research benchmark**. It must **never** be used to plan, guide,
> validate, or inform a real medical procedure, and it is **not** a medical
> device. It is built only on **de-identified, open** data. Nothing in this
> repository constitutes medical advice or clinical endorsement.

This document states the benchmark's limitations honestly. Read it before
interpreting any TRACE-3D result.

---

## 1. The metric is a geometric proxy, not a clinical endorsement

TRACE-3D scores a **straight-line needle** against **static segmentation meshes**.
This deliberately ignores almost everything that makes a real procedure hard:

- **respiration** and patient motion (the liver moves centimetres with breathing);
- **tissue deformation** and organ shift as the needle is advanced;
- **needle bending / deflection** — real needles are not straight lines;
- **contrast phases** — vessel conspicuity and apparent anatomy change with phase.

A high `path_safe` score means the path clears the **static model**, not that the
real procedure would be safe. The geometry is a tractable, deterministic proxy
for a few literature-validated criteria (tissue-traversal distance, vital-
structure avoidance) — nothing more.

## 2. Segmentation blindness

The default pipeline uses TotalSegmentator at default resolution (117-class
task). That segmentation **does not label** many of the small vessels and bowel
loops that drive the **majority of percutaneous complications**. Because the
scene graph is built from these labels, the **safety metric is structurally blind
to exactly the structures most likely to be injured in reality**. Passing
TRACE-3D's safety check therefore says nothing about avoiding small unlabeled
vessels. This is a property of the data, not a bug, and it is reported as a
limitation rather than hidden.

## 3. Synthetic lesions are approximations

In the default synthetic path, lesions are **procedurally inserted and flagged as
synthetic**. Localization is anchored to **real labeled anatomical structures**,
which keeps the spatial geometry meaningful, but differential diagnoses derived
from synthetic cases are **treated cautiously** and should not be read as
clinically realistic disease distributions.

## 4. Clinical authority gap (hard gate)

The gold differentials, management decisions, safety partial-orders, and rubrics
are constructed by non-clinicians for engineering purposes. **At least one
qualified clinician must review the gold before any published claim** about
clinical validity. This is a **hard gate**: results may be computed for
engineering iteration, but no clinical-validity claim may be made without that
review.

## 5. Data ethics and privacy

- Only **de-identified, open** datasets are used.
- **Raw scans are never redistributed.** TRACE-3D ships the build pipeline and
  license-permitted **derived** artifacts (e.g. scene graphs), never the source
  imaging.
- Share-alike obligations (CC BY-SA sources such as AMOS/MSD) **propagate** to any
  redistributed derived scene graphs — see `../DATA_LICENSES.md`.
- **MIMIC-IV** is credentialed and **not redistributable**: it is optional,
  **local-only, and gated**, and is never shipped.
- Non-commercial sources (e.g. MedQA-CS seeds, CC BY-NC) must only be used
  non-commercially.

## 6. Fairness

Fairness is tested via **demographic-swap invariance**: for applicable cases we
generate variants that swap demographic attributes and assert that the **gold
management / trajectory does not change**. The reported `fairness_gap` measures
whether the agent's behavior drifts across the swap.

- Fairness testing **never alters gold**. It only audits invariance.
- A non-zero `fairness_gap` is a finding about the **agent**, not a license to
  change the benchmark's gold answers.

## 7. Contamination resistance

Cases are **procedurally generated from a private held-out seed**, and we ship the
**generator** rather than only static cases. This lets evaluators regenerate
fresh, unseen cases and reduces train/test contamination. Do **not** benchmark a
model on cases generated from a seed that model has been trained on.

## 8. Intended use and prohibited use

**Intended:** measuring LLM-agent performance on 3D-grounded, closed-loop
clinical-action safety, spatial-belief fidelity, and confidence calibration, in a
research setting.

**Prohibited:**

- any clinical, diagnostic, or procedure-planning use;
- presenting TRACE-3D scores as evidence of real-world procedural safety;
- redistributing raw source scans or credentialed (e.g. MIMIC) data;
- making clinical-validity claims without clinician review of gold.

## 9. Responsible reporting

When reporting TRACE-3D results, always report `deterministic_fraction` (target
> 0.65) alongside scores, disclose whether cases were synthetic or real,
disclose the segmentation resolution / task used, and state that the safety
metric is a static-geometry proxy blind to unlabeled small structures. Report the
calibration metrics (`margin_calibration_error`, `safety_calibration_error`,
`overconfident_near_vessel`)
alongside safety — a safer trajectory with miscalibrated confidence is the
benchmark's central cautionary finding, not a success.
