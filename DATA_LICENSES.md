# TRACE-3D Data & License Matrix

TRACE-3D's **code** is MIT (see `LICENSE`). The data sources it can ingest carry
their **own** licenses with their own obligations. This matrix is normative for
what may be redistributed.

> Principle: **ship the build pipeline and license-permitted derived artifacts —
> never raw scans.** When in doubt, regenerate locally from the original source
> under that source's license.

---

## License matrix

| Source / component | License | Commercial? | Redistribution in TRACE-3D | Notes |
|---|---|---|---|---|
| **TRACE-3D code** | MIT | Yes | Shipped | Generator, builder, orchestrator, scorer. |
| **TotalSegmentator outputs** (default 117-class task) | Apache-2.0 | Yes | Derived labels permitted | Default pipeline uses only the Apache-2.0 task. |
| **TotalSegmentator** subtasks: `tissue_types`, `heartchambers_highres`, `brain_aneurysm` | Non-commercial | No | **BLOCKED** in default pipeline | Explicitly disabled by default; non-commercial only. |
| **AMOS22** (source CT) | CC BY-SA 4.0 | Yes (share-alike) | **Raw scans never shipped**; derived scene graphs only, with attribution + share-alike | Share-alike + attribution **propagate** to redistributed derivatives. |
| **MSD** (Medical Segmentation Decathlon, source CT) | CC BY-SA 4.0 | Yes (share-alike) | **Raw scans never shipped**; derived scene graphs only, with attribution + share-alike | Same propagation as AMOS. |
| **MedQA-CS** case seeds | CC BY-NC | **No** | Non-commercial use only | Non-commercial seeds; keep out of any commercial pipeline. |
| **MIMIC-IV** | PhysioNet credentialed | No | **NOT redistributable** | Optional, **local-only, gated**; never shipped. |
| **MedQA** | MIT | Yes | Permitted | Open. |
| **MedMCQA** | MIT/open | Yes | Permitted | Open. |
| **PubMedQA** | MIT/open | Yes | Permitted | Open. |

---

## Key obligations

### Share-alike propagation (AMOS22 / MSD, CC BY-SA 4.0)

CC BY-SA 4.0 requires **attribution** and is **share-alike**: any **derived**
artifact you redistribute (e.g. a scene graph computed from an AMOS/MSD scan)
must carry attribution to the source **and** be released under a compatible
share-alike license. TRACE-3D therefore:

1. ships the **build pipeline** so users derive scene graphs locally; and
2. only redistributes derived artifacts that are **license-permitted**, with the
   required attribution and share-alike terms attached;
3. **never** redistributes raw scans.

### TotalSegmentator task gating

The default pipeline uses **only** the **Apache-2.0** default 117-class task. The
**non-commercial** subtasks `tissue_types`, `heartchambers_highres`, and
`brain_aneurysm` are **blocked by default** and must not be enabled in any
commercial or redistributed pipeline.

### Non-commercial seeds (MedQA-CS, CC BY-NC)

CC BY-NC permits **non-commercial** use only. Keep MedQA-CS-derived case seeds
out of any commercial deployment.

### Credentialed data (MIMIC-IV)

MIMIC-IV requires PhysioNet credentialing and is **not redistributable**. It is
supported only as an **optional, local-only, gated** input and is never included
in any shipped artifact.

---

## Practical guidance

- To reproduce **real** cases, obtain AMOS22 / MSD yourself under CC BY-SA 4.0,
  run them through the `real` extra locally, and keep raw scans local.
- If you **redistribute** any derived scene graph from a CC BY-SA source, include
  the source attribution and release under compatible share-alike terms.
- The **synthetic** default path has **no source-data obligations** beyond the
  MIT code license and needs no downloads.
