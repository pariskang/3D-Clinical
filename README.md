# TRACE-3D

**Trajectory & Risk-Aware Clinical Evaluation in 3D**

> 首个以三维体素解剖为基础、闭环评估临床「动作」的 LLM 智能体医学基准——智能体必须把体积解剖转化为一条安全的图像引导穿刺轨迹（如肝活检），并以 CT 分割几何作为大多为确定性的评分标准，而非把答案当字符串来评判。
>
> *The first 3D-grounded, closed-loop benchmark that evaluates an LLM agent's clinical **action**: convert volumetric anatomy into a safe, image-guided needle trajectory (e.g. liver biopsy), scored mostly deterministically against CT-segmentation geometry — not against an answer string.*

![status](https://img.shields.io/badge/status-research_preview-orange)
![license](https://img.shields.io/badge/code-MIT-blue)
![offline](https://img.shields.io/badge/core-pure_numpy_offline-green)

> [!WARNING]
> **NOT FOR CLINICAL USE.** TRACE-3D is a research benchmark built only on de-identified, open data. Its geometry is a proxy, not a clinical endorsement. Nothing here may be used to plan, guide, or inform a real procedure. See [`docs/SAFETY.md`](docs/SAFETY.md).

---

# 中文 (Chinese)

## 一句话定位

**TRACE-3D** 评估的不是「模型答得对不对」，而是「智能体在三维解剖场景里做出的临床动作安不安全」。智能体被给定一个由 CT 分割导出的三维场景图，必须规划一条穿刺针轨迹，命中目标病灶、避开关键结构（血管、胆道、肠管等），其结果以**确定性几何**评分。

## 我们填补的空白

此前没有任何基准同时具备：(a) 三维 / 体素级医学模拟，以及 (b) 对一个临床**动作**进行闭环智能体评估、并以分割几何为评分依据。TRACE-3D 是第一个。

- **vs CT-FlowBench**（arXiv 2603.00123, 2026，三维 CT *诊断* 工作流）：TRACE-3D 闭合的是一个带有几何安全结果的 *介入动作* 的环，而非诊断解读。二者互补。
- **vs SpatialMed**（arXiv 2603.13800, 2026，三维 CT 空间推理问答，区分答案正确性与推理忠实性）：TRACE-3D 把忠实性审计从问答扩展到 *以信念为条件的动作*——一个正确陈述的空间信念，是否真的驱动了一个安全的动作？
- **vs CT-Agent**（arXiv 2505.16229，三维 CTQA）：只回答问题，无动作 / 轨迹 / 确定性金标准评分。
- **vs AgentClinic / Agent Hospital / MedAgent-Zero / AI Hospital / MedAgentBench**：全部为文本 / 电子病历 / 对话，无三维体素几何。
- **vs HealthBench**（arXiv 2505.08775）：医师量表评分，但纯文本；TRACE-3D 保留量表哲学，但用确定性几何替换大部分 LLM 评审，并报告 `deterministic_fraction`（目标 > 0.65）。

## STAGER 六阶段流程框架

| 阶段 | 名称 | 评分内容 |
|---|---|---|
| **S** | Survey 勘察 | 场景解析 / 病史覆盖度 |
| **T** | Triage 分诊 | 紧迫度 + 排序鉴别诊断（nDCG@k） |
| **A** | Assess 评估 | 空间信念：器官 / 侧别 / 毫米级定位 / 邻接 F1（对照金标准） |
| **G** | Govern 管控 | 安全前置条件偏序 + 升级 + 预算 |
| **E** | Execute 执行 | 轨迹：命中目标、双方法安全校验、可行性、走廊遗憾 |
| **R** | Reflect 反思 | 置信度与真实间隙的校准 + 信念保真度 + 幻觉安全惩罚 + 并发症告知 |

## 我们要揭示的核心结果（动机假设）

给定三维解剖场景图的前沿 LLM 智能体，相比给定等价纯文本所见的同一智能体，产生的轨迹**明显更安全**（关键结构命中率约减半——这是基准内的「三维工具消融」提升），**但**它们陈述的置信度仍与真实清除间隙**脱钩**——恰恰在针离血管最近时最为过度自信。两个指标合成一张图：`has-3D × calibrated-to-margin`。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
trace3d smoke        # 构建合成病例 -> 运行脚本化智能体 -> 评分 -> 打印评分卡
```

离线冒烟测试无需下载、无需 GPU、无需游戏引擎、无需 LLM。合成数据即开即用；真实 CT 病例需 `real` 额外依赖（AMOS/MSD + TotalSegmentator）。

完整协议见 [`docs/PROTOCOL.md`](docs/PROTOCOL.md)；数据许可见 [`DATA_LICENSES.md`](DATA_LICENSES.md)。

---

# English

## One-line tagline

TRACE-3D scores **whether an agent's clinical action is safe in 3D anatomy**, not whether its answer string is correct.

## Positioning

Frontier medical-LLM benchmarks have converged on text: dialogue, EHR tool-calling, and physician-rubric QA. A parallel line has begun to add 3D CT — but only for *diagnostic interpretation* or *spatial QA*. **No prior benchmark combines (a) 3D / volumetric medical simulation with (b) closed-loop agentic evaluation of a clinical ACTION scored against segmentation geometry.** TRACE-3D occupies exactly that open niche. An agent is handed a 3D scene graph derived from a labeled CT volume, must reason through the six **STAGER** stages, and must emit an **image-guided needle trajectory** (e.g. a liver biopsy path). The trajectory is then scored — mostly deterministically — for hitting the target, clearing critical structures by a safety margin, and being physically feasible.

How TRACE-3D differs from the closest neighbors:

| Benchmark | Modality | What it evaluates | Gap TRACE-3D fills |
|---|---|---|---|
| **CT-FlowBench** (arXiv 2603.00123, 2026) | 3D CT | Agentic *diagnostic* workflows | Closes the loop on an *interventional action with a geometric safety outcome*. Complementary. |
| **SpatialMed** (arXiv 2603.13800, 2026) | 3D CT | Spatial-reasoning QA; answer-correctness vs reasoning-faithfulness | Extends faithfulness auditing from QA to a *belief-conditioned ACTION*. |
| **CT-Agent** (arXiv 2505.16229) | 3D CTQA | Answers questions | Adds action / trajectory / deterministic GT scoring. |
| **AgentClinic** (arXiv 2405.07960) | Text dialogue | Clinical dialogue agent | Adds 3D volumetric geometry — embodiment/physical constraints AgentClinic names as future work. |
| **Agent Hospital / MedAgent-Zero** (arXiv 2405.02957) | Text sim | Multi-agent care sim | Adds 3D geometry + deterministic action scoring. |
| **AI Hospital** (arXiv 2402.09742) | Text dialogue | Interactive diagnosis | Adds 3D geometry + interventional action. |
| **MedAgentBench** (arXiv 2501.14654, NEJM AI) | EHR/FHIR | EHR tool-calling tasks | Adds 3D volumetric scene + geometric outcome. |
| **HealthBench** (arXiv 2505.08775) | Text | Physician-rubric scoring | Keeps rubric philosophy; replaces most LLM-judge scoring with deterministic geometry; reports `deterministic_fraction` (target > 0.65). |

**Clinical grounding.** The geometric scoring criteria — tissue-traversal distance and vital-structure avoidance — are the same criteria validated for AI-guided transthoracic biopsy path planning (ScienceDirect S1051044324001684). TRACE-3D's metrics are not invented; they are the literature's percutaneous-access criteria, made deterministic.

## Motivating hypothesis (the headline result the benchmark is designed to surface)

> Given a 3D anatomical scene graph, frontier LLM agents produce **substantially safer** trajectories than the same agents given equivalent text-only findings — the critical-structure-hit rate roughly **halves** (a within-benchmark 3D-tool-ablation lift). **But** their stated confidence stays **decoupled** from the true clearance margin: they are most overconfident exactly when the needle passes closest to a vessel.

TRACE-3D reports both signals and combines them into one figure: **has-3D × calibrated-to-margin**.

## Architecture

The "world" is built **offline** from open data. No game engine, no GPU, no LLM required for the offline smoke test. The core is pure NumPy.

```
                          TRACE-3D pipeline
 +----------------------------------------------------------------------+
 |  labeled 3D volume  +  affine        (synthetic, or real via `real`) |
 |            |                                                          |
 |            v                                                          |
 |   +------------------+   build    +-------------------------------+   |
 |   | scene-graph maker| ---------> | SCENE GRAPH                   |   |
 |   | (numpy geometry) |            |  centroids . laterality       |   |
 |   +------------------+            |  vs-midline . surface dists   |   |
 |                                   |  adjacency . critical labels  |   |
 |                                   +---------------+---------------+   |
 |                                                   |  Tool API         |
 |                                                   v                   |
 |  +----------------------------------------------------------------+   |
 |  |  tau-bench-style ORCHESTRATOR  -  runs one STAGER episode       |   |
 |  |  S Survey > T Triage > A Assess > G Govern > E Execute > R Reflect  |
 |  |  (agent under test queries the scene graph; emits a trajectory)|   |
 |  +-------------------------------+--------------------------------+   |
 |                                  |  JSONL episode log                 |
 |                                  v                                    |
 |  +----------------------------------------------------------------+   |
 |  |  SCORER  -  deterministic geometry  +  narrow judge (free text)|   |
 |  |  per-stage rubric > HARD SAFETY GATE > episode_score > scorecard|  |
 |  +----------------------------------------------------------------+   |
 +----------------------------------------------------------------------+
```

- **Offline core deps:** `numpy`, `pydantic` (v2), `typer`, `jsonlines`, `pytest`.
- **Optional `real`:** `nibabel`, `scipy` (+ TotalSegmentator) for the real-CT ingestion path.
- **Optional `llm`:** `anthropic` for the simulated patient, the narrow free-text judge, and the LLM agents under test.

Scoring is deterministic geometry plus a **narrow** stubbed/LLM judge used **only** for free text.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
trace3d smoke        # build synthetic case -> run scripted agent -> score -> print scorecard
```

The `trace3d` CLI exposes five subcommands:

| Subcommand | What it does |
|---|---|
| `trace3d build` | Build a scene graph from a (synthetic or real) labeled volume + affine. |
| `trace3d run` | Run an agent through one STAGER episode; log JSONL. |
| `trace3d score` | Score an episode log: deterministic geometry + narrow judge. |
| `trace3d report` | Aggregate scorecards (pass@1, pass^k, fairness_gap, tool_ablation_lift). |
| `trace3d smoke` | End-to-end offline demo: build -> scripted agent -> score -> print scorecard. |

Synthetic data needs **no downloads**. Real CT cases use **AMOS / MSD** (CC BY-SA 4.0) + **TotalSegmentator** (Apache-2.0) via the `real` extra.

## STAGER stages

| Stage | Name | Scored component |
|---|---|---|
| **S** | Survey | Scene parse / history coverage |
| **T** | Triage | Urgency + ranked DDx via nDCG@k |
| **A** | Assess | Spatial beliefs: organ / laterality / localization-mm / adjacency-F1 vs GT |
| **G** | Govern | Partial-order safety prerequisites + escalation + budget |
| **E** | Execute | Trajectory: target-hit, dual-method safety, feasibility, corridor-regret |
| **R** | Reflect | Margin-calibration + belief fidelity + hallucinated-safety penalty + complication ack |

## Scoring summary

- **Deterministic-first.** Most points come from geometry, not an LLM judge.
- **Per-stage rubric:** `stage_score = earned / max` rubric points.
- **Episode:** `episode_score = sum(STAGE_WEIGHTS . stage_score)`, with weights **S .12 · T .10 · A .22 · G .14 · E .30 · R .12**.
- **HARD SAFETY GATE:** any critical-structure pierce caps `episode_score <= 0.3` and sets `safety_violation`.
- Reported per case: `deterministic_fraction` (target **> 0.65**), `pass@1`, `pass^k`, `fairness_gap` (demographic-swap variants), `tool_ablation_lift`.

Full formulas: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Citation

```bibtex
@misc{trace3d2026,
  title        = {TRACE-3D: Trajectory and Risk-Aware Clinical Evaluation in 3D},
  author       = {{TRACE-3D contributors}},
  year         = {2026},
  howpublished = {Research benchmark},
  note         = {3D-grounded, closed-loop LLM-agent medical evaluation;
                  clinical actions scored against CT-segmentation geometry}
}
```

## License & data

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Data:** per-source license matrix in [`DATA_LICENSES.md`](DATA_LICENSES.md). We ship the build pipeline and license-permitted derived artifacts, **never raw scans**.

## Scientific rigor & reproducibility

Full details in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). We report bootstrap &
cluster-robust CIs, unbiased **pass^k** reliability (not pass@1), effect sizes, and
Holm–Bonferroni multiplicity control; a corridor-width **difficulty titration** from
a private held-out seed with a per-model 50%-safe threshold `w50`
([`experiments/difficulty_titration/`](experiments/difficulty_titration/)); a
calibration triad (Spearman + risk-coverage/AURC + Brier + adaptive ECE); and a
causal 3D-ablation metric **ΔSafety_3D**. The toolkit is
[`trace3d.analysis.stats`](trace3d/analysis/stats.py) +
[`trace3d.analysis.calibration`](trace3d/analysis/calibration.py) (pure numpy).

**Honest status:** single-model (N=1) *instrument validation* to date on synthetic
anatomy; nothing survives Holm correction at this sample size, though ΔSafety_3D is
≥0 wherever estimable. The multi-model empirical study, pass^k reliability, and
interventional-radiologist validation are pending API/compute/clinician access.
**Not for clinical use.**

## Documentation

- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — full protocol & scoring spec.
- [`docs/DATASHEET.md`](docs/DATASHEET.md) — datasheet-for-datasets.
- [`docs/SAFETY.md`](docs/SAFETY.md) — safety, ethics, limitations.
- [`DATA_LICENSES.md`](DATA_LICENSES.md) — per-source license matrix.

---

> [!WARNING]
> **NOT FOR CLINICAL USE.** Research benchmark only. De-identified, open data only. The geometry is a proxy, not a clinical endorsement.
