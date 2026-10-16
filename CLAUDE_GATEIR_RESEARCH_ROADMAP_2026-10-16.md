# GATE-IR: Next Research Directions and Claude Action Plan (16 October 2026)

> **Status:** Research recommendations and execution prompt, **not experimental results**.
> **Scope:** No new mathematical theory requested. Focus on the original code, reproducibility, practical experiments, failure analysis, and defensible publication opportunities.
> **Instruction to Claude:** Independently **verify, challenge, and if necessary reject** every recommendation in this document. Do not treat this document, previous ChatGPT advice, or existing council reports as scientific authority.

## 1. Sources to inspect before taking action

Primary repositories:
- Distillation/student code: https://github.com/abishek-c367/FineTuning-YOLOv8l-on-PseudoRGB-Images
- 2026 audit branch, if still present: https://github.com/abishek-c367/FineTuning-YOLOv8l-on-PseudoRGB-Images/tree/audit/2026-10
- Weather gate / preprocessing implementation: https://github.com/zer0-data/GATE-IR

Read the audit's `audit_2026/audit/PROJECT_COMPONENT_MAP.md`, `CODE_AUDIT_2026.md`, `CLAIM_EVIDENCE_LEDGER.md`, `PRIOR_ART_NOVELTY_MATRIX.md`, `RECOMMENDATIONS_COUNTERCHECK.md`, and `REPRODUCTION_PROTOCOL.md`. Review the **actual code and tests**, not only the audit prose. If files have changed, identify branch and exact commit before relying on them.

The current `audit_2026/paper/DRAFT.md` concerns **a separate RAW-precision/AGC hypothesis**. Preserve it without treating its proposed results as evidence. The `EXPERIMENT_MATRIX.md`, `COUNCIL_RECORD.md`, and `DRAFT.md` disagree on some protocol details; do not run that full protocol before reconciliation.

## 2. What is implemented versus unverified

As reported in the 15 October audit (verify against the current repository):
- **Implemented/reported executed:** FLIR 3-class thermal data pipeline, CycleGAN thermal-to-pseudo-RGB generation, pseudo-RGB cache, frozen YOLOv8-L teacher feature extraction, YOLO-style thermal student, P3/P4 feature-mimic loss, distillation training loop, and checkpoint-writing code. The collaborator reported experiments, but original data/weights/logs have not been independently reproduced by the auditor.
- **Original trained student configuration:** `include_p2=True`, `use_transformer_neck=False`. Do **not** claim the ViT neck was used in the reported training.
- **Identified issues:** original notebook could silently load zero trained weights (key `student_state` and architecture mismatch); offline evaluation used training-mode BatchNorm; conflicting class maps; fragile cache/resume logic; untrained weather gate and rain-removal module; unverified teacher fine-tuning and class labels. Audit branch includes patches, but re-test them and create fixed-code regression tests rather than assuming all tests pass.
- **Not established:** a reproducible thermal-only baseline, an independently evaluated CycleGAN teacher/student comparison, real-weather routing performance, small-object improvement, and real-time deployment claims.
- **Data caution:** FLIR v2 does not provide per-frame weather labels for the proposed three-class study. Keep real-weather claims separate; verify external datasets, annotations, sensor/camera differences, and usage licenses.

## 3. Research tracks to evaluate (ranked by reuse of current code)

### Track A — WHEN does pseudo-RGB distillation help or harm? [FIRST PRIORITY]

**Central question:** Does an *unpaired* CycleGAN pseudo-RGB teacher confer a reproducible advantage over a no-teacher thermal student or a thermal teacher, and in which object/image conditions?

**Minimal fair comparisons:**
1. Thermal student trained without teacher or distillation.
2. Same thermal student architecture with a thermal-trained teacher.
3. Same thermal student with a COCO-pretrained RGB teacher receiving replicated grayscale thermal input.
4. Same thermal student with that teacher receiving CycleGAN-generated pseudo-RGB.
5. Optional only if paired image geometry and annotation validity are verified: a teacher receiving **genuinely paired** visible imagery as a separate reference arm, never confused with the unpaired setting.

Match splits, labels, architecture, evaluation, training budget, teacher feature levels, seed policy, and augmentation wherever meaningful. Document any unavoidable parameter-count/compute mismatch. Test an established stronger thermal-only and/or cross-modal KD baseline if reproducible. Teacher detection accuracy is useful but **teacher feature quality** may be different; inspect both.

**Report:** COCO mAP50-95, AP50, AP-small, per-class AP (person, car, bicycle), training stability, runtime/memory, and results across several seeds. Hold videos, not just shuffled frames, disjoint across splits. Perform qualitative matched-example review without cherry-picking.

**Value:** A failure-mode/mechanism investigation is more defensible than simply combining CycleGAN + YOLO + feature mimic. A result of 'no benefit' is scientifically informative but is not automatically publication-worthy unless scope, controls, evidence, and novelty are strong.

### Track B — Translation artifacts and object preservation [SECOND PRIORITY]

**Question:** Does CycleGAN erase tiny pedestrians, alter object boundaries, amplify background texture, or create misleading patterns that hurt teacher guidance?

**First tests:** On the same thermal frames, compare native input, pseudo-RGB output, matched boxes, teacher detections, teacher feature responses, and final student errors. Stratify by object size, contrast, occlusion, scene/video, and distance proxy only if reliably observable. Human-review a **blinded, pre-specified** sample of correct/incorrect cases.

**Go/no-go:** If artifacts plausibly explain systematic errors *and* a controlled intervention improves detection across videos/seeds, consider an object-preserving translation or selective distillation follow-up. Do not claim causal evidence solely from pictures.

### Track C — Small and distant target detection

Compare stock thermal YOLO, P2 head, optional ViT-neck architecture, and selected distillation variants using identical data and fair compute. Audit shows P2-on-FLIR and ViT-YOLO already exist in literature; adding these modules alone is not a novel contribution. Test whether translation/teacher supervision interacts meaningfully with fine-resolution targets. Require improvements in AP-small without unacceptable overall AP/latency losses.

### Track D — Does enhancement help detection enough to justify a router?

**Do this before training a weather gate.** Evaluate identity, gamma, contrast/local normalization and any valid trained enhancer on the *same fixed detector*. Examine recovered/missed detections and whether a diagnostic per-image **oracle** choice suggests headroom. This oracle uses labels and is only an offline upper bound, not a deployable model. Compare with ordinary robustness augmentation.

The old gamma rule reacts to mean image value, not directly to thermal fog-related contrast loss. The untrained rain-removal model is **not** evidence of effective thermal deraining. Do not import visible-rain streak assumptions to LWIR without physical and empirical justification.

**Go/no-go:** If the best offline choice provides negligible reliable gain, stop router development. If there is meaningful, reproducible gain and a cheap predictor can approach it, consider a separate routing paper against strong published routing baselines.

### Track E — Temporal consistency across thermal video

After a good detector exists, test changes in confidence, temporary missed detections, gain/AGC changes, camera motion, and scene content on consecutive frames. Use ground-truth tracks where available; otherwise clearly label estimated tracks. Keep this a secondary study unless valid video annotations and multiple independent recordings support stronger claims.

## 4. Immediate execution order — small, falsification-first

**Gate 0: access and provenance (no large training).**
- Check collaborator consent/authorship and whether original FLIR data access/terms permit planned derivatives.
- Request and inventory `cyclegan_epoch_45.pth`, the **actual** teacher checkpoint (stock or fine-tuned), `checkpoints_distill/*.pth`, training logs, config/requirements, and evaluator output.
- Fingerprint the exact original runs and hashes. Do not claim any result was reproduced merely because code ran on synthetic batches.
- Verify all split IDs, annotation class mapping, frame uniqueness, any genuinely paired RGB/thermal subset, and video overlap.
- Confirm GPU, RAM, free disk, and **other active research jobs** before allocating resources.

**Gate 1: regression audit (small CPU/GPU smoke tests).**
- Validate strict checkpoint round-trip and exact trained architecture.
- Evaluate in `eval()`, confirm BatchNorm stats are unchanged, and compare to pycocotools.
- Correct class IDs, teacher label conversion, CycleGAN cache/resume and fallback; add tests that pass under the **patched** implementation (the old T05 test partly characterizes the broken code).
- Verify loss/backprop on the pinned Ultralytics release, and quantify student parameter count and inference latency honestly.

**Gate 2: reproducible baseline and pilot.**
- Run a tiny **video-grouped** pilot with no teacher, then pseudo-RGB teacher only if real teacher and CycleGAN weights are available and valid.
- Confirm both are trainable, losses finite, evaluation meaningful, and saved checkpoints reload correctly. Pilot results are diagnostic, not headline.
- Next execute a bounded, comparable no-teacher vs thermal-teacher vs grayscale-RGB-teacher vs pseudo-RGB-teacher experiment. Freeze exact protocol before test evaluation.

**Gate 3: stop or scale.**
- Only request authorization for multi-seed, full training when Gate 0–2 pass and a genuine open research question survives prior-art review.
- Predefine budgets, stopping rules, number of seeds/videos, evaluation and uncertainty; do not tune protocol to obtain positive significance.
- If the teacher does not help, diagnose negative transfer and decide whether Track B, C or D offers a measurable stronger contribution. Preserve negative and inconclusive outcomes.

## 5. Independent literature and publication review

Re-check exact claims and applicability (dataset version/class subsets, annotations, teacher type, translator at train vs test, metrics, code availability) against at least:
- Borrow from Anywhere (2019), https://arxiv.org/abs/1905.08789
- CGDet (2025), https://arxiv.org/abs/2511.01435
- Thermal-Det (CVPR 2026), https://arxiv.org/abs/2605.10130
- FreqKD (2026), https://arxiv.org/abs/2606.11572
- YOLO-APDM (2024), https://pmc.ncbi.nlm.nih.gov/articles/PMC11598205/
- GDIP-YOLO (2023), https://arxiv.org/abs/2209.14922
- Relevant 2026 studies available at audit time, and independently search for later relevant work.

Separate **published facts**, **code-observed behavior**, **synthetic smoke-test outcomes**, **hypotheses**, and **actual detector experiments**. Explicitly correct earlier ChatGPT/Claude assertions that do not survive the check.

A routine combination of existing methods is unlikely to qualify as a strong method paper. Target an applied journal/workshop **only after experiments establish a reproducible and well-positioned finding**. Do not invent numerical improvements, acceptance probabilities, 'first' claims, or results tables.

## 6. Required outputs from Claude

Create or update on a **separate working branch** (do not alter an ongoing experiment):
- `audit_2026/followup/CURRENT_STATE_AND_BLOCKERS.md` — exact verified commits, available/missing data and weights, tested versus only reported behaviors.
- `audit_2026/followup/PRIOR_ART_COUNTERCHECK.md` — sources, matching and differences, refuted/unsupported suggestions.
- `audit_2026/followup/PILOT_PROTOCOL.md` — smallest fair teacher comparison, leakage prevention, resources, abort criteria, approval gates.
- `audit_2026/followup/EXPERIMENT_RESULTS.md` — only completed runs with immutable config, seed, checkpoint hash, metrics and comparison; otherwise mark **NOT RUN**.
- `audit_2026/followup/PUBLICATION_OPTIONS.md` — candid ranking of Tracks A–E and recommendation with reasons **including reasons not to proceed**.

Do not automatically write a positive-results paper from planned experiments. Avoid adding new mathematics or theoretical results: the requested work is experimental and evidence-led.

## 7. Boundaries and permissions

- Do not force-push, rewrite history, merge into `main`, delete branches, change repository visibility, or publish datasets/checkpoints without the relevant permissions.
- Do not interrupt ongoing experiments (including separate research projects or users' GPU/CPU work); no mass downloads or compute-intensive runs without explicit user authorization.
- Keep collaborator source attribution and licensing intact; verify ownership/consent before releasing any original assets.
- Distinguish a research **recommendation** from a **validated finding** in every output.
- If materials are missing or an experiment cannot be run, report the blocker and a precise retrieval/request list rather than fabricating numbers.

**Instruction:** Start at Gate 0. Independently verify the above against current repositories and literature, report disagreements, and propose the smallest informative pilot before requesting any expensive training.
