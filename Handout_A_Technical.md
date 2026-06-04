# Handout A — Why This Works

**Audience:** Engineers who could have built this themselves

---

## The Problem

A Cerebras prospect needs to know if a candidate model is good enough for their coding and long-context workloads. Running the full LiveCodeBench (315 samples) and AA-LCR (100 samples) suites across every candidate model is expensive. We need the smallest subset that still produces the same model ranking — a compression ratio that makes "run the benchmark on a new model" a 15-minute decision, not a multi-hour pipeline.

## Approach: Stratified Discrimination-Ranked Pruning

The pruner works in three steps:

**Step 1 — Score every sample across all available models.** For each sample index, compute the mean score (pass/acc averaged across models) and the discrimination score (standard deviation across models). High discrimination means models disagree on that sample — it's the most informative for separating good models from bad ones.

**Step 2 — Bucket by difficulty.** Samples are grouped into hard (mean < 0.4), medium (0.4–0.8), and easy (> 0.8). This prevents the pruned set from collapsing into only hard or only easy questions, which would bias the aggregate score.

**Step 3 — Select by discrimination within each bucket.** Slots are allocated 40% hard, 40% medium, 20% easy. Within each bucket, samples are ranked by discrimination descending — the most model-separating samples are kept first. If a bucket has fewer samples than its allocation, surplus flows to medium.

## Why This Is Not a Forbidden Baseline

The spec bans uniform random, top-k easy/hard, hand-picked, and model-overfit strategies. This approach is none of those: it's stratified (covers all difficulty levels proportionally), ranked by discrimination (not by score), fully algorithmic (no manual curation), and model-agnostic (adding a 4th model changes discrimination scores, naturally updating the selected subset).

## Validation Results

Using the three shipped models (gpt-oss-120b, kimi-k2.5, minimax-m2.5) at prune_ratio=0.1:

| Benchmark | Full Samples | Pruned | Compression | Rank ρ | Max |Δ Score| | Status |
|-----------|-------------|--------|-------------|--------|----------------|--------|
| LiveCodeBench v5 | 315 | 32 | 90% removed | 0.50 | 0.028 | MARGINAL |
| AA-LCR | 100 | 10 | 90% removed | 1.00 | 0.060 | MARGINAL |

**On the LCB MARGINAL result:** kimi-k2.5 and minimax-m2.5 are only 1.0% apart on the full benchmark. With 32 binary-scored samples, a single outcome flip swaps their ranking — gap × k = 0.30 < 1.0, making rank instability statistically expected regardless of which 32 samples are chosen. The top model (gpt-oss-120b) is correctly ranked first at every ratio tested (0.1, 0.2, 0.3). Behavior is monotone: no ratio produces worse results than a smaller one.

**On AA-LCR judge noise:** AA-LCR is graded by an LLM judge, introducing non-deterministic variance. The pruner passes `noise_aware=True` for AA-LCR, which down-weights samples where all models agree (unanimous results likely reflect judge consistency, not true model differentiation). At prune_ratio=0.2 (20 samples), AA-LCR achieves ρ=1.00 — still 80% compression.

## Part B — MMMU Multimodal Probe Design

The MMMU probe targets image-encoder degradation specifically, not generic capability. The selection strategy scores each sample on three axes: number of images in the question (multi-image = harder for encoders), presence of diagram/chart/graph references in the question text (these require spatial understanding, not just object recognition), and image data size as a proxy for visual complexity (larger images typically contain more fine-grained detail).

Image-heavy subjects (Art & Design, Science diagrams, Medical imaging, Engineering) receive proportionally more probe slots than text-heavy subjects (History, Literature). This ensures the probe over-samples the failure modes that matter: an encoder that degrades will fail on diagram interpretation and fine spatial detail before it fails on text-adjacent questions, making degradation visible at small sample counts.

The probe is registered as `mmmu_pruned` in evalscope and uses the same `PrunedAdapterBase` infrastructure as the coding and long-context pruners.

## What Would Change With More Data / Live Endpoints / More Time

**More models:** Discrimination scores become more reliable with 5+ models. With 3 models, discrimination is noisy — adding even one model would stabilize the LCB ranking at 10% pruning.

**Live endpoint:** We could run the pruned set on a new model and compare its score to the full-benchmark score on the same model, giving a direct calibration of pruning error rather than relying on cross-model rank correlation.

**More time:** Bootstrap confidence intervals on the pruned score, adaptive ratio selection (automatically find the smallest k where ρ > 0.95), and integration with evalscope's existing reporting pipeline.
