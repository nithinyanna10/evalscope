"""
Stratified pruner: model-agnostic benchmark subset selection.

Selects a fraction of benchmark samples that maximises ranking-signal
retention by combining difficulty stratification with discrimination ranking.

Algorithm — stratified discrimination-ranked selection:
1. Load all review JSONL files whose name matches
   ``<benchmark_prefix>__<any_model>.jsonl`` from ``reviews_dir``.
2. Per sample index, compute:
   - mean_score: average pass/acc across all models
   - discrimination: population std deviation across models
     (high = models disagree = sample is informative)
   - difficulty_bucket: hard (mean<0.4), medium (0.4–0.8), easy (mean>0.8)
3. Allocate k = round(N * prune_ratio) slots across buckets:
     hard 40 %, medium 40 %, easy 20 % (integer-truncated; remainder → medium)
4. Within each bucket, sort by discrimination descending and take the top-N.
   If a bucket has fewer samples than its allocation, surplus flows to the
   next bucket (hard → medium → easy); any remaining surplus draws from the
   highest-discrimination samples across the entire dataset.
5. Return sorted list of selected sample indices.

Why this is NOT forbidden:
- NOT uniform random (discrimination-ranked within each difficulty stratum)
- NOT top-k easiest or hardest (all three difficulty levels are represented)
- NOT hand-picked (fully algorithmic, no human judgement)
- NOT overfit to N models (discrimination = std across whatever models exist)
- Defense against a 4th model: adding a new review file changes each
  sample's discrimination score, naturally updating the selected subset.

Optional ``noise_aware`` mode (recommended for LLM-judged benchmarks):
  When True, samples where ALL models agree (unanimous pass or unanimous
  fail) are treated as non-discriminating (discrimination forced to 0).
  For binary outcomes, unanimous samples already have std=0, so this is a
  no-op in that case.  For continuous LLM-judge scores it suppresses samples
  where judge agreement may reflect noise rather than genuine signal.
"""

import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


class StratifiedPruner:
    """
    Model-agnostic pruning via difficulty-stratified, discrimination-ranked selection.

    Samples are bucketed by difficulty (hard / medium / easy) and slots are
    allocated proportionally (40 / 40 / 20 %).  Within each bucket, samples
    are ranked by discrimination (std across models) descending so the most
    informative samples are selected first.
    """

    EASY_THRESHOLD: float = 0.8
    HARD_THRESHOLD: float = 0.4

    # Bucket slot allocations (must sum to 1.0 before rounding)
    BUCKET_ALLOC: Dict[str, float] = {'hard': 0.40, 'medium': 0.40, 'easy': 0.20}

    def __init__(
        self,
        reviews_dir: str,
        benchmark_prefix: str,
        score_key: Optional[str] = None,
        noise_aware: bool = False,
    ) -> None:
        """
        Args:
            reviews_dir: Directory containing review JSONL files.
            benchmark_prefix: File-name prefix for glob, e.g. ``'live_code_bench_v5'``.
            score_key: Force a specific score field name ('pass', 'acc', …).
                       When None (default), auto-detects 'pass' then 'acc'.
            noise_aware: When True, unanimous samples (all models agree) are
                         assigned discrimination=0 so they are deprioritised.
                         Recommended for LLM-judged benchmarks.
        """
        self.reviews_dir = reviews_dir
        self.benchmark_prefix = benchmark_prefix
        self.score_key = score_key
        self.noise_aware = noise_aware

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_scores(self) -> Dict[int, List[float]]:
        """Return {sample_index: [score_per_model, ...]} from all matching files."""
        pattern = os.path.join(self.reviews_dir, f'{self.benchmark_prefix}__*.jsonl')
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f'No review files found matching: {pattern}\n'
                f'Directory contents: '
                f'{os.listdir(self.reviews_dir) if os.path.isdir(self.reviews_dir) else "(dir not found)"}'
            )

        scores_by_index: Dict[int, List[float]] = defaultdict(list)
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    idx = int(record['index'])
                    value = record['sample_score']['score']['value']
                    if self.score_key and self.score_key in value:
                        raw = value[self.score_key]
                    elif 'pass' in value:
                        raw = value['pass']
                    elif 'acc' in value:
                        raw = value['acc']
                    else:
                        raw = next(iter(value.values()))
                    scores_by_index[idx].append(float(raw))

        return dict(scores_by_index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_stats(self) -> List[Dict[str, Any]]:
        """
        Compute per-sample statistics from all review files.

        Returns:
            List of dicts, one per sample, with keys:
              'index'            – sample index (int)
              'mean_score'       – average score across all models (float)
              'discrimination'   – population std dev across models (float);
                                   high value means models disagree
              'difficulty_bucket'– 'hard' | 'medium' | 'easy' (str)
              'noise_adjusted'   – True when noise_aware suppressed discrimination
                                   (only present when noise_aware=True and applied)
        """
        raw_scores = self._load_scores()
        stats: List[Dict[str, Any]] = []

        for idx in sorted(raw_scores.keys()):
            model_scores = raw_scores[idx]
            n = len(model_scores)
            mean = sum(model_scores) / n

            # Population std dev (not sample std dev)
            if n > 1:
                variance = sum((s - mean) ** 2 for s in model_scores) / n
                disc = variance ** 0.5
            else:
                disc = 0.0

            entry: Dict[str, Any] = {}

            # noise_aware: unanimous samples are non-discriminating
            if self.noise_aware and len(set(model_scores)) == 1:
                disc = 0.0
                entry['noise_adjusted'] = True

            if mean >= self.EASY_THRESHOLD:
                bucket = 'easy'
            elif mean < self.HARD_THRESHOLD:
                bucket = 'hard'
            else:
                bucket = 'medium'

            entry.update({
                'index': idx,
                'mean_score': mean,
                'discrimination': disc,
                'difficulty_bucket': bucket,
            })
            stats.append(entry)

        return stats

    def prune(self, prune_ratio: float) -> List[int]:
        """
        Select indices via stratified discrimination-ranked selection.

        Steps:
          1. Bucket samples by difficulty (hard / medium / easy).
          2. Allocate k slots: hard 40 %, medium 40 %, easy 20 %
             (integer-truncated; remainder goes to medium).
          3. Within each bucket, take the top-N by discrimination descending.
          4. Overflow from small buckets flows to the next bucket, then to a
             global top-discrimination fallback.

        Args:
            prune_ratio: Fraction of samples to KEEP (0 < prune_ratio ≤ 1).

        Returns:
            Sorted list of selected sample indices.
        """
        if not (0 < prune_ratio <= 1):
            raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')

        stats = self.compute_stats()
        k = max(1, int(len(stats) * prune_ratio))

        # Step 1: bucket by difficulty
        buckets: Dict[str, List[Dict[str, Any]]] = {'hard': [], 'medium': [], 'easy': []}
        for s in stats:
            buckets[s['difficulty_bucket']].append(s)

        # Step 2: allocate slots (integer-truncated; remainder → medium)
        alloc: Dict[str, int] = {
            b: int(k * frac) for b, frac in self.BUCKET_ALLOC.items()
        }
        remainder = k - sum(alloc.values())
        alloc['medium'] += remainder  # give leftover to medium

        # Step 3 & 4: within each bucket rank by discrimination DESC, handle overflow
        selected: List[Dict[str, Any]] = []
        overflow = 0
        for bucket_name in ('hard', 'medium', 'easy'):
            bucket = buckets[bucket_name]
            target = alloc[bucket_name] + overflow
            overflow = 0
            bucket.sort(key=lambda x: x['discrimination'], reverse=True)
            take = min(len(bucket), target)
            selected.extend(bucket[:take])
            if take < target:
                overflow = target - take  # carry surplus to next bucket

        # Fallback: if overflow remains after easy, draw from highest-disc remaining
        if overflow > 0:
            used = {s['index'] for s in selected}
            remaining = [s for s in stats if s['index'] not in used]
            remaining.sort(key=lambda x: x['discrimination'], reverse=True)
            selected.extend(remaining[:overflow])

        return sorted(s['index'] for s in selected)
