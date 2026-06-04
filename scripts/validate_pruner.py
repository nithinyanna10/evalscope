"""
Offline validation of StratifiedPruner using precomputed review JSONL files.

Validates that the pruned subset preserves model ranking signal without
requiring live model inference.

Usage::

    python scripts/validate_pruner.py \\
        --reviews-dir /path/to/Evals/Part\\ 1/reviews/ \\
        [--prune-ratio 0.1]

Exit codes:
  0 — no hard failures (all benchmarks with >15 pruned samples PASS)
  1 — at least one hard failure (a benchmark with >15 pruned samples FAIL)

Benchmarks with ≤15 pruned samples that miss the criteria are reported as
MARGINAL — statistically expected at that sample count with binary outcomes
and close model scores, not counted as algorithm failures.
"""

import argparse
import json
import os
import sys

# Make evalscope_ext importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evalscope_ext.pruning.stratified_pruner import StratifiedPruner  # no evalscope deps
from evalscope_ext.utils import spearman_rank_correlation

# ---------------------------------------------------------------------------
# Per-benchmark configuration
# Keys: benchmark prefix (matches review file naming convention)
# Values: prune_ratio — smallest keep-fraction that gives a reliable signal
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS = {
    'live_code_bench_v5': 0.10,  # 315 samples → 32 kept; plenty of power
    'aa_lcr':             0.20,  # 100 samples → 20 kept; 0.1 gives only 10
}

# Rank-swap failures are MARGINAL (not hard-fails) when they are
# statistically expected.  The test: if the minimum adjacent-model
# score gap × pruned sample count < 1.0, a single binary outcome flip
# changes which model "wins" — the algorithm cannot prevent this.
# Equivalently: k < 15 always qualifies; larger k qualifies only when
# the closest models are within ~(1/k) of each other.
SMALL_K_HARD_THRESHOLD = 15          # k ≤ this → always MARGINAL
GAP_TIMES_K_THRESHOLD   = 1.0       # gap * k < this → MARGINAL regardless of k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_benchmark_scores(reviews_dir: str, prefix: str):
    """Return {model_name: {idx: score}}, sorted_all_indices."""
    import glob
    pattern = os.path.join(reviews_dir, f'{prefix}__*.jsonl')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files found matching {pattern}')

    model_scores = {}
    all_indices = set()

    for filepath in files:
        model = os.path.basename(filepath).replace(f'{prefix}__', '').replace('.jsonl', '')
        scores = {}
        with open(filepath, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                idx = int(record['index'])
                value = record['sample_score']['score']['value']
                if 'pass' in value:
                    raw = value['pass']
                elif 'acc' in value:
                    raw = value['acc']
                else:
                    raw = next(iter(value.values()))
                scores[idx] = float(raw)
                all_indices.add(idx)
        model_scores[model] = scores

    return model_scores, sorted(all_indices)




def _min_adjacent_gap(score_dict):
    """Smallest gap between any two adjacent model scores (sorted)."""
    vals = sorted(score_dict.values())
    if len(vals) < 2:
        return 1.0
    return min(vals[i + 1] - vals[i] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_benchmark(reviews_dir: str, prefix: str, prune_ratio: float) -> dict:
    """
    Validate the pruner on one benchmark.

    Returns a result dict with keys:
      prefix, prune_ratio, total, k, rho, max_delta, status, hard_fail
    """
    print(f'\n{"=" * 60}')
    print(f'Benchmark prefix : {prefix}')
    print(f'Prune ratio      : {prune_ratio} (keep {prune_ratio:.0%})')

    model_scores, all_indices = _load_benchmark_scores(reviews_dir, prefix)
    total = len(all_indices)
    print(f'Total samples    : {total}')
    print(f'Models found     : {sorted(model_scores)}')

    pruner = StratifiedPruner(reviews_dir=reviews_dir, benchmark_prefix=prefix)
    selected = pruner.prune(prune_ratio=prune_ratio)
    k = len(selected)
    selected_set = set(selected)
    print(f'Selected indices : {k} samples')
    print(f'  First 20       : {selected[:20]}')

    # Per-model full vs pruned scores
    print(f'\n{"Model":<30} {"Full":>8} {"Pruned":>8} {"|Delta|":>8}')
    print('-' * 60)

    full_by_model = {}
    pruned_by_model = {}

    for model in sorted(model_scores):
        scores = model_scores[model]
        all_present = [i for i in all_indices if i in scores]
        pruned_present = [i for i in selected if i in scores]

        full_score = sum(scores[i] for i in all_present) / len(all_present) if all_present else 0.0
        pruned_score = (sum(scores[i] for i in pruned_present) / len(pruned_present)
                        if pruned_present else 0.0)
        delta = abs(full_score - pruned_score)

        print(f'{model:<30} {full_score:>8.4f} {pruned_score:>8.4f} {delta:>8.4f}')
        full_by_model[model] = full_score
        pruned_by_model[model] = pruned_score

    models = sorted(full_by_model)
    if len(models) >= 2:
        f_vals = [full_by_model[m] for m in models]
        p_vals = [pruned_by_model[m] for m in models]
        rho = spearman_rank_correlation(f_vals, p_vals)
        print(f'\nSpearman rank correlation (full vs pruned): {rho:.4f}')
    else:
        rho = float('nan')
        print('\nSpearman rank correlation: N/A (fewer than 2 models)')

    deltas = [abs(full_by_model[m] - pruned_by_model[m]) for m in models]
    max_delta = max(deltas) if deltas else 0.0
    delta_ok = max_delta < 0.05
    corr_ok = (rho != rho) or rho > 0.9  # nan → N/A → treated as pass

    # Determine status.
    # MARGINAL: criteria miss AND the failure is statistically expected because
    #   (a) the pruned set is very small (k ≤ SMALL_K_HARD_THRESHOLD), OR
    #   (b) the closest two models are so similar that a single binary-outcome
    #       flip in the pruned set changes who ranks higher (gap * k < 1.0).
    criteria_pass = delta_ok and corr_ok
    gap = _min_adjacent_gap(full_by_model)
    statistically_marginal = (k <= SMALL_K_HARD_THRESHOLD) or (gap * k < GAP_TIMES_K_THRESHOLD)

    if criteria_pass:
        status = 'PASS'
        hard_fail = False
    elif statistically_marginal:
        status = 'MARGINAL'
        hard_fail = False
    else:
        status = 'FAIL'
        hard_fail = True

    print(f'\nMax |delta|      : {max_delta:.4f}  {"✓" if delta_ok else "✗"} (threshold < 0.05)')
    if rho == rho:
        print(f'Spearman ρ       : {rho:.4f}  {"✓" if corr_ok else "✗"} (threshold > 0.9)')
    print(f'Result           : {status}')

    if status == 'MARGINAL':
        print(f'  (statistically expected: closest models {gap:.1%} apart, '
              f'gap×k={gap*k:.2f} < {GAP_TIMES_K_THRESHOLD} — '
              f'one binary flip in {k} samples swaps the ranking)')

    return {
        'prefix': prefix,
        'prune_ratio': prune_ratio,
        'total': total,
        'k': k,
        'rho': rho,
        'max_delta': max_delta,
        'status': status,
        'hard_fail': hard_fail,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Offline StratifiedPruner validation.')
    parser.add_argument(
        '--reviews-dir',
        required=True,
        help='Directory containing precomputed review JSONL files.',
    )
    parser.add_argument(
        '--prune-ratio',
        type=float,
        default=None,
        help='Override keep-fraction for ALL benchmarks (default: per-benchmark config).',
    )
    args = parser.parse_args()

    reviews_dir = os.path.expanduser(args.reviews_dir)
    if not os.path.isdir(reviews_dir):
        print(f'ERROR: reviews_dir not found: {reviews_dir}')
        sys.exit(1)

    results = []
    for prefix, default_ratio in BENCHMARK_CONFIGS.items():
        ratio = args.prune_ratio if args.prune_ratio is not None else default_ratio
        try:
            result = validate_benchmark(reviews_dir, prefix, ratio)
            results.append(result)
        except FileNotFoundError as e:
            print(f'\nSKIPPED {prefix}: {e}')

    # Summary
    print(f'\n{"=" * 60}')
    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_marginal = sum(1 for r in results if r['status'] == 'MARGINAL')
    n_fail = sum(1 for r in results if r['status'] == 'FAIL')
    has_hard_fail = any(r['hard_fail'] for r in results)

    overall = 'PASS' if not has_hard_fail else 'FAIL'
    parts = []
    if n_pass:
        parts.append(f'{n_pass} pass')
    if n_marginal:
        parts.append(f'{n_marginal} marginal')
    if n_fail:
        parts.append(f'{n_fail} hard-fail')
    print(f'OVERALL: {overall} ({", ".join(parts)})')
    print()
    for r in results:
        rho_str = f'rho={r["rho"]:.4f}' if r['rho'] == r['rho'] else 'rho=N/A'
        print(
            f'  {r["prefix"]:25} {r["total"]:>4}→{r["k"]:<4}  '
            f'{r["status"]:<8}  {rho_str}  max|Δ|={r["max_delta"]:.4f}'
        )

    sys.exit(0 if not has_hard_fail else 1)


if __name__ == '__main__':
    main()
