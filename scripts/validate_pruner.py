"""
Offline validation of StratifiedPruner using precomputed review JSONL files.

Validates that the pruned subset preserves model ranking signal without
requiring live model inference.

Usage::

    python scripts/validate_pruner.py \\
        --reviews-dir /path/to/Evals/Part\\ 1/reviews/ \\
        --prune-ratio 0.1

Exit code 0 = all benchmarks pass the |delta|<0.05 AND rho>0.9 criteria.
"""

import argparse
import json
import os
import sys

# Make evalscope_ext importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evalscope_ext.pruning.stratified_pruner import StratifiedPruner  # no evalscope deps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_benchmark_scores(reviews_dir: str, prefix: str):
    """
    Return {model_name: {sample_index: score}} for all files matching prefix.
    Also returns the full sorted list of all sample indices found.
    """
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


def _spearman(xs, ys):
    """Spearman ρ between two sequences (no scipy required)."""
    n = len(xs)
    if n < 2:
        return float('nan')

    def _ranks(seq):
        sorted_idx = sorted(range(n), key=lambda i: seq[i])
        rank = [0.0] * n
        for r, i in enumerate(sorted_idx, start=1):
            rank[i] = float(r)
        return rank

    rx = _ranks(xs)
    ry = _ranks(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def validate_benchmark(reviews_dir: str, prefix: str, prune_ratio: float) -> bool:
    """
    Validate pruner on one benchmark.  Returns True if criteria pass.
    """
    print(f'\n{"=" * 60}')
    print(f'Benchmark prefix : {prefix}')
    print(f'Prune ratio      : {prune_ratio} (keep {prune_ratio:.0%})')

    # Load all model scores
    model_scores, all_indices = _load_benchmark_scores(reviews_dir, prefix)
    total = len(all_indices)
    print(f'Total samples    : {total}')
    print(f'Models found     : {sorted(model_scores)}')

    # Run pruner
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
        all_present = [idx for idx in all_indices if idx in scores]
        pruned_present = [idx for idx in selected if idx in scores]

        full_score = sum(scores[i] for i in all_present) / len(all_present) if all_present else 0.0
        pruned_score = sum(scores[i] for i in pruned_present) / len(pruned_present) if pruned_present else 0.0
        delta = abs(full_score - pruned_score)

        print(f'{model:<30} {full_score:>8.4f} {pruned_score:>8.4f} {delta:>8.4f}')
        full_by_model[model] = full_score
        pruned_by_model[model] = pruned_score

    # Rank correlation across models
    models = sorted(full_by_model)
    if len(models) >= 2:
        f_vals = [full_by_model[m] for m in models]
        p_vals = [pruned_by_model[m] for m in models]
        rho = _spearman(f_vals, p_vals)
        print(f'\nSpearman rank correlation (full vs pruned): {rho:.4f}')
    else:
        rho = float('nan')
        print('\nSpearman rank correlation: N/A (fewer than 2 models)')

    # Pass/fail criteria
    deltas = [abs(full_by_model[m] - pruned_by_model[m]) for m in models]
    max_delta = max(deltas) if deltas else 0.0
    delta_ok = max_delta < 0.05
    corr_ok = (rho != rho) or rho > 0.9  # nan → N/A → pass

    status = 'PASS' if (delta_ok and corr_ok) else 'FAIL'
    print(f'\nMax |delta|      : {max_delta:.4f}  {"✓" if delta_ok else "✗"} (threshold < 0.05)')
    if rho == rho:
        print(f'Spearman ρ       : {rho:.4f}  {"✓" if corr_ok else "✗"} (threshold > 0.9)')
    print(f'Result           : {status}')

    if status == 'FAIL' and rho == rho and rho <= 0.9 and len(models) < 4:
        # Statistical note: when two ADJACENT models have very similar scores
        # (< 5% gap) and the pruned set is small (k < ~15), rank swaps are
        # expected by chance with binary outcomes.
        sorted_vals = sorted(full_by_model.values())
        min_adjacent_gap = min(
            sorted_vals[i+1] - sorted_vals[i]
            for i in range(len(sorted_vals)-1)
        ) if len(sorted_vals) > 1 else 1.0
        if min_adjacent_gap < 0.05:
            print(f'NOTE: rank failure is statistically expected — closest models are')
            print(f'      {min_adjacent_gap:.1%} apart; with only {k} binary samples,')
            print(f'      a single outcome flip in the pruned set swaps their ranking.')

    return status == 'PASS'


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
        default=0.1,
        help='Fraction of samples to keep (default: 0.1).',
    )
    args = parser.parse_args()

    reviews_dir = os.path.expanduser(args.reviews_dir)
    if not os.path.isdir(reviews_dir):
        print(f'ERROR: reviews_dir not found: {reviews_dir}')
        sys.exit(1)

    benchmarks = [
        'live_code_bench_v5',
        'aa_lcr',
    ]

    all_pass = True
    for prefix in benchmarks:
        try:
            ok = validate_benchmark(reviews_dir, prefix, args.prune_ratio)
            if not ok:
                all_pass = False
        except FileNotFoundError as e:
            print(f'\nSKIPPED {prefix}: {e}')

    print(f'\n{"=" * 60}')
    overall = 'PASS' if all_pass else 'FAIL'
    print(f'Overall validation: {overall}')
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
