"""
compare_runs — compare full vs pruned evalscope evaluation runs.

Usage::

    python -m evalscope_ext.tools.compare_runs \\
        --full  ./results_full/ \\
        --pruned ./results_pruned/ \\
        [--output compare_report.json]

Input directories are expected to contain evalscope output trees::

    <dir>/
      reports/
        <model_name>/
          <benchmark_name>.json   # evalscope Report JSON

For each benchmark found in both directories the tool computes:
  - full_score:     aggregate score from the full run
  - pruned_score:   aggregate score from the pruned run
  - score_delta:    abs(full_score - pruned_score)
  - rank_corr:      Spearman ρ between model rankings (full vs pruned),
                    or N/A when fewer than 2 models are available
  - status:         PASS when |delta| < 0.05 AND (ρ > 0.9 OR N/A)

A compare_report.json is written alongside the console table.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evalscope_ext.utils import spearman_rank_correlation

PASS_DELTA_THRESHOLD = 0.05
PASS_CORR_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def _find_report_files(root: str) -> List[Path]:
    """Recursively find all *.json files under <root>/reports/."""
    reports_root = Path(root) / 'reports'
    if not reports_root.exists():
        # Fallback: search the whole tree for json files
        reports_root = Path(root)
    return list(reports_root.rglob('*.json'))


def _parse_report(path: Path) -> Optional[Tuple[str, str, float]]:
    """
    Parse a single report JSON.

    Returns (model_name, benchmark_name, score) or None on failure.
    """
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        model_name = data.get('model_name', '')
        dataset_name = data.get('dataset_name', '')
        score = float(data.get('score', 0.0))
        if model_name and dataset_name:
            return model_name, dataset_name, score
    except Exception:
        pass
    return None


def _collect_scores(root: str) -> Dict[str, Dict[str, float]]:
    """
    Return {benchmark_name: {model_name: score}} from all reports under root.
    """
    scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for path in _find_report_files(root):
        parsed = _parse_report(path)
        if parsed:
            model, benchmark, score = parsed
            scores[benchmark][model] = score
    return dict(scores)


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare(full_dir: str, pruned_dir: str) -> List[dict]:
    """
    Compare full vs pruned runs and return a list of per-benchmark result dicts.
    """
    full_scores = _collect_scores(full_dir)
    pruned_scores = _collect_scores(pruned_dir)

    # Canonicalise benchmark names: pruned benchmark may append '_pruned'
    # Build mapping: canonical_name → (full_benchmark_key, pruned_benchmark_key)
    def _canonical(name: str) -> str:
        return name.removesuffix('_pruned')

    full_canon = {_canonical(k): k for k in full_scores}
    pruned_canon = {_canonical(k): k for k in pruned_scores}

    common = sorted(set(full_canon) & set(pruned_canon))
    if not common:
        print('WARNING: no matching benchmarks found between full and pruned directories.')
        print(f'  Full benchmarks:   {sorted(full_scores)}')
        print(f'  Pruned benchmarks: {sorted(pruned_scores)}')

    results = []
    for canonical in common:
        fk = full_canon[canonical]
        pk = pruned_canon[canonical]

        f_model_scores = full_scores[fk]
        p_model_scores = pruned_scores[pk]

        # Models present in both runs
        shared_models = sorted(set(f_model_scores) & set(p_model_scores))

        # Aggregate full and pruned scores (mean over shared models, or single model)
        if shared_models:
            full_score = sum(f_model_scores[m] for m in shared_models) / len(shared_models)
            pruned_score = sum(p_model_scores[m] for m in shared_models) / len(shared_models)
        else:
            # Fall back to whatever is available
            full_score = (sum(f_model_scores.values()) / len(f_model_scores)
                          if f_model_scores else 0.0)
            pruned_score = (sum(p_model_scores.values()) / len(p_model_scores)
                            if p_model_scores else 0.0)
            shared_models = []

        score_delta = abs(full_score - pruned_score)

        # Rank correlation across models
        if len(shared_models) >= 2:
            f_vals = [f_model_scores[m] for m in shared_models]
            p_vals = [p_model_scores[m] for m in shared_models]
            rank_corr = spearman_rank_correlation(f_vals, p_vals)
        else:
            rank_corr = float('nan')

        delta_ok = score_delta < PASS_DELTA_THRESHOLD
        corr_ok = (rank_corr != rank_corr) or rank_corr > PASS_CORR_THRESHOLD  # nan → N/A → pass
        status = 'PASS' if (delta_ok and corr_ok) else 'FAIL'

        # Estimate pruned_ratio from sample counts in report if available
        pruned_ratio = _estimate_pruned_ratio(full_scores, pruned_scores, fk, pk)

        results.append({
            'benchmark': canonical,
            'full_score': round(full_score, 4),
            'pruned_score': round(pruned_score, 4),
            'score_delta': round(score_delta, 4),
            'rank_corr': round(rank_corr, 4) if rank_corr == rank_corr else None,
            'pruned_ratio': pruned_ratio,
            'status': status,
            'models_compared': len(shared_models),
        })

    return results


def _estimate_pruned_ratio(
    full_scores: Dict, pruned_scores: Dict, fk: str, pk: str
) -> Optional[float]:
    """Try to estimate pruned_ratio from sample counts; returns None if unavailable."""
    # This would require access to report num field; skip for now
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_score(v: float) -> str:
    return f'{v:.4f}'


def _fmt_corr(v) -> str:
    if v is None:
        return '  N/A  '
    return f'{v:+.4f}'


def _print_table(results: List[dict]) -> None:
    if not results:
        print('No results to display.')
        return

    col_widths = {
        'benchmark': max(16, max(len(r['benchmark']) for r in results)),
        'full': 10,
        'pruned': 10,
        'delta': 10,
        'corr': 9,
        'status': 6,
    }

    def _row(*cells):
        return ('│ ' + ' │ '.join(str(c).ljust(w) for c, w in zip(cells, col_widths.values())) + ' │')

    sep_top = ('┌' + '┬'.join('─' * (w + 2) for w in col_widths.values()) + '┐')
    sep_mid = ('├' + '┼'.join('─' * (w + 2) for w in col_widths.values()) + '┤')
    sep_bot = ('└' + '┴'.join('─' * (w + 2) for w in col_widths.values()) + '┘')

    title = ' CEREBRAS EVAL COMPRESSION REPORT '
    total_width = sum(w + 3 for w in col_widths.values()) + 1
    print('┌' + title.center(total_width - 2, '─') + '┐')
    print(sep_top.replace('┌', '├').replace('┐', '┤'))
    print(_row('Benchmark', 'Full Score', 'Pruned Score', 'Δ Score', 'Rank ρ', 'Status'))
    print(sep_mid)

    for r in results:
        print(_row(
            r['benchmark'],
            _fmt_score(r['full_score']),
            _fmt_score(r['pruned_score']),
            _fmt_score(r['score_delta']),
            _fmt_corr(r['rank_corr']),
            r['status'],
        ))

    print(sep_bot)

    all_pass = all(r['status'] == 'PASS' for r in results)
    overall = 'PASS' if all_pass else 'FAIL'
    print(f'\nOverall: {overall} — pruned set {"preserves" if all_pass else "does NOT preserve"} ranking signal')
    if not all_pass:
        failed = [r['benchmark'] for r in results if r['status'] != 'PASS']
        print(f'  Failed benchmarks: {", ".join(failed)}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare full vs pruned evalscope evaluation runs.'
    )
    parser.add_argument('--full', required=True, help='Directory containing full-run evalscope outputs.')
    parser.add_argument('--pruned', required=True, help='Directory containing pruned-run evalscope outputs.')
    parser.add_argument(
        '--output',
        default='compare_report.json',
        help='Path for the JSON comparison report (default: compare_report.json).',
    )
    args = parser.parse_args(argv)

    results = compare(args.full, args.pruned)
    _print_table(results)

    # Write JSON report
    report = {
        'full_dir': os.path.abspath(args.full),
        'pruned_dir': os.path.abspath(args.pruned),
        'pass_criteria': {
            'max_score_delta': PASS_DELTA_THRESHOLD,
            'min_rank_corr': PASS_CORR_THRESHOLD,
        },
        'results': results,
        'overall': 'PASS' if all(r['status'] == 'PASS' for r in results) else 'FAIL',
    }
    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(f'\nJSON report written to: {args.output}')

    sys.exit(0 if report['overall'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
