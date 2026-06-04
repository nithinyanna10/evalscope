"""Pruned variant of LiveCodeBench using stratified sample selection."""

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope_ext.pruning.pruned_adapter_base import PrunedAdapterBase


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

Pruned variant of LiveCodeBench (v5 subset) using stratified sample selection.
Keeps ~10% of samples (configurable via prune_ratio) that maximise ranking-signal
retention by selecting discriminative problems across difficulty buckets.

Use reviews_dir to point at precomputed model-review JSONL files.
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template='### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy to use. Currently only "stratified" is supported.',
                'value': 'stratified',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to KEEP (e.g. 0.1 keeps 10 %).',
                'value': 0.1,
            },
            'reviews_dir': {
                'type': 'str',
                'description': 'Directory containing precomputed review JSONL files.',
                'value': '',
            },
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems starting from this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {},
            },
        },
    )
)
class LiveCodeBenchPrunedAdapter(PrunedAdapterBase, LiveCodeBenchAdapter):
    """
    LiveCodeBench with stratified sample pruning.

    MRO: LiveCodeBenchPrunedAdapter → PrunedAdapterBase → LiveCodeBenchAdapter
         → DefaultDataAdapter → DataAdapter

    PrunedAdapterBase (listed first) intercepts load_dataset(), lets the
    parent chain handle all LCB-specific loading and prompt formatting, then
    filters the resulting DatasetDict to the strategically selected indices.
    """

    _reviews_benchmark_prefix: str = 'live_code_bench_v5'
