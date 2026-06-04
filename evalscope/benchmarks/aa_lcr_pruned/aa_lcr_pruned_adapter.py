"""Pruned variant of AA-LCR using stratified sample selection."""

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.constants import Tags
from evalscope_ext.pruning.pruned_adapter_base import PrunedAdapterBase


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

Pruned variant of AA-LCR (Artificial Analysis Long Context Retrieval) using
stratified sample selection.  Keeps ~10% of samples (configurable via
prune_ratio) that maximise ranking-signal retention.

Use reviews_dir to point at precomputed model-review JSONL files.
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template="""
BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
""",
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
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files; if null will auto-download.',
                'value': None,
            },
        },
    )
)
class AaLcrPrunedAdapter(PrunedAdapterBase, AALCRAdapter):
    """
    AA-LCR with stratified sample pruning.

    MRO: AaLcrPrunedAdapter → PrunedAdapterBase → AALCRAdapter
         → DefaultDataAdapter → DataAdapter

    PrunedAdapterBase (listed first) intercepts load_dataset(), lets the
    parent chain handle AA-LCR-specific loading (including text_dir
    resolution), then filters the DatasetDict to the selected indices.
    """

    _reviews_benchmark_prefix: str = 'aa_lcr'
    _noise_aware: bool = True  # LLM-judged: unanimous samples may reflect judge noise
