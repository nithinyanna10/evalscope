"""
Pruned MMMU variant that specifically stresses image encoders.

Uses MMMUImageEncoderProbe to select questions where the answer REQUIRES
reading the image precisely (diagrams, charts, fine spatial detail), making
encoder degradation visible at small sample counts.
"""

from typing import Dict, List, Set

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter
from evalscope.constants import Tags
from evalscope_ext.pruning.pruned_adapter_base import PrunedAdapterBase


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Image-Encoder Probe)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

Pruned MMMU variant that specifically stresses image encoders rather than
general capability.  Questions are selected where the answer REQUIRES reading
the image precisely — diagrams, charts, fine spatial detail.

An encoder that degrades will fail these first while passing text-adjacent
questions, making degradation visible at small sample counts.

## Selection Strategy

- Subject-stratified: proportional slots per MMMU subject
- Image-heavy subjects (Art, Science, Medical, Engineering) receive boosted weight
- Within each subject, samples ranked by image complexity:
    score = 0.4 * norm(n_images) + 0.3 * has_diagram + 0.3 * norm(file_size)
- Minimum 2 samples per subject regardless of prune_ratio
""",
        dataset_id='AI-ModelScope/MMMU',
        subset_list=[
            'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
            'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
            'Clinical_Medicine', 'Computer_Science', 'Design',
            'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
            'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
            'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering',
            'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology',
        ],
        metric_list=['acc'],
        eval_split='validation',
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy. Use "image_encoder_probe" for MMMU.',
                'value': 'image_encoder_probe',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to KEEP per subject (e.g. 0.1 keeps ~10%).',
                'value': 0.1,
            },
            'reviews_dir': {
                'type': 'str',
                'description': 'Unused for MMMU (no model review files required).',
                'value': '',
            },
        },
    )
)
class MMMUPrunedAdapter(PrunedAdapterBase, MMMUAdapter):
    """
    MMMU with image-encoder-focused sample pruning.

    MRO: MMMUPrunedAdapter → PrunedAdapterBase → MMMUAdapter
         → VisionLanguageAdapter → DefaultDataAdapter → DataAdapter

    PrunedAdapterBase (listed first) intercepts load_dataset() and delegates
    index selection to MMMUImageEncoderProbe via the overridden
    _get_selected_indices_per_subset().
    """

    def _get_selected_indices_per_subset(
        self,
        subset_keys: List[str],
        datasets,
    ) -> Dict[str, Set[int]]:
        from evalscope_ext.pruning.mmmu_probe import MMMUImageEncoderProbe

        probe = MMMUImageEncoderProbe()
        return probe.select(
            subset_keys=subset_keys,
            datasets=datasets,
            prune_ratio=self.prune_ratio,
        )
