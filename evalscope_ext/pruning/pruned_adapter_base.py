"""
PrunedAdapterBase: universal mixin for benchmark-agnostic sample pruning.

Subclass both this and the target benchmark adapter, with PrunedAdapterBase
listed FIRST so Python's MRO places it before DefaultDataAdapter::

    class MyPrunedAdapter(PrunedAdapterBase, MyBenchmarkAdapter):
        _reviews_benchmark_prefix = 'my_benchmark_v5'

The mixin intercepts load_dataset(), calls the parent chain to load and
process all samples, then filters each subset down to the indices selected
by the pruning strategy.

Required dataset-args:
  pruning_strategy: 'stratified'   (only supported value)
  prune_ratio:      0.1            (fraction to KEEP)
  reviews_dir:      '/path/to/reviews/'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set

from evalscope.utils.logger import get_logger

from .stratified_pruner import StratifiedPruner

if TYPE_CHECKING:
    from evalscope.api.dataset.dataset import DatasetDict

logger = get_logger()


class PrunedAdapterBase:
    """
    Mixin that adds stratified pruning to any DefaultDataAdapter subclass.

    Must be listed FIRST in the MRO so it intercepts load_dataset() before
    DefaultDataAdapter::

        class PrunedX(PrunedAdapterBase, XAdapter): ...

    Class attribute ``_reviews_benchmark_prefix`` must be set by each
    concrete pruned adapter to the file-name prefix used in the review
    JSONL files (e.g. ``'live_code_bench_v5'``).
    """

    _reviews_benchmark_prefix: str = ''

    # ------------------------------------------------------------------
    # These read from self.extra_params which is available after the
    # parent DataAdapter.__init__ runs (via the cooperative super() chain).
    # ------------------------------------------------------------------

    @property
    def _prune_ratio(self) -> float:
        return float(self.extra_params.get('prune_ratio', 0.1))  # type: ignore[attr-defined]

    @property
    def _reviews_dir(self) -> str:
        return self.extra_params.get('reviews_dir', '') or ''  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Override point for subclasses that use a different probe
    # ------------------------------------------------------------------

    def _get_selected_indices_per_subset(
        self,
        subset_keys: List[str],
        datasets: 'DatasetDict',
    ) -> Dict[str, Set[int]]:
        """
        Return {subset_key: set_of_selected_sample_ids}.

        Default implementation applies StratifiedPruner to all subsets
        with the same index set.  Override in subclasses (e.g. MMMU
        pruned) to perform per-subset selection.
        """
        if not self._reviews_dir:
            logger.warning(
                f'{self.name}: reviews_dir not set — skipping pruning, keeping all samples'  # type: ignore[attr-defined]
            )
            return {key: set(range(len(datasets[key]))) for key in subset_keys}

        prefix = self._reviews_benchmark_prefix or self.name  # type: ignore[attr-defined]
        pruner = StratifiedPruner(reviews_dir=self._reviews_dir, benchmark_prefix=prefix)
        selected = set(pruner.prune(prune_ratio=self._prune_ratio))
        return {key: selected for key in subset_keys}

    # ------------------------------------------------------------------
    # load_dataset hook
    # ------------------------------------------------------------------

    def load_dataset(self) -> 'DatasetDict':
        """Load the full dataset via the parent chain, then apply pruning."""
        # Lazy import to avoid circular-import issues at module load time.
        from evalscope.api.dataset.dataset import MemoryDataset

        dataset_dict = super().load_dataset()  # type: ignore[misc]

        subset_keys = list(dataset_dict.keys())
        selected_per_subset = self._get_selected_indices_per_subset(subset_keys, dataset_dict)

        total_before = sum(len(dataset_dict[k]) for k in subset_keys)
        for key in subset_keys:
            ds = dataset_dict[key]
            selected = selected_per_subset.get(key, set())
            kept = [s for s in ds if s.id in selected]
            dataset_dict[key] = MemoryDataset(
                samples=kept,
                name=ds.name,
                location=ds.location,
            )
        total_after = sum(len(dataset_dict[k]) for k in subset_keys)

        logger.info(
            f'Pruned {self.name} from {total_before} → {total_after} samples '  # type: ignore[attr-defined]
            f'({self._prune_ratio:.0%} kept)'
        )
        return dataset_dict
