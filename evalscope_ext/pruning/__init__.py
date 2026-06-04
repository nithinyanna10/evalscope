"""Pruning utilities: stratified sample selection and adapter base."""

from .stratified_pruner import StratifiedPruner

# PrunedAdapterBase depends on evalscope; import lazily to avoid pulling the
# full evalscope import chain when only StratifiedPruner is needed.

__all__ = ['StratifiedPruner']
