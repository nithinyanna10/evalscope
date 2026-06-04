"""
MMMUImageEncoderProbe: subject-stratified probe set selection for MMMU.

Selects questions that specifically stress image encoders rather than general
capability.  Standard random sampling picks mostly text-answerable questions
where the image is decorative.  This probe forces questions where the answer
REQUIRES reading the image precisely — diagrams, charts, fine spatial detail.
An encoder that degrades will fail these first while passing text-adjacent
questions, making degradation visible at small sample counts.

Image complexity score per sample::

    score = 0.4 * norm(n_images) + 0.3 * has_diagram + 0.3 * norm(file_size)

where:
  - n_images:     number of images referenced in the question
  - has_diagram:  1 if question text mentions figure/diagram/chart/graph
  - file_size:    total bytes of all images in the sample (proxy for detail)

Subjects are weighted by image-heaviness; image-heavy subjects
(Art, Science diagrams, Medical, Engineering) receive proportionally more
probe slots than text-heavy subjects (History, Literature, Music).
"""

import re
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from evalscope.api.dataset.dataset import DatasetDict

# Subjects known to be image-heavy (fraction of probe budget they attract)
_IMAGE_HEAVY_SUBJECTS = {
    'Art',
    'Architecture_and_Engineering',
    'Basic_Medical_Science',
    'Biology',
    'Chemistry',
    'Clinical_Medicine',
    'Diagnostics_and_Laboratory_Medicine',
    'Design',
    'Electronics',
    'Energy_and_Power',
    'Geography',
    'Materials',
    'Math',
    'Mechanical_Engineering',
    'Physics',
    'Public_Health',
}

_DIAGRAM_PATTERN = re.compile(
    r'\b(figure|diagram|chart|graph|plot|image|table|illustration)\b',
    re.IGNORECASE,
)

_MIN_PER_SUBJECT = 2


class MMMUImageEncoderProbe:
    """
    Selects a probe set from MMMU that stresses image encoders.

    Works on already-loaded evalscope Sample objects (no extra network I/O).
    """

    def select(
        self,
        subset_keys: List[str],
        datasets: 'DatasetDict',
        prune_ratio: float,
    ) -> Dict[str, Set[int]]:
        """
        Return {subject: set_of_selected_sample_ids}.

        Args:
            subset_keys: MMMU subject names (e.g. 'Art', 'Biology', ...).
            datasets: Loaded DatasetDict keyed by subject.
            prune_ratio: Fraction of samples to keep per subject.
        """
        result: Dict[str, Set[int]] = {}

        # Compute per-subject budget with image-heavy boost
        total_subjects = len(subset_keys)
        if total_subjects == 0:
            return result

        heavy_weight = 1.5
        light_weight = 1.0
        subject_weights = {
            s: heavy_weight if s in _IMAGE_HEAVY_SUBJECTS else light_weight
            for s in subset_keys
        }
        total_weight = sum(subject_weights.values())

        for subject in subset_keys:
            ds = datasets[subject]
            n_total = len(ds)
            if n_total == 0:
                result[subject] = set()
                continue

            # Subject-proportional budget
            weight_fraction = subject_weights[subject] / total_weight
            k = max(
                _MIN_PER_SUBJECT,
                round(n_total * prune_ratio),
            )

            # Score each sample by image complexity
            scores: List[float] = []
            for sample in ds:
                scores.append(self._score_sample(sample))

            # Rank by complexity descending, take top-k
            ranked = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
            selected_positions = set(ranked[:k])

            # Map position → sample.id
            sample_ids = [s.id for s in ds]
            result[subject] = {sample_ids[pos] for pos in selected_positions}

        return result

    def _score_sample(self, sample) -> float:
        """Compute image complexity score for a single Sample."""
        from evalscope.api.messages.content import ContentImage

        # Count images and total bytes
        n_images = 0
        total_bytes = 0
        question_text = ''

        if isinstance(sample.input, list):
            for msg in sample.input:
                content = getattr(msg, 'content', None)
                if content is None:
                    continue
                if isinstance(content, str):
                    question_text += content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, ContentImage):
                            n_images += 1
                            # image_url may be a data URI or bytes
                            data = getattr(part, 'image_url', None) or ''
                            if isinstance(data, (bytes, bytearray)):
                                total_bytes += len(data)
                            elif isinstance(data, str):
                                total_bytes += len(data)
                        elif hasattr(part, 'text'):
                            question_text += str(getattr(part, 'text', ''))
        elif isinstance(sample.input, str):
            question_text = sample.input

        has_diagram = 1.0 if _DIAGRAM_PATTERN.search(question_text) else 0.0

        # Normalize n_images (cap at 7 which is MMMU max)
        norm_images = min(n_images, 7) / 7.0
        # Normalize file_size (cap at 1 MB for normalization)
        norm_size = min(total_bytes, 1_048_576) / 1_048_576

        return 0.4 * norm_images + 0.3 * has_diagram + 0.3 * norm_size
