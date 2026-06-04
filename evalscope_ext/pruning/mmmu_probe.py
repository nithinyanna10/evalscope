"""
MMMUImageEncoderProbe: subject-stratified probe set selection for MMMU.

Selects questions that specifically stress image encoders rather than general
capability.  Standard random sampling picks mostly text-answerable questions
where the image is decorative.  This probe forces questions where the answer
REQUIRES reading the image precisely — diagrams, charts, fine spatial detail.
An encoder that degrades will fail these first while passing text-adjacent
questions, making degradation visible at small sample counts.

Image complexity score per sample::

    score = 0.4 * norm(n_images) + 0.3 * has_diagram + 0.3 * size_score

where:
  - n_images:    number of images in the question (capped at 7, the MMMU max)
  - has_diagram: 1.0 if question text contains figure/diagram/chart/graph/etc.
  - size_score:  proxy for image detail —
                   bytes-based (data URI or raw bytes): bytes / 1 MB, capped at 1.0
                   pixel-based (PIL Image): width×height / (1024×1024), capped at 1.0
                   URL string (size unknown without fetching): 0.5 (neutral)

Subject weighting:
  Image-heavy subjects (Art, Science diagrams, Medical, Engineering) receive
  proportionally more probe slots than text-heavy subjects (History,
  Literature, Music).  The total budget is split proportionally to each
  subject's weight, with a floor of _MIN_PER_SUBJECT samples per subject.
"""

import re
from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from evalscope.api.dataset.dataset import DatasetDict

# Subjects known to be image-heavy
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

_HEAVY_WEIGHT = 1.5
_LIGHT_WEIGHT = 1.0
_MIN_PER_SUBJECT = 2


class MMMUImageEncoderProbe:
    """
    Selects a probe set from MMMU that stresses image encoders.

    The total probe budget (total_samples × prune_ratio) is distributed
    across subjects proportionally to their image-heaviness weight.
    Within each subject, samples are ranked by image complexity score and
    the top-k are selected (minimum _MIN_PER_SUBJECT per subject).
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
            prune_ratio: Fraction of total samples to keep.
        """
        if not subset_keys:
            return {}

        # Per-subject weights (image-heavy subjects attract more slots)
        subject_weights = {
            s: _HEAVY_WEIGHT if s in _IMAGE_HEAVY_SUBJECTS else _LIGHT_WEIGHT
            for s in subset_keys
        }
        total_weight = sum(subject_weights.values())

        # Total probe budget across all subjects
        total_n = sum(len(datasets[s]) for s in subset_keys)
        total_budget = max(_MIN_PER_SUBJECT * len(subset_keys),
                           round(total_n * prune_ratio))

        result: Dict[str, Set[int]] = {}

        for subject in subset_keys:
            ds = datasets[subject]
            n_total = len(ds)
            if n_total == 0:
                result[subject] = set()
                continue

            # Allocate slots proportional to this subject's weight
            k = max(
                _MIN_PER_SUBJECT,
                int(total_budget * subject_weights[subject] / total_weight),
            )
            k = min(k, n_total)  # can't take more than we have

            # Score each sample by image complexity and select top-k
            scores: List[float] = [self._score_sample(sample) for sample in ds]
            ranked = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
            selected_positions = set(ranked[:k])

            sample_ids = [s.id for s in ds]
            result[subject] = {sample_ids[pos] for pos in selected_positions}

        return result

    def _score_sample(self, sample) -> float:
        """Compute image complexity score for a single Sample.

        score = 0.4 * norm(n_images) + 0.3 * has_diagram + 0.3 * size_score
        """
        from evalscope.api.messages.content import ContentImage

        n_images = 0
        size_score = 0.0   # accumulates across all images; averaged at the end
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
                            size_score += self._image_size_score(part)
                        elif hasattr(part, 'text'):
                            question_text += str(getattr(part, 'text', ''))
        elif isinstance(sample.input, str):
            question_text = sample.input

        has_diagram = 1.0 if _DIAGRAM_PATTERN.search(question_text) else 0.0

        # Normalise n_images (MMMU max is 7)
        norm_images = min(n_images, 7) / 7.0

        # Average size_score across images (neutral 0.5 when no images)
        avg_size = (size_score / n_images) if n_images > 0 else 0.0

        return 0.4 * norm_images + 0.3 * has_diagram + 0.3 * avg_size

    @staticmethod
    def _image_size_score(part) -> float:
        """Return a [0, 1] size proxy for one ContentImage part.

        - Raw bytes / bytearray: bytes / 1 MB (capped at 1.0)
        - Data URI string (base64): len(string) / 1_400_000 (≈ 1 MB base64)
        - PIL Image: width × height / (1024 × 1024) (capped at 1.0)
        - Plain URL string: 0.5 (neutral — size unknown without fetching)
        """
        data = getattr(part, 'image_url', None)

        if data is None:
            return 0.5  # no data available → neutral

        if isinstance(data, (bytes, bytearray)):
            return min(len(data) / 1_048_576, 1.0)

        if isinstance(data, str):
            if data.startswith('data:'):
                # Base64 data URI — use string length as byte proxy
                # (~4/3 overhead means 1 MB ≈ 1,365,333 chars)
                return min(len(data) / 1_365_333, 1.0)
            else:
                # Plain URL — size unknown without fetching; use neutral score
                return 0.5  # URL images: size unknown without fetching; use neutral score

        # PIL Image or similar object with .size attribute
        size_attr = getattr(data, 'size', None)
        if size_attr is not None:
            try:
                w, h = size_attr
                return min(w * h / (1024 * 1024), 1.0)
            except (TypeError, ValueError):
                pass

        return 0.5  # fallback: neutral
