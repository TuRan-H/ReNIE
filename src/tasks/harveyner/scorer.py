from typing import Dict, List, Type

from src.tasks.harveyner.prompts import ENTITY_DEFINITIONS
from src.tasks.utils_scorer import SpanScorer
from src.tasks.utils_typing import Entity


class HarveyNEREntityScorer(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}
