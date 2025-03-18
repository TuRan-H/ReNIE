from typing import List, Type
from src.tasks.fewrel.prompts import COARSE_RELATION_DEFINITIONS
from src.tasks.utils_scorer import RelationScorer


class FewRelCoarseRelationScorer(RelationScorer):
    valid_types: List[Type] = COARSE_RELATION_DEFINITIONS


    def __call__(self, reference: list[list], predictions: list[list], **kwargs: dict):
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_reversed_match = True
                for elem in pre_ann: elem._allow_reversed_match = True

        return super().__call__(reference, predictions)

