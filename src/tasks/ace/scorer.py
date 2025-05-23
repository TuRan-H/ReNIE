from typing import Dict, List, Type

from src.tasks.ace.prompts import (
    COARSE_EVENT_DEFINITIONS,
    COARSE_RELATION_DEFINITIONS,
    ENTITY_DEFINITIONS,
    EVENT_DEFINITIONS,
    RELATION_DEFINITIONS,
    VALUE_DEFINITIONS,
)
from src.tasks.utils_scorer import EventScorer, RelationScorer, SpanScorer
from src.tasks.utils_typing import Entity, Value


class ACEEntityScorer(SpanScorer):
    """ACE Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class ACEValueScorer(SpanScorer):
    """ACE Values identification and classification scorer."""

    valid_types: List[Type] = VALUE_DEFINITIONS

    def __call__(self, reference: List[Value], predictions: List[Value], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"values": output["spans"]}


class ACECoarseRelationScorer(RelationScorer):
    """ACE Relation identification scorer."""

    valid_types: List[Type] = COARSE_RELATION_DEFINITIONS


class ACERelationScorer(RelationScorer):
    """ACE Relation identification scorer."""

    valid_types: List[Type] = RELATION_DEFINITIONS


class ACEEventScorer(EventScorer):
    """ACE Event and argument classification scorer."""

    valid_types: List[Type] = COARSE_EVENT_DEFINITIONS


class ACEEventArgumentScorer(EventScorer):
    """ACE Event and argument classification scorer."""

    valid_types: List[Type] = EVENT_DEFINITIONS
