from typing import Dict, List, Type

from src.tasks.crossner.prompts_ai import ENTITY_DEFINITIONS_AI, ENTITY_DEFINITIONS_AI_woMISC
from src.tasks.crossner.prompts_literature import ENTITY_DEFINITIONS_LITERATURE, ENTITY_DEFINITIONS_LITERATURE_woMISC
from src.tasks.crossner.prompts_music import ENTITY_DEFINITIONS_MUSIC, ENTITY_DEFINITIONS_MUSIC_woMISC
from src.tasks.crossner.prompts_natural_science import (
    ENTITY_DEFINITIONS_NATURAL_SCIENCE,
    ENTITY_DEFINITIONS_NATURAL_SCIENCE_woMISC,
)
from src.tasks.crossner.prompts_politics import ENTITY_DEFINITIONS_POLITICS, ENTITY_DEFINITIONS_POLITICS_woMISC
from src.tasks.utils_scorer import SpanScorer
from src.tasks.utils_typing import Entity


class CrossNERPoliticsEntityScorer(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_POLITICS

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERMusicEntityScorer(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_MUSIC

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERLiteratureEntityScorer(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_LITERATURE

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERAIEntityScorer(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_AI

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERNaturalScienceEntityScorer(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_NATURAL_SCIENCE

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERPoliticsEntityScorer_woMISC(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_POLITICS_woMISC

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERMusicEntityScorer_woMISC(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_MUSIC_woMISC

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERLiteratureEntityScorer_woMISC(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_LITERATURE_woMISC

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERAIEntityScorer_woMISC(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_AI_woMISC

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}


class CrossNERNaturalScienceEntityScorer_woMISC(SpanScorer):
    """CoNLL03 Entity identification and classification scorer."""

    valid_types: List[Type] = ENTITY_DEFINITIONS_NATURAL_SCIENCE_woMISC

    def __call__(self, reference: List[Entity], predictions: List[Entity], **kwargs: dict) -> Dict[str, Dict[str, float]]:
        if kwargs.get("scorer_config", False):
            for ref_ann, pre_ann in zip(reference, predictions):
                for elem in ref_ann: elem._allow_partial_match = True
                for elem in pre_ann: elem._allow_partial_match = True
        output = super().__call__(reference, predictions)
        return {"entities": output["spans"]}
