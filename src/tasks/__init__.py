from typing import Dict, List

from . import (
    ace,
    bc5cdr,
    broadtwitter,
    casie,
    conll03,
    crossner,
    diann,
    e3c,
    fabner,
    fewrel,
    harveyner,
    mitmovie,
    mitrestaurant,
    multinerd,
    ncbidisease,
    ontonotes,
    rams,
    tacred,
    wikievents,
    wnut,
)


TASK_ID_TO_CONFIGS = {
    "ACE05_NER": ("configs/data_configs/ace_config.json", "NER"),
    "ACE05_VER": ("configs/data_configs/ace_config.json", "VER"),
    "ACE05_RE": ("configs/data_configs/ace_config.json", "RE"),
    "ACE05_RC": ("configs/data_configs/ace_config.json", "RC"),
    "ACE05_EE": ("configs/data_configs/ace_config.json","EE"),
    "ACE05_EAE": ("configs/data_configs/ace_config.json","EAE"),
    "RAMS_EAE": ("configs/data_configs/rams_config.json", "EAE"),
    "CoNLL03_NER": ("configs/data_configs/conll03_config.json", "NER"),
    "CASIE_EE": ("configs/data_configs/casie_config.json", "EE"),
    "CASIE_EAE": ("configs/data_configs/casie_config.json", "EAE"),
    "Europarl_NER": ("configs/data_configs/europarl_config.json", "NER"),
    "TACRED_SF": ("configs/data_configs/tacred_config.json", "SF"),
    "OntoNotes5_NER": ("configs/data_configs/ontonotes_config.json", "NER"),
    "NcbiDisease_NER": ("configs/data_configs/ncbidisease_config.json", "NER"),
    "DIANN_NER": ("configs/data_configs/diann_config.json", "NER"),
    "WNUT17_NER": ("configs/data_configs/wnut17_config.json", "NER"),
    "MultiNERD_NER": ("configs/data_configs/multinerd_config.json", "NER"),
    "WikiEvents_NER": ("configs/data_configs/wikievents_config.json", "NER"),
    "WikiEvents_EE": ("configs/data_configs/wikievents_config.json", "EE"),
    "WikiEvents_EAE": ("configs/data_configs/wikievents_config.json", "EAE"),
    "FabNER_NER": ("configs/data_configs/fabner_config.json", "NER"),
    "FewRel_RE": ("configs/data_configs/fewrel_config.json", "RE"),
    "E3C_NER": ("configs/data_configs/fewrel_config.json", "NER"),
    "BC5CDR_NER": ("configs/data_configs/bc5cdr_config.json", "NER"),
    "BroadTwitter_NER": ("configs/data_configs/broadtwitter_config.json", "NER"),
    "HarveyNER_NER": ("configs/data_configs/harveyner_config.json", "NER"),
    "MITMovie_NER": ("configs/data_configs/mitmovie_config.json", "NER"),
    "MITRestaurant_NER": ("configs/data_configs/mitrestaurant_config.json", "NER"),
    "CrossNER_CrossNER_AI": ("configs/data_configs/crossner_ai_config.json", "CrossNER_AI"),
    "CrossNER_CrossNER_POLITICS": ("configs/data_configs/crossner_politics_config.json", "CrossNER_POLITICS"),
    "CrossNER_CrossNER_NATURAL_SCIENCE": ("configs/data_configs/crossner_science_config.json", "CrossNER_NATURAL_SCIENCE"),
    "CrossNER_CrossNER_LITERATURE": ("configs/data_configs/crossner_literature_config.json", "CrossNER_LITERATURE"),
    "CrossNER_CrossNER_MUSIC": ("configs/data_configs/crossner_music_config.json", "CrossNER_MUSIC"),
    "CrossNER_woMISC_CrossNER_AI": ("configs/data_configs/crossner_ai_wo_misc_config.json", "CrossNER_AI"),
    "CrossNER_woMISC_CrossNER_POLITICS": ("configs/data_configs/crossner_politics_wo_misc_config.json", "CrossNER_POLITICS"),
    "CrossNER_woMISC_CrossNER_NATURAL_SCIENCE": ("configs/data_configs/crossner_science_wo_misc_config.json", "CrossNER_NATURAL_SCIENCE"),
    "CrossNER_woMISC_CrossNER_LITERATURE": ("configs/data_configs/crossner_literature_wo_misc_config.json", "CrossNER_LITERATURE"),
    "CrossNER_woMISC_CrossNER_MUSIC": ("configs/data_configs/crossner_music_wo_misc_config.json", "CrossNER_MUSIC"),
}


TASK_ID_TO_TASKS = {
    "ACE05_NER": "src.tasks.ace",
    "ACE05_VER": "src.tasks.ace",
    "ACE05_RE": "src.tasks.ace",
    "ACE05_RC": "src.tasks.ace",
    "ACE05_EE": "src.tasks.ace",
    "ACE05_EAE": "src.tasks.ace",
    "RAMS_EAE": "src.tasks.rams",
    "CoNLL03_NER": "src.tasks.conll03",
    "CASIE_EE": "src.tasks.casie",
    "CASIE_EAE": "src.tasks.casie",
    "Europarl_NER": "src.tasks.conll03",
    "TACRED_SF": "src.tasks.tacred",
    "OntoNotes5_NER": "src.tasks.ontonotes",
    "NcbiDisease_NER": "src.tasks.ncbidisease",
    "DIANN_NER": "src.tasks.diann",
    "WNUT17_NER": "src.tasks.wnut",
    "MultiNERD_NER": "src.tasks.multinerd",
    "WikiEvents_NER": "src.tasks.wikievents",
    "WikiEvents_EE": "src.tasks.wikievents",
    "WikiEvents_EAE": "src.tasks.wikievents",
    "FabNER_NER": "src.tasks.fabner",
    "FewRel_RE": "src.tasks.fewrel",
    "E3C_NER": "src.tasks.e3c",
    "BC5CDR_NER": "src.tasks.bc5cdr",
    "BroadTwitter_NER": "src.tasks.broadtwitter",
    "HarveyNER_NER": "src.tasks.harveyner",
    "MITMovie_NER": "src.tasks.mitmovie",
    "MITRestaurant_NER": "src.tasks.mitrestaurant",
    "CrossNER_CrossNER_AI": "src.tasks.crossner",
    "CrossNER_CrossNER_POLITICS": "src.tasks.crossner",
    "CrossNER_CrossNER_NATURAL_SCIENCES": "src.tasks.crossner",
    "CrossNER_CrossNER_LITERATURE": "src.tasks.crossner",
    "CrossNER_CrossNER_MUSIC": "src.tasks.crossner",
    "CrossNER_woMISC_CrossNER_AI": "src.tasks.crossner",
    "CrossNER_woMISC_CrossNER_POLITICS": "src.tasks.crossner",
    "CrossNER_woMISC_CrossNER_NATURAL_SCIENCES": "src.tasks.crossner",
    "CrossNER_woMISC_CrossNER_LITERATURE": "src.tasks.crossner",
    "CrossNER_woMISC_CrossNER_MUSIC": "src.tasks.crossner",
}

__all__ = [
    "ace",
    "rams",
    "conll03",
    "casie",
    "tacred",
    "ontonotes",
    "ncbidisease",
    "bc5cdr",
    "diann",
    "wnut",
    "multinerd",
    "wikievents",
    "fabner",
    "fewrel",       # 新加入数据集: fewrel
    "e3c",
    "broadtwitter",
    "harveyner",
    "mitmovie",
    "mitrestaurant",
    "crossner",
    "TASK_ID_TO_TASKS",
    "task_id_to_guidelines",
]


def task_id_to_prompts(task_id: str) -> str:
    """
    Returns the prompts path for a given task.

    Args:
        task_id (str): The task id

    Returns:
        str: The path to the prompts
    """

    if task_id.upper() == "CASIE_EE":
        return "src.tasks.casie.prompts_ed"

    elif task_id.upper() == "CASIE_EAE":
        return "src.tasks.casie.prompts_eae"
    elif task_id.upper() == "CrossNER_CrossNER_POLITICS".upper():
        return "src.tasks.crossner.prompts_politics"
    elif task_id.upper() == "CrossNER_CrossNER_AI".upper():
        return "src.tasks.crossner.prompts_ai"
    elif task_id.upper() == "CrossNER_CrossNER_NATURAL_SCIENCE".upper():
        return "src.tasks.crossner.prompts_natural_science"
    elif task_id.upper() == "CrossNER_CrossNER_LITERATURE".upper():
        return "src.tasks.crossner.prompts_literature"
    elif task_id.upper() == "CrossNER_CrossNER_MUSIC".upper():
        return "src.tasks.crossner.prompts_music"
    elif task_id.upper() == "CrossNER_woMISC_CrossNER_POLITICS".upper():
        return "src.tasks.crossner.prompts_politics"
    elif task_id.upper() == "CrossNER_woMISC_CrossNER_AI".upper():
        return "src.tasks.crossner.prompts_ai"
    elif task_id.upper() == "CrossNER_woMISC_CrossNER_NATURAL_SCIENCE".upper():
        return "src.tasks.crossner.prompts_natural_science"
    elif task_id.upper() == "CrossNER_woMISC_CrossNER_LITERATURE".upper():
        return "src.tasks.crossner.prompts_literature"
    elif task_id.upper() == "CrossNER_woMISC_CrossNER_MUSIC".upper():
        return "src.tasks.crossner.prompts_music"

    # Default case
    else:
        return TASK_ID_TO_TASKS[task_id] + ".prompts"


def task_id_to_guidelines(task_id: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Return the guidelines for a given task.

    Args:
        task_id (str): The task id.

    Returns:
        The guidelines for the task.
    """
    if task_id.lower() == "ace05":
        from src.tasks.ace.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "rams":
        from src.tasks.rams.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "conll03":
        from src.tasks.conll03.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "casie":
        from src.tasks.casie.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "tacred":
        from src.tasks.tacred.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "ontonotes5":
        from src.tasks.ontonotes.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "ncbidisease":
        from src.tasks.ncbidisease.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "bc5cdr":
        from src.tasks.bc5cdr.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "diann":
        from src.tasks.diann.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "wnut17":
        from src.tasks.wnut.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "multinerd":
        from src.tasks.multinerd.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "wikievents":
        from src.tasks.wikievents.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "fabner":
        from src.tasks.fabner.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "e3c":
        from src.tasks.e3c.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "broadtwitter":
        from src.tasks.broadtwitter.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "harveyner":
        from src.tasks.harveyner.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "mitmovie":
        from src.tasks.mitmovie.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "mitrestaurant":
        from src.tasks.mitrestaurant.guidelines_gold import GUIDELINES

        return GUIDELINES
    elif task_id.lower() == "crossner":
        from src.tasks.crossner.guidelines_gold import GUIDELINES

        return GUIDELINES
    else:
        raise ValueError(f"Task {task_id} not supported.")
