from typing import Tuple, Union
from datasets import load_dataset

from ..utils_data import DatasetLoader, Sampler
from .guidelines import GUIDELINES
from .prompts import (
	COARSE_RELATION_DEFINITIONS,
	REL2CLSMAPPING
)


class FewRelLoader(DatasetLoader):
	"""
	Loader for FewRel dataset

	Args:
		split (str): Split of the dataset. Can be either "train" or "validation"
		**kwargs: Additional arguments
	"""
	RELATION_TO_CLASS_MAPPING = REL2CLSMAPPING

	def __init__(self, split: str, **kwargs):
		# 数据类, 用于存储数据
		self.elements = {}

		assert split in ["train", "validation"], f"Invalid split: {split}"
		self.split = split

		dataset:dict = load_dataset("./download/few_rel", split=self.split)

		key = 0
		for data in dataset:
			text = data['tokens']
			relation_str = data['names'][0]
			# 过滤所有relation不在wikidata中的数据
			if relation_str not in self.RELATION_TO_CLASS_MAPPING:
				continue
			relation_cls = self.RELATION_TO_CLASS_MAPPING[relation_str]
			args_1 = [text[i] for i in data['head']['indices'][0]]
			args_1 = " ".join(args_1)
			args_2 = [text[i] for i in data['tail']['indices'][0]]
			args_2 = " ".join(args_2)
			text = " ".join(text)
			relation = relation_cls(args_1, args_2)

			self.elements[key] = {
				"id": key,
				"doc_id": key,
				"text": text,
				"coarse_relations": [relation],
				"gold": [relation]
			}
			key += 1

class FewRelSampler(Sampler):
	"""
	Sampler for FewRel dataset

	Args:
	"""
	def __init__(
		self,
		dataset_loader: FewRelLoader,
		task: str = None,
		split: str = "train",
		parallel_instances: Union[int, Tuple[int, int]] = 1,
		max_guidelines: int = -1,
		guideline_dropout: float = 0.0,
		seed: float = 0,
		ensure_positives_on_train: bool = False,
		dataset_name: str = None,
		scorer: str = None,
		sample_only_gold_guidelines: bool = False,
		**kwargs,
	):
		# ! 这里的RE和ace05中的RE不一样, ace05中的RE时给定entity, 提取出entity中的relation
		# ! 而这里的RE是给定relation, 提取出entity
		assert task == "RE", f"fewrel dataset only support RE task, got {task}"

		task_definitions, task_target, task_template = [COARSE_RELATION_DEFINITIONS, "coarse_relations", "templates/prompt_fewrel_re.txt"]

		ensure_positives_on_train = True
		sample_total_guidelines = 6
		max_guidelines = 6
		

		super().__init__(
			dataset_loader=dataset_loader,
			task=task,
			split=split,
			parallel_instances=parallel_instances,
			sample_total_guidelines=sample_total_guidelines,
			max_guidelines=max_guidelines,
			guideline_dropout=guideline_dropout,
			seed=seed,
			prompt_template=task_template,
            ensure_positives_on_train=ensure_positives_on_train,
            sample_only_gold_guidelines=sample_only_gold_guidelines,
            dataset_name=dataset_name,
            scorer=scorer,
            task_definitions=task_definitions,
            task_target=task_target,
            is_coarse_to_fine=False,
            definitions=GUIDELINES,
            examples=None,
            **kwargs,
		)


if __name__ == '__main__':
	dataset = FewRelLoader("train")