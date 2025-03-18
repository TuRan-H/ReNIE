import inspect
import logging
import math
import random
import re
import black
import numpy as np
from string import Formatter
from typing import (
	Any,
	Dict,
	List,
	Set,
	Tuple,
	Type,
	Union
)
from jinja2 import Template
from src.tasks.utils_typing import cast_to
from src.utils import (
	DemonstrationRetriever,
	SynonymSampling,
	BoundaryDisturbance
)
from transformers import(
	AutoTokenizer,
	AutoModel,
	PreTrainedModel
) 
import copy


def filter_demonstration_by_guidelines(demonstrations, guideline, target, class_label_re):
	"""
	根据guideline中的类定义, 过滤出demonstration中的annotation

	Args:
		demonstrations (List[Dict]): 一个demonstration列表
		guideline (List[str]): 包含类定义的guideline列表
		target (str): 当前数据集的目标任务, 比如说NER --> "entities", VER --> "values"
		class_label_re (Pattern[str]): 用于提取类标签的正则表达式模式

	Returns:
		List[Dict[str, Any]]: 过滤后的demonstration的annotation列表
	"""
	annotations = []
	# 获取demonstration
	demonstration = demonstrations[0]
	# 获取demonstration中的每一个annotation, 判断annotation是否存在于_guideline中
	for ann in demonstration[target]:
		ann_cls_name = ann.__class__.__name__
		for guideline_cls in guideline:
			guideline_cls_name = class_label_re.findall(guideline_cls)[0]
			if ann_cls_name == guideline_cls_name:
				annotations.append(ann)
				break
	
	return annotations


class DatasetLoader:
	"""An abstract class for dataset loaders."""

	def __iter__(self):
		for elem in self.elements.values():
			yield elem

	def __len__(self):
		return len(self.elements)

	def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
		if isinstance(idx, int) or isinstance(idx, slice):
			return list(self.elements.values())[idx]  # Not very efficient
		else:
			return self.elements[idx]


class Sampler:
	"""
	A generic data `Sampler` class.

	Args:
		dataset_loader (`DatasetLoader`):
			The dataset loader that contains the data information.
		task (`str`, optional):
			The task to sample. Defaults to `None`.
		split (`str`, optional):
			The split to sample. It must be one of the following: "train", "dev" or
			"test". Depending on the split the sampling strategy differs. Defaults to
			`"train"`.
		parallel_instances (`Union[int, Tuple[int, int]]`, optional):
			The number of sentences sampled in parallel. Options: | 控制并行采样的数量
				
				* **`int`**: The amount of elements that will be sampled in parallel.
				* **`tuple`**: The range of elements that will be sampled in parallel.

			Defaults to 1.
		max_guidelines (`int`, optional):
			The number of guidelines to append to the example at the same time. If `-1`
			is given then all the guidelines are appended. Defaults to `-1`.
		sample_total_guidelines (`int`, optional):
			The total number of guidelines to sample. If `-1` is given then all the
			guidelines are sampled. Defaults to `-1`.
		guideline_dropout (`float`, optional):
			The probability to dropout a guideline definition for the given example. This
			is only applied on training. Defaults to `0.0`.
		seed (`float`, optional):
			The seed to sample the examples. Defaults to `0`.
		prompt_template (`str`, optional):
			The path to the prompt template. Defaults to `"templates/prompt_eae.txt"`.
		ensure_positives_on_train (bool, optional):
			Whether to ensure that the guidelines of annotated examples are not removed.
			Defaults to `True`.
		dataset_name (str, optional):
			The name of the dataset. Defaults to `None`.
		scorer (`str`, optional):
		   The scorer class import string. Defaults to `None`.
		sample_only_gold_guidelines (`bool`, optional):
			Whether to sample only guidelines of present annotations. Defaults to `False`.
		task_definitions (`List[Type]`, optional):
			The task definitions or guidelines. Defaults to `None`.
		task_target (`str`, optional):
			The key of the target task annotations in the dict outputed by the
			`DatasetLoader`. This is useful when the `DataLoader` returns annotations for
			different tasks. Defaults to "labels".
		remove_guidelines (`bool`, optional):
			Whether or not to remove guideline information. This is usefull for building the
			baseline. Defaults to `False`.
		is_coarse_to_fine (`bool`, optional):
			Whether or not the task is coarse_to_fine classification. Defaults to `False`.
		coarse_to_fine (`Dict[Type, List[Type]]`, optional):
			If `is_coarse_to_fine` this argument contains the information to map from coarse
			labels to fine labels. Defaults to `None`.
		fine_to_coarse (`Dict[Type, Type]`, optional):
			If `is_coarse_to_fine` this argument contains the information to map from fine
			labels to coarse labels. Defaults to `None`.
		lang (`str`, optional):
			Language of the guidelines to sample. Defaults to `"en"`.
		definitions (`Dict[str, Any]`, optional):
			Dictionary from where to sample the guideline definitions. Defaults to None.
		include_examples_prob (float, optional):
			Whether or not include examples in the guidelines. Defaults to `0.0`.
		examples (`Dict[str, Any]`, optional):
			Dictionary from where to sample the examples. Defaults to None.
		label_noise_prob (`float`, optional):
			The probability to hide the label names. Defaults to `0.0`.

	Raises:
		ValueError:
			raised when no task definitions are given.
	"""

	def __init__(
		self,
		dataset_loader: DatasetLoader,
		task: str = None,
		split: str = "train",
		parallel_instances: Union[int, Tuple[int, int]] = 1,
		max_guidelines: int = -1,
		sample_total_guidelines: int = -1,
		guideline_dropout: float = 0.0,
		seed: float = 0,
		prompt_template: str = "templates/prompt.txt",
		ensure_positives_on_train: bool = False,
		sample_only_gold_guidelines: bool = False,
		dataset_name: str = None,
		scorer: str = None,
		task_definitions: List[Type] = None,
		task_target: str = "labels",
		remove_guidelines: bool = False,
		is_coarse_to_fine: bool = False,
		coarse_to_fine: Dict[Type, List[Type]] = None,
		fine_to_coarse: Dict[Type, Type] = None,
		lang: str = "en",
		definitions: Dict[str, Any] = None,
		include_examples_prob: float = 0.0,
		examples: Dict[str, Any] = None,
		label_noise_prob: float = 0.0,
		coarse_dropout: float = 0.0,
		**kwargs,
	) -> None:
		self.loader = dataset_loader
		self.task = task
		assert split in [
			"train",
			"dev",
			"test",
		], f"{split} must be either 'train', 'dev' or 'test'."
		self.split = split
		if isinstance(parallel_instances, int):
			parallel_instances = (1, parallel_instances)
		self.parallel_instances = tuple(parallel_instances)
		self.guideline_dropout = guideline_dropout
		self.coarse_dropout = coarse_dropout
		self.seed = seed
		if not task_definitions or not len(task_definitions):
			raise ValueError("task_definitions argument must not be None or empty")
		self.task_definitions = task_definitions
		self.task_target = task_target

		if max_guidelines < 0 or max_guidelines > len(self.task_definitions):
			self.max_guidelines = len(self.task_definitions)
		else:
			self.max_guidelines = max_guidelines
		if sample_total_guidelines < 0 or sample_total_guidelines > len(self.task_definitions):
			self.sample_total_guidelines = len(self.task_definitions)
		else:
			self.sample_total_guidelines = sample_total_guidelines
		self.ensure_positives_on_train = ensure_positives_on_train
		self.sample_only_gold_guidelines = sample_only_gold_guidelines

		with open(prompt_template, "rt") as f:
			self.template = Template(f.read())

		self.dataset_name = dataset_name
		self.scorer_cls = scorer

		# Maping information for coarse --> fine tasks such as EAE or RC
		self.is_coarse_to_fine = is_coarse_to_fine
		self._coarse_to_fine = coarse_to_fine
		self._fine_to_coarse = fine_to_coarse

		self._black_mode = black.Mode()
		self.remove_guidelines = remove_guidelines
		# self._remove_guidelines_re = re.compile(r'"""(.+\n?)*"""')
		self._remove_guidelines_re = re.compile(r'"""[^"]+"""')
		self._remove_guidelines_fn = lambda x: self._remove_guidelines_re.sub("", x).replace("\n    \n", "\n")

		self._remove_comments_re = re.compile(r"#.+?\n")
		self._remove_comments_fn = lambda x: self._remove_comments_re.sub("\n", x)

		self._remove_empty_comments_re = re.compile(r"#()*\n")
		self._remove_empty_comments_fn = lambda x: self._remove_empty_comments_re.sub("\n", x)
		self._formatter = Formatter()

		self.lang = lang
		self.definitions = definitions
		if not self.definitions:
			raise ValueError("You must provide definitions for your guidelines!")
		self.include_examples_prob = include_examples_prob
		# Make 1.0 prob on example sampling in evaluation for reproducibility
		if self.include_examples_prob > 0 and self.split != "train":
			self.include_examples_prob = 1.0
		
		# examples: inline examples, 也就是每一个attribute的例子
		self.examples = examples
		if include_examples_prob > 0 and not self.examples:
			logging.warning(
				"`include_examples_prob` is > 0 but `examples` is None. If you want to include examples, you must"
				" provide examples. `include_examples_prob` has been changed to 0.0"
			)
			self.include_examples_prob = 0

		self.label_noise_prob = label_noise_prob
		self._class_label_re = re.compile(r"class (\w+)\(\w+\)")

		# 构造DemonstrationRetriver, SynonymSampling, BoundaryDisturbance对象
		self.add_demonstrations = kwargs.get('add_demonstrations', False)
		self.robustness_enhancement = kwargs.get('robustness_enhancement', False)
		self.top_k = kwargs.get("top_k", 1)
		self.device = kwargs.get("device", "cpu")
		self.force_random = kwargs.get("force_random", False)
		self.force_SLR = kwargs.get("force_SLR", False)

		if self.add_demonstrations:
			dataloader_cls = kwargs.get("dataloader_cls", None)
			extra_train_file = kwargs.get("extra_train_file", None)
			if self.split == 'test':
				if self.force_random is True:
					self.demonstration_retriever = DemonstrationRetriever(
						dataloader = self.loader,
						retrieval_strategy = "random",
						demo_loader_cls = dataloader_cls,
						demo_loader_arguments = extra_train_file,
						tasks = [self.task],
						top_k = self.top_k,
					)
				else:
					# 测试集使用最近邻检索构造demonstrations
					self.demonstration_retriever = DemonstrationRetriever(
						dataloader = self.loader,
						retrieval_strategy = "knn",
						demo_loader_cls = dataloader_cls,
						demo_loader_arguments = extra_train_file,
						tasks = [self.task],
						top_k = self.top_k,
						device = self.device
					)
			else:
				if self.force_SLR:
					self.demonstration_retriever = DemonstrationRetriever(
						dataloader = self.loader,
						retrieval_strategy = "knn",
						demo_loader_cls = dataloader_cls,
						demo_loader_arguments = extra_train_file,
						task = [self.task],
						top_k = self.top_k,
						device = self.device,
						is_train_dev = True
					)
				else:
					# 训练集和验证集使用随机检索构造demonstrations, 用于进行task adaptation
					self.demonstration_retriever = DemonstrationRetriever(
						dataloader = self.loader,
						retrieval_strategy = "random",
						top_k = self.top_k,
						device = self.device
					)

		if self.robustness_enhancement:
			simcse_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="./model/sup-simcse-roberta-large")
			simcse_model: PreTrainedModel = AutoModel.from_pretrained("./model/sup-simcse-roberta-large")
			simcse_model.to(self.device)
			self.synonym_sampler = SynonymSampling(simcse_tokenizer, simcse_model, device=self.device)
			self.boundary_disturbancer = BoundaryDisturbance(simcse_tokenizer, simcse_model)
		

	def _sample(self, instances, demonstrations):# -> Any:
		"""
		根据预处理的数据集, 进行样本的构建

		Args:
			instances (List[Dict]): 一个实例列表
			demonstrations (List[Dict]): 一个demonstration列表
		"""
		# 如果没有检索到demonstrations, 不构建demonstrations
		add_demonstrations = self.add_demonstrations
		if demonstrations == [None] or demonstrations == []: add_demonstrations = False

		# _gold refers to specifc gold information that is used in the template (depends on the task)
		_gold: List[Any] = [gold for inst in instances for gold in inst["gold"]]
		# positive_guidelines: referst just to the guidelines definitions of the labels in the example
		positive_guidelines: Set[Type] = {type(ann) for inst in instances for ann in inst[self.task_target]}
		# 对positive_guidelines进行处理, 将demonstration中出现过的guideline加入到_positive_guidelines中
		if add_demonstrations:
			positive_guidelines = positive_guidelines.union(
				type(ann) for inst in demonstrations for ann in inst[self.task_target]
			)
		# 若启用了is_coarse_to_fine, 则将positive_guidelines中的 fine guideline -> coarse guidelines
		if self.is_coarse_to_fine:		
			coarse_guidelines: Set[Type] = {self._fine_to_coarse[_def] for _def in positive_guidelines}

		# *** 构造guidelines
		# guidelines: 每一个guideline的类变量
		guidelines: List[Type] = [*self.task_definitions]
		# The variable all_guidelines makes compatible the coarse-to-fine with normal tasks
		all_guidelines = [guidelines] if not self.is_coarse_to_fine else coarse_guidelines
		for guidelines in all_guidelines:
			if self.is_coarse_to_fine:
				if self.coarse_dropout and random.random() < self.coarse_dropout:
					continue
				# 获取coarse guideline所对应的所有fine guideline
				# In case of `is_coarse_to_fine` the guidelines variable is a single type
				coarse_type = guidelines
				guidelines = self._coarse_to_fine[coarse_type]

			# sample_only_gold_guidelines: 仅采样那些在instances中出现过的guidelines
			# This may defer with `positive_guidelines` because we can apply this after coarse-to-fine conversion
			if self.sample_only_gold_guidelines:
				guidelines = [
					definition
					for definition in guidelines
					if any(isinstance(ann, definition) for inst in instances for ann in inst[self.task_target])
				]

			# 若设定了 `sample_total_guidelines`, 并且 `sample_total_guidlelines`的长度小于所有guidelines的长度, 则对guidelines进行dropout
			# Reduce the ammount of labels by sampling. We can make sure positive guidelines are sampled using `ensure_positives_on_train`
			if self.sample_total_guidelines < len(guidelines) and not self.sample_only_gold_guidelines:
				p = np.asarray(
					[
						(100.0 if _def in positive_guidelines and self.ensure_positives_on_train else 0.0)
						for _def in guidelines
					]
				)		# type: ignore
				# 对概率分布p进行平滑处理和归一化
				p += 1.0 / p.shape[0]
				p /= p.sum()
				guidelines = np.random.choice(
					np.asarray(guidelines), # type: ignore
					size=(self.sample_total_guidelines,),
					replace=False,
					p=p,
				).tolist()

			# Shuffle the guidelines
			random.shuffle(guidelines)

			splits = math.ceil(len(guidelines) / self.max_guidelines)
			for i in range(splits):
				_guidelines = guidelines[i * self.max_guidelines : (i + 1) * self.max_guidelines]
				if self.split == "train":
					# 在训练集中, 对_guidelines进行dropout, 并且保证positive_guidelines不会被dropout
					_guidelines_dropout = [
						_def
						for _def in _guidelines
						if random.random() > self.guideline_dropout
						or (_def in positive_guidelines and self.ensure_positives_on_train)
					]

					# Ensure at least one guideline is used
					if len(_guidelines_dropout) == 0 and len(_guidelines) > 0:
						_guidelines_dropout.append(random.choice(_guidelines))
					_guidelines = _guidelines_dropout

				# *** 构造_annotation部分
				_ann = [ann for inst in instances for ann in inst[self.task_target] if type(ann) in _guidelines]

				# *** 构造text部分
				_text = " ".join([inst["text"] for inst in instances]).strip()

				# 如果当前的任务设计粗粒度到细粒度的转换 (RC 和 EAE), 则将annotations转化为粗粒度的annotations
				if self.is_coarse_to_fine:
					_gold = [cast_to(ann, coarse_type) for ann in _ann]

				# Remove the chances for hallucination because the task is classification
				if self.is_coarse_to_fine and not len(_ann):
					continue

				# 获取_guidelines中类定义的源代码
				_guidelines = [inspect.getsource(definition) for definition in _guidelines]

				# *** 构造_definitions部分. 这里的definitions是guidelines中的类定义
				_definitions = {
					key: random.choice(value[self.lang]) if self.split == "train" else value[self.lang][0]
					for key, value in self.definitions.items()
				}

				# *** 构造_examples部分. (guidelines中每一个类的例子)
				# Sample few-shot examples if train (add epsilon for not sampling a 0.0)
				if min(random.random() + 1e-6, 1.0) <= self.include_examples_prob:
					_examples = {
						key: (
							f"""Such as: "{'", "'.join(random.sample(value[self.lang], k=min(5,len(value[self.lang]))))}" """
							if self.split == "train"
							else f"""Such as: "{'", "'.join(value[self.lang][:5])}" """
						)
						for key, value in self.examples.items()
					}
				else:
					# _examples = {key: "" for key in self.examples.keys()}
					_examples = {
						key[1]: ""
						for definition in _guidelines
						for key in self._formatter.parse(definition)
						if key[1] is not None and "example" in key[1]
					}

				# *** 结合_examples和_definitions, 解析_guidelines
				_repl = {**_examples, **_definitions}
				_guidelines = [definition.format(**_repl) for definition in _guidelines]

				# If no examples are provide, empty comments are created, the following line removes them
				_guidelines = {self._remove_empty_comments_fn(definition) for definition in _guidelines}

				# Remove definitions for baseline
				if self.remove_guidelines:
					_guidelines = [self._remove_guidelines_fn(definition) for definition in _guidelines]
					_guidelines = [self._remove_comments_fn(definition) for definition in _guidelines]
				
				# *** 构造demonstrations部分
				if add_demonstrations:
					_demonstration_list = self.build_demonstrations(
						demonstrations,
						_guidelines,
						coarse_type if self._coarse_to_fine else None
					)
				else:
					_demonstration_list = []
				
				# *** 构造negative instances部分
				if self.robustness_enhancement and _demonstration_list:
					for demon in _demonstration_list:
						negative_instance = self.build_negative_instances(demon)
						if negative_instance: demon['negative_instances'] = negative_instance

				text = self.template.render(
					guidelines=_guidelines,
					text=_text,
					annotations=_ann,
					gold=_gold,
					demonstrations=_demonstration_list
				)

				# 对于训练集, 执行 label noise. 如果remove_guidelines为True, 则不执行label noise
				if self.split == "train" and self.label_noise_prob > 0.0 and not self.remove_guidelines:
					# 仅当添加了demonstrations时, 才会使用这种方式找出文本中的各个部分
					pretext_idx = text.index("# This is the text to analyze\ntext =")
					results_idx = re.search(r'(?<!demonstrations_)result = \[', text).start()
					_pretext = text[:pretext_idx]
					_intext = text[pretext_idx:results_idx]
					_postext = text[results_idx:]
					# 获取在文本开始到 "\ntext =" 之间的所有类名
					class_names = self._class_label_re.findall(_pretext)
					random.shuffle(class_names)
					i = 1
					for name in class_names:
						if random.random() <= self.label_noise_prob:
							_pretext = _pretext.replace(f"{name}", f"LABEL_{i}")
							_postext = _postext.replace(f"{name}(", f"LABEL_{i}(")
							i += 1
					text = _pretext + _intext + _postext


				# *** 构造final_example
				final_example = {
					"ids": [inst["id"] for inst in instances],
					"task_id": f"{self.dataset_name}_{self.task}",
					"scorer_cls": self.scorer_cls,
					"labels": black.format_str(_ann.__repr__(), mode=self._black_mode),
					"text": black.format_str(text, mode=self._black_mode),
					"unlabelled_sentence": _text,
				}

				yield final_example


	def __iter__(self):
		random.seed(self.seed)
		np.random.seed(self.seed)
		instances = list()
		demonstrations = list()
		# 每一次采样, 总共生成的实例个数, 从 `self.parallel_instances` 中随机选择一个数
		total_inst = random.randint(*self.parallel_instances)
		prev_id = None
		for index, elem in enumerate(self.loader):
			# Prevent mixing sentences from different documents.
			if (len(instances) == total_inst) or (prev_id is not None and elem["doc_id"] != prev_id):
				for samp in self._sample(instances, demonstrations):
					yield samp
				instances = []
				demonstrations = []
				total_inst = random.randint(*self.parallel_instances)

			# *** 获取demonstration
			if self.add_demonstrations:
				demonstrations = self.demonstration_retriever(index)
			instances.append(elem)
			prev_id = elem["doc_id"]

		# 在迭代结束后, 如果instances中还有元素, 对其调用_sample, yield出去
		if len(instances):
			for samp in self._sample(instances, demonstrations):
				yield samp


	def build_demonstrations(self, demonstrations: list, guideline: list, coarse_type):
		"""
		构造demonstrations

		Args:
			demonstrations (list): 一个demonstration列表
			guideline (list): 一个guideline列表
			coarse_type (Type): 粗粒度的guideline类型
		"""
		# _demonstration_list: 用于构造Jinja2模板的demonstration列表
		demonstration_list = []

		for demo in demonstrations:
			_demonstration = {
				'text': str(),
				'annotations': list()
			}
			_demonstration['text'] = demo['text']
			_demonstration['annotations'] = filter_demonstration_by_guidelines(
				[demo],
				guideline,
				self.task_target,
				self._class_label_re,
			)
			if coarse_type:
				_demonstration['gold'] = [cast_to(ann, coarse_type) for ann in _demonstration['annotations']]
			else:
				_demonstration['gold'] = [gold for gold in demo['gold']]

			# if self.is_coarse_to_fine:
			# 	if _demonstration['gold'] == []: continue
			
			demonstration_list.append(_demonstration)
		
		return demonstration_list


	def	build_negative_instances(self, demonstration):
		"""
		为每一个demonstration构造两个负样本: 1. boundary disturbance 2. synonym sampling

		Args:
			demonstration_list (dict): 一个demonstration (dict), 包含text, annotations, gold三个key
			max_negative_instances (int): 负例最大的个数

		Return:
			negative_instance_list (list): 列表, 
			类似于demonstration_list的数据结构, 不过每一个demonstration中的annotations都是被构造好的负样本
		"""
		negative_instance_list = []

		synonym = copy.deepcopy(demonstration)
		boundary = copy.deepcopy(demonstration)

		# NER task
		if self.task_target == "entities":
			for syn_ann, boun_ann in zip(synonym['annotations'], boundary['annotations']):
				setattr(syn_ann,'span',self.synonym_sampler(synonym['text'], getattr(syn_ann, 'span')))
				setattr(boun_ann,'span',self.boundary_disturbancer(boundary['text'], getattr(boun_ann, 'span')))

			negative_instance_list.extend(
				random.sample(
					synonym['annotations']+boundary['annotations'],
					len(demonstration['annotations'])
				)
			)

			negative_instance_list = self.remove_duplicates_and_demonstrations(
				negative_instance_list,
				demonstration['annotations']
			)
		
		# EE task, EE任务是在文中找能够mention一个event的token
		elif self.task_target == "events":
			for syn_ann, boun_ann in zip(synonym['annotations'], boundary['annotations']):
				setattr(syn_ann, 'mention', self.synonym_sampler(synonym['text'], getattr(syn_ann, 'mention')))
				setattr(boun_ann,'mention',self.boundary_disturbancer(boundary['text'], getattr(boun_ann, 'mention')))
			
			negative_instance_list.extend(
				random.sample(
					synonym['annotations']+boundary['annotations'],
					len(demonstration['annotations'])
				)
			)
			negative_instance_list = self.remove_duplicates_and_demonstrations(
				negative_instance_list,
				demonstration['annotations']
			)

		# EAE 任务, EAE任务是找到一个任务中每个event attribute的内容
		elif self.task_target == "arguments":
			for syn_ann, boun_ann in zip(synonym['annotations'], boundary['annotations']):
				ann_attr = [
					attr 
					for attr in dir(syn_ann)
					if not attr.startswith("__")
					and not callable(getattr(syn_ann, attr))
					and not attr.startswith("_")
				]
				for attr in ann_attr:
					if attr == "mention": continue
					attr_content = getattr(syn_ann, attr)
					# 对象中某个attribute可能是list, 这里避免这种问题
					list_tag = False
					if isinstance(attr_content, list):
						if len(attr_content) > 0:
							attr_content = attr_content[0]
							list_tag = True
						else: continue
					syn = self.synonym_sampler(synonym['text'], attr_content)
					boun = self.boundary_disturbancer(boundary['text'], attr_content)
					if list_tag: syn, boun = [syn], [boun]
					setattr(syn_ann, attr, syn)
					setattr(boun_ann, attr, boun)

			negative_instance_list.extend(
				random.sample(
					synonym['annotations']+boundary['annotations'],
					len(demonstration['annotations'])
				)
			)

		# coarse_relations: RE task, RE任务是给定NER的结果, 判断两个实体之间的coarse_relations
		elif self.task_target == "coarse_relations":
			for syn_ann, boun_ann in zip(synonym['annotations'], boundary['annotations']):
				for attr in ['arg1', 'arg2']:
					setattr(syn_ann, attr, self.synonym_sampler(synonym['text'], getattr(syn_ann, attr)))
					setattr(boun_ann, attr, self.boundary_disturbancer(boundary['text'], getattr(boun_ann, attr)))
			
			negative_instance_list.extend(
				random.sample(
					synonym['annotations']+boundary['annotations'],
					len(demonstration['annotations'])
				)
			)
			negative_instance_list = self.remove_duplicates_and_demonstrations(
				negative_instance_list,
				demonstration['annotations']
			)

		# relations: RC task, RC任务是根据粗粒度的relation, 判断细粒度的relation
		# 不能更改类名, 这里不进行RES
		elif self.task_target == "relations":
			pass

		return negative_instance_list
	

	@staticmethod
	def remove_duplicates_and_demonstrations(original_list, exclude_items: list = None):
		"""
		确保列表中没有重复的元素
		如果提供了exclude_items, 则从original_list中排除exclude_items中的元素

		Args:
			original_list (list): 一个列表, 列表中的元素都是类实例, 例如 GPE(span="New York")
			exclude_items (list): 一个列表, 包含需要排除的元素
		"""
		unique_list = []
		for item in original_list:
			if item not in unique_list:
				unique_list.append(item)
		
		if exclude_items:
			unique_list = [item for item in unique_list if item not in exclude_items]
			
		
		return unique_list