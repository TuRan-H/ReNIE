import copy
import nltk
import random
import nltk.tokenize.treebank
import os
import json
import jinja2 as jinja

from tqdm import tqdm
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize import TreebankWordDetokenizer
from torch import cosine_similarity
from collections import OrderedDict


random.seed(24)


def find_sublist_index(main_list, sub_list):
	"""
	找到子表 (sub_list) 在主表 (main_list) 中的位置
	如果没有找到, 则返回 -1
	"""
	sub_len = len(sub_list)
	main_len = len(main_list)

	if sub_len == 0 or main_len < sub_len:
		return -1
	
	for i in range(main_len - sub_len + 1):
		if main_list[i:i+sub_len] == sub_list:
			return list(range(i, i+sub_len))
	
	return -1


def text_insert_elem(text, elem, index):
	if index < 0 or index > len(text):
		raise ValueError("Index out of bounds")

	return text[:index] + elem + text[index:]


class SynonymsSubstitute:
	"""
	同义词替换类
	输入一个字符串, 调用 `SynonymsSubstitute`的 `__call__` 方法, 即可替换字符串中所有词到其同义词
	"""
	def __init__(self, tokenizer, detokenizer:nltk.tokenize.TreebankWordDetokenizer):
		self.wordnet = wordnet
		self.tokenizer = tokenizer
		self.detokenizer = detokenizer

	
	def __call__(self, sentence):
		"""
		使用nltk的WordNet获取单词的同义词, 并返回被替换过的字符串
		"""
		# 生成一个列表, 用来记录那些token是单词, 那些token是标点符号
		tokens = self.tokenizer(sentence)
		new_tokens = list()

		for token in tokens:
			try:
				new_tokens.append(self.get_synoyms(token)[0])
			except:
				new_tokens.append(token)
		
		# 如果替换过的token和原来的token一样, 则返回一个空列表
		if new_tokens == tokens:
			new_tokens = list()
				
		return self.detokenizer(new_tokens)


	def get_synoyms(self, word):
		"""
		获取一个单词的所有同义词, 返回一个列表
		"""
		synonyms = set()
		for syn in self.wordnet.synsets(word):
			for lemma in syn.lemmas():
				if lemma.name().lower != word.lower():
					synonyms.add(lemma.name().replace('_', ''))
		
		return list(synonyms)


class NegativeGenerator:
	"""
	给定一个数据集, 处理数据集中的每一项, 对每一项中的demonstration构造负例
	"""
	def __init__(self, dataset_path) -> None:
		# ./data/processed_w_demonstrations/broadtwitter.ner.test.jsonl
		dataset_property = os.path.basename(dataset_path).split(".")
		# 数据集分为三个部分, 数据集名称, 任务, 数据集划分
		self.dataset_name, self.task, self.split = dataset_property[0], dataset_property[1], dataset_property[2]
		with open(dataset_path, 'r') as fp:
			self.dataset = [json.loads(line) for line in fp.readlines()]

		self.tokenizer = word_tokenize
		self.detokenizer = TreebankWordDetokenizer()
		self.detokenizer = self.detokenizer.detokenize

		self.synonyms_substitute = SynonymsSubstitute(tokenizer=self.tokenizer, detokenizer=self.detokenizer)
		self.elem = list()

		from transformers import AutoTokenizer, AutoModel
		self.simcse_tokenizer = AutoTokenizer.from_pretrained('./model/sup-simcse-roberta-large')
		self.simcse = AutoModel.from_pretrained('./model/sup-simcse-roberta-large')

		self.build()


	def __getitem__(self, index):
		return self.elem[index]
	

	def __len__(self):
		return len(self.elem)


	def build(self):
		"""
		处理数据集
		"""
		bar = tqdm(total=len(self.dataset), desc=self.dataset_name)
		for example in self.dataset:
			text:str = example['text']

			# *** 找出数据集构造的各个部分
			guideline_start_idx = text.index("# The following lines describe the task definition")
			try: demonstrations_start_idx = text.index("# This is the demonstrations")
			except ValueError: demonstrations_start_idx = -1
			input_text_start_idx = text.index("# This is the text to analyze")
			result_start_idx = text.find("# The annotation instances that take place in the text above are listed here")

			if demonstrations_start_idx != -1:
				guideline = text[guideline_start_idx:demonstrations_start_idx]
				demonstrations = text[demonstrations_start_idx:input_text_start_idx]
			else:
				guideline = text[guideline_start_idx:input_text_start_idx]
				demonstrations = ""
			input_text = text[input_text_start_idx:result_start_idx]
			result = text[result_start_idx:]

			# *** 如果demonstration存在, 则对demonstration进行处理, 否则, 跳过这个样本
			if demonstrations != "":
				from src.tasks.utils_typing import Entity
				from dataclasses import dataclass
				exec(guideline, globals(), locals())
				exec_dict = {}
				exec(demonstrations, locals(), exec_dict)
				positive_results = exec_dict.get("demonstrations_result", None)
				demonstrations_text = exec_dict.get("demonstrations", None)

				# 获取demonstrations中所有的正例, positive_instances
				positive_instances = list()
				for pr in positive_results:
					positive_instance_cls, positive_instance_attr = self.extract_attribute(pr)
					positive_instances.append({positive_instance_cls: positive_instance_attr})

				if len(positive_instances) >= 1:
					# 处理所有的正例, 构建负例 negative_instances
					negative_instances = list()
					for pi in positive_instances:
						ni = self.generate_negative_instance(
							pi,
							demonstrations_text
						)
						if ni != None: negative_instances.append(ni)

					# 将负例实例化 negative_results
					negative_results = list()
					for ni in negative_instances:
						for cls, attr in ni.items():
							attr = dict(attr)
							negative_results.append(cls(**attr))

					# 使用Jinja渲染模版
					with open('./templates/add_negative_result.txt', 'r') as fp:
						template = fp.read()
						template = jinja.Template(template)
					new_demonstrations = template.render(demonstrations_text=demonstrations_text, positive_instances=positive_results, negative_instances=negative_results)

					example['text'] = guideline+new_demonstrations+input_text+result
					
			self.elem.append(example)
			bar.update(1)
				
					
		return None


	def generate_negative_instance(self, positive_instance, text):
		"""
		给定一个正例, 根据这个正例构造一个新的负例

		Args:
			positive_instance (dict): 正样本, 一个字典
				key是类变量, 表明这个positive_instance属于哪一个类
				value代表属性, 属性可能有多个, [[属性名, 属性值]]
			construct_manner (str): 负例构造的方式, 可以是"positive_based"或者"boundary"
			text (str): 输入文本
			tokenizer (spacy.tokenizer): 分词器
		
		Returns:
			dict: 负例, 数据的构造形式和positive_instance相同. 一个字典, 字典的key是类变量, value是属性列表
				如果负例不存在, 则返回None
		"""
		construct_manner = random.choice(['synonym', 'boundary']) if random.random() < 0.05 else "none"

		cls = list(positive_instance.keys())[0]
		negative_instance = {cls:list()}
		# 获取一个实例所有的属性
		attributes = positive_instance[cls]

		# 构建负例
		for attribute in attributes:
			attribute_name, attribute_value = attribute

			if construct_manner == "synonym":
				new_attribute_value = self.synonyms_substitute(attribute_value)
			elif construct_manner == "boundary":
				new_attribute_value = self.extend_or_contract_tokens(attribute_value, text)
			else:
				new_attribute_value = None
			
			# 特殊情况不输出negative_instance
			if new_attribute_value == None or new_attribute_value == list() or new_attribute_value == attribute_value or new_attribute_value == '':
				negative_instance = None
			else:
				negative_instance[cls].append([attribute_name, new_attribute_value])

		return negative_instance


	def extend_or_contract_tokens(self, span, text):
		"""
		对给定的span向扩展一个token (token是从原句text中提取的), 或者减去一个token
		对所有构造出来的新span, 计算其与原来的文本的相似度, 选择相似度最高的那个span

		输入一个token列表, 返回一个token列表
		如果不存在, 则返回空

		Args:
			span (str): 表示一个正例
			text (str): 正例span所在的文本
		"""
		span = [d for d in self.tokenizer(span)]
		text = [d for d in self.tokenizer(text)]

		new_spans = []

		construct_manner = ['expand', 'delet']

		for cm in construct_manner:
			if cm == 'expand':
				span_index_in_text = find_sublist_index(text, span)
				direction = ['left', 'right']

				for d in direction:
					new_span = copy.deepcopy(span)
					if d == 'left':
						try: new_span.insert(0, text[span_index_in_text[0]-1])
						except: new_span = None
					elif d == 'right':
						try: new_span.append(text[span_index_in_text[-1]+1])
						except: new_span = None
					if new_span != None: new_spans.append(new_span)
			elif cm == 'delet':
				if len(span) == 1: continue
				for d in direction:
					new_span = copy.deepcopy(span)
					if d == 'left':
						new_span = new_span[1:]
					elif d == 'right':
						new_span = new_span[:-1]
					new_spans.append(new_span)

		# 计算与原句子的cosine similarity
		original_span = self.detokenizer(span)
		original_span_simcse_embedding = self.simcse_tokenizer(original_span, padding=True, truncation=True, return_tensors='pt')
		original_span_simcse_embedding = self.simcse(**original_span_simcse_embedding).pooler_output.squeeze(0)
		
		new_spans_similarity_dict = dict()
		for ns in new_spans:
			new_span = self.detokenizer(ns)
			new_span_simcse_embedding = self.simcse_tokenizer(new_span, padding=True, truncation=True, return_tensors='pt')
			new_span_simcse_embedding = self.simcse(**new_span_simcse_embedding).pooler_output.squeeze(0)

			similarity =  cosine_similarity(original_span_simcse_embedding, new_span_simcse_embedding, dim=0)
			new_spans_similarity_dict[new_span] = similarity
		
		new_spans_similarity_dict = OrderedDict(sorted(new_spans_similarity_dict.items(), key=lambda x: x[1], reverse=True))
		
		if new_spans_similarity_dict != {}:
			return next(iter(new_spans_similarity_dict.keys()))
		else:
			return None


	@staticmethod
	def extract_attribute(instance):
		"""
		抽取一个类的所有实例属性
		"""
		cls = instance.__class__

		attributes = list()
		for attr in dir(instance):
			if not callable(getattr(instance, attr)) and not attr.startswith("__") and not attr.startswith("_"):
				attributes.append([attr, getattr(instance, attr)])

		return cls, attributes




if __name__ == "__main__":
	# output_dir = "./data/processed_w_demonstrations_negative"

	# datasets = [
	# 	'broadtwitter.ner',
	# 	'crossner.crossner_ai',
	# 	'crossner.crossner_literature',
	# 	'crossner.crossner_music',
	# 	'crossner.crossner_natural_science',
	# 	'crossner.crossner_politics',
	# 	'fabner.ner',
	# 	'harveyner.ner',
	# 	"mitmovie.ner",
	# 	"mitrestaurant.ner"
	# ]

	output_dir = "./temp/data"

	datasets = [
		'broadtwitter.ner'
	]

	for dataset in datasets:
		dataset_path = os.path.join("./data/processed_w_demonstrations", dataset+".test.jsonl")
		print(dataset_path)
		negative_generator = NegativeGenerator(dataset_path=dataset_path)
		output_path = os.path.join(output_dir, dataset+".test.jsonl")
		if os.path.exists(output_path): os.remove(output_path)

		with open(output_path, 'w') as fp:
			for example in negative_generator:
				# print(json.dumps(example), file=fp)
				print(json.dumps(example))

