"""
自定义脚本, 用来构造context-aware bert Synonym Substitution的数据集
"""
from torch.utils.data import Dataset
import json
import os
from transformers import BertTokenizer
import re
from tqdm import tqdm
from typing import Union


class MaskingDataset(Dataset):
	def __init__(self, dataset_name: Union[list[str], str], split):
		"""
		初始化数据集

		Args:
			dataset_name: str, 数据集名称
			split: str, 数据集划分, 可选值为["train", "dev", "test"]
		"""
		DATASET_BASE_NAME = "/home/turan/RESEARCH/GoLLIE/data/processed_w_examples"
		self.file_path = []

		# 确定数据集地址
		if isinstance(dataset_name, str):
			dataset_name = [dataset_name]
		for elem in dataset_name:
			if split == "train":
				self.file_path.append(os.path.join(DATASET_BASE_NAME, f"{elem}.train.0.jsonl"))
			elif split == 'dev':
				self.file_path.append(os.path.join(DATASET_BASE_NAME, f"{elem}.dev.jsonl"))
			else:
				self.file_path.append(os.path.join(DATASET_BASE_NAME, f"{elem}.test.jsonl"))

		# 读取数据
		data = []
		for file_path in self.file_path:
			with open(file_path) as fp:
				data.extend([json.loads(line) for line in fp])
		
		# 初始化tokenizer
		self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained("/home/turan/SDA/models/bert-base-uncased")

		# 处理数据
		bar = tqdm(total=len(data), desc=f"Processing data")
		self.processed_example = []
		for example in data:
			self.processed_example.append(self.process_example(example))
			bar.update(1)


	def __len__(self):
		return len(self.processed_example)


	def __getitem__(self, idx):
		return self.processed_example[idx]

	
	def build_tags_mask(self, example, tags: list[str]):
		"""
		生成tags_mask, 用于标记需要mask的token位置
		其中1表示需要mask的token, 0表示不需要mask的token

		Args:
			example: `BatchEncoding`, 一个batch的数据
			tags: `List[str]`, 一个batch的tags
		"""
		# 创建tags_mask
		input_ids = example['input_ids']
		# tags_mask = torch.zeros_like(input_ids, dtype=example['input_ids'].dtype)
		tags_mask = [0] * len(input_ids)

		# 遍历每一个input_id, 以及每一个tag
		for tag in tags:
			# 查找tag在input_id中的位置
			tag_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tag))
			tag_index = find_sublist_index(input_ids, tag_ids)
			if tag_index != -1: 
				for i in tag_index: tags_mask[i] = 1
	
		return tags_mask


	def process_example(self, example):
		"""
		对example进行处理
		"""
		# 提取原数据集的labels字段, 使用正则对其进行处理
		tags = example['labels']
		re_iter = re.finditer(pattern=r"\w+\(span=[\"'](.+?)[\"']\)", string=tags)
		tags = [match.group(1) for match in re_iter]

		# 对原数据集的unlabelled_sentence字段进行分词
		text = example['unlabelled_sentence']
		processed_example = self.tokenizer(text, padding=True, return_tensors=None)

		# 构造tags_token_mask
		processed_example['tags_mask'] = self.build_tags_mask(processed_example, tags)

		return processed_example


def find_sublist_index(main_list, sub_list):
	"""
	在tensor中找到sub_tensor的起始索引 (1D tensor)

	Args:
		tensor: `torch.Tensor`, 原始tensor
		sub_tensor: `torch.Tensor`, 需要查找的子tensor

	Returns:
		index: `int`, 子tensor在原始tensor中的起始索引，如果未找到则返回-1
	"""
	# 获取原始列表和子列表的长度
	sub_len = len(sub_list)
	for i in range(len(main_list) - sub_len + 1):
		if main_list[i:i + sub_len] == sub_list:
			return [j for j in range(i, i+sub_len)]

	return -1


if __name__ == '__main__':
	dataset_list = ['bc5cdr.ner', 'ace05.ner']
	dataset = MaskingDataset(dataset_name=dataset_list, split="train")


	count = 0
	for data in dataset:
		if len(data['input_ids']) != len(data['tags_mask']):
			count += 1
	print(count)