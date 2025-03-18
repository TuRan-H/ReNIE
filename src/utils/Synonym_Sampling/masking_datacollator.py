"""
构造用于训练 BERT synonym substitution 的 DataCollator
"""
import torch
from transformers import BertTokenizer, BatchEncoding
from typing import Mapping
import copy


# 加载 spaCy 语言模型
class MaskingDataCollator:
	def __init__(
		self,
		tokenizer: BertTokenizer,
		max_length: int = None,
	) -> None:
		self.tokenizer = tokenizer
		self.pad_token_id = tokenizer.pad_token_id
		self.mask_token_id = tokenizer.mask_token_id
		self.max_length = max_length
	

	def __call__(self, examples: list[Mapping[str, list]]) -> Mapping[str, torch.Tensor]:
		"""
		对一个batch的数据进行掩码
		"""
		features = copy.deepcopy(examples)		# 深拷贝, 防止修改原数据

		# 构造tags_mask_list, 用以标记需要mask的token位置
		tags_mask_list = [feature.pop('tags_mask') for feature in features]
		tags_mask_list_max_length = max([len(i) for i in tags_mask_list])
		for i in range(len(tags_mask_list)):
			remainder = [0] * (tags_mask_list_max_length - len(tags_mask_list[i]))	# 这里不使用pad_token_id, 因为tags_mask只有0和1
			tags_mask_list[i] += remainder
		tags_mask_list = torch.tensor(tags_mask_list, dtype=torch.bool)
		
		# 对input_ids, token_type_id, attention_mask进行padding
		features:BatchEncoding = self.tokenizer.pad(
			features,
			padding=True,
			max_length=self.max_length,
			return_tensors='pt'
		)

		# 特殊处理: 防止tags_mask_list和input_ids的padding后的长度不同
		# if tags_mask_list.shape[1] != features['input_ids'].shape[1]:
		# 	input_ids_max_length = features['input_ids'].shape[1]
		# 	tags_mask_list = tags_mask_list[:, :input_ids_max_length]

		# 对标签进行处理
		labels = features['input_ids'].clone()
		for tags_mask, label in zip(tags_mask_list, labels):
			label[~tags_mask] = -100	# 将不需要mask的位置设置为-100
		features['labels'] = labels
		
		# 对input_ids进行掩码
		for i in range(len(features['input_ids'])):
			features['input_ids'][i][tags_mask_list[i]] = self.mask_token_id

		return features


if __name__ == '__main__':
	# 示例用法
	from src.utils import MaskingDataset
	from torch.utils.data import DataLoader

	tokenizer = BertTokenizer.from_pretrained("/home/turan/SDA/models/bert-base-uncased")
	collator = MaskingDataCollator(tokenizer=tokenizer)
	dataset_list = ['bc5cdr.ner', 'ace05.ner']
	dataset = MaskingDataset(dataset_name=dataset_list, split="train")

	dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)

	for epoch in range(3):
		for batch in dataloader:
			pass
		
	print("Done!")
	