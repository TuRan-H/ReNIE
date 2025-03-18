"""
同义词采样 (Synonym Sampling) 使用bert生成负样本
"""
import torch
from transformers import (
	AutoTokenizer,
	AutoModelForMaskedLM,
	AutoModel,
	BertTokenizerFast,
	PreTrainedModel
)
from collections import OrderedDict


class SynonymSampling:
	"""
	输入一个句子, 输出句子中实体的同义词

	Args:
		simcse_tokenizer (AutoTokenizer): 用于加载SimCSE模型的tokenizer
		simcse_model (PreTrainedModel): 用于加载SimCSE模型
		device (str): bert model的设备类型, 默认为cpu
	"""
	def __init__(self, simcse_tokenizer, simcse_model: PreTrainedModel, device) -> None:
		self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("./model/masked_bert")
		self.model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained("./model/masked_bert")
		self.model.to(device)
		self.simcse_tokenizer = simcse_tokenizer
		self.simcse = simcse_model


	def __call__(self, context: str, entity: str, top_k = 3):
		"""
		输入一个句子, 输出句子中实体的同义词
		"""
		synonyms_list = self.get_synonyms(context, entity, top_k)
		synonym = self.calculate_simiarity(synonyms_list, entity)
		
		return synonym


	def get_synonyms(self, sentence: str, entity: str, top_k):
		"""
		获取实体的同义词

		Args:
			sentence (str): 输入句子
			entity (str): 实体
			top_k (int): 返回的同义词数量
		
		Returns:
			list: 同义词列表
		"""
		# 将entity替换为[MASK]
		if entity in sentence:
			masked_sentence = sentence.replace(entity, '[MASK]')
		else:
			return []

		# 进行预测
		tokenizer_input: torch.Tensor = self.tokenizer.encode(masked_sentence, return_tensors="pt")
		if tokenizer_input.device != self.model.device:
			tokenizer_input = tokenizer_input.to(self.model.device)
		mask_token_index = torch.where(tokenizer_input == self.tokenizer.mask_token_id)[1]

		with torch.no_grad():
			output = self.model(tokenizer_input)
		
		mask_token_logits = output.logits[0, mask_token_index, :]
		top_3_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
		top_3_tokens = [self.tokenizer.decode([token]) for token in top_3_tokens]
		top_3_tokens = [elem for elem in top_3_tokens if elem.lower() != entity.lower()]

		return top_3_tokens


	def calculate_simiarity(self, synonyms_list: list, original_token: str):
		"""
		计算同义词列表中的词与原词的相似度, 返回相似度最高的那个
		"""
		original_token_simcse_embedding = self.simcse_tokenizer(original_token, padding = True,
			truncation = True,
			return_tensors = "pt"
		)
		original_token_simcse_embedding = {k: v.to(self.simcse.device) for k, v in original_token_simcse_embedding.items()}
		original_token_simcse_embedding = self.simcse(**original_token_simcse_embedding).pooler_output.squeeze(0)

		synonyms_similarity_dict = OrderedDict()
		for synonym in synonyms_list:
			synonym_simcse_embedding = self.simcse_tokenizer(synonym, padding = True,
				truncation = True,
				return_tensors = "pt"
			)
			synonym_simcse_embedding = {k: v.to(self.simcse.device) for k, v in synonym_simcse_embedding.items()}
			# pooler_output表示句子级别的embedding, 通常是[CLS]token所对应的向量
			synonym_simcse_embedding = self.simcse(**synonym_simcse_embedding).pooler_output.squeeze(0)
			similarity = torch.cosine_similarity(original_token_simcse_embedding, synonym_simcse_embedding, dim=0)
			synonyms_similarity_dict[synonym] = similarity
		
		# 由于sorted函数返回的是一个迭代
		synonyms_similarity_dict = sorted(synonyms_similarity_dict.items(), key=lambda x: x[1], reverse=True)

		if synonyms_similarity_dict != {}:
			return synonyms_similarity_dict[0][0]
		else:
			return ""


if __name__ == '__main__':
	sentence = "the man is sleeping on the chair"
	span = "chair"
	sampler = SynonymSampling(
		simcse_tokenizer = AutoTokenizer.from_pretrained('./model/sup-simcse-roberta-large'),
		simcse_model = AutoModel.from_pretrained('./model/sup-simcse-roberta-large'),
		device="cpu"
	)
	result = sampler(sentence, span)
	print(result)