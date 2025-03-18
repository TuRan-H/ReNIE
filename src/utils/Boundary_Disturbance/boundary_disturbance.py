"""
边界扰动 (Boundary Disturbance) 生成负样本
"""
from transformers import AutoTokenizer, AutoModel
from nltk import word_tokenize
import copy
from src.utils import find_sublist_index
from collections import OrderedDict
from torch import cosine_similarity



class BoundaryDisturbance:
	def __init__(self, simcse_tokenizer, simcse_model):
		self.tokenizer = word_tokenize
		self.simcse_tokenizer = simcse_tokenizer
		self.simcse = simcse_model


	def __call__(self, context, entity):
		"""
		输入一个句子, 输出句子中实体的同义词, 采用边界扰动策略
		"""
		all_synonyms = self.extend_or_truncate_tokens(context, entity)
		synonym = self.calculate_similarity(all_synonyms, entity)

		return synonym


	def extend_or_truncate_tokens(self, text, span):
		"""
		对给定的span扩展一个token (token是从原句text中提取的), 或者减去一个token

		输入一个token列表, 返回一个token列表
		如果不存在, 则返回空

		Args:
			span (str): 需要扩展或者减少的span
			text (str): span所在的文本
		"""
		span = [d for d in self.tokenizer(span)]
		text = [d for d in self.tokenizer(text)]
		output_spans = []
		construct_manner = ["extend", "contract"]
		direction = ['left', 'right']

		for manner in construct_manner:
			if manner == 'extend':
				span_index_in_text = find_sublist_index(text, span)

				for d in direction:
					new_span = copy.deepcopy(span)
					if d == 'left':
						try: new_span.insert(0, text[span_index_in_text[0]-1])
						except: new_span = None
					elif d == 'right':
						try: new_span.append(text[span_index_in_text[-1]+1])
						except: new_span = None
					if new_span != None: output_spans.append(new_span)
			else:
				if len(span) == 1: continue
				for d in direction:
					new_span = copy.deepcopy(span)
					if d == 'left':
						new_span = new_span[1:]
					elif d == 'right':
						new_span = new_span[:-1]
					output_spans.append(new_span)

		for index, span in enumerate(output_spans):
			new_span = " ".join(span)
			output_spans[index] = new_span

		return output_spans
	

	def calculate_similarity(self, new_spans: list, original_span: str):
		"""
		计算新span与原span的相似度, 并返回相似度最高的span
		"""
		original_span_simcse_embedding = self.simcse_tokenizer(original_span, padding = True,
			truncation = True,
			return_tensors = "pt"
		)
		original_span_simcse_embedding = {k: v.to(self.simcse.device) for k, v in original_span_simcse_embedding.items()}
		original_span_simcse_embedding = self.simcse(**original_span_simcse_embedding).pooler_output.squeeze(0)

		new_spans_similarity_dict = OrderedDict()
		for ns in new_spans:
			new_span_simcse_embedding = self.simcse_tokenizer(ns, padding = True,
				truncation = True,
				return_tensors = "pt"
			)
			new_span_simcse_embedding = {k: v.to(self.simcse.device) for k, v in new_span_simcse_embedding.items()}
			new_span_simcse_embedding = self.simcse(**new_span_simcse_embedding).pooler_output.squeeze(0)
			similarity = cosine_similarity(original_span_simcse_embedding, new_span_simcse_embedding, dim=0)
			new_spans_similarity_dict[ns] = similarity
		new_spans_similarity_dict = OrderedDict(sorted(new_spans_similarity_dict.items(), key=lambda x: x[1], reverse=False))

		if new_spans_similarity_dict != {}:
			return next(iter(new_spans_similarity_dict.keys()))
		else:
			return ""


if __name__ == '__main__':
	bd = BoundaryDisturbance(
		simcse_tokenizer = AutoTokenizer.from_pretrained('./model/sup-simcse-roberta-large'),
		simcse_model = AutoModel.from_pretrained('./model/sup-simcse-roberta-large')
	)
	sentences = "Microsoft has released the latest Surface Pro laptop"
	span = "Surface Pro laptop"
	result = bd(sentences, span)
	print(result)
