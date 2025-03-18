"""
继承于SimCSE
修改其中某些方法
给定query, 检索demonstrations
"""
import numpy as np
import logging
import torch
from torch import Tensor
from numpy import ndarray
from simcse import SimCSE
from typing import Union
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


class KNN(SimCSE):
	def __init__(
		self, model_name_or_path: str, 
		device: str = None,
		num_cells: int = 100,
		num_cells_in_search: int = 10,
		pooler = None
	):
		super().__init__(model_name_or_path, device=device, pooler=pooler)

	def build_index(
		self, 
		sentences_or_file_path:Union[str, list[str]], 
		use_faiss: bool = None, 
		faiss_fast: bool = False, 
		device: str = None, 
		batch_size: int = 64
	):
		"""
		创建最近邻检索的index

		Args:
			sentences_or_file_path: 数据文件或者数据文件地址
			use_faiss: 是否使用faiss框架进行检索
			faiss_fast: 是否使用faiss_fast框架进行检索, 使用faiss_fast需要进行额外的训练
			device: 将index保存在那个设备上.
			batch_size: build_index时的batch大小
		"""
		if device == None:
			device = self.device
		if use_faiss is None or use_faiss:
			try:
				import faiss
				assert hasattr(faiss, "IndexFlatIP")
				use_faiss = True 
			except:
				logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
				use_faiss = False
		
		# if the input sentence is a string, we assume it's the path of file that stores various sentences
		if isinstance(sentences_or_file_path, str):
			sentences = []
			with open(sentences_or_file_path, "r") as f:
				logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
				# *** 这里tqdm设置为了disable
				for line in tqdm(f, disable=True):
					sentences.append(line.rstrip())
			sentences_or_file_path = sentences
		
		logger.info("Encoding embeddings for sentences...")
		embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

		logger.info("Building index...")
		self.index = {"sentences": sentences_or_file_path}
		
		if use_faiss:
			quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
			if faiss_fast:
				index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path))) 
			else:
				index = quantizer

			# *** 假设device存在, 且不是 `cuda`, 说明device有设备编号, 则取出设备编号
			if device and (device != "cuda" and device != 'cpu'):
				device_index = int(device.split(":")[-1])
			else:
				device_index = 0

			if device != None:
				if hasattr(faiss, "StandardGpuResources"):
					logger.info("Use GPU-version faiss")
					res = faiss.StandardGpuResources()
					res.setTempMemory(20 * 1024 * 1024 * 1024)
					index = faiss.index_cpu_to_gpu(
						provider=res,
						device= device_index,
						index=index
					)
				else:
					logger.info("Use CPU-version faiss")
			else: 
				logger.info("Use CPU-version faiss")

			if faiss_fast:            
				index.train(embeddings.astype(np.float32))
			index.add(embeddings.astype(np.float32))
			index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
			self.is_faiss_index = True
		else:
			index = embeddings
			self.is_faiss_index = False
		
		self.index["index"] = index
		logger.info("Finished")

		# *** 用于删除显存占用
		self.res = locals().get('res', None)

	def encode(
		self, sentence: Union[str, list[str]], 
		device: str = None, 
		return_numpy: bool = False,
		normalize_to_unit: bool = True,
		keepdim: bool = False,
		batch_size: int = 64,
		max_length: int = 128
	) -> Union[ndarray, Tensor]:

		target_device = self.device if device is None else device
		self.model = self.model.to(target_device)
		
		single_sentence = False
		if isinstance(sentence, str):
			sentence = [sentence]
			single_sentence = True

		embedding_list = [] 
		with torch.no_grad():
			total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
			for batch_id in tqdm(range(total_batch), leave=False, position=0, disable=False, desc="KNN Encoding"):
				inputs = self.tokenizer(
					sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
					padding=True, 
					truncation=True, 
					max_length=max_length, 
					return_tensors="pt"
				)
				inputs = {k: v.to(target_device) for k, v in inputs.items()}
				outputs = self.model(**inputs, return_dict=True)
				if self.pooler == "cls":
					embeddings = outputs.pooler_output
				elif self.pooler == "cls_before_pooler":
					embeddings = outputs.last_hidden_state[:, 0]
				else:
					raise NotImplementedError
				if normalize_to_unit:
					embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
				embedding_list.append(embeddings.cpu())
		embeddings = torch.cat(embedding_list, 0)
		
		if single_sentence and not keepdim:
			embeddings = embeddings[0]
		
		if return_numpy and not isinstance(embeddings, ndarray):
			return embeddings.numpy()
		return embeddings