"""
Sentence_Level_Retrieval
给定样本池, 使用KNN+SimCSE检索与query最相似的样本, 作为demonstration
"""
import random
from src.utils import (
	KNN,
	get_class,
)


class DemonstrationRetriever:
	"""
	Retrieves a demonstration from the given dataloader.
	初始化RetrievalDemonstration对象。

	Args:
		dataloader (DatasetLoader): query数据集对象
			id: demonstration的唯一标识符
			doc_id: 文档的唯一标识符
			text: 输入文本
		retrieval_strategy (str): 使用的检索策略。可以是`random`或`knn`
		demon_loader_cls (str): demonstrations的数据加载器的类
			如果没有提供则会将 `dataloader` 作为demonstrations的样本池
		train_loader_argument (Any): demonstrations的数据加载器的参数
		kwargs (dict): 其他参数, top_k: knn检索的最大数量
			task: 当前任务
			top_k: knn检索的最大数量
			device: cpu还是cuda
			exclude_current_query: 是否排除当前query
	"""
	def __init__(self,
			dataloader: dict,
			retrieval_strategy: str,
			demo_loader_cls: str = None,
			demo_loader_arguments = None,
			**kwargs
		) -> None:
		self.dataloader = dataloader
		self.top_k = kwargs.get("top_k", 1)
		self.robustness_enhancement = kwargs.get("robustness_enhancement", False)
		self.device = kwargs.get("device", "cpu")
		self.is_train_dev = kwargs.get("is_train_dev", False)

		assert retrieval_strategy in ['random', 'knn'], "The retrieval strategy should be either `random` or `knn`." 
		self.retrieval_strategy = retrieval_strategy

		# 如果retrieval_strategy是knn, 那么需要加载训练集, 并且构建KNN检索, 如果是random, 那么不需要
		if self.retrieval_strategy == "knn":
			demo_loader_cls = get_class(demo_loader_cls)		# type: callable
			demo_dataloader = demo_loader_cls(demo_loader_arguments, **kwargs)
			# text2example: 样本池中一个样本的text部分到一个样本的映射
			self.text2example = {item['text'] : item for item in demo_dataloader}
			self.sample_pool = KNN("./model/sup-simcse-roberta-large", device=self.device)
			self.sample_pool.build_index(sentences_or_file_path=[item for item in self.text2example.keys()], device=self.device, use_faiss=True)
		else:
			if demo_loader_cls is None:
				self.sample_pool = dataloader
			else:
				demo_loader_cls = get_class(demo_loader_cls)		# type: callable
				demo_dataloader = demo_loader_cls(demo_loader_arguments, **kwargs)
				self.sample_pool = demo_dataloader


	def __call__(self, index_query: int):
		"""
		给定一个index_query, 根据index_query生成一个demonstration

		Args:
			index_query (int): The index of the query. The demonstration should be different from the query.
		
		Returnes:
			List[Dict[str, Any]]: A list of demonstrations.
		"""
		demonstration = []
		if self.retrieval_strategy == "random":
			pool = [i for i in range(len(self.sample_pool)) if i != index_query]

			selected_index = random.sample(pool, self.top_k)
			for index in selected_index:
				demonstration.append(self.sample_pool[index])

		elif self.retrieval_strategy == "knn":
			if not self.is_train_dev:
				retrieved_demonstrations = self.sample_pool.search(
					queries = self.dataloader[index_query]['text'],
					threshold = 0.2,
					top_k = self.top_k
				)
				for i in range(len(retrieved_demonstrations)):
					demonstration.append(self.text2example[retrieved_demonstrations[i][0]])
			else:
				retrieved_demonstrations = self.sample_pool.search(
					queries = self.dataloader[index_query]['text'],
					threshold = 0.2,
					top_k = self.top_k + 1
				)
				retrieved_demonstrations = [
					demo
					for demo in retrieved_demonstrations 
					if demo[0] != self.dataloader[index_query]['text']
				]
				for i in range(len(retrieved_demonstrations)):
					demonstration.append(self.text2example[retrieved_demonstrations[i][0]])


		return demonstration


if __name__ == '__main__':
	pass