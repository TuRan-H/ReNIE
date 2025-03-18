"""
使用BertLM训练bert
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import (
	BertForMaskedLM, 
	BertTokenizer, 
	Trainer,
	TrainingArguments
)
from src.utils import MaskingDataset, MaskingDataCollator


MODEL_PATH = "/home/turan/SDA/models/bert-base-uncased"
SAVE_PATH = "./results/masked_bert"


if __name__ == '__main__':
	# 导入数据集
	dataset_list = [
		'bc5cdr.ner',
		'ace05.ner',
		'bc5cdr.ner',
		'conll03.ner',
		'diann.ner',
		'ncbidisease.ner',
		'ontonotes5.ner',
		'wnut17.ner'
	]
	dataset = MaskingDataset(dataset_name=dataset_list, split="train")

	# # 导入模型
	model = BertForMaskedLM.from_pretrained(MODEL_PATH)
	tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

	for data in dataset:
		if len(data['input_ids']) != len(data['tags_mask']):
			raise ValueError("The length of input_ids and tags_mask is not equal.")

	
	# # 实例化相关类
	data_collator = MaskingDataCollator(tokenizer)
	training_args = TrainingArguments(
		output_dir=SAVE_PATH,			# 模型保存路径
		overwrite_output_dir=True,		# 是否覆盖之前的结果
		num_train_epochs=3,				# 训练的epoch数
		per_device_train_batch_size=8,	# 每个设备上的batch size
		save_steps=10_000,				# 保存模型的步数
		save_total_limit=2,				# 保存的最大模型数量
		report_to= "none",
		logging_strategy='steps',
		logging_steps=500
		)
	trainer = Trainer(
		model=model,					# BERT模型
		args=training_args,				# 训练参数
		train_dataset=dataset,			# 训练数据
		data_collator=data_collator		# 数据掩码生成器
	)

	# 开始训练
	trainer.train()
