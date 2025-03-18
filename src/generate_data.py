import copy
import json
import logging
# import multiprocessing as mp
import os
from argparse import ArgumentParser
from functools import partial
from itertools import cycle
from typing import Type
# from src.utils.clean_cache import clean_cache

import datasets
from tqdm import tqdm


def generate_batch_configs(all_configs, step):
	for i in range(0, len(all_configs), step):
		yield all_configs[i:i+step]


def get_class(class_path: str) -> Type:
	"""
	递归式的取出class_path所对应的类

	Example:
		>>> get_class("src.tasks.broadtwitter.data_loader.BroadTwitterDataLoader")
		取出BroadTwitterDataLoader类
	"""
	components = class_path.split(".")
	mod = __import__(components[0])
	for comp in components[1:]:
		mod = getattr(mod, comp)

	return mod


def multicpu_generator(args, tqdm_position, config):
	# 从config中动态的获取dataloader和sampler
	dataloader_cls = get_class(config["dataloader_cls"])
	sampler_cls = get_class(config["sampler_cls"])

	# 确定随机数种子 `seeds`
	seeds = config.get("seed", 0)
	if isinstance(seeds, int):
		seeds = [seeds]
	# 确定 `label_noise`
	label_noise = config.get("label_noise_prob", 0.0)
	if isinstance(label_noise, float):		
		label_noise = cycle([label_noise])

	if "train_file" in config:
		dataloader = dataloader_cls(config["train_file"], **config)
		for ie_task in config["tasks"]:
			for seed, noise_prob in zip(seeds, label_noise):
				config["seed"] = seed
				config["label_noise_prob"] = noise_prob
				# Avoid multiple values for keyword argument
				_kwargs = {**config, **config["task_configuration"][ie_task]}
				# 实例化sampler, sampler用来对数据进行后处理
				sampler = sampler_cls(
					dataloader,
					task=ie_task,
					split="train",
					**_kwargs,
				)

				output_name = f"{config['dataset_name'].lower()}.{ie_task.lower()}.train.{seed}.jsonl"

				if os.path.exists(os.path.join(args.output_dir, output_name)) and not args.overwrite_output_dir:
					logging.warning(f"Skipping {output_name} because it already exists.")
					continue

				# 使用sampler将数据写入文件
				with open(os.path.join(args.output_dir, output_name), "w") as _file, tqdm(
					total=len(dataloader),
					desc=f"{config['dataset_name']}-{ie_task}-train-{seed}",
					position=tqdm_position,
				) as progress:
					ids = []
					# 使用sampler生成数据, 并写入文件
					for elem in sampler:
						_file.write(f"{json.dumps(elem, ensure_ascii=False)}\n")
						# ids用来记录当前的数据的id, 若处理到了下一个ids, 则更新进度条
						if ids != elem["ids"]:
							ids = elem["ids"]
							progress.update(len(ids))

				logging.info(f"Data saved to {os.path.abspath(os.path.join(args.output_dir, output_name))}")

	if "dev_file" in config:
		config["seed"] = 0
		dataloader = dataloader_cls(config["dev_file"], **config)
		for task in config["tasks"]:
			_kwargs = {**config, **config["task_configuration"][task]}
			sampler = sampler_cls(
				dataloader,
				task=task,
				split="dev",
				**_kwargs,
			)

			output_name = f"{config['dataset_name'].lower()}.{task.lower()}.dev.jsonl"

			if os.path.exists(os.path.join(args.output_dir, output_name)) and not args.overwrite_output_dir:
				logging.warning(f"Skipping {output_name} because it already exists.")
				continue

			with open(os.path.join(args.output_dir, output_name), "w") as _file, tqdm(
				total=len(dataloader),
				desc=f"{config['dataset_name']}-{task}-dev",
				position=tqdm_position,
			) as progress:
				ids = []
				for elem in sampler:
					_file.write(f"{json.dumps(elem, ensure_ascii=False)}\n")
					if ids != elem["ids"]:
						ids = elem["ids"]
						progress.update(len(ids))

			logging.info(f"Data saved to {os.path.abspath(os.path.join(args.output_dir, output_name))}")


	if "test_file" in config:
		config["seed"] = 0
		dataloader = dataloader_cls(config["test_file"], **config)
		for task in config["tasks"]:
			_kwargs = {**config, **config["task_configuration"][task]}
			sampler = sampler_cls(
				dataloader,
				task=task,
				split="test",
				**_kwargs,
			)

			output_name = f"{config['dataset_name'].lower()}.{task.lower()}.test.jsonl"

			if os.path.exists(os.path.join(args.output_dir, output_name)) and not args.overwrite_output_dir:
				logging.warning(f"Skipping {output_name} because it already exists.")
				continue

			with open(os.path.join(args.output_dir, output_name), "w") as _file, tqdm(
				total=len(dataloader),
				desc=f"{config['dataset_name']}-{task}-test",
				position=tqdm_position,
			) as progress:
				ids = []
				for elem in sampler:
					_file.write(f"{json.dumps(elem, ensure_ascii=False)}\n")
					if ids != elem["ids"]:
						ids = elem["ids"]
						progress.update(len(ids))

			logging.info(f"Data saved to {os.path.abspath(os.path.join(args.output_dir, output_name))}")


def main(args):
	os.makedirs(args.output_dir, exist_ok=True)

	config_files = args.configs
	# We generate a new config for each train split and task, so we also parallelize over each split and task
	configs = []
	splits = ["train_file", "dev_file", "test_file"]
	# data_info: 用来在终端中输出本次生成的数据集信息
	data_info = {}

	# *** 构建每个任务, 每个split的configs
	for config_file in config_files:
		# 读取config文件中的内容, 比如说 `configs/data_configs/ace_config.json`
		with open(config_file, "rt") as f:
			config = json.load(f)

		# remove_guidelines: 消融实验, 移除guidelines
		config["remove_guidelines"] = args.baseline
		# include_examples_prob: 在训练数据中添加example的概率
		config["include_examples_prob"] = float(args.include_examples)
		# label_noise_prob: 对label打上mask (noise) 的概率, 比如说将标签 GPE 转化为 LABEL_1
		if args.remove_masking: config["label_noise_prob"] = 0.0
		# guideline_dropout: 随机dropout某些guideline的概率
		if args.remove_dropout:
			for task in config["tasks"]:
				config["task_configuration"][task]["guideline_dropout"] = 0.0

		# 添加自定义的config
		config['add_demonstrations'] = args.add_demonstrations
		config['robustness_enhancement'] = args.robustness_enhancement
		config['top_k'] = args.top_k
		config['device'] = args.device
		config["force_random"] = args.force_random
		config['force_SLR'] = args.force_SLR

		# We generate a new config for each train split and task
		# task: 当前的任务类型, 比如NER, RE等
		tasks = config["tasks"]
		for split in splits:
			for task in tasks:
				new_config = copy.deepcopy(config)
				# 删除new_config中除了当前split之外的其他split
				if split in new_config:
					for other_split in splits:
						if other_split != split:
							if other_split in new_config:
								new_config.pop(other_split)

					new_config["tasks"] = [task]

					# 在data_info中添加数据集名称字段
					if config["dataset_name"] not in data_info:
						data_info[config["dataset_name"]] = {}

					# 在data_info中添加任务字段
					if task not in data_info[config["dataset_name"]]:
						data_info[config["dataset_name"]][task] = {
							"train_file": False,
							"dev_file": False,
							"test_file": False,
						}
					data_info[config["dataset_name"]][task][split] = True

					configs.append(new_config)		# 将构造好的new_config添加到configs列表中

	generator_fn = partial(
		multicpu_generator,
		args,
	)

	logging.warning(f"We will generate the following data: {json.dumps(data_info, indent=4)})")

	# *** 记录train_file, 用于后续的KNN检索导入训练集
	configs = sorted(
		configs,
		key=lambda x: x['dataset_name'] + str(x['tasks']) + ('0' if 'train_file' in x else '1' if 'dev_file' in x else '2')
	)
	for config in configs:
		if config['dataset_name'] == "CASIE":
			if "dev_file" in config:
				extra_train_file = config.get("dev_file", None)
			if "test_file" in config:
				config['extra_train_file'] = extra_train_file	# type: ignore

		if "train_file" in config:
			extra_train_file = config.get("train_file", None)
			config['extra_train_file'] = extra_train_file
		if "dev_file" in config:
			config['extra_train_file'] = extra_train_file
		if "test_file" in config:
			config['extra_train_file'] = extra_train_file
		
	# *** 调用generator_fn生成数据
	for config in configs:
		generator_fn(1, config)

	logging.warning(f"Data saved to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
	logging.basicConfig(level=logging.WARNING)
	datasets.logging.set_verbosity_error()
	datasets.logging.disable_progress_bar()
	parser = ArgumentParser("generate_data", description="Generate Code formatted data.")

	parser.add_argument(
		"-c",
		"--configs",
		nargs="+",
		dest="configs",
		type=str,
		help="The list of configuration files.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=str,
		dest="output_dir",
		default="data/processed",
		help="Output directory where files will be saved.",
	)
	parser.add_argument(
		"--overwrite_output_dir",
		action="store_true",
		help="Whether to overwrite the output dir.",
	)
	parser.add_argument(
		"--baseline",
		action="store_true",
		default=False,
		help="Whether to generate baseline data. | 是否生成baslinedata, 也就是什么都没有的data",
	)
	parser.add_argument(
		"--include_examples",
		action="store_true",
		default=False,
		help="Whether to include examples in the data. | 在训练数据中添加example",
	)
	parser.add_argument(
		"--remove_dropout",
		action="store_true",
		default=False,
		help="Remove guideline dropout for the ablation analysis. | ablation analysis选项: 删除guideline的dropout",
	)
	parser.add_argument(
		"--remove_masking",
		action="store_true",
		default=False,
		help="Remove guideline masking for the ablation analysis. | ablation analysis设置: 删除guidelien mask",
	)

	# ************************************************** 自定义参数
	parser.add_argument(
		"--add_demonstrations",
		action="store_true",
		default=False,
		help="whether add demonstration for in-context learning | 是否添加用于in-context learning的demonstration"
	)
	parser.add_argument(
		"--top_k",
		type=int,
		default=1,
		help="top k for KNN retrieval | KNN检索的top k"
	)
	parser.add_argument(
		"--robustness_enhancement",
		action="store_true",
		default=False,
		help="whether to enhance the robustness of the model"
	)
	parser.add_argument(
		"--device",
		type=str,
		default='cpu',
		help="device to store the data and model"
	)
	parser.add_argument(
		"--force_random",
		action="store_true",
		default=False,
		help="在生成测试集数据的时候, 强制使用random检索demonstrations的策略"
	)
	parser.add_argument(
		"--force_SLR",
		action="store_true",
		default=False,
		help="在生成训练集数据的时候, 强制使用SLR检索demonstrations的策略"
	)
	# ************************************************** 自定义参数

	args = parser.parse_args()
	main(args)