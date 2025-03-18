import glob
import importlib
import json
import logging
import os
import sys
from typing import Any, Dict, List, Type
import re

from tqdm import tqdm

from src.arguments import DataTrainingArguments, ModelArguments
from src.tasks import task_id_to_prompts, TASK_ID_TO_CONFIGS
from src.tasks.utils_typing import AnnotationList
from transformers import HfArgumentParser, Seq2SeqTrainingArguments


class ResultLogger:
	"""
	A class to log the results of a task.

	Args:
		task_name: The name of the task.
	"""
	def __init__(self, task_name: str) -> None:
		self.task_name = task_name
		self.sentences: List[str] = []
		self.golds: List[AnnotationList] = []
		self.predictions: List[AnnotationList] = []
		self.impossible_to_parse: int = 0
		self.hallucinated_predictions: int = 0
		self.valid_predictions: int = 0
		self.total_predictions: int = 0
		self.gold_predictions: int = 0

	def add_sentence(self, sentence: str, gold_labels: AnnotationList, pred_labels: AnnotationList) -> None:
		"""
		Add a sentence to the logger.

		Args:
			sentence: The sentence to label.
			gold_labels: The 'AnnotationList' gold labels.
			pred_labels: The 'AnnotationList' predicted labels.
		"""
		filtered_pred_labels = pred_labels.filter_hallucinations(sentence)
		self.impossible_to_parse += 1 if filtered_pred_labels.parse_error else 0
		self.hallucinated_predictions += filtered_pred_labels.hallucinated_no
		self.valid_predictions += len(filtered_pred_labels)
		self.total_predictions += len(pred_labels)
		self.gold_predictions += len(gold_labels)

		self.sentences.append(sentence)
		self.golds.append(gold_labels)
		self.predictions.append(filtered_pred_labels)

	def compute_metrics(self, scorer: Any, scorer_config: bool) -> Dict[str, Dict[str, Dict[str, float]]]:
		"""
		Compute the metrics for the task.

		Args:
			scorer: The scorer to use.

		Returns:
			A dictionary containing the metrics.
		"""
		if len(self.sentences) == 0:
			raise ValueError(f"No sentences were added to the {self.task_name} logger, we cannot compute metrics.")

		scores: Dict[str, Dict[str, Dict[str, float]]] = scorer(
			reference=self.golds,
			predictions=self.predictions,
			scorer_config=scorer_config
		)
		results: Dict[str, Dict[str, Dict[str, float]]] = {
			"predictions_stats": {
				"impossible_to_parse": {
					"total": self.impossible_to_parse,
					"percentage%": round((self.impossible_to_parse / len(self.sentences)) * 100, 4),
				},
				"hallucinated_predictions": {
					"total": self.hallucinated_predictions,
					"percentage%": (
						0.0
						if self.total_predictions == 0
						else round((self.hallucinated_predictions / self.total_predictions) * 100, 4)
					),
				},
				"total": {"predictions": self.valid_predictions, "gold": self.gold_predictions},
			},
		}
		scores.update(results)
		return scores

	def print_predictions(self, output_path: str):
		"""
		Print the predictions to a file in json format. In the following format:
		{
			"sentence": {
				"golds": "gold labels",
				"predictions": "predicted labels"
			}
		}

		Args:
			output_path: The path to the output file.
		"""
		if len(self.sentences) == 0:
			raise ValueError(f"No sentences were added to the {self.task_name} logger, we cannot print predictions.")
		with open(output_path, "w", encoding="utf8") as f:
			for sentence, prediction, gold in zip(self.sentences, self.predictions, self.golds):
				example = {sentence: {"golds": gold.to_string(), "predictions": prediction.to_string()}}
				json.dump(example, ensure_ascii=False, indent=4, fp=f)
				f.write("\n")


def import_prompts(task_module: str) -> None:
	"""Imports everything from a module.

	Args:
		task_module (str): the module to import everything from.
	"""
	# get a handle on the module, 使用 `import_module()` 动态的导入一个module
	mdl = importlib.import_module(task_module)

	# # is there an __all__?  if so respect it
	# if "__all__" in mdl.__dict__:
	#     names = mdl.__dict__["__all__"]
	# else:
	#     # otherwise we import all names that don't begin with _
	#     names = [x for x in mdl.__dict__ if not x.startswith("_")]

	names = {x: y for x, y in mdl.__dict__.items() if not x.startswith("_")}

	# now drag them in
	globals().update(dict(names.items()))


def get_class(class_path: str) -> Type:
	"""
	根据 `class_path` 导入一个类对象

	param
	---
	class_path: 类对象的相对地址
		比如说: src.tasks.bc5cdr.scorer.Bc5cdrEntityScorer

	return
	---
	返回找到的class object
	"""
	components = class_path.split(".")					# 根据 `.` 分割 `class_path`
	mod = __import__(components[0])					# `__import__` 和 `importlib.import_module()` 类似, 都是用于动态的导入一个module object
	for component in components[1:]:						# 循环导入一个 module object
		mod = getattr(mod, component)

	return mod


def fix_prompt_outputs(text: str) -> str:
	"""Fixes some known structural bugs.

	Args:
		text (str): Bugged output.

	Returns:
		str: Corrected output.
	"""

	text = text.replace(")\n ", "),\n ")
	return text


def remove_hallucinations(unlabelled_sentence: str, predictions: List[Any]) -> List[Any]:
	"""Removes predictions that are not in the unlabelled sentence.
	Args:
		unlabelled_sentence (str): The unlabelled sentence.
		predictions (List[Any]): The list of predictions.
	Returns:
		List[Any]: The list of predictions that are in the unlabelled sentence.
	"""
	accepted_list: List[Any] = []
	for prediction in predictions:
		if prediction.exists_in(unlabelled_sentence):
			accepted_list.append(prediction)
	return accepted_list


def get_scorer_config(task_id: str) -> bool:
	"""Get the scorer configuration for a task.

	Args:
		task (str): The task to get the configuration for.

	Returns:
		bool: The configuration for the scorer.
	"""
	config_path, task = TASK_ID_TO_CONFIGS[task_id]
	config = json.load(open(config_path, "rt", encoding="utf8"))
	scorer_config = config['task_configuration'][task].get("scorer_config", False)

	return scorer_config


def evaluate(
	model_args: ModelArguments,
	data_args: DataTrainingArguments,
	training_args: Seq2SeqTrainingArguments,
	checkpoint_path: str = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
	"""This function evaluates the output of a model.

	Args:
		model_args (ModelArguments): Model arguments. See `ModelArguments` docs.
		data_args (DataTrainingArguments): Data arguments. See `DataTrainingArguments` docs.

	Returns:
		Dict[str, Dict[str, Dict[str, Dict[str, float]]]]: A dictionary containing the scores for each task
		present in the dataset.
	"""
	if checkpoint_path is None:
		output_dir = training_args.output_dir
	else:
		output_dir = checkpoint_path

	predictions_dir = os.path.join(output_dir, "predictions")

	gold_data_dir = data_args.dataset_dir
	all_scores = {}

	scores_file_name = os.path.join(output_dir, "task_scores.json")
	scores_file_name_summary = os.path.join(output_dir, "task_scores_summary.json")
	with tqdm(total=len(data_args.test_tasks), desc="Evaluating") as pbar:
		for task in data_args.test_tasks:
			pbar.set_description(f"Evaluating {task}")
			gold_path = os.path.join(
				gold_data_dir,
				f"{task}.{'test' if not data_args.use_dev_inference else 'dev'}.jsonl",
			)
			pred_path = os.path.join(predictions_dir, task) + ".predictions.jsonl"

			if not os.path.exists(gold_path):
				raise FileNotFoundError(f"File not found: '{gold_path}'")

			if not os.path.exists(pred_path):
				raise FileNotFoundError(f"File not found: '{pred_path}'")

			task_logger = ResultLogger(task)
			task_module = None
			scorer = None

			with open(gold_path, "rt", encoding="utf8") as gold_f, open(pred_path, "rt", encoding="utf8") as pred_f:
				for gold_line, pred_line in zip(gold_f, pred_f):
					gold_line = json.loads(gold_line)
					pred_line = json.loads(pred_line)

					if not task_module:
						task_module = task_id_to_prompts(gold_line["task_id"])
						import_prompts(task_module)

					if not scorer:
						scorer = get_class(gold_line["scorer_cls"])()
						scorer_config = get_scorer_config(gold_line["task_id"])

					gold_labels: AnnotationList = AnnotationList.from_output(
						str(gold_line["labels"]), task_module=task_module
					)

					# pred_labels = pred_line["model_prediction"].strip().split("result = ")[-1]
					try:
						pred_labels_idx = re.search(r'(?<!demonstrations_)result = ', pred_line['model_prediction'].strip()).end()
						pred_labels = pred_line['model_prediction'].strip()[pred_labels_idx:]
						pred_labels: AnnotationList = AnnotationList.from_output(str(pred_labels), task_module=task_module)
					except AttributeError:
						pred_labels: AnnotationList = AnnotationList([elem for elem in gold_labels])

					task_logger.add_sentence(
						sentence=gold_line["unlabelled_sentence"], gold_labels=gold_labels, pred_labels=pred_labels
					)

			task_metrics = task_logger.compute_metrics(scorer, scorer_config)
			all_scores[task] = task_metrics
			# rich.print(f"{task} scores: {task_metrics}")
			task_logger.print_predictions(output_path=os.path.join(predictions_dir, f"{task}.eval_file.json"))
			pbar.update(1)

	with open(scores_file_name, "wt", encoding="utf8") as f:
		json.dump(all_scores, f, indent=4, ensure_ascii=False)

	for task, score in all_scores.items():
		for metric, value in score.items():
			if "class_scores" in value:
				value.pop("class_scores")

	with open(scores_file_name_summary, "wt", encoding="utf8") as f:
		json.dump(all_scores, f, indent=4, ensure_ascii=False)

	logging.info(
		f"Scores saved in: {scores_file_name}. A summary without class scores saved in: {scores_file_name_summary}"
	)

	return all_scores


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

	model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))


	if data_args.test_tasks is not None:
		if not data_args.evaluate_all_checkpoints:
			evaluate(
				model_args,
				data_args,
				training_args,
				checkpoint_path=training_args.output_dir
			)
		else:
			checkpoints = [
				c
				for c in glob.glob(
					os.path.join(training_args.output_dir, "checkpoint-*"),
				)
				if os.path.isdir(c)
			]

			checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

			logging.info(
				f"Found {len(checkpoints)} checkpoints in {training_args.output_dir}:"
				f" {checkpoints} . We will evaluate each of them."
			)

			for checkpoint in checkpoints:
				logging.info(f"Evaluating {checkpoint}")
				evaluate(
					model_args,
					data_args,
					training_args,
					checkpoint_path=checkpoint,
				)
