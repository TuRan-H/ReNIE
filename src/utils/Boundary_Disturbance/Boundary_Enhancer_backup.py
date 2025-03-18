# Import packages
import json
import re
import jinja2 as jinja
import libcst as cst
import black
import random
from typing import List, Optional
from dataclasses import dataclass
from multiprocessing import process
from functools import partial
import spacy
import spacy.tokens


class AddIsPositiveAtt:
	"""
	给定一个guidelines, 将其中所有的类定义重构, 添加 `is_positive` 属性
	"""
	def __init__(self) -> None:
		class AddIsPositiveTransformer(cst.CSTTransformer):
			def leave_ClassDef(self, original_node, updated_node):
				# 创建新的属性节点
				new_field = cst.AnnAssign(
					target=cst.Name("is_positive"),
					annotation=cst.Annotation(cst.Name("bool")),
					value=None
				)

				# 创建带注释的行
				new_field_with_comment = cst.SimpleStatementLine(
					body=[new_field],
					trailing_whitespace=cst.TrailingWhitespace(
						whitespace=cst.SimpleWhitespace(" "),
						comment=cst.Comment("# Indicate if the current instance is positive; otherwise, it is negative")
					)
				)

				# 在类中插入新的属性
				return updated_node.with_changes(
					body=updated_node.body.with_changes(body=[*updated_node.body.body, new_field_with_comment])
				)

		self.transform_function = AddIsPositiveTransformer()


	def transform_class(self, code):
		"""
		给定一个类的定义, 添加 `is_positive` 属性
		"""
		module = cst.parse_module(code)
		modified_module = module.visit(self.transform_function)
		modified_code = modified_module.code

		return modified_code
	

	def __call__(self, instructions) -> str:
		# 使用re找到所有的类名
		label_classes = re.finditer(pattern=r'@dataclass\nclass (\w+)\(.*?\)', string=instructions)
		label_classes = [item.group(1) for item in label_classes]

		# 根据类名重构类的定义, 添加 `is_positive` 属性
		label_definitions = list()
		for label_class in label_classes:
			label_definition = re.search(pattern=rf'@dataclass\nclass {label_class}\(.*?\):.*?(?=@dataclass|$)', string=instructions, flags=re.DOTALL)
			label_definition = label_definition.group(0).strip('\n')
			# 重构该类的定义
			label_definition = self.transform_class(label_definition)
			label_definitions.append(label_definition)
		
		modified_instructions = str()
		for label_definition in label_definitions:
			modified_instructions += label_definition + '\n\n'

		modified_instructions = "# The following lines describe the task definition\n"+ modified_instructions

		return modified_instructions


class AddIsPositiveArg:
	"""
	给定一个annotations列表, 重构其中类实例化, 添加一个关键字参数 is_positive=True
	"""
	def __init__(self) -> None:
		class AddIsPositiveTransformer(cst.CSTTransformer):
			def leave_Call(self, original_node, updated_node):
				# 创建新的参数节点
				new_arg = cst.Arg(
					value=cst.Name(value="True"),
					keyword=cst.Name(value="is_positive")
				)

				# 检查是否已经存在is_positive参数
				if any(arg.keyword and arg.keyword.value == "is_positive" for arg in updated_node.args):
					return updated_node

				# 插入新的参数
				new_args = list(updated_node.args) + [new_arg]

				# 返回更新后的节点
				return updated_node.with_changes(args=new_args)

		self.transform_function = AddIsPositiveTransformer()

	def transform_code(self, code: str) -> str:
		"""
		给定一个代码字符串，添加 `is_positive` 参数，同时保留注释
		"""
		module = cst.parse_module(code)
		modified_module = module.visit(self.transform_function)
		modified_code = modified_module.code

		return modified_code

	def __call__(self, code: str) -> str:
		modified_code = self.transform_code(code)
		return modified_code


def create_negative_samples(tokenizer, example, guideline, model_predictions):
	"""
	Create negative samples by modifying the original model predictions.

	Args:
		tokenizer (Tokenizer): The tokenizer used to tokenize the sentences.
		example (dict): 输入的样本
			样本中主要存在以下几个key
			* unlabelled_sentence
			* text
			* labels

		original_model_predictions (str): 没有加上 is_positive 的模型输出.
		guideline (str): 可能存在的类型定义.
		model_predictions (str): 加上了is_positive, 但是没有添加负例的模型输出.

	Returns:
		str: The modified model predictions with negative instances added.
	"""
	matches = re.finditer(r'(\w+)\(span="(.+?)", is_positive = True\)', model_predictions)

	positive_instances_dict = {"label_name":list(), "span":list()}
	negative_instances_dict = {"label_name":list(), "span":list()}
	for matching in matches:
		positive_instances_dict['label_name'].append(matching.group(1))
		positive_instances_dict['span'].append(matching.group(2))

	# *** 构建negative_instances_dict
	for label_name, span in zip(positive_instances_dict['label_name'], positive_instances_dict['span']):
		unlabelled_sentence = example['unlabelled_sentence']
		unlabelled_sentence = [d for d in tokenizer(unlabelled_sentence)]
		unlabelled_sentence_text = [d.text for d in unlabelled_sentence]

		try: 
			if random.randint(0, 1) < 0.6:
				span = [d.text for d in tokenizer(span)]
				if len(span) == 1:
					if random.choice(['left', 'right']) == 'left':
						span = add_extra_token(
							span=span, 
							unlabelled_sentence=unlabelled_sentence, 
							unlabelled_sentence_text=unlabelled_sentence_text, 
							direction='left'
						)
					else:
						span = add_extra_token(
							span=span, 
							unlabelled_sentence=unlabelled_sentence, 
							unlabelled_sentence_text=unlabelled_sentence_text, 
							direction='right'
						)
				elif len(span) > 1:
					# choice==1: 随机删除一个边界token, choice==2: 随机加上一个边界token
					choice = random.choice([1, 2])
					if choice == 1:
						if random.choice(['left', 'right']) == 'left':
							span = span[1:]
						else:
							span = span[:-1]
					elif choice == 2:
						if random.choice(['left', 'right']) == 'left':
							span = add_extra_token(
								span=span, 
								unlabelled_sentence=unlabelled_sentence, 
								unlabelled_sentence_text=unlabelled_sentence_text, 
								direction='left'
							)
						else:
							span = add_extra_token(
								span=span, 
								unlabelled_sentence=unlabelled_sentence, 
								unlabelled_sentence_text=unlabelled_sentence_text, 
								direction='right'
							)
				negative_instances_dict['label_name'].append(label_name)
				negative_instances_dict['span'].append(span)
		except (ValueError, IndexError):
			pass


	# *** 对negative_instances加上一个is_positive=False
	negative_instances = list()
	negative_instances__template = "{label_name}(span=\"{span}\", is_positive=False)"
	jinja_template = jinja.Template("\nresult = [\n    {% for n in negative_instances %}\n\t{{ n }},\n    {% endfor %}\n]\n")
	for label_name, span in zip(negative_instances_dict['label_name'], negative_instances_dict['span']):
		span = " ".join(span).strip(" ")
		negative_instances.append(negative_instances__template.format(label_name=label_name, span=span))
	negative_instances = jinja_template.render(negative_instances=negative_instances)

	# *** 合并positive_instances和negative_instances
	from src.tasks.utils_typing import Entity
	local_vars = {"Entity":Entity}
	exec(guideline, globals(), local_vars)
	exec(model_predictions, globals(), local_vars)
	instances:list = local_vars.get('result', None)
	exec(negative_instances, globals(), local_vars)
	negative_instances = local_vars.get('result', None)
	instances.extend(negative_instances)
	random.shuffle(instances)
	
	# 使用jinja2构造prompt
	template = jinja.Template('# The annotation instances that take place in the text above are listed here\nresult = [\n{%- for ann in annotations %}\n    {{ ann }},\n{%- endfor %}\n]\n')
	model_predictions = template.render(annotations=instances)

	return model_predictions


def add_extra_token(
	span: list[str], 
	unlabelled_sentence: list[spacy.tokens.token.Token], 
	unlabelled_sentence_text: list[str], 
	direction: str
):
	"""
	Adds an extra token to the given span in the specified direction.

	Args:
		span (list[str]): The span of tokens to which the extra token will be added.
		unlabelled_sentence (list[spacy.tokens.token.Token]): The list of tokens in the unlabelled sentence.
		unlabelled_sentence_text (list[str]): The list of token texts in the unlabelled sentence.
		direction (str): The direction in which the extra token will be added. Can be either 'left' or 'right'.

	Returns:
		list[str]: The updated span with the extra token added.
	"""
	if direction == 'left':
		extra_token_index = unlabelled_sentence_text.index(span[0])-1
		extra_token = unlabelled_sentence[extra_token_index]
		while not extra_token.is_alpha:
			if "\"" in extra_token.text:
				extra_token_index -= 1
				extra_token = unlabelled_sentence[extra_token_index]
				continue
			span.insert(0, extra_token.text)
			extra_token_index -= 1
			extra_token = unlabelled_sentence[extra_token_index]
		span.insert(0, extra_token.text)
	elif direction == 'right':
		extra_token_index = unlabelled_sentence_text.index(span[-1])+1
		extra_token = unlabelled_sentence[extra_token_index]
		while not extra_token.is_alpha:
			if "\"" in extra_token.text:
				extra_token_index += 1
				extra_token = unlabelled_sentence[extra_token_index]
				continue
			span.insert(len(span), extra_token.text)
			extra_token_index += 1
			extra_token = unlabelled_sentence[extra_token_index]
		span.insert(len(span), extra_token.text)
	
	return span


def boundary_enhancer(example, **kwargs):
	"""
	提升样本边界的策略
	在构造数据集的时候, 添加一个 `is_positive` 属性, 用于指示当前实例是否为正例
	sample出一些负例, 将模型的生成任务构造为对比学习任务

	Args:
		example (dict): The example data containing the text, instructions, and results.
		kwargs (dict)
			* add_negative_instances (bool): 是否在results部分添加负样本	
		
	Returns:
		dict: The updated example data.
	"""
	add_negative_instances = kwargs.get('add_negative_instances', False)
	if add_negative_instances:
		tokenizer = kwargs.get("tokenizer", None)

	text:str = example['text']

	guideline_start = 0
	demonstration_start = text.find("# This is the demonstrations\ntext = ")
	input_text_start = text.find("# This is the text to analyze\ntext = ")
	model_predictions_start = text.find("# The annotation instances that take place in the text above are listed here\nresult = ")

	# *** 对guideline和demonstration部分添加 `is_positive` 属性
	if demonstration_start != -1:
		guideline = text[guideline_start:demonstration_start]
		guideline = AddIsPositiveAtt()(guideline)
		demonstration = text[demonstration_start:input_text_start]
		demonstration = AddIsPositiveArg()(demonstration)
	else:
		guideline = text[guideline_start:input_text_start]
		guideline = AddIsPositiveAtt()(guideline)

	input_text = text[input_text_start:model_predictions_start]

	# *** 对model_predictions部分添加 `is_positive` 参数
	model_predictions = text[model_predictions_start:]
	model_predictions = AddIsPositiveArg()(model_predictions)
	
	# *** sample负例, 并添加到model_predictions部分中
	if add_negative_instances:
		if tokenizer == None:
			raise("You want to add negative instances in model predictions, please provide a tokenizer.")

		model_predictions = create_negative_samples(
			tokenizer=tokenizer, 
			example=example,
			guideline=guideline,
			model_predictions=model_predictions
		)

	# *** 使用black对代码进行format
	try:
		text = guideline+demonstration+input_text+model_predictions
	except UnboundLocalError:
		text = guideline+input_text+model_predictions

	example['text'] = black.format_str(text, mode=black.Mode())

	return example



if __name__ == "__main__":
	# 读取数据集, 获取数据集的不同部分
	with open('./data/processed_w_examples/wnut17.ner.train.42.jsonl') as fp:
		examples = fp.readlines()
		examples = [json.loads(line) for line in examples]

	index = random.randint(0, len(examples))
	example = boundary_enhancer(examples[index])

	print(example['text'])