"""为FewRel数据集生成prompts.py文件和guidelines.py文件"""

import json
from json import JSONDecodeError
import jinja2
import os
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm



JINJA_TEMPLATE = """from ..utils_typing import Relation, dataclass

{% for item in class_mapping %}
@dataclass
class {{ item[1] }}(Relation):
	\"\"\"{{ '{' }}{{ item[1] }}{{ '}' }}\"\"\"
	arg1: str
	arg2: str

{% endfor %}

COARSE_RELATION_DEFINITIONS: list = [
{%- for relation in class_mapping %}
	{{ relation[1] }},
{%- endfor %}
]

RELATION_TO_CLASS_MAPPING: dict = {
{%- for relation in class_mapping %}
	"{{ relation[3] }}": {{ relation[1] }},
{%- endfor %}
}
"""


SYSTEM_CONTENT = """I will provide you with a relationship that exists in Wikidata, along with a brief description of this relationship. Please generate two detailed description for this relationship and output them in a list.

----Output format----
["description1", "description2"]

----For example----
Your input: "relation: mountain range, brief description: range or subrange to which the geographical item belongs."
your ourput: "["A mountain range provides a broader context to a particular geographic feature, signifying its place within a vast collection of mountains that have formed through similar geological processes. This relationship helps to categorize and link mountains, subranges, or hills by their proximity and geological history. It underscores the collective nature of mountainous regions, where individual features are part of a larger, often interrelated, topographical structure.", "The mountain range relationship highlights the larger geographic framework within which individual mountains or subranges exist. When a geographical item is classified under a specific mountain range, it denotes that this feature shares similar topographical, geological, and climatic characteristics with other elements within the range."]"
"""


def snake_to_camel(snake_str):
	"""
	将蛇形命名转换为驼峰命名
	"""
	components = snake_str.split('_')
	return components[0].title() + ''.join(x.title() for x in components[1:])


def load_all_relations():
	"""
	导入所有在数据集中存在过的关系
	"""
	relation_set = set()
	dataset = load_dataset("./download/few_rel")

	train_dataset = dataset["train"]
	val_dataset = dataset["validation"]

	for item in train_dataset:
		relation_set.add(item["relation"]) # type: ignore
	for item in val_dataset:
		relation_set.add(item["relation"]) # type: ignore
	
	return relation_set


def get_class_mapping(relation_set):
	"""
	生成class_mapping
	
	Args:
		relation_set: set
	
	Returns:
		List, 列表中的每个元素: [pid, relation_name, relation_description, relation_name_original]
	"""
	with open("./src/tasks/fewrel/pid2name.json", 'r') as f:
		pid2name = json.load(f)

	class_mapping = []
	for pid, (relation_name, relation_description) in pid2name.items():
		relation_name_original = relation_name
		if pid in relation_set:
			if "/" in relation_name:
				relation_name = relation_name.replace("/", "or")
			if " " in relation_name:
				relation_name = relation_name.replace(" ", "_")
				relation_name = snake_to_camel(relation_name)
			else:
				relation_name = relation_name.title()
			class_mapping.append([pid, relation_name, relation_description, relation_name_original])

	return class_mapping


def generate_prompts():
	"""
	生成prompts.py文件
	"""
	relation_set = load_all_relations()
	class_mapping = get_class_mapping(relation_set)

	template = jinja2.Template(JINJA_TEMPLATE)
	render_template = template.render(class_mapping=class_mapping)

	print(render_template)

	with open("./src/tasks/fewrel/prompts.py", 'w') as f:
		f.write(render_template)


# 将这个函数改为异步IO
def generate_guidelines(USE_LLM: bool = False, is_gold: bool = False):
	"""
	生成guidelines.py文件

	Args:
		USE_LLM: bool, 是否使用LLM生成额外的guideline
		is_gold: bool, 是否是gold_guideline

	Note: bast practice -- USE_LLM=True, is_gold=False; USE_LLM=False, is_gold=True
	"""
	if is_gold:
		file_path = "./src/tasks/fewrel/guidelines_gold.py"
	else:
		file_path = "./src/tasks/fewrel/guidelines.py"

	GUIDELINES = dict()
	relation_set = load_all_relations()
	class_mapping = get_class_mapping(relation_set)

	bar = tqdm(total=len(class_mapping))
	for _, relation_name, relation_description, relation_name_original in class_mapping:
		GUIDELINES[relation_name] = {
			'en': list()
		}
		GUIDELINES[relation_name]['en'].append(relation_description)
		# 如果使用LLM, 获取额外的guideline
		if USE_LLM:
			GUIDELINES[relation_name]['en'].extend(
				get_extra_guideline(
					relation=relation_name_original, 
					brief_description=relation_description
				)
			)
		bar.update(1)
	
	with open(file_path, 'w') as f:
		json.dump(GUIDELINES, f, indent=4)

	# 向文件的开头加上 "GUIDELINES = "
	with open(file_path, 'r') as f:
		original_content = f.read()
	with open(file_path, 'w') as f:
		f.write("GUIDELINES = ")
		f.write(original_content)


def get_extra_guideline(
	relation: str,
	brief_description: str,
	model: str = "qwen-turbo"
) -> list:
	client = OpenAI(
		api_key=os.getenv("DASHSCOPE_API_KEY"), 
		base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
	)
	user_content = f"relation: {relation}, brief description: {brief_description}."
	completion = client.chat.completions.create(
		model=model,
		messages=[
			{'role': 'system', 'content': SYSTEM_CONTENT},
			{'role': 'user', 'content': user_content}],
		)
	
	model_response = completion.choices[0].message.content
	try:
		extra_guideline = json.loads(model_response)
	except JSONDecodeError:
		extra_guideline = []
	
	return extra_guideline

def swap_first_last_guideline(reversed: bool = False):
	"""
	调换GUIDELINES中每个relation的guideline的第一个和最后一个
	"""
	import random
	from src.tasks.fewrel.guidelines import GUIDELINES
	def swap(lst):
		index = random.randint(1, len(lst)-1)
		lst[0], lst[index] = lst[index], lst[0]
	
	for _, guideline in GUIDELINES.items():
		if len(guideline['en']) > 1 and len(guideline['en'][0]) < len(guideline['en'][-1]):
			swap(guideline['en'])

	GUIDELINES = json.dumps(GUIDELINES, indent=4)
	GUIDELINES = "GUIDELINES = " + GUIDELINES
	with open("./src/tasks/fewrel/guidelines_LLM.py", 'w') as f:
		f.write(GUIDELINES)

		


if __name__ == '__main__':
	# generate_prompts()
	# generate_guidelines(USE_LLM=False, is_gold=True)
	# generate_guidelines(USE_LLM=True, is_gold=False)
	swap_first_last_guideline()