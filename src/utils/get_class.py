from typing import Type

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