"""
使用wordnet进行同义词替换
"""
class SynonymsSubstitute:
	"""
	同义词替换类
	输入一个句子, 调用 `SynonymsSubstitute`的 `__call__` 方法, 即可替换句子中所有词到其同义词
	"""
	def __init__(self):
		from nltk.corpus import wordnet
		self.wordnet = wordnet
		from nltk import word_tokenize as tokenizer
		self.tokenizer = tokenizer
		from nltk.tokenize.treebank import TreebankWordDetokenizer
		self.detokenizer = TreebankWordDetokenizer()

	def __call__(self, sentence):
		"""
		使用nltk的WordNet获取单词的同义词, 并返回被替换过的字符串
		"""
		# 生成一个列表, 用来记录那些token是单词, 那些token是标点符号
		tokens = self.tokenizer(sentence)
		new_tokens = list()

		for token in tokens:
			try:
				new_tokens.append(self.get_synoyms(token)[0])
			except:
				new_tokens.append(token)
		if new_tokens == tokens:
			new_tokens = list()
				
		return self.detokenizer.detokenize(new_tokens)


	def get_synoyms(self, word):
		synonyms = set()
		for syn in self.wordnet.synsets(word):
			for lemma in syn.lemmas():
				if lemma.name().lower != word.lower():
					synonyms.add(lemma.name().replace('_', ''))
		
		return list(synonyms)



if __name__ == "__main__":
	from nltk.corpus import wordnet

	word = "laptop"


	synonyms = set()
	for syn in wordnet.synsets(word):
		for lemma in syn.lemmas():
			if lemma.name().lower != word.lower():
				synonyms.add(lemma.name().replace('_', ''))
	
	print(list(synonyms))
	# result = random.choice(list(synonyms))
	result = list(synonyms)[0]
	if word.lower() == result.lower():
		result = ""

	print(result)