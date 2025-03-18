def find_sublist_index(main_list, sub_list):
	"""
	找到子表 (sub_list) 在主表 (main_list) 中的位置
	如果没有找到, 则返回 -1
	"""
	sub_len = len(sub_list)
	main_len = len(main_list)

	if sub_len == 0 or main_len < sub_len:
		return -1
	
	for i in range(main_len - sub_len + 1):
		if main_list[i:i+sub_len] == sub_list:
			return list(range(i, i+sub_len))
	
	return -1