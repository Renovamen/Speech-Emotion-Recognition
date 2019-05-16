# 数据集整理

import os, shutil

DATA_PATH = "/Users/zou/Desktop/CASIA"

# 批量删除指定路径下所有非.wav文件
def remove(file_path):
	for root, dirs, files in os.walk(file_path):
		for item in files:
			if not item.endswith('.wav'):
				try:
					print("Delete file: ", os.path.join(root, item))
					os.remove(os.path.join(root, item))
				except:
					continue

# 批量按指定格式改名（不然把相同情感的音频整理到同一个文件夹时会重名）
def rename(file_path):

	for root, dirs, files in os.walk(file_path):
		for item in files:
			if item.endswith('.wav'):
				people_name = root.split('/')[-2]
				emotion_name = root.split('/')[-1]
				item_name = item[:-4] # 音频原名（去掉.wav）
				old_path = os.path.join(root, item)
				new_path = os.path.join(root, item_name + '-' + emotion_name + '-'+ people_name + '.wav') # 新音频地址
				try:
					os.rename(old_path, new_path)
					print('converting ', old_path, ' to ', new_path)
				except:
					continue


# 把音频按情感分类，放在不同文件夹下
def move(file_path):
	for root, dirs, files in os.walk(file_path):
		for item in files:
			if item.endswith('.wav'):
				emotion_name = root.split('/')[-1]
				old_path = os.path.join(root, item)
				new_path = file_path + '/' + emotion_name + '/' + item
				try:
					shutil.move(old_path, new_path)
					print("Move ", old_path, " to ", new_path)
				except:
					continue