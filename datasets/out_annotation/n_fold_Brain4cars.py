import random
from glob import glob
from os.path import *
import os
import csv

# 1. 将一个列表分成指定数量的子列表。每个子列表的长度大致相等，除非列表的长度不能被 n_fold 整除。常用于交叉验证或数据集拆分的操作。
# 作用是将一个列表 full_list 分成 n_fold 份，并返回一个包含这些子列表的列表。
def N_fold(full_list, n_fold):
	# 1. 计算每一份的平均长度。它将列表 full_list 的长度除以 n_fold，并将结果赋值给变量 avg。
    avg = len(full_list) / float(n_fold)
	# 2. 创建了两个空变量：out 和 last。out 是用来存储最终结果的列表，而 last 用来追踪当前子列表的结束索引。
    out = []
    last = 0.0
	# 3. 循环，用于生成子列表。迭代直到 last 大于或等于 full_list 的长度。
	# 每次循环中，使用切片操作 full_list[int(last):int(last + avg)] 来获取当前子列表，并将其添加到 out 列表中。
	# 然后，last 值增加 avg，指向下一个子列表的起始位置。
    while last < len(full_list):
        out.append(full_list[int(last):int(last + avg)])
        last += avg
    return out # 函数返回包含所有子列表的列表 out。


# 2. 这段代码的目的是获取指定目录 data_file_path 下具有两级子目录的所有文件路径，并按照字典顺序进行排序。
data_file_path = r'D:\XHX\Driver-Intention-Prediction-master\Brain4Cars\flownet2_road_camera'
# 1. join 函数用于将目录路径和模式 */*/ 进行连接，形成一个新的路径。这个模式表示所有具有两级子目录的文件路径。
# 2. 然后，glob 函数将这个路径传递给，它会返回与该模式匹配的所有文件路径的列表。
# 3. 最后，sorted 函数对这个列表进行排序，并将结果赋值给变量 file_list。
file_list = sorted(glob(join(data_file_path, '*/*/')))
# file_list includes:"'./flownet2_face_camera/rturn/20141220_154451_747_897/',
# './flownet2_face_camera/rturn/20141220_161342_1556_1706/', "


# 3. 这部分代码使用了随机打乱列表、打印字符串以及调用了之前定义的 N_fold 函数。
random.shuffle(file_list) # 随机打乱。改变原始列表的顺序，使得其中的元素被重新排列。
print(join(data_file_path, '*\\*\\')) # 双反斜杠是为了在 Windows 系统下表示目录结构
n_fold_list = N_fold(file_list, 5) # 将随机打乱后的 file_list 列表作为输入，并将其分成 5 份
# print('我的调试：n_fold_list列表为：', n_fold_list)

# 4. 这部分代码根据交叉验证的设置，将数据从不同的训练集和验证集折中提取出来，并将文件路径、标签、帧数和数据集子集写入到不同的 CSV 文件中。
# 这段代码使用两个嵌套的循环，将数据从 n_fold_list 中的每个折中提取出来，并生成用于创建CSV文件的四个列表：
# file_location（文件路径），label（标签），n_frames（帧数）和subset（数据集子集）。
# 外层循环使用变量 i 迭代 0 到 4，表示折的索引。
for i in range (0,5):
	label = []
	subset = []
	n_frames = []
	file_location = []
	# 内层循环使用变量 j 迭代 0 到 4，表示折内的文件列表。
	for j in range(0,5):
		# 根据当前的折索引 i，将要处理的折设置为验证集（'validation'），其他折设置为训练集（'training'）。
		this_subset = 'training'
		if j == i:
			this_subset = 'validation'
		# 然后，遍历当前折中的文件列表 n_fold_list[j]，依次处理每个文件。
		for file in n_fold_list[j]:
			# 首先，使用字符串操作和分割函数提取出文件的相对路径 fbase、目标标签 ftarget 和文件夹中的帧数 fnum。
			fbase = file[len(data_file_path):]
			# print('我的调试：fbase为：', fbase)
			ftarget_idx = fbase.find('/',1)
			ftarget = fbase[1:ftarget_idx].split('\\')[0] # 已修改： 原代码：ftarget = fbase[1:ftarget_idx]
			# print('我的调试：ftarget为：', ftarget)
			fnum = len(os.listdir(file)) # 已修改： 原代码：fnum = len(os.listdir(file)) - 2, 没有-2是因为车内的要减去
			# 然后，将这些信息添加到相应的列表中。
			file_location.append(fbase[1:])
			n_frames.append(fnum)
			label.append(ftarget)
			subset.append(this_subset)

	# 这段代码打开一个名为 'fold%d.csv' % i 的文件
	with open('fold%d.csv'%i, 'a') as outcsv:
		# 使用 CSV 写入器 csv.writer 将数据写入文件中，CSV 文件的每一行包含了文件路径、标签、帧数和数据集子集。
		writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		# 通过使用 zip 函数，将四个列表中的元素逐个配对，并将每个配对作为一行写入 CSV 文件。
		for (w,x,y,z) in zip (file_location,label,n_frames,subset):
			writer.writerow([w,x,y,z])