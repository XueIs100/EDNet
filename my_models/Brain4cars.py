import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from os.path import *
import numpy as np
import random
from glob import glob
import csv
from utils import load_value_file

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#用于设置 PIL（Python Imaging Library）中的 ImageFile 模块对于截断图像的处理方式。
#在加载图像文件时，一些损坏或者格式不正确的图像文件可能会被截断。如果设置为 True，PIL会自动尝试加载该图像文件，并尽可能地展示出可用的部分。
#如果需要严格保证图像完整性，可以将该选项设为 False


# 1. 这个函数的作用是读取指定路径的图像文件并将其转换为RGB格式的图像对象。
# 在函数内部，使用open函数以二进制模式打开路径对应的文件，并使用Image.open函数打开文件作为图像对象img。
# 最后，将图像对象转换为RGB格式，并将其作为函数的返回值。
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# 2. 这个函数的作用是尝试使用accimage库加载指定路径的图像文件，如果导入accimage库失败，则使用PIL库进行加载。
# 在函数内部，首先尝试导入accimage库。若导入成功，则使用accimage.Image函数加载指定路径的图像文件，并将其作为函数的返回值。
# 如果导入accimage库失败，可能是由于解码问题，此时通过捕获IOError异常来处理，并调用pil_loader函数进行加载。
def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


# 3. 这个函数的作用是根据系统配置和可用的图像加载后端，返回相应的默认图像加载器函数。
# 该函数首先导入torchvision库的get_image_backend函数。
# 然后，通过调用get_image_backend函数获取当前系统的图像加载后端。
# 如果返回值是accimage，则说明当前使用的是accimage库，所以返回accimage_loader函数作为默认的图像加载器。
# 如果返回值不是accimage，则说明当前使用的不是accimage库，所以返回pil_loader函数作为默认的图像加载器。
def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


# 4. 从指定目录中加载视频的特定帧图像，根据给定的帧索引列表，使用指定的图像加载器函数加载每个帧的图像，并返回加载的视频帧图像列表。
# 函数接收以下参数：
# video_dir_path：视频帧所在的目录路径
# frame_indices：需要加载的视频帧的索引列表
# image_loader：图像加载器函数
def video_loader(video_dir_path, frame_indices, image_loader):
    # 首先创建一个空列表video，用于存储加载的视频帧图像。
    video = []
    # 使用循环遍历frame_indices中的每个索引i。
    for i in frame_indices:
        # 在每次迭代中，构造相应索引的图像文件路径image_path，通过os.path.join函数拼接视频帧目录路径和图像文件名。
        image_path = os.path.join(video_dir_path, 'image-{:04d}.png'.format(i))
        # 检查该图像文件路径是否存在，如果存在则调用image_loader函数加载图像，并将加载的图像添加到video列表中。
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        # 如果图像文件路径不存在，则打印出该路径，并返回已加载的视频帧图像列表video。
        else:
            print('图像路径不存在', image_path)
            return video
    return video    # 返回加载的视频帧图像列表video。

def out_video_loader(video_dir_path, frame_indices, image_loader):
    # 首先创建一个空列表video，用于存储加载的视频帧图像。
    video = []
    # 使用循环遍历frame_indices中的每个索引i。
    for i in frame_indices:
        # 修改索引i,使其从0开始，即索引值减去1
        i -= 1
        # 在每次迭代中，构造相应索引的图像文件路径image_path，通过os.path.join函数拼接视频帧目录路径和图像文件名。
        image_path = os.path.join(video_dir_path, '{:06d}.png'.format(i))
        # 检查该图像文件路径是否存在，如果存在则调用image_loader函数加载图像，并将加载的图像添加到video列表中。
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        # 如果图像文件路径不存在，则打印出该路径，并返回已加载的视频帧图像列表video。
        else:
            print('图像路径不存在', image_path)
            return video
    return video    # 返回加载的视频帧图像列表video。


# 5. 通过video_loader函数加载指定路径中特定帧的图像，返回加载的所有视频帧图像。
# get_default_image_loader函数获取默认的图像加载器函数，并将其赋值给变量image_loader。
# functools.partial函数创建一个新的函数video_loader_with_default_image_loader：
#   该函数将图像加载器作为image_loader参数传递给video_loader函数；
#   具体地，functools.partial函数会固定video_loader函数的第三个参数image_loader为image_loader变量的值
#   形成一个新的函数video_loader_with_default_image_loader。
def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def out_get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(out_video_loader, image_loader=image_loader)


# 6. 从指定文件路径中加载注释数据，根据给定的折数，读取对应的CSV文件，并将每一行的注释数据保存到字典中，返回加载的注释数据字典。
# 函数接收以下参数：
#   `data_file_path`：数据文件所在的目录路径
#   `fold`：折数
def load_annotation_data(data_file_path, fold):
    # 首先创建一个空字典`database`，用于存储加载的注释数据。
    database = {}
    # 将数据文件路径和折数拼接成完整的文件路径。
    data_file_path = os.path.join(data_file_path, 'fold%d.csv'%fold)
    print('Load from %s'%data_file_path) # 打印出正在加载的数据文件路径。
    # 使用`open`函数打开数据文件，使用`csv.reader`函数读取CSV文件，并指定以逗号作为分隔符。
    with open(data_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # 通过迭代`csv_reader`获取每一行的数据。
        for row in csv_reader:
            # 在每次迭代中，创建一个空字典`value`，用于存储每一行的注释数据。
            # 将第4列的值赋给字典`value`的键`'subset'`，将第2列的值赋给键`'label'`，将第3列的值转换为整数并赋给键`'n_frames'`。
            value = {}
            value['subset'] = row[3]
            value['label'] = row[1]
            value['n_frames'] = int(row[2])
            # 将每一行的注释数据以视频ID（第1列的值）为键，注释数据字典`value`为值，添加到`database`字典中。
            database[row[0]] = value
    return database # 最后，返回加载的注释数据字典`database`。


# 7. 获取类别标签映射字典，将每个类别标签字符串映射为对应的整数值。
def get_class_labels():
#### define the labels map
    # 创建一个空字典`class_labels_map`，用于存储类别标签映射关系。
    class_labels_map = {}
    class_labels_map['end_action'] = 0
    class_labels_map['lchange'] = 1
    class_labels_map['lturn'] = 2
    class_labels_map['rchange'] = 3
    class_labels_map['rturn'] = 4
    return class_labels_map   # 返回类别标签映射字典`class_labels_map`。


# 8. 根据给定的注释数据字典和数据子集名称，获取属于目标子集的视频名称列表以及对应的注释列表。
# 函数接收以下参数：
#   `data`：注释数据字典
#   `subset`：数据子集名称
def get_video_names_and_annotations(data, subset):
    # 首先创建两个空列表，用于存储视频名称和对应的注释。
    video_names = []
    annotations = []
    # 通过迭代`data.items()`获取注释数据字典中的每一个键值对。
    for key, value in data.items():
        # 在每次迭代中，将当前值字典的`'subset'`键的值赋给变量`this_subset`。
        this_subset = value['subset']
        # 如果`this_subset`与输入的数据子集名称`subset`相等，则表示该视频属于目标子集。
        if this_subset == subset:
            # 获取当前值字典的`'label'`键的值，并赋给变量`label`。
            label = value['label']
            # 将当前键（即视频名称）添加到`video_names`列表中。
            video_names.append(key)    ### key = 'rturn/20141220_154451_747_897'
            # 将当前值字典作为一个整体添加到`annotations`列表中。
            annotations.append(value)
    return video_names, annotations


# 9. 根据给定的视频文件根目录、注释文件路径和其他参数，创建数据集。
# 函数接收以下参数：
#   `root_path`：视频文件根目录
#   `annotation_path`：注释文件路径
#   `subset`：数据子集名称
#   `n_samples_for_each_video`：每个视频采样的样本数
#   `end_second`：每个样本的结束时间（以秒为单位）
#   `sample_duration`：每个样本的时长
#   `fold`：折叠次数
def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, end_second, sample_duration, fold):
    # 1. 首先加载注释数据，并赋值给变量`data`。
    data = load_annotation_data(annotation_path, fold)

    # 2. 获取视频名称列表和注释列表，并分别赋值给`video_names`和`annotations`。
    video_names, annotations = get_video_names_and_annotations(data, subset)
    video_names = [name.replace('\\', '/') for name in video_names]
    # print('我的调试：video_names列表为：', video_names)
    # print('我的调试：video_names列表的长度为：', len(video_names))
    for annotation in annotations:
        annotation['label'] = annotation['label'].replace('\\', '/')
    # print('我的调试：annotations注释列表为：', annotations)
    # 3. 调用`get_class_labels`函数获取类别标签映射字典，并将其赋值给`class_to_idx`。
    class_to_idx = get_class_labels()
    # print('我的调试：class_to_idx字典为：', class_to_idx)
    # 4. 创建一个空字典`idx_to_class`来存储整数到类别标签的反向映射关系。
    idx_to_class = {}
    # 5. 通过迭代`class_to_idx.items()`，将每个键值对的键和值互换，然后添加到`idx_to_class`字典中。
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    # 6. 创建一个空列表`dataset`，用于存储数据集
    dataset = []
    # 7. 通过循环迭代`range(len(video_names))`，遍历视频名称列表中的每个视频。
    for i in range(len(video_names)):
        # 8. 在每次迭代中，首先判断是否满足条件`i % 100 == 0`，如果满足，则打印加载数据的进度。
        if i % 100 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        # 9. 根据视频路径拼接视频文件的完整路径，赋值给变量`video_path`。
        video_path = os.path.join(root_path, video_names[i])
        # print("我的调试：root_path为：", root_path)

        # 10. 如果该视频文件不存在，则打印提示信息并继续下一次迭代。
        if not os.path.exists(video_path):
            print('File does not exists: %s'%video_path)
            continue

        # 11. 计算当前视频的帧数，并赋值给变量`n_frames`。
        # 通过列举视频文件夹中的文件数量来实现，其中减去2是为了排除文件夹中可能存在的其他文件（如原始视频文件）。
        l = os.listdir(video_path)
        # If there are other files (e.g. original videos) besides the images in the folder, please abstract.
        # n_frames = len(l)-2 # 针对文件夹内有若干个帧图片和两个其他文件时
        n_frames = len(l) # 针对文件夹内全部是帧图片
        print('我的调试：video_path文件夹内数量：', video_path, n_frames)
        # print('我的调试：n_frames帧图像文件数量为：', n_frames)

        # 如果帧数小于阈值（16 + 25*(end_second-1)），则打印提示信息并继续下一次迭代。
        if n_frames < 16 + 25*(end_second-1):  # 原版的代码
        # if n_frames < 1:
            print('Video is too short: %s'%video_path)
            continue

        # 12. 设置采样的开始时间和结束时间，即`begin_t`和`end_t`，分别为 1 和视频的总帧数。
        begin_t = 1
        end_t = n_frames
        # 13. 创建一个字典`sample`，包含以下键值对：
        #   `'video'`：视频路径
        #   `'segment'`：视频段的起始和结束时间
        #   `'n_frames'`：视频的总帧数
        #   `'video_id'`：从视频名称中提取的视频ID
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        # print('我的调试：sample采样：', sample)

        # 14. 检查注释列表是否为空。
        # 如果不为空，则将当前视频的类别标签映射为对应的整数值，并赋值给`sample['label']`；否则，将`sample['label']`设为-1。
        if len(annotations) != 0:
             sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        # 15. 根据每个视频样本的数量，判断是按照一个还是多个样本来处理。
        # 如果`n_samples_for_each_video`为1，则将`sample['frame_indices']`设置为从1到`n_frames`的列表，
        # 并将`sample`添加到`dataset`列表中；
        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        # 否则，则进行多次迭代，在每次迭代中将`sample['frame_indices']`设置为从1到`n_frames`的列表，
        # 并创建一个`sample_j`变量作为深拷贝的`sample`字典，将`sample_j`添加到`dataset`列表中。
        else:
            if n_samples_for_each_video > 1:
                for j in range(0, n_samples_for_each_video):
                    sample['frame_indices'] = list(range(1, n_frames+1))
                    sample_j = copy.deepcopy(sample)
                    dataset.append(sample_j)
    return dataset, idx_to_class         # 返回数据集列表`dataset`和整数到类别标签的反向映射字典`idx_to_class`。


# 10. 这段代码定义了一个名为`Brain4cars_Inside`的数据集类，用于处理视频数据集。
    # 这个类可以实现对视频数据集进行加载和预处理，并支持索引访问和长度获取功能
class Brain4cars_Inside(data.Dataset):
    # 1. `__init__()`接受：数据集路径、注释文件路径、子集、折数、结束秒数等。使用`make_dataset()`函数生成数据集和类别名称。
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold, 
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)
        # print('我的调试：self.data为：', self.data)
        # 在初始化过程中，设置了空间变换函数、水平翻转函数、时序变换函数和目标变换函数等参数。
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    # 2. `__getitem__(self, index)`方法用于获取数据集中指定索引位置的样本。
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # 1. 首先，根据索引获取视频的路径和帧索引信息。
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        # if isinstance(frame_indices, list):  # 检查是否为列表
        #     print("00000000000000000000000000000是列表00000000000000000000000000000")
        # else:
        #     print("0000000000000000000")
        h_flip = False
        # 2. 接下来，如果存在时序变换函数，则调用该函数对帧索引进行变换。
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        # 3. 然后，使用视频加载器加载指定路径和帧索引的帧序列。
        clip = self.loader(path, frame_indices)
        # 4. 如果存在水平翻转函数，则以一定概率对帧序列进行水平翻转。
        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                h_flip = True
                clip = [self.horizontal_flip(img) for img in clip]
        # 5. 如果存在空间变换函数，则调用其`randomize_parameters`方法随机设置参数，并将其应用于帧序列中的每一帧。
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        # 6. 最后，将帧序列转换为张量形式，并将通道维度调整到第二个维度上。
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        # 7. 同时，获取对应样本的目标标签。
        target = self.data[index]
        # 8. 如果存在目标变换函数，则对目标标签进行变换。
        if self.target_transform is not None:
            target = self.target_transform(target)
        # 9. 在最后的部分，如果进行了水平翻转并且目标标签不为0（表示正常情况），则根据目标标签的值进行相应的调整。
        if (h_flip == True) and (target != 0):
            if target == 1:
                target = 3
            elif target == 3:
                target = 1
            elif target == 2:
                target = 4
            elif target == 4:
                target = 2
        return clip, target  # 最终，返回处理后的帧序列和目标标签作为元组。
    # 2. `__len__(self)`方法用于获取数据集的长度，即样本数量。它直接返回数据集的长度。
    def __len__(self):
        return len(self.data)


# 11. 这个类可以实现对视频数据集进行加载和预处理，并支持索引访问和长度获取功能
class Brain4cars_Outside(data.Dataset):
    # 1. 构造函数__init__()接受:数据集路径、注释文件路径、子集、折数、结束秒数等。它使用make_dataset()函数生成数据集和类别名称。
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold,
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=5,
                 get_loader=out_get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    # 2. __getitem__(self, index)方法用于获取数据集中指定索引位置的样本。
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is an image.
        """
        # 1. 首先，根据索引获取视频的路径和帧索引信息。
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        h_flip = False
        # 2. 接下来，如果存在时序变换函数，则调用该函数对帧索引进行变换。
        if self.temporal_transform is not None:
            frame_indices,target_idc = self.temporal_transform(frame_indices)
        # 3. 然后，使用视频加载器加载指定路径和帧索引的帧序列。
        clip = self.loader(path, frame_indices)
        target = self.loader(path, target_idc)
        # 4. 如果存在水平翻转函数，则以一定概率对帧序列进行水平翻转。
        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                clip = [self.horizontal_flip(img) for img in clip]
                target = [self.horizontal_flip(img) for img in target]
        # 5. 如果存在空间变换函数，则调用其randomize_parameters方法随机设置参数，并将其应用于帧序列中的每一帧。
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0)
        # 6. 最后，将帧序列转换为张量形式，并将通道维度调整到第二个维度上。
        # 同时，获取对应样本的目标标签。如果存在目标变换函数，则对目标标签进行变换。
        # 在最后的部分，如果进行了水平翻转并且目标标签不为0（表示正常情况），则根据目标标签的值进行相应的调整。
        if self.target_transform is not None:
            target = [self.target_transform(img) for img in target]
        target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
        return clip, target         # 最终，返回处理后的帧序列和目标标签作为元组。

    # 3. __len__(self)方法用于获取数据集的长度，即样本数量。它直接返回数据集的长度。
    def __len__(self):
        return len(self.data)