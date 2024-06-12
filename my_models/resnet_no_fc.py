'''
注释掉fc层
'''

# 1. 公共库函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

# 2. 只有在 __all__ 列表中列出的成员才会被导入
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


# 3. 创建 3x3x3 的卷积层，用于构建三维卷积神经网络模型。3x3x3 convolution with padding
def conv3x3x3(in_planes, out_planes, stride=1):
    # 使用PyTorch中的 `nn.Conv3d` 类创建一个三维卷积层：
    # 1. 指定输入通道数、输出通道数、卷积核大小为 3、步幅和填充等参数，返回了一个经过定义的卷积层。
    # 2. 通过设置 `bias=False` 来禁用卷积层的偏置项。
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


# 4. # 实现了基本块的下采样操作，常用于 ResNet 等深度学习模型中。它通过平均池化和通道维度的填充，实现了在时间维度进行下采样的功能。
# 函数的输入参数包括：
#     `x`：输入张量
#     `planes`：输出通道数
#     `stride`：下采样的步幅
def downsample_basic_block(x, planes, stride):
    # 1. 使用 PyTorch 中 `F.avg_pool3d` 函数对输入张量进行 3D 平均池化操作，其中使用了 1x1x1 的池化核和指定的步幅。
    # 这一步的作用是对输入张量在时间维度上进行下采样，以减少时间分辨率。
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    # 2. 创建了一个全零张量 `zero_pads`，用于对池化结果进行通道维度上的扩充。
    # 通过计算输出通道数与池化结果通道数的差异，将 `zero_pads` 扩充到相应的形状。
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    # 3. 根据当前的运行环境（CPU 或 GPU），检查并确保 `zero_pads` 与 `out` 张量在同一个设备上。
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    # 4. 使用 `torch.cat` 在通道维度上将池化结果 `out.data` 和 `zero_pads` 进行拼接，并将结果封装成 `Variable` 类型后返回。
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


# 5. 定义了一个基本块的类 `BasicBlock`，继承自 PyTorch 的 `nn.Module` 类。
# 这个基本块类用于构建深度学习模型，如 ResNet。它包含了两个卷积层和批标准化层，以及跳跃连接和激活函数。在前向传播过程中，输入先经过一个卷积块，然后通过跳跃连接与残差相加，再经过激活函数输出。
class BasicBlock(nn.Module):
    # 1. 在类的属性中，`expansion` 被设置为 1，用于表示扩张因子。
    expansion = 1

    # 2. 在 `__init__` 方法中，初始化函数接收输入参数包括：
    #     `inplanes`：输入通道数
    #     `planes`：输出通道数
    #     `stride`：卷积的步幅，默认为 1
    #     `downsample`：下采样操作，默认为 None
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # 在初始化函数中，定义了一系列模型层和操作：
        super(BasicBlock, self).__init__()
        # 2.1  `self.conv1`：使用之前定义的 `conv3x3x3` 函数创建一个 3D 卷积层。
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        # 2.2  `self.bn1`：使用 `nn.BatchNorm3d` 创建一个 3D 批标准化层，输入通道数为 `planes`。
        self.bn1 = nn.BatchNorm3d(planes)
        # 2.3  `self.relu`：使用 `nn.ReLU` 创建一个 ReLU 激活函数层。
        self.relu = nn.ReLU(inplace=True)
        # 2.4  `self.conv2`：使用之前定义的 `conv3x3x3` 函数创建一个 3D 卷积层，输入通道数和输出通道数都为 `planes`。
        self.conv2 = conv3x3x3(planes, planes)
        # 2.5  `self.bn2`：使用 `nn.BatchNorm3d` 创建一个 3D 批标准化层，输入通道数为 `planes`。
        self.bn2 = nn.BatchNorm3d(planes)
        # 2.6  `self.downsample`：下采样操作，可以是一个用于减少时间维度分辨率的 `downsample_basic_block` 函数。
        self.downsample = downsample
        # 2.7  `self.stride`：卷积的步幅。
        self.stride = stride

    # 3. 在 `forward` 方法中，定义了基本块的前向传播过程：
    def forward(self, x):
        # 3.1  将输入赋值给 `residual`，作为跳跃连接的路径。
        residual = x
        # 3.2  通过 `conv1`、`bn1` 和 ReLU 激活函数对输入进行卷积操作和批标准化，并将结果赋值给 `out`。
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3.3  通过 `conv2` 和 `bn2` 对 `out` 进行卷积操作和批标准化。
        out = self.conv2(out)
        out = self.bn2(out)
        # 3.4  如果存在下采样操作，则调用 `downsample` 函数对输入 `x` 进行下采样，并将结果赋值给 `residual`。
        if self.downsample is not None:
            residual = self.downsample(x)
        # 3.5  将 `out` 和 `residual` 进行相加，并将结果赋值给 `out`。
        out += residual
        # 3.6 再次使用 ReLU 激活函数对 `out` 进行激活，并返回结果。
        out = self.relu(out)

        return out


# 6. 用于构建残差块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 7. 基于 ResNet 架构的视频分类模型，使用了3D卷积和残差块的概念。
# 该模型适用于视频分类任务，可以根据输入的视频序列提取空间和时间特征，并输出对应的分类结果。
# class ResNet(nn.Module):
#     # 1. `__init__` 方法中初始化了模型的各个层和参数。其中包括：
#     def __init__(self,
#                  block,
#                  layers,
#                  sample_size,
#                  sample_duration,
#                  shortcut_type='B',
#                  num_classes=400):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         #   1.1 `self.conv1`：输入特征图的3D卷积层，将输入通道数为3的特征图转换为64个输出通道。
#         self.conv1 = nn.Conv3d(
#             3,
#             64,
#             kernel_size=7,
#             stride=(1, 2, 2),
#             padding=(3, 3, 3),
#             bias=False)
#         #   1.2 `self.bn1`：输入通道数为64的批标准化层。
#         self.bn1 = nn.BatchNorm3d(64)
#         #   1.3 `self.relu`：使用ReLU激活函数对特征图进行非线性变换。
#         self.relu = nn.ReLU(inplace=True)
#         #   1.4 `self.maxpool`：最大池化层，用于降低特征图的时间和空间分辨率。
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
#         #   1.5 `self.layer1`、`self.layer2`、`self.layer3`、`self.layer4`：
#         #   通过 `_make_layer` 方法创建的一系列残差块层，构建了ResNet的主体结构。
#         self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
#         self.layer2 = self._make_layer(
#             block, 128, layers[1], shortcut_type, stride=2)
#         self.layer3 = self._make_layer(
#             block, 256, layers[2], shortcut_type, stride=2)
#         self.layer4 = self._make_layer(
#             block, 512, layers[3], shortcut_type, stride=2)
#         last_duration = int(math.ceil(sample_duration / 16))
#         last_size = int(math.ceil(sample_size / 32))
#         #   1.6  `self.avgpool`：平均池化层，用于将时序维度的长度与空间维度的尺寸缩减到1。
#         self.avgpool = nn.AvgPool3d(
#             (last_duration, last_size, last_size), stride=1)
#         #   1.7  `self.dropout`：随机失活层，用于防止过拟合。
#         self.dropout = nn.Dropout(0.7)
#         #   1.8  `self.fc`：全连接层，将最后的特征映射转换为对应的类别预测。
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     # 2.  `_make_layer` 方法用于构建残差块层。
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     # 3.  `forward` 方法定义了模型的前向传播过程。
#     # 输入首先通过一系列卷积、批标准化和非线性激活操作，然后经过残差块层进行特征提取，再通过池化和全连接层生成最终的预测结果。
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x_out = self.avgpool(x)
#
#         x = x_out.view(x_out.size(0), -1)
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         return x

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.dropout = nn.Dropout(0.7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print(f"conv1 layer: {x.size()}")
        x = self.bn1(x)
        # print(f"bn1 layer: {x.size()}")
        x = self.relu(x)
        # print(f"relu layer: {x.size()}")
        x = self.maxpool(x)
        # print(f"maxpool layer: {x.size()}")
        x = self.layer1(x)
        # print(f"layer1 layer: {x.size()}")
        x = self.layer2(x)
        # print(f"layer2 layer: {x.size()}")
        x = self.layer3(x)
        # print(f"layer3 layer: {x.size()}")
        x = self.layer4(x)
        # print(f"layer4 layer: {x.size()}")
        x_out = self.avgpool(x)
        # print(f"avgpool layer: {x_out.size()}")
        x = x_out.view(x_out.size(0), -1)
        # print(f"x_out layer: {x.size()}")
        x = self.dropout(x)
        # print(f"dropout layer: {x.size()}")
        # x = self.fc(x)
        # print(f"fc layer: {x.size()}")
        return x


# 8. 获取需要进行微调的参数列表，以及其对应的学习率。
# 参数说明：
#    `model`：需要微调的模型。
#    `ft_begin_index`：微调开始的层数，通常是 ResNet 的最后若干个残差块和全连接层。
# 通过控制 `ft_begin_index` 参数，可以方便地选择需要微调的层数，避免微调全网络带来的过拟合等问题。
def get_fine_tuning_parameters(model, ft_begin_index):
    # 1. 首先判断 `ft_begin_index` 的值。如果为 0，则返回模型的所有参数（即不进行微调）。
    if ft_begin_index == 0:
        return model.parameters()
    # 2. 构造需要微调的层的名称列表 `ft_module_names`。
    # 其中包括最后一个全连接层 `fc` 和从第 `ft_begin_index` 层到第 4 个残差块 `layer4` 的所有层。
    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    # ft_module_names.append('fc')
    # 3. 遍历模型的所有参数，对于属于需要微调的层的参数，将其添加到参数列表中，并指定其默认的学习率。
    # 对于不需要微调的层的参数，将其添加到参数列表中，并将其学习率设置为 0。
    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
    # 4. 返回参数列表。
    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


# 包含 4 个残差块，并且每个残差块中的残差单元数目为 [3, 4, 6, 3]。
# 这意味着第一个残差块有 3 个残差单元，第二个残差块有 4 个残差单元，以此类推。
def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


# 同样也包含了 4 个残差块，但每个残差块中的残差单元数目为 [3, 4, 23, 3]。
# 可以看到，第三个残差块有 23 个残差单元，而其他残差块的残差单元数目都与 resnet50 中的相同。
def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
