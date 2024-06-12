# 公共库函数
import os
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

# 目录中已有的py文件
from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformRandomSample, UniformEndSample
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set
from utils import Logger

if __name__ == '__main__':


    # TODO 一、针对opts.py，对各种参数：路径、标准差等进行操作
    # 1. 解析命令行参数并返回一个包含各种选项的对象 opt
    opt = parse_opts()
    # 2. 路径设置部分
    if opt.root_path != '':
        # 使用 os.path.join() 函数将各个路径拼接在一起。它将 opt.root_path 与其他路径拼接，以确保得到完整的路径。
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    # 3. 尺度设置部分：通过循环设置尺度列表 opt.scales，
    opt.scales = [opt.initial_scale] # 初始化为只包含一个元素 opt.initial_scale。
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)# 将列表的最后一个元素乘以 opt.scale_step，得到新的尺度值，并添加列表。
    # 4. 架构设置部分：将 opt.model 和 opt.model_depth 拼接为形如 model-depth 的字符串，并赋值给 opt.arch。
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    # 5. 均值和标准差设置部分：调用 get_mean() 和 get_std() 函数，并将得到的结果赋值给 opt.mean 和 opt.std。
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    # 6. 总结：解析命令行参数，并根据解析的结果对一些路径、尺度、架构、均值和标准差等进行设置和计算，并打印出设置后的 opt 对象的内容。


    # TODO 二、opts.json的生成，随机种子，模型
    # 7. 将 opt 字典以 JSON 格式写入到文件对象 opt_file 中。
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    # 8. 设置随机种子，用于使随机过程在每次运行时产生相同的结果。opt.manual_seed 是从命令行参数中获取的手动设置的种子值。
    torch.manual_seed(opt.manual_seed)
    # 9. generate_model() 函数根据传递的 opt 对象创建一个模型，并返回生成的模型和其对应的参数，赋值给变量 model，parameters。
    model, parameters = generate_model(opt)
    # 10. 总结：通过 print(model) 将模型打印出来，以便查看模型的架构和配置
    print('我的调试：model：', model)
    print('我的调试：parameters：', parameters)



    # TODO 三、权重、损失函数、cpu和gpu的设置
    # 11. 创建权重列表 weights，其中包含了5个权重值。这些权重值用于在计算损失函数时对不同类别进行加权处理。
    weights = [1, 2, 4, 2, 4]
    # 12. 将权重列表转换为浮点张量，并将其移动到 GPU 上（如果可用）。这样做是为了确保在计算损失函数时能够使用 GPU 加速。
    class_weights = torch.FloatTensor(weights).cuda()
    # 13. 创建一个交叉熵损失函数 (CrossEntropyLoss)，并设置权重参数为 class_weights。在计算损失函数时会按照权重对预测结果进行加权。
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 14. 如果命令行参数中没有设置 no_cuda 为真，则将损失函数移动到 GPU 上进行计算（如果可用）
    if not opt.no_cuda:
        criterion = criterion.cuda()
    # 15. 根据命令行参数设置归一化方法 norm_method：
    # 如果设置了 no_mean_norm 为真且 std_norm 为假，则将 norm_method 设置为均值为 [0, 0, 0]，标准差为 [1, 1, 1] 的归一化方法。
    # 如果 std_norm 是假，则将 norm_method 设置为均值为 opt.mean，标准差为 [1, 1, 1] 的归一化方法。
    # 否则，将 norm_method 设置为均值为 opt.mean，标准差为 opt.std 的归一化方法
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    # 16. 总结：用于设置损失函数权重、GPU计算和归一化方法，并为后续的模型训练和评估过程提供了基础。


    # TODO 四、如果没有设置no_train，就是训练过程
    # start：根据训练参数配置了训练数据的预处理和数据加载过程
    # 17. 如果命令行参数中没有设置no_train，则执行以下代码块。这意味着要进行训练过程。
    if not opt.no_train:
        print('我的调试：if not no_train 开始配置：')
        # 18. 确保训练时的裁剪方法参数设置正确。
        assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)  # 多尺度随机裁剪
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)  # 多尺度角落裁剪
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c']) # 多尺度中心裁剪，其中裁剪位置仅为中心位置。
        elif opt.train_crop == 'driver focus':
            crop_method = DriverFocusCrop(opt.scales, opt.sample_size) # 驾驶员关注区域裁剪。
        # 19. 设置训练时的空间变换 train_spatial_transform（图像预处理/数据增强操作）：
            # 首先应用之前选择的裁剪方法 crop_method。
            # 然后使用 MultiScaleRandomCrop 方法进行多尺度随机裁剪。
            # 将图像转换为张量，并进行归一化处理，使用 norm_method 进行均值和标准差的调整。
        train_spatial_transform = Compose([
            crop_method,
            MultiScaleRandomCrop(opt.scales, opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        # 20. 设置训练时的时间变换 （视频预处理/数据增强操作），采样持续时间为 opt.sample_duration，结束时间为 opt.end_second。
        train_temporal_transform = UniformRandomSample(opt.sample_duration, opt.end_second)
        # 21. 设置目标转换 train_target_transform，用于将目标（类别）转换为标签。
        train_target_transform = ClassLabel()
        # 22. 设置训练过程中的水平翻转 train_horizontal_flip。
        train_horizontal_flip = RandomHorizontalFlip()
        # 23. 使用 get_training_set获取训练数据集，并通过传入之前设置的空间变换、水平翻转、时间变换和目标转换等来对数据集进行相应的处理。
        training_data = get_training_set(opt, train_spatial_transform, train_horizontal_flip,
                                         train_temporal_transform, train_target_transform)
        print('我的调试：training_data及其长度：', training_data, len(training_data))
        # 24. 使用 torch.utils.data.DataLoader 创建训练数据加载器 train_loader，用于在训练过程中批量加载训练数据。
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        # 25. 创建记录训练过程日志的Logger实例
        # 日志文件路径为 os.path.join(opt.result_path, 'train.log')，记录的内容为每个 epoch 的损失、准确率和学习率等。
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        # 26. 创建记录训练过程每个批次日志的 Logger 实例 train_batch_logger
        # 日志文件路径为 os.path.join(同上, 'train_batch.log')，记录的内容为每个 epoch 的批次号、迭代号、损失、准确率和学习率等。
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        print('我的调试：训练时的train_loader！', train_loader)
    # end: 总结：用于配置训练数据的预处理和加载过程，以便在后续的模型训练中使用。


    # TODO 五、优化器和学习率调度器的配置，以及验证数据的预处理和加载。
    # 主要包括了优化器和学习率调度器的配置，以及验证数据的预处理和加载。
    # 27. 根据命令行参数 opt.nesterov 的值来设置 dampening 参数。
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    # 28. 使用 optim.SGD 函数创建一个随机梯度下降（SGD）优化器 optimizer，并传入以下参数：
          # parameters：模型的参数列表。
          # lr=opt.learning_rate：学习率，取命令行参数中的 learning_rate 值。
          # momentum=opt.momentum：动量因子，取命令行参数中的 momentum 值。
          # dampening=dampening：阻尼因子，根据上述步骤得到的 dampening 值。
          # weight_decay=opt.weight_decay：权重衰减，取命令行参数中的 weight_decay 值。
          # nesterov=opt.nesterov：Nesterov 动量，取命令行参数中的 nesterov 值。
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    # 29. 使用 lr_scheduler.MultiStepLR 函数创建一个多步骤学习率调度器 scheduler，并传入以下参数：
          # optimizer：优化器对象。
          # milestones=opt.lr_step：学习率调整的里程碑（即在训练的哪些轮次/阶段进行学习率调整），取命令行参数中的 lr_step 值。
                # gamma=0.1：学习率衰减倍数，每个里程碑处的学习率将乘以该倍数。
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.lr_step, gamma=0.1)


    # TODO 六、如果没有设置no_val，就是验证过程。
    # 30. 如果命令行参数中没有设置 no_val 为真，则执行以下代码块。这意味着要进行验证过程。
    if not opt.no_val:
        print('我的调试：if not no_val 开始：')
        # 31. 创建验证数据的空间变换 val_spatial_transform。使用 DriverCenterCrop 方法进行中心裁剪，并转换为张量，并进行归一化处理。
        val_spatial_transform = Compose([
            DriverCenterCrop(opt.scales, opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        # 32. 创建验证数据的时间变换 val_temporal_transform，使用 UniformEndSample 方法进行均匀地从视频中选择帧。
        val_temporal_transform = UniformEndSample(opt.sample_duration, opt.end_second)
        # 33. 创建验证数据的目标转换 val_target_transform，用于将目标（类别）转换为标签。
        val_target_transform = ClassLabel()
        # 34. 使用dataset里面的get_validation_set 函数获取验证数据集，并传入空间变换、时间变换和目标转换等对数据集进行处理。
        validation_data = get_validation_set(
            opt, val_spatial_transform, val_temporal_transform, val_target_transform)
        # print('我的调试：validation_data：', validation_data)
        # 35. 使用 torch.utils.data.DataLoader 创建验证数据加载器 val_loader，用于在验证过程中批量加载验证数据。
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=2,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=None)
        # 36. 创建记录验证过程日志的 Logger实例 val_logger
        # 日志文件路径为 os.path.join(opt.result_path, 'val.log')，记录的内容为每个 epoch 的损失和准确率
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])
        print('我的调试：验证时的val_loader！', val_loader)


    #  TODO 七、如果有resume_path，就在此基础上继续训练。
    # 这段代码主要用于恢复之前的模型训练状态，以便从上一次训练结束的地方继续训练。
    # 37. 如果提供，则加载指定路径下的检查点文件，即恢复之前中断的训练过程。
    if opt.resume_path:
        print('resume_path：loading checkpoint {}'.format(opt.resume_path))
        # 38. 使用 torch.load() 方法加载检查点文件，将加载的数据存储在字典中。
        checkpoint = torch.load(opt.resume_path)
        # print('我的调试：checkpoint：', checkpoint)
        # 39. 检查当前模型的架构与加载的检查点文件中保存的模型架构是否一致。如果不一致，程序会抛出异常并中止执行。
        assert opt.arch == checkpoint['arch']
        # 40. 将命令行参数 begin_epoch 的值设置为加载的检查点文件中保存的最后一个 epoch 的值（即 checkpoint[epoch']）。
        # print('我的调试：opt.begin_epoch：', opt.begin_epoch)
        # print('我的调试：checkpoint[epoch]：', checkpoint['epoch'])
        opt.begin_epoch = checkpoint['epoch']
        # 41. 使用 model.load_state_dict() 方法加载检查点文件中保存的模型权重（即 checkpoint['state_dict']）。
        model.load_state_dict(checkpoint['state_dict'])
        # 42. 如果命令行参数中没有，则执行以下代码块。这意味着要继续训练模型，因此需要加载优化器状态。
        if not opt.no_train:
            # 43. 使用 optimizer.load_state_dict() 方法加载检查点文件中保存的优化器状态（即 checkpoint['optimizer']）。
            optimizer.load_state_dict(checkpoint['optimizer'])
        # print('我的调试：resume_path部分结束！')


    # 迭代多个 epoch。
    print('run') # 打印输出信息 run。
    # 44. 将全局变量 best_prec 的值设为 0，用于记录最佳验证准确率
    global best_prec
    best_prec = 0
    # print('我的调试：best_prec设置完成！')
    # 45. 使用 range 函数循环迭代从 opt.begin_epoch 到 opt.n_epochs + 1 的所有整数值
    # 其中 opt.begin_epoch 指定了起始 epoch，opt.n_epochs 指定了总共需要迭代的 epoch 数量，+1是因为range的右端点是不包含在迭代范围内。
    print('我的调试：opt.begin_epoch：', opt.begin_epoch)
    print('我的调试：opt.n_epochs：', opt.n_epochs)
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        # print('我的调试：epoch循环开始：')
        # 这段代码用于在每个 epoch 进行模型的训练。
        # 46. 如果命令行参数中 no_train 为假，则执行以下代码块。这意味着要进行训练过程。
        if not opt.no_train:
            print('train at epoch {}'.format(epoch))
            # 47. 调用 model.train() 方法将模型设置为训练模式，以启用 Batch Normalization 和 Dropout 等训练特定的操作。
            model.train()
            # 48. 记录每个 batch 的时间、数据加载时间、损失值和准确率。
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            # 49. 使用 time.time() 函数记录当前时间，作为训练开始的时间 end_time。
            end_time = time.time()

            # print('我的调试：train_loader为：', train_loader)
            # 这段代码是训练过程的循环部分，用于遍历训练数据集并进行模型训练。
            # 50. 遍历训练数据集 train_loader 中的每个 batch，并同时获取 batch 的索引 i 和数据 inputs、targets。
            for i, (inputs, targets) in enumerate(train_loader):
                # 51. 更新 data_time 计时器，记录数据加载时间。
                data_time.update(time.time() - end_time)
                # 52. 如果命令行参数中 no_cuda 为假，则将 targets 移动到 GPU 上，以便在 GPU 上进行计算。
                if not opt.no_cuda:
                    targets = targets.cuda(non_blocking=True)
                # 53. 将 inputs 和 targets 封装为 Variable 对象，使其可以自动计算梯度。
                inputs = Variable(inputs)
                targets = Variable(targets)
                # 54. 将 inputs 输入到模型中，得到模型的输出 outputs。
                outputs = model(inputs)
                print('outputs:', outputs)
                # 55. 使用损失函数 criterion 计算输出 outputs 与目标值 targets 之间的损失。
                # 调用自定义函数 calculate_accuracy 计算模型在当前 batch 上的准确率。
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                # 56. 更新 losses 和 accuracies 计算器，记录当前 batch 的损失和准确率。
                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc, inputs.size(0))
                # 57. 清空优化器 optimizer 的梯度。
                optimizer.zero_grad()
                # 58. 使用反向传播计算损失对模型参数的梯度。
                loss.backward()
                optimizer.step() # 使用优化器更新模型参数。
                # 59. 更新 batch_time 计时器，记录当前 batch 的训练时间。
                batch_time.update(time.time() - end_time)
                end_time = time.time() # 更新 end_time 为当前时间，作为下一个 batch 训练开始的时间。
                # 60.使用 train_batch_logger.log() 记录训练过程的相关信息，包括当前的 epoch、batch 等。
                train_batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(train_loader) + (i + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': optimizer.param_groups[0]['lr']
                })
                # 61. 如果 i 满足 i % 5 == 0，即每隔 5 个 batch 打印一次训练过程的详细信息。
                # 打印信息中包括当前的 epoch、batch 的进度、时间、损失和准确率等。
                if i % 5 == 0:
                  print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch,
                          i + 1,
                          len(train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))
                # 以上就是循环遍历训练数据集并进行模型训练的代码。

            # 这段代码是训练过程的末尾部分，用于记录训练过程中的相关信息并保存模型的检查点。
            # 62. 记录当前 epoch 的训练结果，包括 epoch 数、平均损失 losses.avg、平均准确率 accuracies.avg 和优化器的学习率
            train_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })
            # 63. 如果当前 epoch 满足 epoch % opt.checkpoint == 0，即达到了设定的检查点间隔，会执行以下代码：
            if epoch % opt.checkpoint == 0:
                # 64. 构建保存文件的路径 save_file_path，将其命名为 'save_{}.pth'，其中 {} 会使用当前 epoch 数填充。
                save_file_path = os.path.join(opt.result_path,
                                              'save_{}.pth'.format(epoch))
                # 65. 定义一个字典 states，包含了当前 epoch 的一些状态信息，
                # 如epoch数加 1、模型的架构 opt.arch、模型的状态字典model.state_dict()和优化器的状态字典 optimizer.state_dict()。
                # 将 states 字典保存到文件 save_file_path 中，以便在训练结束后可以加载模型的状态和继续训练。
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            # 以上就是记录训练信息和保存模型检查点的代码。

        # 这段代码是在每个 epoch 结束后执行验证（validation）的操作，并根据验证结果保存最佳模型。
        # 66. 首先，检查 opt.no_val 是否为假（即进行验证）。如果需要进行验证，则执行以下操作：
        if not opt.no_val:
            # 67. 输出当前 epoch 的验证信息，打印出 "Validation at epoch {epoch}"。
            print('Validation at epoch {}'.format(epoch))
            # 68. 将模型设置为评估模式，使用 model.eval()。
            model.eval()
            # 69. 初始化计时器和统计指标的平均值，包括批次时间 batch_time、数据加载时间 data_time、损失 losses 和准确率 accuracies。
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            end_time = time.time()
            # 70. 初始化一个全零矩阵 conf_mat，用于存储混淆矩阵。
            conf_mat = torch.zeros(opt.n_finetune_classes, opt.n_finetune_classes)
            output_file = []
            # 71. 对验证数据集的每个批次进行迭代，在循环中执行以下操作：
            for i, (inputs, targets) in enumerate(val_loader):
                # 72. 更新数据加载时间 data_time。
                data_time.update(time.time() - end_time)
                # 73. 如果不禁用 GPU 计算，则将目标变量 targets 移到 GPU 上。
                if not opt.no_cuda:
                    targets = targets.cuda(non_blocking=True)
                # 74. 将输入数据inputs和目标变量targets封装成PyTorch变量（Variable），并设置为 volatile=True，表示不需要计算梯度。
                # inputs = Variable(inputs, volatile=True) 原版的不再兼容，已修改。
                with torch.no_grad():
                    inputs = inputs

                # targets = Variable(targets, volatile=True) 原版的不再兼容，已修改。
                with torch.no_grad():
                    targets = targets

                # 75. 将输入数据传入模型，并得到模型的输出 outputs。
                outputs = model(inputs)
                print('outputs:', outputs)
                # 76. 使用损失函数 criterion 计算模型的输出和目标的损失值。
                # 使用自定义函数 calculate_accuracy 计算输出的准确率。
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)

                ### print out the confusion matrix
                _,pred = torch.max(outputs,1)
                for t,p in zip(targets.view(-1), pred.view(-1)):
                    conf_mat[t,p] += 1
                # 77. 根据预测结果和目标计算混淆矩阵，并将结果累加到 conf_mat。
                losses.update(loss.item(), inputs.size(0))
                # 78. 更新损失值 losses 和准确率 accuracies。
                accuracies.update(acc, inputs.size(0))
                # 79. 更新批次时间 batch_time。
                batch_time.update(time.time() - end_time)
                # 80. 更新结束时间 end_time。
                end_time = time.time()

                # 打印当前批次的训练信息，包括当前 epoch、批次索引、总批次数、批次时间、数据加载时间、损失和准确率。
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                          epoch,
                          i + 1,
                          len(val_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))
            print(conf_mat)# 打印混淆矩阵 conf_mat。
            # 81. 使用 val_logger.log()记录当前epoch的验证结果，包括epoch数、平均损失losses.avg和平均准确率accuracies.avg。
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
            # 判断当前模型是否为最佳模型，并更新最佳准确率 best_prec。如果当前准确率高于最佳准确率，则设置 is_best 为 True。
            is_best = accuracies.avg > best_prec
            best_prec = max(accuracies.avg, best_prec)
            print('\n The best prec is %.4f' % best_prec)
            # 如果 is_best 为 True，则保存模型状态，
            # 包括当前 epoch、模型架构 opt.arch、模型的状态字典 model.state_dict() 和优化器的状态字典 optimizer.state_dict()。
            # 将保存路径设为 save_file_path = os.path.join(opt.result_path, 'save_best.pth')。
            if is_best:
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                 }
                save_file_path = os.path.join(opt.result_path,
                                    'save_best.pth')
                torch.save(states, save_file_path)
        # 接下来，如果训练且验证，则执行 scheduler.step()，用于调整学习率。
        if not opt.no_train and not opt.no_val:
            scheduler.step()