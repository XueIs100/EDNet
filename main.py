# 公共库
import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
import warnings

# 忽略所有的UserWarning警告消息
# warnings.filterwarnings("ignore", category=UserWarning)
# 忽略特定的警告消息
warnings.filterwarnings("ignore", message="Named tensors and all their associated APIs are an experimental feature")
# 自建函数
from my_models.mean import get_mean
from my_models.mean import get_std
from my_models.generate_model import generate_model
from my_models.convlstm_conv import encoder
from my_models.spatial_transforms import Normalize
from my_models.spatial_transforms import MultiScaleRandomCrop
from my_models.spatial_transforms import MultiScaleCornerCrop
from my_models.spatial_transforms import DriverFocusCrop
from my_models.spatial_transforms import Compose
from my_models.temporal_transforms import UniformRandomSample
from my_models.target_transforms import ClassLabel
from my_models.spatial_transforms import RandomHorizontalFlip
from my_models.temporal_transforms import UniformIntervalCrop
from my_models.dataset import in_get_training_set
from my_models.dataset import out_get_training_set
from my_models.utils import Logger
from my_models.temporal_transforms import UniformEndSample
from my_models.spatial_transforms import ToTensor
from my_models.spatial_transforms import Scale
from my_models.spatial_transforms import DriverCenterCrop
from my_models.dataset import in_get_validation_set
from my_models.dataset import out_get_validation_set
from my_models.utils import AverageMeter
from my_models.concat import Net
from my_models.utils import calculate_accuracy


class CombineOpt:
    def __init__(self):
        # 1. 路径：in_video_path & out_video_path(视频数据路径)、annotation_path（注释路径）、result_path（保存路径）
        self.in_video_path = r"D:\XHX\Driver-Intention-Prediction-master\Brain4Cars\face_camera"
        self.out_video_path = r"D:\XHX\Driver-Intention-Prediction-master\Brain4Cars\flownet2_road_camera"
        self.annotation_path = "D:/XHX/Driver-Intention-Prediction-master/test/zhushi_in"
        self.result_path = "D:/XHX/Driver-Intention-Prediction-master/results_combine/"
        self.pretrain_path = ""
        self.in_pretrain_path = ""
        self.out_pretrain_path = ""
        self.in_resume_path = ""
        self.out_resume_path = ""
        self.resume_path = ""

        # 2. 数据类型 & 折数 & 网络：in_dataset & out_dataset等
        self.in_dataset = "Brain4cars_Inside"
        self.out_dataset = "Brain4cars_Outside"
        self.n_fold = 0
        self.model = "resnet"
        self.model_depth = 50
        self.resnet_shortcut = "B"
        self.in_arch = "resnet-50"
        self.out_arch = "ConvLSTM"
        self.arch = "net"
        # 3. 训练部分
        self.batch_size = 2
        self.learning_rate = 0.0001
        self.begin_epoch = 1
        self.n_epochs = 1
        self.no_train = ""
        self.no_val = ""
        # 4. 车内车外都有：checkpoint（模型在指定的周期（epoch）保存一次）、n_threads（多线程数据加载）
        self.in_sample_duration = 16
        self.out_sample_duration = 5
        self.in_n_scales = 3
        self.out_n_scales = 1
        self.checkpoint = 1
        self.n_threads = 4
        self.end_second = 5
        # 5. 只有车内：n_finetune_classes（微调类别数）、ft_begin_index（开始微调残差块索引位置）、train_crop（数据增强的裁剪方法）
        self.n_classes = 5
        self.n_finetune_classes = 5
        self.ft_begin_index = 4

        self.in_train_crop = "driver focus"
        # 6. 只有车外：interval（训练convlstm时从外部图像进行采样的时间间隔，每隔5个时间步采样一次外部图像，取值范围是1-30）
        self.out_train_crop = "corner"

        self.interval = 5
        # 7. opts.py里面用到的:scales、
        #    norm_value（指定输入数据的归一化范围：如果值为1，表示[0-255];如果值为255，[0-1]，即数据进行了归一化处理）
        self.in_scales = [1.0]
        self.out_scales = [1.0]

        self.scale_step = 0.84089641525

        self.in_norm_value = 1
        self.out_norm_value = 255

        self.mean_dataset = "activitynet"
        self.manual_seed = 2
        self.sample_size = 112
        self.no_cuda = "false"
        self.no_mean_norm = "false"
        self.std_norm = "false"
        self.nesterov = "false"
        self.dampening = 0.9
        self.momentum = 0.9
        self.weight_decay = 0.001
        self.lr_step = [30, 60]
        self.n_val_samples = 1

    def generate_opt(self):
        for i in range(1, self.in_n_scales):
            in_scale = self.in_scales[-1] * self.scale_step
            self.in_scales.append(in_scale)
        for i in range(1, self.out_n_scales):
            out_scale = self.out_scales[-1] * self.scale_step
            self.out_scales.append(out_scale)

        self.in_mean = get_mean(self.in_norm_value, dataset=self.mean_dataset)
        self.in_std = get_std(self.in_norm_value)
        self.out_mean = get_mean(self.out_norm_value, dataset=self.mean_dataset)
        self.out_std = get_std(self.out_norm_value)

        with open(os.path.join(self.result_path, 'combine_opts.json'), 'w') as opt_file:
            json.dump(self.__dict__, opt_file)
        return self


if __name__ == '__main__':

    combine_opt = CombineOpt()
    combine_opt.generate_opt()

    torch.manual_seed(combine_opt.manual_seed)

    # 内外模型结构
    in_model = generate_model(combine_opt)
    in_parameters = in_model.parameters()
    out_model = encoder(hidden_channels=[128, 64, 64, 32],
                        sample_size=combine_opt.sample_size,
                        sample_duration=combine_opt.out_sample_duration).cuda()
    out_model = nn.DataParallel(out_model, device_ids=None)
    out_parameters = out_model.parameters()

    # 创建 Net 类的实例，并传递输入数据 (in_inputs, out_inputs)
    combine_net = Net(in_model, out_model)
    combine_parameters = combine_net.parameters()
    '''
    
    # 2.9.3 如果存在pretrain_path。
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.module.classifier = nn.Linear(
                    model.module.classifier.in_features, opt.n_finetune_classes)
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    
    '''

    weights = [1, 2, 4, 2, 4]  # 车内：权重列表，用于在计算损失函数时对不同类别进行加权处理。
    class_weights = torch.FloatTensor(weights).cuda()

    # 两种结合的采取交叉熵损失函数
    in_criterion = nn.CrossEntropyLoss(weight=class_weights)  # 车内：交叉熵损失函数 (CrossEntropyLoss)
    out_criterion = nn.MSELoss()  # 车外：均方误差，用于衡量预测值与目标值之间的差距。
    combine_criterion = nn.CrossEntropyLoss(weight=class_weights)

    if not combine_opt.no_cuda:
        # in_criterion = in_criterion.cuda()
        # out_criterion = out_criterion.cuda()
        combine_criterion = combine_criterion.cuda()

    if combine_opt.no_mean_norm and not combine_opt.std_norm:
        in_norm_method = Normalize([0, 0, 0], [1, 1, 1])
        out_norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not combine_opt.std_norm:
        in_norm_method = Normalize(combine_opt.in_mean, [1, 1, 1])
        out_norm_method = Normalize(combine_opt.out_mean, [1, 1, 1])
    else:
        in_norm_method = Normalize(combine_opt.in_mean, combine_opt.in_std)
        out_norm_method = Normalize(combine_opt.out_mean, combine_opt.out_std)

    if not combine_opt.no_train:

        assert combine_opt.in_train_crop in ['random', 'corner', 'center', 'driver focus']
        assert combine_opt.out_train_crop in ['random', 'corner', 'center', 'driver focus']

        if combine_opt.in_train_crop == 'random':
            in_crop_method = MultiScaleRandomCrop(combine_opt.in_scales, combine_opt.sample_size)  # 多尺度随机裁剪
        elif combine_opt.in_train_crop == 'corner':
            in_crop_method = MultiScaleCornerCrop(combine_opt.in_scales, combine_opt.sample_size)  # 多尺度角落裁剪
        elif combine_opt.in_train_crop == 'center':  # 多尺度中心裁剪，其中裁剪位置仅为中心位置。
            in_crop_method = MultiScaleCornerCrop(combine_opt.in_scales, combine_opt.sample_size, crop_positions=['c'])
        elif combine_opt.in_train_crop == 'driver focus':
            in_crop_method = DriverFocusCrop(combine_opt.in_scales, combine_opt.sample_size)  # 驾驶员关注区域裁剪
        ############################################ 车内 & 车外 ：数据增强 ######################################################
        if combine_opt.out_train_crop == 'random':
            out_crop_method = MultiScaleRandomCrop(combine_opt.out_scales, combine_opt.sample_size)
        elif combine_opt.out_train_crop == 'corner':
            out_crop_method = MultiScaleCornerCrop(combine_opt.out_scales, combine_opt.sample_size)
        elif combine_opt.out_train_crop == 'center':
            out_crop_method = MultiScaleCornerCrop(combine_opt.out_scales, combine_opt.sample_size,
                                                   crop_positions=['c'])
        elif combine_opt.out_train_crop == 'driver focus':
            out_crop_method = DriverFocusCrop(combine_opt.out_scales, combine_opt.sample_size)

        in_train_spatial_transform = Compose([  # 空间变换（预处理/数据增强）：裁剪方法、多尺度随机裁剪、将图像转换为张量
            in_crop_method,
            MultiScaleRandomCrop(combine_opt.in_scales, combine_opt.sample_size),
            ToTensor(combine_opt.in_norm_value), in_norm_method])
        in_train_temporal_transform = UniformRandomSample(combine_opt.in_sample_duration, combine_opt.end_second)
        in_train_target_transform = ClassLabel()  # 设置目标转换，用于将目标（类别）转换为标签
        in_train_horizontal_flip = RandomHorizontalFlip()  # 设置训练过程中的水平翻转
        ############################################ 车内 & 车外 ：训练时 ######################################################
        out_train_spatial_transform = Compose([Scale(combine_opt.sample_size), ToTensor(combine_opt.out_norm_value)])
        out_train_temporal_transform = UniformIntervalCrop(combine_opt.out_sample_duration, combine_opt.interval)
        out_train_target_transform = Compose([Scale(combine_opt.sample_size), ToTensor(combine_opt.out_norm_value)])
        out_train_horizontal_flip = RandomHorizontalFlip()

        # 获取训练数据集，并通过传入之前设置的空间变换、水平翻转、时间变换和目标转换等来对数据集进行相应的处理。
        in_training_data = in_get_training_set(combine_opt, in_train_spatial_transform, in_train_horizontal_flip,
                                               in_train_temporal_transform, in_train_target_transform)
        # 创建训练数据加载器 train_loader，用于在训练过程中批量加载训练数据。
        in_train_loader = torch.utils.data.DataLoader(
            in_training_data,
            batch_size=combine_opt.batch_size, shuffle=True,
            num_workers=combine_opt.n_threads, pin_memory=True)
        ############################################ 车内 & 车外 ：加载训练数据 #################################################
        out_training_data = out_get_training_set(combine_opt, out_train_spatial_transform, out_train_horizontal_flip,
                                                 out_train_temporal_transform, out_train_target_transform)
        out_train_loader = torch.utils.data.DataLoader(
            out_training_data,
            batch_size=combine_opt.batch_size, shuffle=True,
            num_workers=combine_opt.n_threads, pin_memory=True)

        #         in_train_logger = Logger(
        #             os.path.join(combine_opt.result_path, 'in-train.log'),
        #             ['epoch', 'loss', 'acc', 'lr'])
        #         in_train_batch_logger = Logger(
        #             os.path.join(combine_opt.result_path, 'in-train_batch.log'),
        #             ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        # ############################################ 车内 & 车外 ：创建日志 #################################################
        #         out_train_logger = Logger(
        # 			os.path.join(combine_opt.result_path, 'out-train.log'),
        # 			['epoch', 'loss', 'lr'])
        #         out_train_batch_logger = Logger(
        # 			os.path.join(combine_opt.result_path, 'out-train_batch.log'),
        # 			['epoch', 'batch', 'iter', 'loss', 'lr'])
        ############################################ 总的 ##################################################################
        combine_train_logger = Logger(
            os.path.join(combine_opt.result_path, 'combine-train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        combine_train_batch_logger = Logger(
            os.path.join(combine_opt.result_path, 'combine-train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    ############################################ 优化器和学习率调度器的配置 ###################################################
    if combine_opt.nesterov:
        dampening = 0
    else:
        dampening = combine_opt.dampening
    # 随机梯度下降（SGD）优化器 optimizer，传入参数：模型的参数列表、学习率、动量因子、阻尼因子、权重衰减、动量
    # in_optimizer = optim.SGD(
    #     in_parameters,
    #     lr=combine_opt.learning_rate,
    #     momentum=combine_opt.momentum,
    #     dampening=dampening,
    #     weight_decay=combine_opt.weight_decay,
    #     nesterov=combine_opt.nesterov)
    # out_optimizer = optim.SGD(
    #     out_parameters,
    #     lr=combine_opt.learning_rate,
    #     momentum=combine_opt.momentum,
    #     dampening=dampening,
    #     weight_decay=combine_opt.weight_decay,
    #     nesterov=combine_opt.nesterov)
    combine_optimizer = optim.SGD(
        combine_parameters,
        lr=combine_opt.learning_rate,
        momentum=combine_opt.momentum,
        dampening=dampening,
        weight_decay=combine_opt.weight_decay,
        nesterov=combine_opt.nesterov)

    # 创建一个多步骤学习率调度器 scheduler，并传入参数：
    # 优化器对象、学习率调整的里程碑（即在训练的哪些轮次/阶段进行学习率调整）、学习率衰减倍数，每个里程碑处的学习率将乘以该倍数。
    # in_scheduler = lr_scheduler.MultiStepLR(
    #     in_optimizer, milestones=combine_opt.lr_step, gamma=0.1)
    # out_scheduler = lr_scheduler.MultiStepLR(
    #     out_optimizer, milestones=combine_opt.lr_step, gamma=0.1)
    combine_scheduler = lr_scheduler.MultiStepLR(
        combine_optimizer, milestones=combine_opt.lr_step, gamma=0.1)

    if not combine_opt.no_val:
        # 创建验证数据的空间变换，使用 DriverCenterCrop 方法进行中心裁剪，并转换为张量，并进行归一化处理。
        in_val_spatial_transform = Compose([
            DriverCenterCrop(combine_opt.in_scales, combine_opt.sample_size),
            ToTensor(combine_opt.in_norm_value), in_norm_method])
        # 创建验证数据的时间变换，使用 UniformEndSample 方法进行均匀地从视频中选择帧。
        in_val_temporal_transform = UniformEndSample(combine_opt.in_sample_duration, combine_opt.end_second)
        # 创建验证数据的目标转换，用于将目标（类别）转换为标签。
        in_val_target_transform = ClassLabel()
        ######################################  车内 & 车外 ：一些转换 #######################################################
        out_val_spatial_transform = Compose([Scale(combine_opt.sample_size), ToTensor(combine_opt.out_norm_value)])
        out_val_temporal_transform = UniformIntervalCrop(combine_opt.out_sample_duration, combine_opt.interval)
        out_val_target_transform = out_val_spatial_transform

        # 获取验证数据集，并传入空间变换、时间变换和目标转换等对数据集进行处理。
        in_validation_data = in_get_validation_set(
            combine_opt, in_val_spatial_transform, in_val_temporal_transform, in_val_target_transform)
        # 创建验证数据加载器，用于在验证过程中批量加载验证数据。
        in_val_loader = torch.utils.data.DataLoader(
            in_validation_data,
            batch_size=2,
            shuffle=False,
            num_workers=combine_opt.n_threads,
            pin_memory=None)
        ######################################  车内 & 车外 ：验证集 #######################################################
        out_validation_data = out_get_validation_set(
            combine_opt, out_val_spatial_transform, out_val_temporal_transform, out_val_target_transform)
        out_val_loader = torch.utils.data.DataLoader(
            out_validation_data,
            batch_size=1,
            shuffle=True,
            num_workers=combine_opt.n_threads,
            pin_memory=True)

        # 创建记录验证过程日志
        # in_val_logger = Logger(os.path.join(combine_opt.result_path, 'in-val.log'), ['epoch', 'loss', 'acc'])
        # out_val_logger = Logger(
        #     os.path.join(combine_opt.result_path, 'out-val.log'), ['epoch', 'loss', 'ssim', 'psnr'])
        combine_val_logger = Logger(
            os.path.join(combine_opt.result_path, 'combine-val.log'), ['epoch', 'loss', 'acc'])

    # # 在此基础上继续训练。
    # if combine_opt.in_resume_path:
    #     print('in_resume_path：loading checkpoint {}'.format(combine_opt.in_resume_path))
    #     # 加载检查点文件，存储在字典中。
    #     in_checkpoint = torch.load(combine_opt.in_resume_path)
    #     # 检查当前模型的架构与加载的检查点文件中保存的模型架构是否一致。
    #     assert combine_opt.in_arch == in_checkpoint['arch']
    #     # print('我的调试：opt.begin_epoch：', opt.begin_epoch)
    #     # print('我的调试：checkpoint[epoch]：', checkpoint['epoch'])
    #     combine_opt.begin_epoch = in_checkpoint['epoch']
    #     # 加载检查点文件中保存的模型权重。
    #     in_model.load_state_dict(in_checkpoint['state_dict'])
    #     if not combine_opt.no_train:
    #         in_optimizer.load_state_dict(in_checkpoint['optimizer'])
    # if combine_opt.out_resume_path:
    #     print('out_resume_path：loading checkpoint {}'.format(combine_opt.out_resume_path))
    #     out_checkpoint = torch.load(combine_opt.out_resume_path)
    #     assert combine_opt.out_arch == out_checkpoint['arch']
    #     combine_opt.begin_epoch = out_checkpoint['epoch']
    #     out_model.load_state_dict(out_checkpoint['state_dict'])
    #     if not combine_opt.no_train:
    #         out_optimizer.load_state_dict(out_checkpoint['optimizer'])
    if combine_opt.resume_path:
        print('resume_path：loading checkpoint {}'.format(combine_opt.resume_path))
        checkpoint = torch.load(combine_opt.resume_path)
        combine_opt.begin_epoch = checkpoint['epoch']
        combine_net.load_state_dict(checkpoint['state_dict'])
        if not combine_opt.no_train:
            combine_optimizer.load_state_dict(checkpoint['optimizer'])



    print('run')  # 打印输出信息 run
    global best_prec
    best_prec = 0
    # global out_best_loss
    # out_best_loss = torch.tensor(float('inf'))
    print('我的调试：combine_opt.begin_epoch：', combine_opt.begin_epoch)
    print('我的调试：combine_opt.n_epochs：', combine_opt.n_epochs)
    for epoch in range(combine_opt.begin_epoch, combine_opt.n_epochs + 1):
        # 训练过程
        if not combine_opt.no_train:
            print('train at epoch {}'.format(epoch))
            combine_net.train()
            # 记录每个 batch 的时间、数据加载时间、损失值和准确率。
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            end_time = time.time()

            combined_loader = zip(in_train_loader, out_train_loader)  # 待测试
            for i, ((in_inputs, in_targets), (out_inputs, out_targets)) in enumerate(combined_loader):

                # 更新 data_time 计时器，记录数据加载时间。
                data_time.update(time.time() - end_time)
                # 将 targets 移动到 GPU 上，以便在 GPU 上进行计算。
                if not combine_opt.no_cuda:
                    in_targets = in_targets.cuda(non_blocking=True)
                    out__targets = out_targets.cuda(non_blocking=True)
                # 将 inputs 和 targets 封装为 Variable 对象，使其可以自动计算梯度。
                in_inputs = Variable(in_inputs)
                in_targets = Variable(in_targets)
                out_inputs = Variable(out_inputs)
                out_targets = Variable(out_targets)

                outputs = combine_net((in_inputs, out_inputs))  # 计算模型输出
                outputs = outputs.to("cuda:0")

                in_targets = in_targets.to("cuda:0")

                # 使用损失函数计算输出 outputs 与目标值 targets 之间的损失、计算模型在当前 batch 上的准确率。
                loss = combine_criterion(outputs, in_targets)
                acc = calculate_accuracy(outputs, in_targets)

                losses.update(loss.item(), in_inputs.size(0))
                accuracies.update(acc, in_inputs.size(0))

                # print('in_inputs.size(0):', in_inputs.size(0))
                # print('out_inputs.size(0):', out_inputs.size(0))

                combine_optimizer.zero_grad()  # 清空优化器的梯度。
                loss.backward()  # 使用反向传播计算损失对模型参数的梯度。
                combine_optimizer.step()  # 使用优化器更新模型参数。

                # 更新 batch_time 计时器，记录当前 batch 的训练时间。
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                combine_train_batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(in_train_loader) + (i + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': combine_optimizer.param_groups[0]['lr']
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
                        len(in_train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accuracies))

            combine_train_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': combine_optimizer.param_groups[0]['lr']
            })

            # 如果达到了设定的检查点间隔，会执行以下代码：
            if epoch % combine_opt.checkpoint == 0:
                # 构建保存文件的路径 save_file_path，将其命名为 'save_{}.pth'，其中 {} 会使用当前 epoch 数填充。
                save_file_path = os.path.join(combine_opt.result_path, 'combine_save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': combine_opt.arch,
                    'state_dict': combine_net.state_dict(),
                    'optimizer': combine_optimizer.state_dict(),
                }
                torch.save(states, save_file_path)


        if not combine_opt.no_val:
            print('Validation at epoch {}'.format(epoch))
            combine_net.eval()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            end_time = time.time()

            # 初始化一个全零矩阵 conf_mat，用于存储混淆矩阵。
            conf_mat = torch.zeros(combine_opt.n_finetune_classes, combine_opt.n_finetune_classes)
            output_file = []

            combined_loader = zip(in_train_loader, out_train_loader)  # 待测试
            for i, ((in_inputs, in_targets), (out_inputs, out_targets)) in enumerate(combined_loader):

                data_time.update(time.time() - end_time) # 更新数据加载时间 data_time。
                if not combine_opt.no_cuda:
                    in_targets = in_targets.cuda(non_blocking=True)
                    out__targets = out_targets.cuda(non_blocking=True)

                # 将输入数据和目标变量封装成PyTorch变量（Variable），并设置为 volatile=True，表示不需要计算梯度。
                with torch.no_grad():
                    in_inputs = in_inputs
                    out_inputs = out_inputs
                with torch.no_grad():
                    in_targets = in_targets
                    out_targets = out_targets

                outputs = combine_net((in_inputs, out_inputs))  # 计算模型输出
                outputs = outputs.to("cuda:0")

                # 损失值、输出的准确率
                loss = combine_criterion(outputs, in_targets)
                acc = calculate_accuracy(outputs, in_targets)

                ### print out the confusion matrix
                _,pred = torch.max(outputs,1)
                for t,p in zip(in_targets.view(-1), pred.view(-1)):
                    conf_mat[t,p] += 1

                # 根据预测结果和目标计算混淆矩阵，并将结果累加到 conf_mat。
                losses.update(loss.item(), in_inputs.size(0))
                # 更新损失值 losses 和准确率 accuracies。
                accuracies.update(acc, in_inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                # 打印当前批次的训练信息，包括当前 epoch、批次索引、总批次数、批次时间、数据加载时间、损失和准确率。
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                          epoch,
                          i + 1,
                          len(in_val_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))
            print(conf_mat)# 打印混淆矩阵 conf_mat。

            combine_val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
            # 判断当前模型是否为最佳模型，并更新最佳准确率 best_prec。如果当前准确率高于最佳准确率，则设置 is_best 为 True。
            is_best = accuracies.avg > best_prec
            best_prec = max(accuracies.avg, best_prec)
            print('\n The best prec is %.4f' % best_prec)

            # 如果 is_best 为 True，则保存模型状态，包括当前 epoch、模型架构 opt.arch、模型的状态字典和优化器的状态字典
            if is_best:
                states = {
                    'epoch': epoch + 1,
                    'arch': combine_opt.arch,
                    'state_dict': combine_net.state_dict(),
                    'optimizer': combine_optimizer.state_dict(),
                 }
                save_file_path = os.path.join(combine_opt.result_path, 'save_best_combine.pth')
                torch.save(states, save_file_path)


        # 如果训练且验证，则执行 scheduler.step()，用于调整学习率。
        if not combine_opt.no_train and not combine_opt.no_val:
            combine_scheduler.step()