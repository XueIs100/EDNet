# 1. 公共库函数
import torch
from torch import nn

# 2. 目录中已有的py文件
from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet

def generate_model(opt):
    # 2.3 确保opt.model的取值在 'resnet'、'preresnet'、'wideresnet'、'resnext' 或 'densenet' 中。否则触发AssertionError异常。
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    # 2.4 resnet-10, 18, 34, 50, 101, 152, 200
    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        # 2.4.1 models目录中已有的py文件
        from models.resnet import get_fine_tuning_parameters

        # 2.4.2 构建一个 ResNet-X 模型，并指定模型的一些超参数。
        #      `num_classes`：模型最后一层输出的类别数目，即分类任务的类别数目。
        #      `shortcut_type`：残差块中采用的捷径连接类型，可以选择 "A" 或 "B"。
        #      `sample_size`：输入视频帧的空间尺寸。
        #      `sample_duration`：输入视频帧的时间长度。
        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    # 2.5 wideresnet-50
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        from models.wide_resnet import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    # 2.6 resnext-50, 101, 152
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    # 2.7 preresnet-18, 34, 50, 101, 152, 200
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        from models.pre_act_resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    # 2.8 densenet-121, 169, 201, 264
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        from models.densenet import get_fine_tuning_parameters

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    # 2.9  如果使用GPU，`opt.no_cuda`表示禁用 GPU 加速。
    if not opt.no_cuda:
        # 2.9.1 将模型model加载到 GPU 的显存中。
        model = model.cuda()
        # 2.9.2 使用 `nn.DataParallel` 对模型进行并行化处理。通过该函数，将模型复制到多个 GPU 上，并在每个 GPU 上执行相同的操作。
        model = nn.DataParallel(model, device_ids=None) # `device_ids=None` 表示自动检测并使用当前系统中所有可用的 GPU 设备。
        # 2.9.3 如果存在pretrain_path。
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            # 2.9.3.1 使用断言语句确保当前模型的架构（opt.arch）与预训练模型中存储的架构（pretrain['arch']）相同。
            assert opt.arch == pretrain['arch']
            # 2.9.3.2 将预训练模型中的参数载入到当前模型中。
            # pretrain['state_dict'] 是一个字典，包含了预训练模型中的所有参数和对应的键值对。
            # 该函数会将预训练模型的参数复制到当前模型中，并覆盖掉当前模型原有的参数。
            # 这样，当前模型便拥有了与预训练模型相同的参数，可以直接进行 fine-tuning。
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

    return model, model.parameters()
