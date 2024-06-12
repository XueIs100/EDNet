from datasets.Brain4cars import Brain4cars_Inside, Brain4cars_Outside
#引入Brain4cars_Inside 和 Brain4cars_Outside 两个数据加载器。

# 1. 这段代码接受一个opt参数以及其他几个参数，用于生成训练数据集。
def get_training_set(opt, spatial_transform, horizontal_flip, temporal_transform,
                     target_transform):
    # 1. 代码断言（assert）opt.dataset的值必须是'Brain4cars_Inside'或者'Brain4cars_Outside'，否则会引发异常。
    assert opt.dataset in ['Brain4cars_Inside', 'Brain4cars_Outside']

    # 2.. 如果为'Brain4cars_Inside'，则创建一个Brain4cars_Inside对象，传入以下参数：
    # 视频文件的路径、注释文件的路径、训练集、交叉验证的折数、【end_second：视频处理的结束时间（单位：秒）】、数据集的采样率
    # 空间变换函数、是否进行水平翻转、时间变换函数、目标变换函数
    if opt.dataset == 'Brain4cars_Inside':
        training_data = Brain4cars_Inside(
                opt.video_path,
                opt.annotation_path,
                'training',
                opt.n_fold,
                opt.end_second,
                1,
                spatial_transform=spatial_transform,
                horizontal_flip=horizontal_flip,
                temporal_transform=temporal_transform,
                target_transform=target_transform)

    # 3. 如果opt.dataset为'Brain4cars_Outside'，则创建一个Brain4cars_Outside对象，传入以下参数：
    # opt.video_path：视频文件的路径
    # opt.annotation_path：注释文件的路径
    # 'training'：数据集类型，这里是训练集
    # opt.n_fold：交叉验证的折数
    # opt.end_second：视频处理的结束时间（单位：秒）
    # 10：数据集的采样率
    # spatial_transform：空间变换函数
    # horizontal_flip：是否进行水平翻转
    # target_transform：目标变换函数
    # temporal_transform：时间变换函数
    elif opt.dataset == 'Brain4cars_Outside':
        training_data = Brain4cars_Outside(
            opt.video_path,
            opt.annotation_path,
            'training',
            opt.n_fold,
            opt.end_second,
            10,
            spatial_transform=spatial_transform,
            horizontal_flip=horizontal_flip,
            target_transform=target_transform,
            temporal_transform=temporal_transform)
    # 4. 返回生成的训练数据集对象。
    return training_data



# 2. 这段代码接受一个opt参数以及其他几个参数，用于生成验证数据集。
def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['Brain4cars_Inside', 'Brain4cars_Outside']
    # 1. 如果opt.dataset为'Brain4cars_Inside'，则创建一个Brain4cars_Inside对象，传入以下参数：
    if opt.dataset == 'Brain4cars_Inside':
        # opt.video_path：视频文件的路径
        # opt.annotation_path：注释文件的路径
        # 'validation'：数据集类型，这里是验证集
        # opt.n_fold：交叉验证的折数
        # opt.end_second：视频处理的结束时间（单位：秒）
        # opt.n_val_samples：验证数据集的样本数
        # spatial_transform：空间变换函数
        # None：水平翻转为空
        # temporal_transform：时间变换函数
        # target_transform：目标变换函数
        # sample_duration=opt.sample_duration：样本持续时间，通过opt.sample_duration指定
        validation_data = Brain4cars_Inside(
                opt.video_path,
                opt.annotation_path,
                'validation',
                opt.n_fold,
                opt.end_second,
                opt.n_val_samples,
                spatial_transform,
                None,
                temporal_transform,
                target_transform,
                sample_duration=opt.sample_duration)
    # 2. 如果opt.dataset为'Brain4cars_Outside'，则创建一个Brain4cars_Outside对象，传入以下参数：
    elif opt.dataset == 'Brain4cars_Outside':
        # opt.video_path：视频文件的路径
        # opt.annotation_path：注释文件的路径
        # 'validation'：数据集类型，这里是验证集
        # opt.n_fold：交叉验证的折数
        # opt.end_second：视频处理的结束时间（单位：秒）
        # opt.n_val_samples：验证数据集的样本数
        # spatial_transform：空间变换函数
        # horizontal_flip=None：水平翻转为空
        # temporal_transform：时间变换函数
        # target_transform：目标变换函数
        # sample_duration=opt.sample_duration：样本持续时间，通过opt.sample_duration指定
        validation_data = Brain4cars_Outside(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_fold,
            opt.end_second,
            opt.n_val_samples,
            spatial_transform=spatial_transform,
            horizontal_flip=None,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    # 3. 最后，返回生成的验证数据集对象。
    return validation_data