#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PYTHON="C:\ProgramData\Anaconda3\envs\pytorch_gpu\python.exe"  # 指定可执行 Python 解释器的路径

$PYTHON main_inside.py \
   --root_path "D:/XHX/Driver-Intention-Prediction-master" \
   --video_path "D:/XHX/Driver-Intention-Prediction-master/Brain4Cars/face_camera" \
   --annotation_path "D:/XHX/Driver-Intention-Prediction-master/datasets/annotation" \
   --result_path "results" \
   --dataset "Brain4cars_Inside" \
   --n_classes 5 \
   --n_finetune_classes 5 \
   --ft_begin_index 4 \
   --model "resnet" \
   --model_depth 50 \
   --resnet_shortcut "B" \
   --batch_size 12 \
   --n_threads 4 \
   --checkpoint 5 \
   --n_epochs 1 \
   --resnext_cardinality 32 \
   --begin_epoch 1 \
   --sample_duration 16 \
   --end_second 5 \
   --train_crop "driver focus" \
   --n_scales 3 \
   --learning_rate 0.1 \
   --n_fold 0 \
   --pretrain_path "D:/XHX/Driver-Intention-Prediction-master/path_to_models/save_best_3DResNet50.pth" \
