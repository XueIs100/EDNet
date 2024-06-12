#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PYTHON="C:\ProgramData\Anaconda3\envs\pytorch_gpu\python.exe"  # 指定可执行 Python 解释器的路径

$PYTHON main_outside.py \
   --root_path "D:/XHX/Driver-Intention-Prediction-master" \
   --video_path "D:/XHX/Driver-Intention-Prediction-master/Brain4cars/flownet2_road_camera" \
   --annotation_path "D:/XHX/Driver-Intention-Prediction-master/datasets/out_annotation" \
   --result_path "results_outside" \
   --dataset "Brain4cars_Outside" \
   --batch_size 8 \
   --n_threads 4 \
   --checkpoint 5  \
   --n_epochs 50 \
   --begin_epoch 1 \
   --sample_duration 5 \
   --end_second 5 \
   --interval 5 \
   --n_scales 1 \
   --learning_rate 0.01 \
   --norm_value 255 \
   --n_fold 0 \
   --pretrain_path "D:/XHX/Driver-Intention-Prediction-master/path_to_models/0_fifth.pth" \
   --no_train \