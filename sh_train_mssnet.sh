#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_dp.py \
                --train_datalist '/home/ubuntu/MSSNET_train_final/MSSNet/datalist/datalist_kitti_train.txt'\
                --data_root_dir '/home/ubuntu/MSSNET_train_final/MSSNet/dataset'\
                --checkdir './checkpoint/MSSNet'\
                --max_epoch 3000\
                --wf 20\
                --scale 40\
                --vscale 40\
                --mgpu
