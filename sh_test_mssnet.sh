#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
                --test_datalist '/home/ubuntu/MSSNET_train_final/MSSNet/datalist/datalist_kitti_test_crop.txt'\
                --data_root_dir './dataset'\
                --load_dir '/home/ubuntu/MSSNET_train_final/MSSNet/checkpoint/MSSNet/model_03000E_gopro.pt'\
                --outdir './result/MSSNet_pretrained'\
                --wf 20\
                --scale 40\
                --vscale 40\
                --is_eval\
                --is_save
