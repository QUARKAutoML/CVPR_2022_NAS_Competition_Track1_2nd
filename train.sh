#!/bin/bash

pip install gast-0.4.0-py3-none-any.whl
pip install paddlepaddle_gpu-2.1.3.post101-cp37-cp37m-linux_x86_64.whl

python train_supernet.py --note 4_path_d_5_15_30_v3_c_5_15_40 --lr 0.001 --warmup 0 --epochs 70 --resume ./resnet48.pdparams --sample_num_per_step 4 --depth_end_epoch 30 --save_epochs 14 29 34 44