#!/bin/bash

pip install gast-0.4.0-py3-none-any.whl
pip install paddlepaddle_gpu-2.1.3.post101-cp37-cp37m-linux_x86_64.whl

python train_supernet_nsas.py --note 4_path_d_5_15_30_v3_c_5_15_40_finetune_1_path_nsas_30epoch --lr 0.001 --warmup 0 --epochs 30 --sample_num_per_step 1 --resume save_temp/4_path_d_5_15_30_v3_c_5_15_40-model.th --save_every 10
