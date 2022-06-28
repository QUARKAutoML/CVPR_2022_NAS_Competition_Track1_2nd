#!/bin/bash

pip install gast-0.4.0-py3-none-any.whl
pip install paddlepaddle_gpu-2.1.3.post101-cp37-cp37m-linux_x86_64.whl

python train_supernet_split_separate_multi_stage.py --note nsas_2_finetune_1_path_few_shot_with_nsas_multi_stage_6969_4_2_4_lr0003 --lr 0.003 --warmup 0 --epochs 6 --sample_num_per_step 1 --resume save_temp/4_path_d_5_15_30_v3_c_5_15_40_finetune_1_path_nsas_30epoch-model.th --save_every 3 --resume_type 0 --split_stage 34 --loss_ratio 0.4 0.2 0.4 --distill