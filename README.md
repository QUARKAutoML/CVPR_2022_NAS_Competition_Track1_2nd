# CVPR_2022_NAS_Competition_Track1_2nd

本仓库是CVPR 2022 NAS 赛道1第2名方案代码

## 方案复现流程
### 1.1 启动训练及推理的方式：  
#### 训练超网：  
    第1步：  
    sh train.sh  
    第2步：  
    sh train_finetune_nsas.sh  
    第3步：  
    sh train_finetune_few_shot_with_nsas_multi_stage.sh  
**备注：**  
*1、请在根目录下准备好gast-0.4.0-py3-none-any.whl和paddlepaddle_gpu-2.1.3.post101-cp37-cp37m-linux_x86_64.whl，文件太大，不便上传*  
*2、 参数--resume，--note请根据实际情况修改*  
*3、第3步在第2步基础上resume， 第2步在第1步基础上resume*  
#### 结果评估：  
    python result_supernet_calibrate_bn_split_custom.py --split_stage 34 --note “” --resume save_temp/nsas_2_finetune_1_path_few_shot_with_nsas_multi_stage_6969_4_2_4_lr0003-model.th   
**备注：**  
*1、因为校准存在随机性，所以结果会有微小差异*  
*2、根据实际需求修改Line182~Line188（result_supernet_calibrate_bn_split_custom.py）*  
*3、参数--resume，--note请根据实际情况修改*  

### 1.2 最后一次提交的Json  
    results/paddle-result_4_path_d_5_15_30_v3_c_5_15_40_nsas_ft_30e_few_shot_distill.json  

## 参考
https://github.com/JiahuiYu/slimmable_networks  
https://github.com/mit-han-lab/once-for-all  
https://github.com/xiteng01/CVPR_2022_Track1_demo  
https://github.com/MiaoZhang0525/NSAS_FOR_CVPR  
