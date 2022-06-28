import os
import time
import argparse
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.nn import CrossEntropyLoss
from paddle.vision.transforms import RandomHorizontalFlip, RandomResizedCrop, Compose, Normalize, CenterCrop, Resize
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.vision.datasets import DatasetFolder
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, LinearWarmup

from hnas.utils.transforms import ToArray
from supernet_convert import Convert, supernet
from super_layers import SuperConv2D, SuperBatchNorm2D

import json
import pickle
import few_shot_depth_model as model
import autoslim
import utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--arch', '-a', metavar='ARCH', default='558511111111111111111111111111111111111111111111111')
parser.add_argument('--resume', default='save_temp/*.th', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--save_dir', help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=30)
parser.add_argument('--file_name', default='./Track1_final_archs.json', type=str, metavar='PATH', help='path to json file')
parser.add_argument('--image_dir', default='/home/notebook/data/public/Dataset/overt/eb14f1a2f3fa526fa96111e13fb33f18r/imagenet-pytorch', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--warmup', default=2, type=int)
parser.add_argument('--sample_num_per_step', default=4, type=int)
parser.add_argument('--note', type=str, default='', help='experiment note')
parser.add_argument('--local_rank', default=0, type=int, help='local gpu id (default: 0)')
parser.add_argument('--gpus_per_node', default=0, type=int,
                    help='local gpu id (default: 0)')
parser.add_argument('--rank', default=0, type=int,
                    help='cur node id (default: 0)')
parser.add_argument('--nodes_num', default=1, type=int,
                    help='the number of nodes(default: 1)')
parser.add_argument('--start_num', default=0, type=int, help='the number of arch to skip evaluate')
parser.add_argument('--evaluate_arch_num', default=5000, type=int, help='the number of arch to evaluate')  
parser.add_argument('--split_stage', default=10, type=int, help='the stage to split')
parser.add_argument('--cali_fix', action='store_true', default=False, help='fix cali')

args = parser.parse_args()

# Check the save_dir exists or not
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

logger = utils.get_logger(os.path.join(args.save_dir, "{}-{}-{}.log".format('result_resnet48', args.note, time.strftime("%Y%m%d-%H%M%S"))))
logger.info("args:{}".format(args))

def validate(val_loader, model, criterion, arch):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = paddle.metric.Accuracy(topk=(1,), name='top1')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input
            target_var = target

            # compute output8
            output = model(input_var, arch)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1 = top1.compute(output, target)
            losses.update(loss.numpy()[0], input.shape[0])
            top1.update(prec1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss avg:{loss.avg}\t'
                    'Prec@1 avg:{2}'.format(
                        i, len(val_loader), top1.accumulate(), loss=losses, batch_time=batch_time))

    logger.info(' * Prec@1 avg {}'.format(top1.accumulate()))

    return top1.accumulate()

def get_arch_dict(filename):
    result_dict = None
    with open(filename) as f:
        result_dict = json.load(f)

    return result_dict

def main():

    IMAGE_MEAN = (0.485,0.456,0.406)
    IMAGE_STD = (0.229,0.224,0.225)

    if args.cali_fix:
        transforms = Compose([
            Resize(256),
            CenterCrop(224),
            ToArray(),
            Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
    else:
        transforms = Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToArray(),
            Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
    val_transforms = Compose([Resize(256), CenterCrop(224), ToArray(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    train_set = DatasetFolder(os.path.join(args.image_dir, 'train'), transform=transforms)
    val_set = DatasetFolder(os.path.join(args.image_dir, 'val'), transform=val_transforms)

    if args.cali_fix:
        train_loader = paddle.io.DataLoader(train_set, places='cpu', return_list=True, batch_size=args.batch_size, shuffle=False,  num_workers=args.workers)
    else:
        train_loader = paddle.io.DataLoader(train_set, places='cpu', return_list=True, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
    val_loader = paddle.io.DataLoader(val_set, places='cpu', return_list=True, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.split_stage==34:
        supernet_num = 5
    else:
        assert False

    resnet48 = model.Few_Shot_Depth_Model('1'+args.arch, supernet_num)
    resnet48.split_stage=args.split_stage

    sp_net_config = supernet()
    super_model = Convert(sp_net_config).convert(resnet48)

    if args.resume:
        if dist.get_rank() == 0:
            logger.info('load from {}'.format(args.resume))
        state_dict = paddle.load(args.resume)
        super_model.set_state_dict(state_dict)

    for split_model in super_model.split_models:
        for module in split_model.sublayers():
            if isinstance(module, SuperConv2D):
                split_model.super_modules.append(module)
        logger.info('len of super_modules: {}'.format(len(split_model.super_modules)))

    criterion = CrossEntropyLoss()

    ## store training data
    calibrate_steps=50

    train_loader_calibrate=[]
    for i, (input, target) in enumerate(train_loader):
        if i<calibrate_steps:
            train_loader_calibrate.append(input)
        else:
            break

    def calibrate_bn(arch):
        super_model.train()
        with paddle.no_grad():
            for input_var in train_loader_calibrate:
                _ = super_model(input_var, arch)

    result_dict = get_arch_dict('./results/CVPR_2022_NAS_Track1_test.json')
    result_dict_part = result_dict

    for item_name in result_dict_part:
        # if True:
        if item_name in ['arch717', 'arch213', 'arch470', \
                        'arch471', 'arch748', 'arch837', \
                        'arch781', 'arch875', 'arch90', \
                        'arch615', 'arch788', 'arch1278', \
                        'arch172', 'arch425', 'arch555'
                        ]:
            logger.info(item_name)

            if int(item_name[4:])<args.start_num:
                result_dict_part[item_name]['acc'] = 0.1
                continue

            arch_code = result_dict[item_name]['arch']
            arch = arch_code[1:]

            # prec1 = 0.1
            calibrate_bn(arch)
            prec1 = validate(val_loader, super_model, criterion, arch)

            result_dict_part[item_name]['acc'] = prec1

    result_acc = json.dumps(result_dict_part)
    f = open('./results/paddle-result_{}.json'.format(args.note),'w')
    f.write(result_acc)

if __name__=="__main__":
    main()