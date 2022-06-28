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

import few_shot_depth_model as model
import autoslim
import utils
import arch_nsas as ns
import json

parser = argparse.ArgumentParser(description='')
parser.add_argument('--arch', '-a', metavar='ARCH', default='558511111111111111111111111111111111111111111111111')
parser.add_argument('--resume', default='save_temp/*.th', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_type', default=0, type=int, metavar='N', help='resume model type')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--save_dir', help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=30)
parser.add_argument('--file_name', default='./Track1_final_archs.json', type=str, metavar='PATH', help='path to json file')
parser.add_argument('--image_dir', default='/home/notebook/data/public/Dataset/overt/eb14f1a2f3fa526fa96111e13fb33f18r/imagenet-pytorch', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--warmup', default=2, type=int)
parser.add_argument('--sample_num_per_step', default=4, type=int)
parser.add_argument('--note', type=str, default='', help='experiment note')
parser.add_argument('--local_rank', default=0, type=int, help='local gpu id (default: 0)')
parser.add_argument('--depth_end_epoch', default=0, type=int, help='ps the end of depth shrinking')
parser.add_argument('--split_stage', default=10, type=int, help='the stage to split')
parser.add_argument('--distill', action='store_true', default=False, help='use distillation')
parser.add_argument("--loss_ratio", nargs='+', type=float, default=[])

args = parser.parse_args()

if dist.get_rank() == 0:
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = utils.get_logger(os.path.join(args.save_dir, "{}-{}-{}.log".format('resnet48', args.note, time.strftime("%Y%m%d-%H%M%S"))))

    aliyun_dir='/home/notebook/data/group_cpfs/imagenet-pytorch'
    if os.path.exists(aliyun_dir):
        print('use aliyun datasheet')
        args.image_dir=aliyun_dir

    logger.info("args:{}".format(args))


def train(train_loader, model, criterion, optimizer, epoch, warmup_step, supernet_id, arch_nsas_list, teacher_model, distill, map_dict):
    # switch to train mode
    model.train()

    prec1 = paddle.metric.Accuracy(topk=(1,), name='top1')

    for i, (input, target) in enumerate(train_loader):
        # input_var = paddle.to_tensor(input)
        # target_var = paddle.to_tensor(target)
        cur = epoch*len(train_loader)+i

        input_var = input
        target_var = target

        # compute gradient and do SGD step
        optimizer.clear_grad()
        for j in range(args.sample_num_per_step):
            
            #depth_coding = autoslim.get_depth_of_stage(supernet_id+2, args.split_stage)
            #channel_coding = autoslim.get_channels_no_ps(depth_coding)
            #arch = depth_coding+channel_coding
            arch = model._layers.samplers_total_archs[j][cur]
            if dist.get_rank() == 0:
                if arch[2:4] not in map_dict:
                    print(arch)
                    print(map_dict)
                    RuntimeError("arch not match!")
            # compute output
            output = model(input_var, arch)

            loss = args.loss_ratio[0]*criterion(output, target_var)

            # measure accuracy and record loss
            cur_prec1 = prec1.compute(output, target)
            cur_prec1 = prec1.update(cur_prec1)

            #loss.backward()

            if i % 500 == 0: #refresh select arch per 500 iter
                arch_nsas_list[supernet_id].set_select_arch(arch_nsas_list[supernet_id].arch_list)
                if dist.get_rank() == 0:
                    print('supernet_id step is {}'.format(supernet_id))
                    print(arch_nsas_list[supernet_id].select_arch)

            nsas_alp = args.loss_ratio[1]
            if distill:
                # if dist.get_rank() == 0:
                #     print('use distill')
                with paddle.no_grad():
                    output_t = teacher_model.clean_forward(input_var)
                loss_distill = paddle.nn.functional.cross_entropy(output, paddle.nn.functional.softmax(output_t), soft_label=True)
                loss_distill = loss_distill * args.loss_ratio[2]
                loss += loss_distill
                #loss_distill.backward()
                # nsas_alp = 0.3

            loss.backward()

            arch_nsas_list[supernet_id].update_recent_arch(arch)
            arch_nsas_list[supernet_id].update_select_arch(arch)
            arch_temp = arch
            for arch in arch_nsas_list[supernet_id].select_arch:
                if dist.get_rank() == 0:
                    if arch[2:4] not in map_dict:
                        print(arch)
                        print(map_dict)
                        RuntimeError("arch not match!")
                output = model(input_var, arch)
                loss = (nsas_alp/arch_nsas_list[supernet_id].select_num)*criterion(output, target_var)
                loss.backward()

            arch = arch_temp
            if dist.get_rank() == 0:
                if i % args.print_freq == 0:
                    logger.info('Epoch: [{}][{}/{}] path:{} arch:{}\t'
                        'Loss {}\t'
                        'Prec@1 {}'.format(
                            epoch, i, len(train_loader), j, arch, loss.numpy()[0], cur_prec1))

        if len(train_loader)*epoch+i<warmup_step:
            optimizer._learning_rate.step()

        optimizer.step()


    # for i in range(4):
    #     cur = epoch*len(train_loader)+i
    #     arch = model._layers.samplers_total_archs[0][cur]
    #     print(arch)     

def validate(val_loader, model, criterion):
    batch_time = utils.AverageMeter()
    losses = [utils.AverageMeter() for _ in range(1 if args.sample_num_per_step==1 else 2)]
    top1 = [paddle.metric.Accuracy(topk=(1,), name='top1') for _ in range(1 if args.sample_num_per_step==1 else 2)]

    # switch to evaluate mode
    # model.eval()

    end = time.time()
    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input
            target_var = target

            for j in range(1 if args.sample_num_per_step==1 else 2):
                if j==0:  ## Widest network
                    arch='558511111111111111111111111111111111111111111111111'
                elif j==1: ## Thinnest network
                    arch='222277777000000777700000000000077770000007777000000'
                else:
                    continue

                # compute output
                output = model(input_var, arch)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1 = top1[j].compute(output, target)
                losses[j].update(loss.numpy()[0], input.shape[0])
                top1[j].update(prec1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if dist.get_rank() == 0:
                for j in range(1 if args.sample_num_per_step==1 else 2):
                    if i % args.print_freq == 0:
                        logger.info('Test: [{0}/{1}] path {2}\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss avg:{loss.avg}\t'
                            'Prec@1 avg:{3}'.format(
                                i, len(val_loader), j, top1[j].accumulate(), loss=losses[j], batch_time=batch_time))
    if dist.get_rank() == 0:
        logger.info(' * Prec@1 avg {}'.format(top1[-1].accumulate()))

    return top1[-1].accumulate()


def get_archs(filename):
    archs=[]
    with open(filename) as f:
        result_dict = json.load(f)

    for item_name in result_dict:
        arch_code = result_dict[item_name]['arch']
        arch = arch_code[1:]

        archs.append(arch)
    return archs


def main():
    dist.init_parallel_env()

    paddle.set_device('gpu:{}'.format(dist.ParallelEnv().device_id))

    IMAGE_MEAN = (0.485,0.456,0.406)
    IMAGE_STD = (0.229,0.224,0.225)

    args.lr = args.lr * args.batch_size * dist.get_world_size() / 256
    warmup_step = int(1281024 / (args.batch_size * dist.get_world_size())) * args.warmup
    # warmup_step = args.warmup
    print('lr:{} warmup_step:{} world_size:{}'.format(args.lr, warmup_step, dist.get_world_size()))

    transforms = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToArray(),
        Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    val_transforms = Compose([Resize(256), CenterCrop(224), ToArray(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    train_set = DatasetFolder(os.path.join(args.image_dir, 'train'), transform=transforms)
    val_set = DatasetFolder(os.path.join(args.image_dir, 'val'), transform=val_transforms)

    train_sampler = DistributedBatchSampler(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader = paddle.io.DataLoader(train_set, batch_sampler=train_sampler, places='cpu', return_list=True, num_workers=args.workers)

    val_sampler = DistributedBatchSampler(val_set, batch_size=args.batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(val_set, batch_sampler=val_sampler, places='cpu', return_list=True, num_workers=args.workers)

    if args.split_stage==34:
        supernet_num = 5
    else:
        assert False

    resnet48 = model.Few_Shot_Depth_Model('1'+args.arch, supernet_num)
    resnet48.split_stage=args.split_stage

    resnet48_teacher = []
    if args.distill:
        if dist.get_rank() == 0:
            print('use distill')
        resnet48_teacher = model.Model('1'+args.arch)
        logger.info('teacher load from {}'.format('./resnet48.pdparams'))
        state_dict = paddle.load('./resnet48.pdparams')
        resnet48_teacher.set_state_dict(state_dict)
        resnet48_teacher.eval()

    sp_net_config = supernet()
    super_model = Convert(sp_net_config).convert(resnet48)

    if args.resume:
        if dist.get_rank() == 0:
            logger.info('load from {}'.format(args.resume))
        state_dict = paddle.load(args.resume)
        if args.resume_type == 0:
            super_model.load_state_dict(state_dict)
            if dist.get_rank() == 0:
                print('load normal model')
        elif args.resume_type == 1:
            super_model.set_state_dict(state_dict)
            if dist.get_rank() == 0:
                print('load few_shot model')
        else:
            assert False

    for split_model in super_model.split_models:
        for module in split_model.sublayers():
            if isinstance(module, SuperConv2D):
                split_model.super_modules.append(module)
        if dist.get_rank() == 0:
            logger.info('len of super_modules: {}'.format(len(split_model.super_modules)))

    super_model = paddle.DataParallel(super_model, find_unused_parameters=True)

    criterion = CrossEntropyLoss()

    archs = get_archs('results/CVPR_2022_NAS_Track1_test.json')
    archs_list = [[] for _ in range(supernet_num-1)]

    map_dict = {'22': 0,  '23': 0, '24': 0, \
        '32': 1,  '42': 1, '52': 1, '62':1, '72': 1, '82':1, \
        '83': 2,  '84': 2, '85': 2, \
        '25': 3,  '35': 3, '45': 3, '55':3, '65': 3, '75':3
    }

    if dist.get_rank() == 0:
        print(map_dict)

    for arch in archs:
        stage_depth_code = arch[2:4]
        if stage_depth_code in map_dict:
            idx = map_dict[stage_depth_code]
            archs_list[idx].append(arch)
        # else:
        #     assert False

    arch_nsas_list = []
    for i in range(0, supernet_num-1):
        arch_nsas_list.append(ns.Arch_Nsas_Manager(knn_num=6, select_num=7, recent_num=80, arch_list=archs_list[i]))


    for supernet_id in range(supernet_num-1):
        epochs = args.epochs
        if supernet_id == 1 or supernet_id == 3:
            epochs = int(args.epochs*3/2) #epoch change
        if dist.get_rank() == 0:
            print('supernet_id {} epoch is {}'.format(supernet_id, epochs))

        optimizer = paddle.optimizer.Momentum(learning_rate=LinearWarmup(
        CosineAnnealingDecay(args.lr, epochs), warmup_step, 0., args.lr),
        momentum=args.momentum,
        parameters=super_model.parameters(),
        weight_decay=args.weight_decay)

        total_steps = epochs*len(train_loader)*(args.sample_num_per_step)
        super_model._layers.get_archs_samplers(archs_list[supernet_id], total_steps, args.sample_num_per_step)

        for epoch in range(epochs):
            # train for one epoch
            if dist.get_rank() == 0:
                logger.info('current epoch:{} lr {:.5e}'.format(epoch, optimizer.get_lr()))
            train(train_loader, super_model, criterion, optimizer, epoch, warmup_step, supernet_id, arch_nsas_list, resnet48_teacher, args.distill, map_dict)

            optimizer._learning_rate.step()

            # evaluate on validation set
            prec1 = validate(val_loader, super_model, criterion)

            is_best=True

            if dist.get_rank() == 0:
                if epoch > 0 and (epoch+1) % args.save_every == 0:
                    utils.save_checkpoint(super_model._layers, is_best, filename=os.path.join(args.save_dir, '{}-{}-checkpoint.th'.format(args.note, epoch)))

                utils.save_checkpoint(super_model._layers, is_best, filename=os.path.join(args.save_dir, '{}-model.th'.format(args.note)))

if __name__=="__main__":
    # dist.spawn(main, args=(), nprocs=1, gpus='0')
    # dist.spawn(main, args=(), nprocs=4, gpus='0,1,2,3')
    dist.spawn(main, args=(), nprocs=8, gpus='0,1,2,3,4,5,6,7')