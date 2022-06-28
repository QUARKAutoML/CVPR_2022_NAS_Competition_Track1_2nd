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
import model
import autoslim
import utils
import arch_nsas as ns

parser = argparse.ArgumentParser(description='')
parser.add_argument('--arch', '-a', metavar='ARCH', default='558511111111111111111111111111111111111111111111111')
parser.add_argument('--resume', default='save_temp/*.th', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--save_dir', help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=5)
parser.add_argument('--file_name', default='./Track1_final_archs.json', type=str, metavar='PATH', help='path to json file')
parser.add_argument('--image_dir', default='/home/notebook/data/public/Dataset/overt/eb14f1a2f3fa526fa96111e13fb33f18r/imagenet-pytorch', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--warmup', default=2, type=int)
parser.add_argument('--sample_num_per_step', default=1, type=int)
parser.add_argument('--note', type=str, default='4_path_no_ps_few_shot_5000_scratch_nsas', help='experiment note')
parser.add_argument('--local_rank', default=0, type=int, help='local gpu id (default: 0)')
parser.add_argument('--depth_end_epoch', default=0, type=int, help='ps the end of depth shrinking')

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


def get_archs(filename):
    archs=[]
    with open(filename) as f:
        result_dict = json.load(f)

    for item_name in result_dict:
        arch_code = result_dict[item_name]['arch']
        arch = arch_code[1:]

        archs.append(arch)
    return archs


def train(train_loader, model, criterion, optimizer, epoch, warmup_step, arch_nsas):
    # switch to train mode
    model.train()

    prec1 = paddle.metric.Accuracy(topk=(1,), name='top1')
    for i, (input, target) in enumerate(train_loader):
        # input_var = paddle.to_tensor(input)
        # target_var = paddle.to_tensor(target)

        input_var = input
        target_var = target

        cur = epoch*len(train_loader)+i
        # compute gradient and do SGD step
        optimizer.clear_grad()
        for j in range(args.sample_num_per_step):
            # depth_coding = autoslim.get_depth_randomly(epoch)
            # channel_coding = autoslim.get_channels_randomly(epoch, args.depth_end_epoch, depth_coding)
            # arch = depth_coding+channel_coding

            arch = model._layers.samplers_total_archs[j][cur]

            # compute output
            output = model(input_var, arch)

            loss = 0.5*criterion(output, target_var)
            loss.backward()
            # measure accuracy and record loss
            cur_prec1 = prec1.compute(output, target)
            cur_prec1 = prec1.update(cur_prec1)

            if i % 500 == 0: #refresh select arch per 500 iter
                arch_nsas.set_select_arch(arch_nsas.arch_list)

            arch_nsas.update_recent_arch(arch)
            arch_nsas.update_select_arch(arch)

            for arch in arch_nsas.select_arch:
                output = model(input_var, arch)
                loss = (0.5/arch_nsas.select_num)*criterion(output, target_var)
                loss.backward()

            # if j == 0:
            #     if dist.get_rank() == 0:
            #         print('gpu0: arch {} train_loader {} target_var {}'.format(arch, len(train_loader), target_var.shape[0]))
            #     if dist.get_rank() == 1:
            #         print('gpu1: arch {} train_loader {} target_var {}'.format(arch, len(train_loader), target_var.shape[0]))

            if dist.get_rank() == 0:
                if i % args.print_freq == 0:
                    logger.info('Epoch: [{}][{}/{}] path:{} arch:{}\t'
                        'Loss {}\t'
                        'Prec@1 {}'.format(
                            epoch, i, len(train_loader), j, arch, loss.numpy()[0], cur_prec1))

        if len(train_loader)*epoch+i < warmup_step:
            optimizer._learning_rate.step()

        # for module in model.sublayers():
        #     if isinstance(module, SuperBatchNorm2D):
        #         print('mean:{}'.format(module._mean))
        #         print('variance:{}'.format(module._variance))
        #         break

        optimizer.step()


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
    # train_set=paddle.io.Subset(train_set, list(range(dist.get_world_size()*args.batch_size)))
    # val_set=paddle.io.Subset(val_set, list(range(dist.get_world_size()*args.batch_size)))

    train_sampler = DistributedBatchSampler(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader = paddle.io.DataLoader(train_set, batch_sampler=train_sampler, places='cpu', return_list=True, num_workers=args.workers)

    val_sampler = DistributedBatchSampler(val_set, batch_size=args.batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(val_set, batch_sampler=val_sampler, places='cpu', return_list=True, num_workers=args.workers)

    resnet48 = model.Model('1'+args.arch)

    sp_net_config = supernet()
    super_model = Convert(sp_net_config).convert(resnet48)

    if args.resume:
        logger.info('load from {}'.format(args.resume))
        state_dict = paddle.load(args.resume)
        super_model.set_state_dict(state_dict)

    super_model = paddle.DataParallel(super_model, find_unused_parameters=True)

    for module in super_model.sublayers():
        if isinstance(module, SuperConv2D):
            super_model._layers.super_modules.append(module)
    if dist.get_rank() == 0:
        logger.info('len of super_modules: {}'.format(len(super_model._layers.super_modules)))

    optimizer = paddle.optimizer.Momentum(
            learning_rate=LinearWarmup(
            CosineAnnealingDecay(args.lr, args.epochs), warmup_step, 0., args.lr),
            momentum=args.momentum,
            parameters=super_model.parameters(),
            weight_decay=args.weight_decay)

    criterion = CrossEntropyLoss()

    archs = get_archs('results/CVPR_2022_NAS_Track1_test.json')

    arch_nsas = ns.Arch_Nsas_Manager(knn_num=9, select_num=10, recent_num=100, arch_list=archs)

    total_steps = args.epochs*len(train_loader)*(args.sample_num_per_step)
    super_model._layers.get_archs_samplers(archs, total_steps, args.sample_num_per_step)

    for epoch in range(args.epochs):
        # train for one epoch
        if dist.get_rank() == 0:
            logger.info('current lr {:.5e}'.format(optimizer.get_lr()))
        train(train_loader, super_model, criterion, optimizer, epoch, warmup_step, arch_nsas)

        optimizer._learning_rate.step()
        #lr_scheduler.on_epoch_end(epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, super_model, criterion)

        best_prec1=0
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if dist.get_rank() == 0:
            if epoch > 0 and (epoch+1) % args.save_every == 0:
                utils.save_checkpoint(super_model._layers, is_best, filename=os.path.join(args.save_dir, '{}-{}-checkpoint.th'.format(args.note, epoch)))

            utils.save_checkpoint(super_model._layers, is_best, filename=os.path.join(args.save_dir, '{}-model.th'.format(args.note)))


if __name__=="__main__":
    #dist.spawn(main, args=(), nprocs=1, gpus='0')
    #dist.spawn(main, args=(), nprocs=2, gpus='0,1')
    #dist.spawn(main, args=(), nprocs=4, gpus='0,1,2,3')
    dist.spawn(main, args=(), nprocs=8, gpus='0,1,2,3,4,5,6,7')
    # main()
