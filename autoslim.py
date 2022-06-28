import numpy as np
import paddle
import paddle.nn as nn
import copy
import random

random.seed(123)

def get_depth_randomly(epoch):
    depth_coding=[]
    depth_end=[5, 5, 8, 5]

    # setting 15
    if epoch<5:
        depth_start=[4, 4, 6, 2]
    elif epoch<15:
        depth_start=[3, 3, 4, 2]
    else: # 30
        depth_start=[2, 2, 2, 2]

    for s, e in zip(depth_start, depth_end):
        depth_coding.append(random.choice(list(range(s, e+1))))

    return ''.join([str(d_c) for d_c in depth_coding])

def get_channels_randomly(epoch, depth_end_epoch, depth_coding):
    channels_coding=[]
    depth_end=[5, 5, 8, 5]

    channel_start = [1, 1, 1]

    # setting 15
    if epoch<35:
       channel_end = [3, 3, 3]
    elif epoch<45:
       channel_end = [3, 5, 5]
    else:
       channel_end = [3, 5, 7]

    for i in range(len(depth_end)):
        for j in range(1, int(depth_end[i])+1):
            if j<=int(depth_coding[i]):
                if epoch<depth_end_epoch:
                    channels_coding.append(1)
                else:
                    if i<2:
                        channels_coding.append(random.choice(list(range(channel_start[0], channel_end[0]+1))))
                    elif i<3:
                        channels_coding.append(random.choice(list(range(channel_start[1], channel_end[1]+1))))
                    else:
                        channels_coding.append(random.choice(list(range(channel_start[2], channel_end[2]+1))))
            else:
                channels_coding.append(0)

    ## broadcast channels
    idxs = [
        list(range(5, -1,-1)),
        list(range(10, 5,-1)),
        list(range(18,10,-1)),
        list(range(23,18,-1))
    ]
    for i in range(len(idxs)-1, -1, -1):
        if epoch<depth_end_epoch:
            choice=1
        else:
            if i<2:
                choice = random.choice(list(range(channel_start[0], channel_end[0]+1)))
            elif i<3:
                choice = random.choice(list(range(channel_start[1], channel_end[1]+1)))
            else:
                choice = random.choice(list(range(channel_start[2], channel_end[2]+1)))
        for loc in idxs[i]:
            channels_coding.insert(loc, choice)

    for i in range(len(channels_coding)-1, 0, -1):
        if channels_coding[i-1]==0:
            channels_coding[i]=0

    return ''.join([str(c_c) for c_c in channels_coding])


if __name__ == "__main__":
    pass