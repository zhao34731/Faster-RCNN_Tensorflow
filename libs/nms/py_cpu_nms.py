# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    #get all bbox location seperatively and socres
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #calculate all areas of different bbox
    order = scores.argsort()[::-1]
    #rank the bbox scores index
    keep = []
    while order.size > 0:
        i = order[0] #get the top score
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

#NMS 算法的计算方式：
#输入为一系列bbox框，先对框进行排序，得到从大到小的顺序，之后每次从里面选出一个概率值最大的框，计算该框与其他剩下的框的IoU交叠率，
#将那些交叠率大于thresh的框抑制掉，也就是选出去那些交叠率小于thresh的框，作为新的order，再次反复执行，并将这里order里面的序号是在
#刚开始就指定好的，因此不会发生混乱，最终，当order里面没没有序号的时候，则循环完成，达标将所有交叠率大于某个thresh的框都进行了抑制。
