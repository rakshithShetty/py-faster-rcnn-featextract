#!/usr/bin/env python
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import numpy as np
import argparse
import json
import scipy.integrate as integs
from collections import defaultdict


def gauss2d(y, x, m, sig):
    return np.exp(-(((x - m[0]) ** 2.0) / (2.0 * (sig[0] ** 2.0)) + ((y - m[1]) ** 2.0) / (2.0 * (sig[1] ** 2.0))))


def getRectCoord(args):
    gridSz = args.grid
    if args.gridtype == 'reg':
        gridRectCord = np.zeros((gridSz * gridSz, 4))
        gridA = (1.0 / (gridSz * gridSz)) * np.ones((gridSz * gridSz, 1))
        for i in xrange(gridSz):
            for j in xrange(gridSz):
                gridRectCord[i * gridSz + j][0] = (1.0 / gridSz) * i
                gridRectCord[i * gridSz + j][1] = (1.0 / gridSz) * j
                gridRectCord[i * gridSz + j][2] = (1.0 / gridSz) * (i + 1)
                gridRectCord[i * gridSz + j][3] = (1.0 / gridSz) * (j + 1)
    elif args.gridtype == 'rect':
        gridRectCord = np.zeros((2 * gridSz, 4))
        gridA = (1.0 / gridSz) * np.ones((2 * gridSz, 1))
        for i in xrange(gridSz):
            gridRectCord[i][0] = 0.0
            gridRectCord[i][1] = (1.0 / gridSz) * i
            gridRectCord[i][2] = 1.0
            gridRectCord[i][3] = (1.0 / gridSz) * (i + 1)
        for i in xrange(gridSz):
            gridRectCord[gridSz + i][0] = (1.0 / gridSz) * i
            gridRectCord[gridSz + i][1] = 0.0
            gridRectCord[gridSz + i][2] = (1.0 / gridSz) * (i + 1)
            gridRectCord[gridSz + i][3] = 1.0
    return gridRectCord, gridA


def dump_clasDetFeat(dets, catToIdx, args, max_idx):
    if args.appendtofeat == None:
        spatMap = np.zeros((max_idx, len(catToIdx)))
    else:
        spatMap = np.load(open(args.appendtofeat, 'rb'))
        args.resFile = args.appendtofeat

    for ann in dets:
        cid = catToIdx[ann['category_id']]
        imgid = ann['image_index']

        if ann['score'] > spatMap[imgid][cid]:
            spatMap[imgid][cid] = ann['score']

    spatMapSmall = spatMap.astype(np.float16)
    np.save(args.resFile, spatMapSmall)


def dump_spatMapFeat(detDict, catToIdx, args, max_idx):
    gridRectCord, gridA = getRectCoord(args)
    gridlen = gridRectCord.shape[0]

    if args.appendtofeat == None:
        spatMap = np.zeros((max_idx, len(catToIdx), gridlen))
    else:
        spatMap = np.load(open(args.appendtofeat, 'rb')).reshape(max_idx, len(catToIdx), gridlen)
        args.resFile = args.appendtofeat

    for _, dects_per_im in detDict.iteritems():
        for ann in dects_per_im:
            nC = [ann['bbox'][0] / ann['width'], ann['bbox'][1] / ann['height'],
                  ann['bbox'][2] / ann['width'], ann['bbox'][3] / ann['height']]
            bC = [nC[0], nC[1], nC[0] + nC[2], nC[1] + nC[3]]
            bA = nC[2] * nC[3]
            cid = catToIdx[ann['category_id']]

            if args.use_gauss_weight == 1:
                gM = (nC[0] + nC[2] / 2, nC[1] + nC[3] / 2)
                gS = (np.sqrt(nC[1] ** 2 + nC[3] ** 2), np.sqrt(nC[1] ** 2 + nC[3] ** 2))

            for i in xrange(gridRectCord.shape[0]):
                ov_x = [min(bC[2], gridRectCord[i][2]), max(bC[0], gridRectCord[i][0])]
                ov_y = [min(bC[3], gridRectCord[i][3]), max(bC[1], gridRectCord[i][1])]
                sI = max(0, ov_x[0] - ov_x[1]) * max(0, ov_y[0] - ov_y[1])

                if sI > 0:
                    imgid = ann['image_index']

                    if args.use_gauss_weight == 0:
                        sU = gridA[i] + bA - sI
                        assert (sU > sI)
                        spatMap[imgid][cid][i] += (ann['score'] ** args.scale_by_det) * sI / sU
                    else:
                        spatMap[imgid][cid][i] += (ann['score'] ** args.scale_by_det) * integs.dblquad(
                            gauss2d, ov_x[1], ov_x[0], lambda x: ov_y[1],
                            lambda x: ov_y[0], (gM, gS))[0]

    spatMap.resize((spatMap.shape[0], spatMap.shape[1] * spatMap.shape[2]))
    spatMapSmall = spatMap.astype(np.float16)
    np.save(args.resFile, spatMapSmall)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Dump features')
    parser.add_argument('jsonFile',
                        help='detections json file')
    parser.add_argument('-o', dest='resFile', type=str,
                        default='dummpy.npy',
                        help='detections json file')
    parser.add_argument('--categories', dest='categories_file',
                        type=str, default='categories.json',
                        help='Categories file. This come in place of the categories database')
    parser.add_argument('--grid', dest='grid', type=int, default=4,
                        help='grid size, gxg')
    parser.add_argument('--gridtype', dest='gridtype', type=str, default='reg',
                        help='grid type, reg or rect')
    parser.add_argument('--use_gauss_weight', dest='use_gauss_weight', type=int,
                        default=0, help='grid size, gxg')
    parser.add_argument('--scale_by_det', dest='scale_by_det', type=int,
                        default=1, help='grid size, gxg')
    parser.add_argument('--dump_class_only', dest='dump_class_only', type=int,
                        default=0, help='grid size, gxg')
    parser.add_argument('--appendtofeat', dest='appendtofeat',
                        type=str, default=None,
                        help='add features to already existing file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.jsonFile) as f:
        detections = json.load(f)
    print('Loaded dets from {:s}'.format(args.jsonFile))

    with open(args.categories_file) as f:
        categories = json.load(f)
    print('Loaded cats from {:s}'.format(args.categories_file))

    cat_idx = sorted(list(set([c['id'] for c in categories])))
    catToIdx = dict(zip(cat_idx, range(len(cat_idx))))
    iilist = [d['image_index'] for d in detections]
    lenlbls = 1
    if len(iilist):
        lenlbls = max(iilist) + 1

    CONF_THRESH = 0.8

    if args.dump_class_only == 1:
        dump_clasDetFeat(detections, catToIdx, args, lenlbls)
    else:
        detDict = defaultdict(list)
        for ann in detections:
            if ann['score'] > CONF_THRESH:
                detDict[ann['image_index']].append(ann)

        dump_spatMapFeat(detDict, catToIdx, args, lenlbls)
