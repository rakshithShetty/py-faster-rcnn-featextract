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

import matplotlib as mpl
# mpl.use('Agg')
import _init_paths
from datasets.factory import get_imdb
from fast_rcnn.config import cfg
import _init_paths
from utils.timer import Timer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fast_rcnn.nms_wrapper import nms
import numpy as np
import os, cv2
import argparse
import cPickle as pkl
import json
import re
import scipy.integrate as integs
from collections import defaultdict


def gauss2d(y, x, m, sig):
    return np.exp(-(((x - m[0]) ** 2.0) / (2.0 * (sig[0] ** 2.0)) + ((y - m[1]) ** 2.0) / (2.0 * (sig[1] ** 2.0))))


def dump_clasDetFeat(dets, imdb, coco2imgid, args, max_idx):
    if args.appendtofeat == None:
        spatMap = np.zeros((max_idx, len(imdb.classes) - 1))
    else:
        spatMap = np.load(open(args.appendtofeat, 'rb'))
        args.resFile = args.appendtofeat

    catToIdx = {}
    for cat in imdb._class_to_coco_cat_id.keys():
        catToIdx[imdb._class_to_coco_cat_id[cat]] = imdb._class_to_ind[cat] - 1

    for ann in dets:
        cid = catToIdx[ann['category_id']]
        if ann['score'] > spatMap[coco2imgid[ann['image_id']]][cid]:
            spatMap[coco2imgid[ann['image_id']]][cid] = ann['score']

    spatMapSmall = spatMap.astype(np.float16)
    np.save(open(args.resFile, 'wb'), spatMapSmall)


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


def dump_spatMapFeat(detDict, imdb, coco2imgid, args, max_idx):
    gridRectCord, gridA = getRectCoord(args)
    gridlen = gridRectCord.shape[0]
    if args.appendtofeat == None:
        spatMap = np.zeros((max_idx, len(imdb.classes) - 1,
                            gridlen))
    else:
        spatMap = np.load(open(args.appendtofeat, 'rb')).reshape(
            len(coco2imgid), len(imdb.classes) - 1, gridlen)
        args.resFile = args.appendtofeat

    catToIdx = {}
    for cat in imdb._class_to_coco_cat_id.keys():
        catToIdx[imdb._class_to_coco_cat_id[cat]] = imdb._class_to_ind[cat] - 1

    szs = imdb._get_sizes()

    for im_ind, index in enumerate(imdb.image_index):
        for ann in detDict[index]:
            # assert(((ann['bbox'][0] +ann['bbox'][2] )<=szs[im_ind][0]) &
            #         ((ann['bbox'][1] + ann['bbox'][3])<=szs[im_ind][1]))
            nC = [ann['bbox'][0] / szs[im_ind][0], ann['bbox'][1] / szs[im_ind][1],
                  ann['bbox'][2] / szs[im_ind][0], ann['bbox'][3] / szs[im_ind][1]]
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
                    if args.use_gauss_weight == 0:
                        sU = gridA[i] + bA - sI
                        assert (sU > sI)
                        spatMap[coco2imgid[ann['image_id']]][cid][i] += (ann['score'] ** args.scale_by_det) * sI / sU
                    else:
                        # print ov_x, ov_y, gM, gS
                        spatMap[coco2imgid[ann['image_id']]][cid][i] += (ann['score'] ** args.scale_by_det) * \
                                                                        integs.dblquad(
                                                                            gauss2d, ov_x[1], ov_x[0],
                                                                            lambda x: ov_y[1],
                                                                            lambda x: ov_y[0], (gM, gS))[0]
        if im_ind % 500 == 1:
            print('Now at %d' % (im_ind))

    spatMap.resize((spatMap.shape[0], spatMap.shape[1] * spatMap.shape[2]))
    spatMapSmall = spatMap.astype(np.float16)
    np.save(open(args.resFile, 'wb'), spatMapSmall)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('jsonFile',
                        help='detections json file')
    parser.add_argument('-o', dest='resFile', type=str,
                        default='dummpy.npy',
                        help='detections json file')
    parser.add_argument('--imdb', dest='imdb_name',
                        type=str, default='coco_2014_train',
                        help='Which database was this run on')
    parser.add_argument('--nsave', dest='nsave',
                        type=int, default=100,
                        help='How many images to write into pdf')
    parser.add_argument('--pdfname', dest='pdfappend',
                        type=str, default='dummy',
                        help='str to append to the pdfName')
    parser.add_argument('--labels', dest='labels',
                        type=str, default='/projects/databases/coco/labels.txt',
                        help='str to append to the pdfName')
    parser.add_argument('--featfromlbl', dest='featfromlbl',
                        type=str, default='',
                        help='should we use lables.txt, if yes which feature?')
    parser.add_argument('--usecococls', dest='usecococls',
                        type=int, default=1,
                        help='wether to use coco class')
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

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    cfg.USE_GPU_NMS = False

    imdb = get_imdb(args.imdb_name)

    if args.usecococls != 0:
        CLASSES = imdb.classes
    num_images = len(imdb.image_index)

    # Load the images file
    dets = json.load(open(args.jsonFile, 'r'))
    print '\n\nLoaded dets from {:s}'.format(args.jsonFile)

    resAll = []

    lbls = open(args.labels, 'r').read().splitlines()
    coco2imgid = {}
    for lb in lbls:
        lbParts = lb.split()
        lbParts[1] = lbParts[1][1:-1]
        if (len(lbParts[1].split(':')) == 1):
            if args.featfromlbl == '':
                coco2imgid[int(lbParts[1])] = int(lbParts[0][1:])
        elif re.match(args.featfromlbl, lbParts[1].split(':')[1]):
            coco2imgid[int(lbParts[1].split(':')[0])] = int(lbParts[0][1:])
    # import pdb;pdb.set_trace()

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    if args.dump_class_only == 1:
        dump_clasDetFeat(dets, imdb, coco2imgid, args, len(lbls))
    else:
        detDict = defaultdict(list)
        for ann in dets:
            if ann['score'] > CONF_THRESH:
                detDict[ann['image_id']].append(ann)

        dump_spatMapFeat(detDict, imdb, coco2imgid, args, len(lbls))

        # plt.show()
