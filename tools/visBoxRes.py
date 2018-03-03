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
mpl.use('Agg')
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

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def vis_detections(im, class_name, dets, thresh=0.5, fig = None):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return fig 

    im = im[:, :, (2, 1, 0)]
    if fig == None:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
    else:
        ax = fig.axes[0]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                      fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    return fig


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('jsonFile',
                        help='detections json file')
    parser.add_argument('--imdb', dest='imdb_name',
                        type=str, default='coco_2014_minival',
                        help='Which database was this run on')
    parser.add_argument('--nsave', dest='nsave',
                        type=int, default=100,
                        help='How many images to write into pdf')
    parser.add_argument('--pdfname', dest='pdfappend',
                        type=str, default='dummy',
                        help='str to append to the pdfName')
    parser.add_argument('--usecococls', dest='usecococls',
                        type=int, default=1,
                        help='wether to use coco class')

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
    detsJson = json.load(open(args.jsonFile,'r'))
    print '\n\nLoaded dets from {:s}'.format(args.jsonFile)
    
    pp = PdfPages(args.imdb_name+'_'+args.pdfappend + '.pdf')
    resAll = []

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for i in np.random.choice(np.arange(num_images),size=args.nsave, replace=False):
        # Visualize detections for each class
        im = cv2.imread(imdb.image_path_at(i))
        fig = None
        for j in xrange(1, len(CLASSES)):
            if type(dets[j][i]) != list:
                fig = vis_detections(im, CLASSES[j], dets[j][i], thresh=CONF_THRESH, fig=fig)

        if fig != None: 
            pp.savefig(fig)
    
    pp.close()
    #plt.show()

