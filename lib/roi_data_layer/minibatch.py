# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob, 'im_info': [], 'gt_boxes': []}

  for im_i in np.arange(num_images):
    # assert len(im_scales) == 1, "Single batch only"
    # assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      gt_inds = np.where(roidb[im_i]['gt_classes'] != 0)[0]
    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
      gt_inds = np.where(roidb[im_i]['gt_classes'] != 0 & np.all(roidb[im_i]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 6), dtype=np.float32)
    gt_boxes[:, 0] = im_i  # batch inds
    gt_boxes[:, 1:5] = roidb[im_i]['boxes'][gt_inds, :] * im_scales[im_i]
    gt_boxes[:, 5] = roidb[im_i]['gt_classes'][gt_inds]
    blobs['gt_boxes'].append(np.reshape(gt_boxes, (-1, 6)))
    im_info_temp = np.array(
      [im_blob.shape[1], im_blob.shape[2], im_scales[im_i]],
      dtype=np.float32)
    blobs['im_info'].append(np.reshape(im_info_temp, (-1, 3)))
  blobs['gt_boxes'] = np.concatenate(blobs['gt_boxes'])
  blobs['im_info'] = np.concatenate(blobs['im_info'])

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
