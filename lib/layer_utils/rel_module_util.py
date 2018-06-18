# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Add relation module by Chengpeng Chen
# --------------------------------------------------------

# the layer to encoding the geometirc features
# refer to paper: Relation network for object detection && Attention is all you need

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
eps = 1e-3


def pos_encoding(rois, fea_dim):
  """
  calculate the relative-position encoding for each location pair
  note: ignore the _feat_stride because of the zero order transformation
  :param rois: rois with [bbox_num, 5], (im_inds, x1, y1, x2, y2)
  :param fea_dim: the dimension of the output of each location pair
  :return: encoding of the relative-position, [im_num, bbox_num, bbox_num]
  """
  # change form to ctr_x, ctr_y, widths, heights
  rois[:, 3] = rois[:, 3] - rois[:, 1] + 1
  rois[:, 4] = rois[:, 4] - rois[:, 2] + 1
  rois[:, 1] = rois[:, 1] + 0.5*rois[:, 3]
  rois[:, 2] = rois[:, 2] + 0.5*rois[:, 4]

  rois_num_total = rois.shape[0]
  im_num = np.alen(np.unique(rois[:, 0]))
  rois_num = rois_num_total // im_num

  assert fea_dim % 8 == 0, 'fea_dim must be divided by 8'

  pos_encode_list = []

  for im_i in np.arange(im_num):
    rois_im_i = rois[im_i*rois_num:(im_i+1)*rois_num, 1:]  # [rois_num, 4]
    pos_encode_tmp = np.zeros(shape=[rois_num, rois_num, 4], dtype=np.float32)

    pos_encode_tmp[:, :, 0] = np.divide(rois_im_i[:, 0][:, np.newaxis] - rois_im_i[:, 0][np.newaxis],
                                        rois_im_i[:, 2][:, np.newaxis])
    pos_encode_tmp[:, :, 0] = np.maximum(np.abs(pos_encode_tmp[:, :, 0]), eps)
    pos_encode_tmp[:, :, 1] = np.divide(rois_im_i[:, 1][:, np.newaxis] - rois_im_i[:, 1][np.newaxis],
                                        rois_im_i[:, 3][:, np.newaxis])
    pos_encode_tmp[:, :, 1] = np.maximum(np.abs(pos_encode_tmp[:, :, 1]), eps)
    pos_encode_tmp[:, :, 2] = np.divide(rois_im_i[:, 2][np.newaxis], rois_im_i[:, 2][:, np.newaxis])
    pos_encode_tmp[:, :, 3] = np.divide(rois_im_i[:, 3][np.newaxis], rois_im_i[:, 3][:, np.newaxis])

    pos_encode_tmp = np.log(pos_encode_tmp)
    # print(pos_encode_tmp[:2, :2, :])

    pos_encode_tmp = _pos_encode(pos_encode_tmp, fea_dim)
    # pos_encode_tmp = np.reshape(pos_encode_tmp, [rois_num, rois_num, -1])

    pos_encode_list.append(pos_encode_tmp[np.newaxis])

  pos_encode_total = np.concatenate(pos_encode_list)

  return np.float32(pos_encode_total)


def _pos_encode(position_mat, feat_dim, wave_length=1000):
  """
  return sin and cos encodeing of the position code, refer to code in mxnet
  """
  # position_mat, [num_rois, nongt_dim, 4]
  rois_num = position_mat.shape[0]
  feat_range = np.arange(0, feat_dim / 8)
  dim_mat = np.power(np.full((1,), wave_length),
                     (8. / feat_dim) * feat_range)
  dim_mat = np.reshape(dim_mat, [1, 1, 1, -1])
  position_mat = np.expand_dims(100.0 * position_mat, axis=3)
  div_mat = np.divide(position_mat, dim_mat)
  sin_mat = np.sin(div_mat)
  cos_mat = np.cos(div_mat)
  # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
  embedding = np.concatenate([sin_mat, cos_mat], axis=3)
  # embedding, [num_rois, nongt_dim, feat_dim]
  embedding = np.reshape(embedding, [rois_num, rois_num, feat_dim])
  return embedding


def main():
  #
  # to eval the position encoding
  #
  rois = np.array([
    [0, 100, 200, 300, 400],
    [0, 4, 5, 6, 7],
    [0, 6, 7, 8, 9],
    [0, 5, 6, 7, 8],
    [0, 0, 3, 5, 7],
    [0, 50, 60, 100, 130]
  ], dtype=np.float32)
  fea_dim = 64

  pos_encode_total = pos_encoding(rois, fea_dim)
  print(pos_encode_total.shape)
  # print(pos_encode_total)
  print('===')
  print(pos_encode_total[0, 1, 2, :])
  # print(pos_encode_total[0, 2, 1, :])
  # for i in np.arange(1):
  #   print(pos_encode_total[0, i, i, :])
  # print('===')
  # for i in np.arange(5):
  #   print(pos_encode_total[0, i, i+1, :])
  # print('===')
  # for i in np.arange(1, 6):
  #   print(pos_encode_total[0, i, i-1, :])


if __name__ == main():
    main()
