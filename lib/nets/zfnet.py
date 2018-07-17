# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

# """ZF net support"""
# from kaffe.tensorflow import Network
#
# class Zeiler_Net(Network):
#     def setup(self):
#         (self.feed('data')
#              .conv(7, 7, 96, 2, 2, name='conv1')
#              .lrn(1, 1.66666666667e-05, 0.75, name='norm1')
#              .max_pool(3, 3, 2, 2, name='pool1')
#              .conv(5, 5, 256, 2, 2, name='conv2')
#              .lrn(1, 1.66666666667e-05, 0.75, name='norm2')
#              .max_pool(3, 3, 2, 2, padding=None, name='pool2')
#              .conv(3, 3, 384, 1, 1, name='conv3')
#              .conv(3, 3, 384, 1, 1, name='conv4')
#              .conv(3, 3, 256, 1, 1, name='conv5')
#              .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
#              .fc(4096, name='fc6')
#              .fc(4096, name='fc7')
#              .fc(1000, relu=False, name='fc8')
#              .softmax(name='prob'))

class zfnet(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'zfnet'

  def _image_to_head(self, is_training, reuse=None):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net = slim.conv2d(self._image, 96, kernel_size=[7, 7], stride=[2, 2],
                        trainable=is_training, scope='conv1')
      net = tf.nn.local_response_normalization(net, depth_radius=1, alpha=1.66666666667e-05, beta=0.75, name='norm1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')
      net = slim.conv2d(net, 256, kernel_size=[5, 5], stride=[2, 2],
                        trainable=is_training, scope='conv2')
      net = tf.nn.local_response_normalization(net, depth_radius=1, alpha=1.66666666667e-05, beta=0.75, name='norm2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='SAME')

      net = slim.conv2d(net, 384, kernel_size=[3, 3], stride=[1, 1],
                        trainable=is_training, scope='conv3')
      net = slim.conv2d(net, 384, kernel_size=[3, 3], stride=[1, 1],
                        trainable=is_training, scope='conv4')
      net = slim.conv2d(net, 256, kernel_size=[3, 3], stride=[1, 1],
                        trainable=is_training, scope='conv5')

    self._act_summaries.append(net)
    self._layers['head'] = net

    return net

  def _head_to_tail(self, pool5, is_training, reuse=None, average_pool=True):
    # use average_pool=True to be compatible with resnet and mobilenet
    if not average_pool:
      return pool5
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                           scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                           scope='dropout7')

    return fc7

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == (self._scope + '/fc6/weights:0') or \
         v.name == (self._scope + '/fc7/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv, 
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                            self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                            self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
