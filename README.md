# tf-faster-rcnn-relation-module
A reimplementation of relation module for faster rcnn in Tensorflow (refer to paper [Relation Network for Object Detection](https://arxiv.org/abs/1711.11575)). This repository is based on the multi-image version of Faster RCNN available [here](https://github.com/ChengpengChen/tf-faster-rcnn-multi-img). The codes for relation module are based on the original mxnet implementation.

**Note**: 
  - Only the ``relation module`` (between fc layers for feature enhance) is implemented for present. The one for ``duplicate process`` (to learn NMS) in not support yet.

