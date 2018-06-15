# tf-faster-rcnn-relation-module
A reimplementation of relation module for faster rcnn in Tensorflow (refer to paper [Relation Network for Object Detection](https://arxiv.org/abs/1711.11575)). This repository is based on the multi-image version of Faster RCNN available [here](https://github.com/ChengpengChen/tf-faster-rcnn-multi-img). The codes for relation module are based on the original mxnet implementation.

  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101 RM.ENABLE_INSTANCE True TRAIN.LEARNING_RATE 0.000125
  ```

As above shell shown, we set ``RM.ENABLE_INSTANCE`` as ``True`` to enable ``relation module``. The learning rate is set according to the original mxnet implementation. More configuration for ``relation module`` can be viewed in ``lib/model/config.py``

**Note**: 
  - Only the ``relation module`` (between fc layers for feature enhance) is implemented for present. The one for ``duplicate process`` (to learn NMS) in not support yet.
  - ``relation module`` is impletemented in ``lib/nets/network.py``.
  - The position embedding code are implemented in python, and reside in ``lib/layer_utils/rel_module_util.py``.

