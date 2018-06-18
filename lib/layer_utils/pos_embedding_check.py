# check the position embedding in mxnet
# compare to lib/layer_utils/rel_module_util.py

import mxnet as mx
import math

def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
    # position_mat, [num_rois, nongt_dim, 4]
    feat_range = mx.sym.arange(0, feat_dim / 8)
    dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                     rhs=(8. / feat_dim) * feat_range)
    dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, -1))
    position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=3)
    div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
    sin_mat = mx.sym.sin(data=div_mat)
    cos_mat = mx.sym.cos(data=div_mat)
    # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
    embedding = mx.sym.concat(sin_mat, cos_mat, dim=3)
    # embedding, [num_rois, nongt_dim, feat_dim]
    embedding = mx.sym.Reshape(embedding, shape=(0, 0, feat_dim))
    return embedding

def extract_position_matrix(bbox, nongt_dim):
    """ Extract position matrix

    Args:
        bbox: [num_boxes, 4]

    Returns:
        position_matrix: [num_boxes, nongt_dim, 4]
    """
    xmin, ymin, xmax, ymax = mx.sym.split(data=bbox,
                                          num_outputs=4, axis=1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = mx.sym.broadcast_minus(lhs=center_x,
                                     rhs=mx.sym.transpose(center_x))
    delta_x = mx.sym.broadcast_div(delta_x, bbox_width)
    delta_x = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_x), 1e-3))
    delta_y = mx.sym.broadcast_minus(lhs=center_y,
                                     rhs=mx.sym.transpose(center_y))
    delta_y = mx.sym.broadcast_div(delta_y, bbox_height)
    delta_y = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_y), 1e-3))
    delta_width = mx.sym.broadcast_div(lhs=bbox_width,
                                       rhs=mx.sym.transpose(bbox_width))
    delta_width = mx.sym.log(delta_width)
    delta_height = mx.sym.broadcast_div(lhs=bbox_height,
                                        rhs=mx.sym.transpose(bbox_height))
    delta_height = mx.sym.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = mx.sym.slice_axis(sym, axis=1, begin=0, end=nongt_dim)
        concat_list[idx] = mx.sym.expand_dims(sym, axis=2)
    position_matrix = mx.sym.concat(*concat_list, dim=2)
    return position_matrix

def main():
    rois = mx.nd.array([
      [100., 200., 300., 400.],
      [4, 5, 6, 7],
      [6, 7, 8, 9],
      [5, 6, 7, 8],
      [0, 3, 5, 7],
      [50, 60, 100, 130]
    ])
    
    rois_mx = mx.sym.Variable('rois')
    rois_mx = mx.sym.Reshape(rois_mx, shape=(-1, 4))
    sliced_rois = mx.sym.slice_axis(rois_mx, axis=1, begin=0, end=None)    

    pos_matrix = extract_position_matrix(sliced_rois, nongt_dim=rois.shape[0])
    pos_embedding = extract_position_embedding(pos_matrix, feat_dim=64)
    ex = pos_matrix.bind(mx.cpu(), {'rois': rois})
    ex2 = pos_embedding.bind(mx.cpu(), {'rois': rois})

    y = ex.forward()
    y_np = y[0].asnumpy()
    y2 = ex2.forward()
    y2_np = y2[0].asnumpy()
    print('===position matrix===')
    print(y_np.shape)
    print(y_np[:2, :2, :])
    print('===position embedding===')
    print(y2_np.shape)
    print(y2_np[2, 1, :])


if __name__ == main():
    main()
