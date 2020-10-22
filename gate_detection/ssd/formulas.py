import numpy as np
from numpy.linalg import norm

eps = 1e-10

def rot_matrix(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st],[st, ct]])

def rbox_to_polygon(rbox):
    cx, cy, w, h, theta = rbox
    box = np.array([[-w,h],[w,h],[w,-h],[-w,-h]]) / 2.
    box = np.dot(box, rot_matrix(theta))
    box += rbox[:2]
    return box

def polygon_to_rbox3(xy):
    # two points at the center of the left and right edge plus heigth
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, db = tr - tl, bl - br
    # height is mean between distance from top to bottom right and distance from top edge to bottom left
    h = (norm(np.cross(dt, tl - br)) + norm(np.cross(dt, tr - bl))) / (
        2 * (norm(dt) + eps)
    )
    p1 = (tl + bl) / 2.0
    p2 = (tr + br) / 2.0
    return np.hstack((p1, p2, h))


def rbox3_to_polygon(rbox):
    x1, y1, x2, y2, h = rbox
    alpha = np.arctan2(x1 - x2, y2 - y1)
    dx = -h * np.cos(alpha) / 2.0
    dy = -h * np.sin(alpha) / 2.0
    xy = np.reshape(
        [x1 - dx, y1 - dy, x2 - dx, y2 - dy, x2 + dx, y2 + dy, x1 + dx, y1 + dy],
        (-1, 2),
    )
    return xy


def iou(box, priors):
    """Compute intersection over union for the box with all priors.
    # Arguments
        box: Box, numpy tensor of shape (4,).
            (x1 + y1 + x2 + y2)
        priors:
    # Return
        iou: Intersection over union,
            numpy tensor of shape (num_priors).
    """
    # compute intersection
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:4], box[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    return iou


def non_maximum_suppression(boxes, confs, overlap_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.

    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        overlap_threshold:
        top_k: Maximum number of returned indices.

    # Return
        List of remaining indices.

    # References
        - Girshick, R. B. and Felzenszwalb, P. F. and McAllester, D.
          [Discriminatively Trained Deformable Part Models, Release 5](http://people.cs.uchicago.edu/~rbg/latent-release5/)
    """
    eps = 1e-15

    boxes = boxes.astype(np.float64)

    pick = []
    x1, y1, x2, y2 = boxes.T

    idxs = np.argsort(confs)
    area = (x2 - x1) * (y2 - y1)

    while len(idxs) > 0:
        i = idxs[-1]

        pick.append(i)
        if len(pick) >= top_k:
            break

        idxs = idxs[:-1]

        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        I = w * h

        overlap = I / (area[idxs] + eps)
        # as in Girshick et. al.

        # U = area[idxs] + area[i] - I
        # overlap = I / (U + eps)

        idxs = idxs[overlap <= overlap_threshold]

    return pick
