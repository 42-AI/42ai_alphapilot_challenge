import numpy as np
import tensorflow as tf
import keras.backend as K


def smooth_l1_loss(y_true, y_pred):
    """Compute L1-smooth loss.
    # Arguments
        y_true: Ground truth bounding boxes,
            tensor of shape (?, num_boxes, 4).
        y_pred: Predicted bounding boxes,
            tensor of shape (?, num_boxes, 4).
    # Returns
        l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).
    # References
        https://arxiv.org/abs/1504.08083
    """
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred) ** 2
    loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(loss, axis=-1)


def focal_loss(y_true, y_pred, gamma=2.0, alpha=1.0):
    """Compute binary focal loss.

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).

    # Returns
        focal_loss: Focal loss, tensor of shape (?, num_boxes).
    # References
        https://arxiv.org/abs/1708.02002
    """
    # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.0 - eps)
    # loss = - K.pow(1-y_pred, gamma) * y_true*tf.log(y_pred) - K.pow(y_pred, gamma) * (1-y_true)*tf.log(1-y_pred)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -K.pow(1.0 - pt, gamma) * K.log(pt)
    loss = alpha * loss
    return tf.reduce_sum(loss, axis=-1)


def compute_metrics(class_true, class_pred, conf, top_k=100):
    """Compute precision, recall, accuracy and f-measure for top_k predictions.

    from top_k predictions that are TP FN or FP (TN kept out)
    """

    top_k = tf.cast(top_k, tf.int32)
    eps = K.epsilon()

    mask = tf.greater(class_true + class_pred, 0)
    # mask = tf.logical_or(tf.greater(class_true, 0), tf.greater(class_pred, 0))
    mask_float = tf.cast(mask, tf.float32)

    vals, idxs = tf.nn.top_k(conf * mask_float, k=top_k)

    top_k_class_true = tf.gather(class_true, idxs)
    top_k_class_pred = tf.gather(class_pred, idxs)

    true_mask = tf.equal(top_k_class_true, top_k_class_pred)
    false_mask = tf.logical_not(true_mask)
    pos_mask = tf.greater(top_k_class_pred, 0)
    neg_mask = tf.logical_not(pos_mask)

    tp = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pos_mask), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, pos_mask), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, neg_mask), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, neg_mask), tf.float32))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    fmeasure = 2 * (precision * recall) / (precision + recall + eps)

    return precision, recall, accuracy, fmeasure


class FocalLoss(object):
    def __init__(self, lambda_conf=1000.0, lambda_offsets=1.0):
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.metrics = []

    def compute(self, y_true, y_pred):
        # y.shape (batches, priors, 4 x bbox_offset + 8 x quadrilaterals + 5 x rbbox_offsets + n x class_label)

        batch_size = tf.shape(y_true)[0]
        num_priors = tf.shape(y_true)[1]
        num_classes = tf.shape(y_true)[2] - 12
        eps = K.epsilon()

        # confidence loss
        conf_true = tf.reshape(y_true[:, :, 12:], [-1, num_classes])
        conf_pred = tf.reshape(y_pred[:, :, 12:], [-1, num_classes])

        class_true = tf.argmax(conf_true, axis=1)
        class_pred = tf.argmax(conf_pred, axis=1)
        conf = tf.reduce_max(conf_pred, axis=1)

        neg_mask_float = conf_true[:, 0]
        neg_mask = tf.cast(neg_mask_float, tf.bool)
        pos_mask = tf.logical_not(neg_mask)
        pos_mask_float = tf.cast(pos_mask, tf.float32)
        num_total = tf.cast(tf.shape(conf_true)[0], tf.float32)
        num_pos = tf.reduce_sum(pos_mask_float)
        num_neg = num_total - num_pos

        conf_loss = focal_loss(conf_true, conf_pred, alpha=[0.002, 0.998])
        conf_loss = tf.reduce_sum(conf_loss)

        conf_loss = conf_loss / (num_total + eps)

        # offset loss, bbox, quadrilaterals, rbbox
        loc_true = tf.reshape(y_true[:, :, 0:12], [-1, 12])
        loc_pred = tf.reshape(y_pred[:, :, 0:12], [-1, 12])

        loc_loss = smooth_l1_loss(loc_true, loc_pred)
        # loc_loss = smooth_l1_loss(loc_true[:,:4], loc_pred[:,:4])
        pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float)  # only for positives

        loc_loss = pos_loc_loss / (num_pos + eps)

        # total loss
        total_loss = self.lambda_conf * conf_loss + self.lambda_offsets * loc_loss

        # metrics
        precision, recall, accuracy, fmeasure = compute_metrics(
            class_true, class_pred, conf, top_k=100 * batch_size
        )

        def make_fcn(t):
            return lambda y_true, y_pred: t

        for name in [
            "conf_loss",
            "loc_loss",
            "precision",
            "recall",
            "accuracy",
            "fmeasure",
            "num_pos",
            "num_neg",
        ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)

        return total_loss
