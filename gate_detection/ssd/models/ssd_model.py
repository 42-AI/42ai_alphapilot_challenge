"""Keras implementation of SSD."""

from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

from .head.ssd_head import multibox_head
from .body.ssd_dense_body import dsod300_body, dsod512_body
from .body.ssd_resnet_body import ssd512_resnet_body
from .body.ssd_body import ssd300_body, ssd512_body


def SSD300(input_shape=(300, 300, 3), num_classes=21, softmax=True):
    """SSD300 architecture.
    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.

    # Notes
        In order to stay compatible with pre-trained models, the parameters
        were chosen as in the caffee implementation.

    # References
        https://arxiv.org/abs/1512.02325
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd300_body(x)

    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 4, 4]
    normalizations = [20, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(
        source_layers, num_priors, num_classes, normalizations, softmax
    )
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [
        [1, 2, 1 / 2],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2],
        [1, 2, 1 / 2],
    ]
    model.minmax_sizes = [
        (30, 60),
        (60, 111),
        (111, 162),
        (162, 213),
        (213, 264),
        (264, 315),
    ]
    model.steps = [8, 16, 32, 64, 100, 300]
    model.special_ssd_boxes = True

    return model


def SSD512(input_shape=(512, 512, 3), num_classes=21, softmax=True):
    """SSD512 architecture.
    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.

    # Notes
        In order to stay compatible with pre-trained models, the parameters
        were chosen as in the caffee implementation.

    # References
        https://arxiv.org/abs/1512.02325
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_body(x)

    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, -1, -1, -1, -1, -1, -1]
    output_tensor = multibox_head(
        source_layers, num_priors, num_classes, normalizations, softmax
    )
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [
        [1, 2, 1 / 2],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 3, 1 / 2, 1 / 3],
        [1, 2, 1 / 2],
        [1, 2, 1 / 2],
    ]
    # model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
    model.minmax_sizes = [
        (20.48, 51.2),
        (51.2, 133.12),
        (133.12, 215.04),
        (215.04, 296.96),
        (296.96, 378.88),
        (378.88, 460.8),
        (460.8, 542.72),
    ]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True

    return model


def DSOD300(input_shape=(300, 300, 3), num_classes=21, activation="relu", softmax=True):
    """DSOD, DenseNet based SSD300 architecture.
    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
        activation: Type of activation functions.

    # References
        https://arxiv.org/abs/1708.01241
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod300_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(
        source_layers, num_priors, num_classes, normalizations, softmax
    )
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.aspect_ratios = [
        [1, 2, 1 / 2],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2],
        [1, 2, 1 / 2],
    ]
    model.minmax_sizes = [
        (30, 60),
        (60, 111),
        (111, 162),
        (162, 213),
        (213, 264),
        (264, 315),
    ]
    model.steps = [8, 16, 32, 64, 100, 300]
    model.special_ssd_boxes = True

    return model


SSD300_dense = DSOD300


def DSOD512(input_shape=(512, 512, 3), num_classes=21, activation="relu", softmax=True):
    """DSOD, DenseNet based SSD512 architecture.
    # Arguments
        input_shape: Shape of the input image.
        num_classes: Number of classes including background.
        activation: Type of activation functions.

    # References
        https://arxiv.org/abs/1708.01241
    """
    x = input_tensor = Input(shape=input_shape)
    source_layers = dsod512_body(x, activation=activation)

    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20, 20]

    output_tensor = multibox_head(
        source_layers, num_priors, num_classes, normalizations, softmax
    )
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    model.aspect_ratios = [
        [1, 2, 1 / 2],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 3, 1 / 2, 1 / 3],
        [1, 2, 1 / 2],
        [1, 2, 1 / 2],
    ]
    model.minmax_sizes = [
        (35, 76),
        (76, 153),
        (153, 230),
        (230, 307),
        (307, 384),
        (384, 460),
        (460, 537),
    ]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True

    return model


SSD512_dense = DSOD512


def SSD512_resnet(input_shape=(512, 512, 3), num_classes=21, softmax=True):

    # TODO: it does not converge!

    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_resnet_body(x)

    # Add multibox head for classification and regression
    num_priors = [4, 6, 6, 6, 6, 4, 4]
    normalizations = [20, 20, 20, 20, 20, 20, 20]
    output_tensor = multibox_head(
        source_layers, num_priors, num_classes, normalizations, softmax
    )
    model = Model(input_tensor, output_tensor)
    model.num_classes = num_classes

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers
    # stay compatible with caffe models
    model.aspect_ratios = [
        [1, 2, 1 / 2],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 1 / 2, 3, 1 / 3],
        [1, 2, 3, 1 / 2, 1 / 3],
        [1, 2, 1 / 2],
        [1, 2, 1 / 2],
    ]
    model.minmax_sizes = [
        (35, 76),
        (76, 153),
        (153, 230),
        (230, 307),
        (307, 384),
        (384, 460),
        (460, 537),
    ]
    model.steps = [8, 16, 32, 64, 128, 256, 512]
    model.special_ssd_boxes = True

    return model
