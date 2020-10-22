"""Keras implementation of SSD."""
import numpy as np

from keras.engine.topology import Layer
from keras.engine.topology import InputSpec

import keras.backend as K

from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import concatenate
from keras.layers import Reshape


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Default feature scale.
    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)
    # Output shape
        Same as input
    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    # TODO
        Add possibility to have one scale for all features.
    """

    def __init__(self, scale, **kwargs):
        if K.image_dim_ordering() == "tf":
            self.axis = 3
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name=self.name + "_gamma")
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


def deprecated_multibox_head(
    source_layers, num_priors, num_classes, normalizations=None, softmax=True
):

    class_activation = "softmax" if softmax else "sigmoid"

    mbox_conf = []
    mbox_loc = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split("/")[0]

        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + "_norm"
            x = Normalize(normalizations[i], name=name)(x)

        # confidence
        name1 = name + "_mbox_conf"
        x1 = Conv2D(num_priors[i] * num_classes, 3, padding="same", name=name1)(x)
        x1 = Flatten(name=name1 + "_flat")(x1)
        mbox_conf.append(x1)

        # location
        name2 = name + "_mbox_loc"
        x2 = Conv2D(num_priors[i] * 4, 3, padding="same", name=name2)(x)
        x2 = Flatten(name=name2 + "_flat")(x2)
        mbox_loc.append(x2)

    mbox_loc = concatenate(mbox_loc, axis=1, name="mbox_loc")
    mbox_loc = Reshape((-1, 4), name="mbox_loc_final")(mbox_loc)

    mbox_conf = concatenate(mbox_conf, axis=1, name="mbox_conf")
    mbox_conf = Reshape((-1, num_classes), name="mbox_conf_logits")(mbox_conf)
    mbox_conf = Activation(class_activation, name="mbox_conf_final")(mbox_conf)

    predictions = concatenate([mbox_loc, mbox_conf], axis=2, name="predictions")

    return predictions


def multibox_head(
    source_layers, num_priors, num_classes=2, normalizations=None, softmax=True
):

    class_activation = "softmax" if softmax else "sigmoid"

    mbox_conf = []
    mbox_loc = []
    mbox_quad = []
    mbox_rbox = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split("/")[0]

        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + "_norm"
            x = Normalize(normalizations[i], name=name)(x)

        # confidence
        name1 = name + "_mbox_conf"
        x1 = Conv2D(num_priors[i] * num_classes, (3, 5), padding="same", name=name1)(x)
        x1 = Flatten(name=name1 + "_flat")(x1)
        mbox_conf.append(x1)

        # location, Delta(x,y,w,h)
        name2 = name + "_mbox_loc"
        x2 = Conv2D(num_priors[i] * 4, (3, 5), padding="same", name=name2)(x)
        x2 = Flatten(name=name2 + "_flat")(x2)
        mbox_loc.append(x2)

        # quadrilateral, Delta(x1,y1,x2,y2,x3,y3,x4,y4)
        name3 = name + "_mbox_quad"
        x3 = Conv2D(num_priors[i] * 8, (3, 5), padding="same", name=name3)(x)
        x3 = Flatten(name=name3 + "_flat")(x3)
        mbox_quad.append(x3)

        # rotated rectangle, Delta(x1,y1,x2,y2,h)
        '''
        name4 = name + "_mbox_rbox"
        x4 = Conv2D(num_priors[i] * 5, (3, 5), padding="same", name=name4)(x)
        x4 = Flatten(name=name4 + "_flat")(x4)
        mbox_rbox.append(x4)
        '''

    mbox_conf = concatenate(mbox_conf, axis=1, name="mbox_conf")
    mbox_conf = Reshape((-1, num_classes), name="mbox_conf_logits")(mbox_conf)
    mbox_conf = Activation(class_activation, name="mbox_conf_final")(mbox_conf)

    mbox_loc = concatenate(mbox_loc, axis=1, name="mbox_loc")
    mbox_loc = Reshape((-1, 4), name="mbox_loc_final")(mbox_loc)

    mbox_quad = concatenate(mbox_quad, axis=1, name="mbox_quad")
    mbox_quad = Reshape((-1, 8), name="mbox_quad_final")(mbox_quad)

    '''
    mbox_rbox = concatenate(mbox_rbox, axis=1, name="mbox_rbox")
    mbox_rbox = Reshape((-1, 5), name="mbox_rbox_final")(mbox_rbox)
    '''

    predictions = concatenate(
        [mbox_loc, mbox_quad, mbox_conf], axis=2, name="predictions"
    )

    return predictions
