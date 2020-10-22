import numpy as np
import keras.backend as K

class CustomLoss(object):
    def __init__(self, lambda_roll=5.0, lambda_pitch=1.0, lambda_yaw=5.0, lambda_thrust=1.0, loss_func="mse"):
        self.lambda_roll = lambda_roll
        self.lambda_pitch = lambda_pitch
        self.lambda_yaw = lambda_yaw
        self.lambda_thrust = lambda_thrust
        self.loss_func = loss_func
        self.metrics = []

    def compute(self, y_true, y_pred):
        # y.shape (batches, priors, 4 x bbox_offset + 8 x quadrilaterals + 5 x rbbox_offsets + n x class_label)

        # angle loss
        roll_true = y_true[:, 0]
        roll_pred = y_pred[:, 0]
        pitch_true = y_true[:, 1]
        pitch_pred = y_pred[:, 1]
        yaw_true = y_true[:, 2]
        yaw_pred = y_pred[:, 2]

        # thrust loss
        thrust_true = y_true[:, 3]
        thrust_pred = y_pred[:, 3]

        if self.loss_func == "mse":
            roll_loss = K.mean(K.square(roll_pred - roll_true), axis=-1)
            pitch_loss = K.mean(K.square(pitch_pred - pitch_true), axis=-1)
            yaw_loss = K.mean(K.square(yaw_pred - yaw_true), axis=-1)
            thrust_loss = K.mean(K.square(thrust_pred - thrust_true), axis=-1)
        if self.loss_func == "mae":
            roll_loss = K.mean(K.abs(roll_pred - roll_true), axis=-1)
            pitch_loss = K.mean(K.abs(pitch_pred - pitch_true), axis=-1)
            yaw_loss = K.mean(K.abs(yaw_pred - yaw_true), axis=-1)
            thrust_loss = K.mean(K.abs(thrust_pred - thrust_true), axis=-1)

        # total loss
        total_loss = self.lambda_roll * roll_loss
        total_loss += self.lambda_pitch * pitch_loss
        total_loss += self.lambda_yaw * yaw_loss
        total_loss += self.lambda_thrust * thrust_loss

        def make_fcn(t):
            return lambda y_true, y_pred: t

        for name in [
            "roll_loss",
            "pitch_loss",
            "yaw_loss",
            "thrust_loss",
        ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)

        return total_loss
