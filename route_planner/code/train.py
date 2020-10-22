import matplotlib  # Mac bug fix
matplotlib.use("TkAgg")

# Basic imports
import talos as ta
import cPickle
import os
import numpy as np
import math
import datetime

# Keras imports
from keras import backend as K
from keras.optimizers import Adam
from keras.optimizers import Adam
from keras.callbacks import (
    Callback,
    TensorBoard,
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau
)
from keras.models import load_model

# Local imports
from training.supervised_manager import SupervisedManager
from models.rnn import RNN
from training.ground_truth import GroundTruth
from training.input_generator import InputGenerator
from training.custom_loss import CustomLoss

# Custom functions
def step_decay(epoch):
    ''' Decrease learning rate by 1e-1 every 10 epochs
    '''
    initial_lrate = 1e-3
    drop = 1e-1
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
        math.floor((1+epoch)/epochs_drop))
    return lrate

class PrintLR(Callback):
    ''' Print the learning rate
    '''
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("\nlr={}\n\n".format(K.eval(lr_with_decay)))

class SupervisedTalosTraining(SupervisedManager):
    """docstring for SupervisedTraining."""

    def __init__(self, experimentname=None):
        super(SupervisedTalosTraining, self).__init__(experimentname)

    def setupDir(self, params):
        train_name = self.experimentname
        directory = "./graph/{}_{}".format(train_name, datetime.datetime.now().strftime("%m%d%H%M%S"))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def processParams(self, params):
        # Directory for experience
        directory = self.setupDir(params)
        # Set up generator
        generator = InputGenerator(
            self.gt,
            chunk_size=params["nb_timesteps"],
            batch_size=params["batch_size"],

        )
        # Set up model
        model = RNN(params)
        model = model.model
        # Set up loss
        loss = CustomLoss(
            lambda_roll=params["lambda_roll"],
            lambda_pitch=params["lambda_pitch"],
            lambda_yaw=params["lambda_yaw"],
            lambda_thrust=params["lambda_thrust"],
            loss_func=params["loss_func"],
        )
        decay =  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-6)
        # Set up loss
        optimizer = Adam(
            lr=params["lr"],
            decay=params["decay"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
        )

        # Set up callbacks
        mcp = ModelCheckpoint(
            directory + "/model.{epoch:03d}.h5",
            verbose=1,
            save_weights_only=False,
        )
        lrp = PrintLR()
        #lrs = LearningRateScheduler(step_decay)
        callbacks = [mcp, lrp, decay]
        # return all params
        return generator, model, loss, optimizer, callbacks


    def trainTalos(self, x_train, y_train, x_val, y_val, params, initial_epoch=0):
        # Talos dynamic model set up
        generator, model, loss, optimizer, callbacks = self.processParams(params)

        model.summary()
        # compile the model
        model.compile(optimizer=optimizer, loss="mse", metrics=loss.metrics)

        history = model.fit_generator(
            generator.generate(set="train"),
            steps_per_epoch=generator.steps_per_epoch_train,
            validation_data=generator.generate(set="valid"),
            validation_steps=generator.steps_per_epoch_valid,
            epochs=params["epochs"],
            callbacks=callbacks,
            max_queue_size=10,
            workers=1,
            verbose=1,
            initial_epoch=initial_epoch,
        )

        return history, model

    def search(self, params):
        t = ta.Scan(
            x=np.zeros((16, 5, 20)),  # Add dummy X or talos crashes
            y=np.zeros((16, 4)),  # Same for y
            model=self.trainTalos,
            params=params,
            experiment_no="1",
        )
        return t


if __name__ == "__main__":

    experimentname = "MaximeSuperScien_v2"
    PATH = "../resources/data_maxime.pkl"

    params = {
        "nb_layers": [4],               # model
        "nb_neurons": [256],            # model
        "cell": ["gru"],                # model
        "dropout": [0],               # model
        "dense_layer": [False],         # model
        "nb_features": [32],            # input shape
        "nb_timesteps": [8],            # input shape
        "loss_func": ["mse"],           # loss
        "lambda_roll": [100.0],         # loss
        "lambda_pitch": [100.0],        # loss
        "lambda_yaw": [100.0],          # loss
        "lambda_thrust": [100.0],       # loss
        "lr": [1e-3],                   # learning
        "decay": [1e-5],                # learning
        "batch_size": [32],             # learning
        "optimizer": ["adam"],          # learning
        "epochs": [30],                 # learning
    }

    # Initialisetraining session
    manager = SupervisedTalosTraining(experimentname)

    # Load data
    with open(PATH, "rb") as f:
        manager.gt = cPickle.load(f)

    # Train
    t = manager.search(params)
