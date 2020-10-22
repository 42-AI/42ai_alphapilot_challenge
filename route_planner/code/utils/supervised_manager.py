# Basic imports
import talos as ta
import numpy as np
import cPickle
import os

# Keras imports
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

# Local imports
from models.rnn import RNN
from models.nn import NN
from training.ground_truth import GroundTruth
from training.input_generator import InputGenerator


class SupervisedManager(object):
    """docstring for SupervisedManager."""
    def __init__(self, experimentname=None):
        # class attributes
        self._i = 0
        self.experimentname = experimentname
        self.model = None
        self.weightpath = None
        self.loss = None
        self.callbacks = []

    def setWeights(self, model, weightpath=None, multi_gpu=False):
        self.model = model
        self.weightpath = weightpath
        if weightpath:
            print("[*] Loading weights...")
            self.model.load_weights(weightpath, by_name=True)
            print("Done\n")

    def setModel(self, modelpath=None):
        self.model = load_model(modelpath)

    def setFreeze(self, freeze):
        self.freeze = freeze
        for layer in self.model.layers:
            layer.trainable = not layer.name in freeze

    def setLoss(self, loss):
        self.loss = loss

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss
        )

    def setGroundTruth(self, datapath, ratio_valid=0.1):
        # PATH = "../resources"
        self.gt = GroundTruth(path=datapath, split=ratio_valid)

    def setInputGenerator(self, nb_timesteps, batch_size):
        self.batch_size = batch_size
        self.generator = InputGenerator(
            self.gt, chunk_size=nb_timesteps, batch_size=batch_size
        )

    def setCallbacks(self, callbacks):
        self.callbacks.append(callbacks)

    def train(self, epochs=15, initial_epoch=0):
        history = self.model.fit_generator(
            self.generator.generate(set="train"),
            steps_per_epoch=self.generator.steps_per_epoch_train,
            validation_data=self.generator.generate(set="valid"),
            validation_steps=self.generator.steps_per_epoch_valid,
            epochs=epochs,
            callbacks=self.callbacks,
            max_queue_size=10,
            workers=1,
            verbose=1,
            initial_epoch=initial_epoch,
        )
        return history

if __name__ == "__main__":

    experimentname = "NN"
    PATH = "../resources/data_killian.pkl"

    train_name = experimentname
    directory = "./graph/{}".format(train_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    manager = SupervisedManager(experimentname)

    with open(PATH, "rb") as f:
        manager.gt = cPickle.load(f)

    manager.model = NN().model
    manager.setInputGenerator(nb_timesteps=None, batch_size=16)
    manager.setLoss("mse")
    manager.setOptimizer("adam")
    manager.compile()
    manager.model.summary()

    tbd = TensorBoard(
        log_dir=directory, histogram_freq=0, write_graph=True, write_images=True
    )
    manager.setCallbacks(tbd)
    mcp = ModelCheckpoint(
        directory + "/model.{epoch:03d}.h5", verbose=1, save_weights_only=False
    )
    manager.setCallbacks(mcp)

    # 6. train le model
    manager.train()
