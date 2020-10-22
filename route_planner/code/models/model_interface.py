import abc


class ModelInterface(object):
    """Parent class to every type of model to be implemented."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.params = {
            "nb_features": 20,
            "lr": 1e-4,
            "batch_size": 16,
            "optimizer": "adam",
            "loss": "mse",
            "epochs": 5,
        }

    @abc.abstractmethod
    def updateParams(self):
        pass

    @abc.abstractmethod
    def initModel(self):
        pass
