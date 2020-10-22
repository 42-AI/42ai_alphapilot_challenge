import keras
from model_interface import ModelInterface

from keras.models import Model, model_from_json, Sequential
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Lambda, LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM


class NN(ModelInterface):
    """Base class for RNN model architecture."""

    def __init__(self, params=None):
        ModelInterface.__init__(self)
        if params:
            self.params = params
        else:
            self.updateParams()
        self.model = self.initModel()

    def updateParams(self):
        NNparams = {"nb_layers": 3, "nb_neurons": 256, "dropout": 0.1}
        self.params.update(NNparams)

    def initModel(self):
        """
        Function used to initialize a NNModel.
        Input shape:
            (batch_size, time_steps, input_length)
        """
        input_shape = (self.params["nb_features"],)
        x = input_tensor = Input(input_shape)
        x = Dense(self.params["nb_neurons"], activation="relu")(x)
        for i in range(2, self.params["nb_layers"] + 1):
            x = Dense(self.params["nb_neurons"], activation="relu")(x)
            if self.params["dropout"]:
                x = Dropout(self.params["dropout"])(x)
        x = output_tensor = Dense(4)(x)
        model = Model(input_tensor, output_tensor)
        return model


if __name__ == "__main__":
    rnn = NN().model
    rnn.summary()
