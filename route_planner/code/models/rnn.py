import keras
from model_interface import ModelInterface

from keras.models import Model, model_from_json, Sequential
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Lambda, LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM


class RNN(ModelInterface):
    """Base class for RNN model architecture."""

    def __init__(self, params=None):
        ModelInterface.__init__(self)
        if params:
            self.params = params
        else:
            self.updateParams()
        self.cell = self.getcell()
        self.model = self.initModel()

    def updateParams(self):
        RNNparams = {
            "nb_timesteps": 6,
            "nb_layers": 3,
            "nb_neurons": 128,
            "dense_layer": False,
            "cell": "gru",
            "dropout": 0.2,
        }
        self.params.update(RNNparams)

    def getcell(self):
        if self.params["cell"] == "lstm":
            return LSTM
        if self.params["cell"] == "gru":
            return GRU
        if self.params["cell"] == "plstm":
            return PLSTM

    def initModel(self):
        """
        Function used to initialize a RNNModel.
        Input shape:
            (batch_size, time_steps, input_length)
        """
        input_shape = (self.params["nb_timesteps"], self.params["nb_features"])
        x = input_tensor = Input(input_shape)
        for i in range(1, self.params["nb_layers"] + 1):
            # Make sure the LSTM respects many-to-one architecture
            return_sequences = True if i < self.params["nb_layers"] else False
            x = self.cell(
                self.params["nb_neurons"],
                return_sequences=return_sequences,
                dropout=self.params["dropout"],
            )(x)
        if self.params['dense_layer'] == True:
            x = Dense(20)(x)
        x = output_tensor = Dense(4)(x)
        model = Model(input_tensor, output_tensor)
        return model


if __name__ == "__main__":
    rnn = RNN().model
    rnn.summary()
    plot_model(rnn, to_file='model.png', show_shapes=True)
