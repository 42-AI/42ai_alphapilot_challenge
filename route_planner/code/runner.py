#!/home/ubuntu/AlphaPilot/route_planner/venv/bin/python
# Basic imports
import rospy
import numpy as np
import yaml
import os

# Keras imports
from keras.models import load_model

# Local imports
from controllers.drone_controller import DroneController
from training.custom_loss import CustomLoss
from utils.data_formater import DataFormater

class Runner(object):
    """docstring for Runner."""

    def __init__(self, model=None):
        super(Runner, self).__init__()
        self.__initVars(model)

    def __initVars(self, model):
        self.drone_controller = DroneController()
        self.model = self.loadModel(model)
        self.nominal_gates_ref = self.loadNominalGatesRef()
        self.nb_timestep = 8  # can be setted via model input shape
        self.nb_features = 22  # can be setted via model input shape
        self.nb_gates_nominal = 2
        self.nb_gates_ir = 1
        self.delta_t = 0.1
        self.formater = DataFormater(nb_timesteps=self.nb_timestep)

    def loadNominalGatesRef(self):
        with open("../resources/nominal_gate_locations.yaml") as f:
            return yaml.load(f)

    def loadModel(self, model_path):
        compute = CustomLoss().compute
        model = load_model(model_path, custom_objects={"compute": compute})
        model.summary()
        return model

    def prepro(self, last_observation, observation):
        out = self.formater.sensorToData(
            last_observation,
            observation,
            self.nb_gates_nominal,
            self.nb_gates_ir,
            self.nominal_gates_ref,
        )
        return out

    def pickStates(self):
        """ Select nb_timesteps frames to feed them the RNN for prediction. """
        out = []
        states = self.all_observations
        t = states[-1]["secs"]
        out[0:0] = [self.formater.encode(self.formater.format_input(states[-1]))]
        for state in reversed(states[:-1]):
            if len(out) == self.nb_timestep:
                return np.array(out).reshape(1, self.nb_timestep, self.nb_features)
            if t - state["secs"] > self.delta_t:
                out[0:0] = [self.formater.encode(self.formater.format_input(state))]
                t = state["secs"]
        return None

    def run(self):
        """ Main loop used to launch the model prediction loop. """
        # Setting up our environment
        action = None

        last_observation = self.drone_controller.reset()
        observation, _, _ = self.drone_controller.step(None)
        self.all_observations = []
        while not rospy.is_shutdown():
            curr_state = self.prepro(last_observation, observation)
            self.all_observations.append(curr_state)
            states = self.pickStates()
            if states is not None:
                action = self.model.predict(states)[0]
                action = self.formater.decode(action)

            last_observation = observation
            observation, _, done = self.drone_controller.step(action)

            if done:
                # Drone end of run
                # give the time to reset the drone position
                # reset after training
                self.all_observations = []
                observation = self.drone_controller.reset()
                action = None

if __name__ == "__main__":
    model_path = os.path.dirname(os.path.realpath(__file__)) +'/weights/model.049.h5'
    rospy.init_node("alpha", anonymous=True)

    Runner(model=model_path).run()
