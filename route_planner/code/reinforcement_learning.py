# Basic imports
import signal
import rospy
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
import os
import datetime
import sys
import scipy.io as sio
import math
import argparse

# Tensorboard
from keras.callbacks import TensorBoard, ModelCheckpoint

# Keras imports
import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import glorot_uniform
from keras.models import load_model

# Local imports
from controllers.drone_controller import DroneController
from controllers.simulation_controller import SimulationController
from training.custom_loss import CustomLoss
from utils.data_formater import DataFormater

GOOD_DISP = '\033[92m\033[1m'
BLUE_DISP = '\033[94m\033[1m'
PURP_DISP = '\033[93m\033[1m'
BAD_DISP = '\033[91m\033[1m'
RESET_DISP = '\033[0m'

class ReinforcementLearning(object):
    """docstring for ReinforcementLearning."""

    def __init__(self, params, model=None):
        super(ReinforcementLearning, self).__init__()
        self.__initVars(params, model)

    def __initVars(self, params, model):
        """
        """
        self.experimentname = params["experimentname"]
        self.metrics = {}
        self.sigmas = params["sigmas"]
        self.gamma = params["gamma"]
        self.sigma_decay = params["sigma_decay"]

        self.drone_controller = DroneController()
        self.simulation_controller = SimulationController()

        self.model = self.__getPolicyModel(model)

        self.directory = self.__setupDir()
        self.nominal_gates_ref = self.__getNominalGatesRef()

        self.callbacks = [
            TensorBoard(
                log_dir=self.directory+"/tensorboard",
                histogram_freq=0,
                write_graph=True,
                write_images=True,
            ),
        ]

        self.nb_timestep = 8  # can be setted via model input shape
        self.nb_features = 22  # can be setted via model input shape

        self.nb_gates_nominal = 2
        self.nb_gates_ir = 1
        self.delta_t = 0.1
        self.render = True
        self.max_stack_size = 2000
        self.formater = DataFormater(nb_timesteps=self.nb_timestep)
        if self.sigma_decay:
            self.__initSigmas()

    def __initSigmas(self):
        """ Sigmas for greedyPolicy deviation
        """
        self.weight_gates = [0 for _ in range(11)]
        self.good_gate = 50
        self.bad_gate = -10
        self.sigma = 5e-2
        self.sigmas = [self.sigma for _ in range(11)]
        self.sigma_gamma = 0.5
        self.sigma_delta = 1.0005
        self.weight_max = np.ones([11, 1]) * 10000  # max is 10000
        self.weight_min = np.ones([11, 1]) * 0  # min is     0

    def __updateSigmas(self):
        """ Function to update the sigmas of the model training RL.
            in function of `self.weight_gates`
        """
        for i in range(11):
            self.sigmas[i] = self.sigma / self.sigma_delta ** self.weight_gates[i]

    def __updateWeightsGates(self, id_gate, gate_passed=True):
        """ Function to update the weights of the gate and propagate it to the
            previous gate in function of the `self.sigma_gamma`.

            Call the __updateSigmas() function at the end.
        """
        modif = self.good_gate if gate_passed else self.bad_gate
        while id_gate >= 0 and id_gate < len(self.weight_gates):
            self.weight_gates[id_gate] += modif
            modif = math.floor(modif * self.sigma_gamma)
            if modif == 0:
                break
            id_gate -= 1
        # Check limits
        a = np.array(self.weight_gates)
        a = np.min(np.concatenate((a.reshape(-1, 1), self.weight_max), axis=1), axis=1)
        a = np.max(np.concatenate((a.reshape(-1, 1), self.weight_min), axis=1), axis=1)
        self.weight_gates = a.tolist()
        self.__updateSigmas()

    def __getNominalGatesRef(self):
        """ Docstring for __getNominalGatesRef. """
        with open("../resources/nominal_gate_locations.yaml") as f:
            return yaml.load(f)

    def __getPolicyModel(self, model_path):
        """ Load last model, using our 'compute' custom loss. """
        compute = CustomLoss().compute
        model = load_model(model_path, custom_objects={"compute": compute})
        model.summary()
        return model

    def __setupDir(self):
        """ Docstring for __setupDir. """
        directory = "./graph/{}_{}".format(
            self.experimentname, datetime.datetime.now().strftime("%m%d%H%M")
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(directory+"/tensorboard"):
            os.makedirs(directory+"/tensorboard")
        return directory

    def __prepareTraining(self, x_stack, y_stack, reward_stack, run_data):
        # Order by run reward sum
        run_data = sorted(run_data, key=lambda x: x[0])
        #Check min and max reward and update
        #min, max = run_data[0][0], run_data[-1][0]
        #self.max_reward = max if max > self.max_reward else self.max_reward
        #self.min_reward = min if min > self.min_reward else self.min_reward
        # add best runs to the stack
        #mean_reward = (self.max_reward - self.min_reward) / 4. + self.min_reward
        size = len(run_data)
        if x_stack is not None:
            x_train, y_train, r_train = x_stack.copy(), y_stack.copy(), reward_stack.copy()
        else:
            x_train, y_train, r_train = None, None, None
        for _ in range(15): # DEBUG: STACK BEST
            x_train, y_train, r_train = self.__stackFrames(x_train, y_train, r_train, run_data)
            # add to the global saving stack
            if len(run_data) * 2 > size:
                x_stack, y_stack, reward_stack = self.__stackFrames(x_stack, y_stack, reward_stack, run_data)
            del run_data[-1]
            if len(run_data) == 0:
                break
        # Mix up data
        c = list(zip(x_train.tolist(), y_train.tolist(), r_train.tolist()))
        random.shuffle(c)
        x_train, y_train, r_train = zip(*c)
        # Normalize the reward
        r_train = np.array(r_train)
        r_train -= np.mean(r_train)
        r_train /= np.std(r_train) # idem
        return x_stack, y_stack, reward_stack, np.array(x_train), np.array(y_train), r_train

    def __stackFrames(self, x, y, r, data):
        _, data = data[-1]
        x_train, y_train, r_train = data
        #print(x_train, y_train, r_train)
        if x is None:
            x, y, r = np.array(x_train), np.array(y_train), np.array(r_train)
            return x, y, r
        x = np.array(np.concatenate((x, np.array(x_train))))
        y = np.array(np.concatenate((y, np.array(y_train))))
        r = np.array(np.concatenate((r, np.array(r_train))))
        return x, y, r

    def __diluteFrames(self, x_stack, y_stack, reward_stack):
        c = list(zip(x_stack.tolist(), y_stack.tolist(), reward_stack.tolist()))
        random.shuffle(c)
        x_stack, y_stack, reward_stack = zip(*c)
        x_stack, y_stack, reward_stack = np.array(x_stack), np.array(y_stack), np.array(reward_stack)
        size = x_stack.shape
        if size < self.max_stack_size:
            return x_stack, y_stack, reward_stack
        size = self.max_stack_size
        return x_stack[:size], y_stack[:size], reward_stack[:size]

    def __saveMetrics(self, episode_id, run_id, observation):
        # Display
        # print("Total reward: {}".format(int(self.reward_sum)))
        key_episode = str(episode_id % 10)
        name = key_episode+'_'+str(run_id)
        self.metrics[name] = {
            'reward': self.reward_sum,
            'nb_gate': observation['next_true_gate_id'],
            'elapsed_time': self.drone_controller.sensors.secs - self.drone_controller.start_time,
        }

    def __doneCallbackSigmas(self, observation):
        if self.sigma_decay:
            if observation["next_true_gate_id"] == 10:
                self.__updateWeightsGates(observation["next_true_gate_id"])
            else:
                self.__updateWeightsGates(
                    observation["next_true_gate_id"], gate_passed=False
                )

    def loadSigmas(self, file_name):
        """ Load sigma settings. """
        data = sio.loadmat(file_name)
        self.weight_gates = data["weight_gates"][0]
        self.good_gate = data["good_gate"][0][0]
        self.bad_gate = data["bad_gate"][0][0]
        self.sigma = data["sigma"][0][0]
        self.sigmas = data["sigmas"][0]
        self.sigma_gamma = data["sigma_gamma"][0][0]
        self.sigma_delta = data["sigma_delta"][0][0]
        self.weight_max = np.ones([11, 1]) * data["weight_max"][0][0]
        self.weight_min = np.ones([11, 1]) * data["weight_min"][0][0]

    def saveSigmas(self, name=""):
        """ Save sigma settings. """
        if self.sigma_decay:
            data = {
                "weight_gates": self.weight_gates,
                "good_gate": self.good_gate,
                "bad_gate": self.bad_gate,
                "sigma": self.sigma,
                "sigmas": self.sigmas,
                "sigma_gamma": self.sigma_gamma,
                "sigma_delta": self.sigma_delta,
                "weight_max": self.weight_max[0],
                "weight_min": self.weight_min[0],
            }
            sio.savemat(
                file_name="{}/{}_sigmas.mat".format(self.directory.rstrip("/"), name),
                mdict=data,
            )

    def prepro(self, last_observation, observation):
        """ Get input vector from raw sensor data:
            input:
                - last_observation: sensor data from previous state, to compute speed
                - observation: sensor data to pre-process
            output:
                - out: the model's input vector containing processed data
        """
        out = self.formater.sensorToData(
            last_observation,
            observation,
            self.nb_gates_nominal,
            self.nb_gates_ir,
            self.nominal_gates_ref,
        )
        return out

    def pickStates(self):
        """ Select states according to the delta_t our lstm is trained on.
            This is necessary because our step method does not record states with
            the same frequency as our data recorder.
            Set self.delta_t to 0 to deactivate.
        """
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

    def forwardPass(self, states):
        """ Perform one forward pass from policy model, on nb_timesteps observations from all_observations.
            input:
                - self.all_observations: the list containing all the observed frames
            output:
                - action: a decision made on nb_timesteps observations
        """
        action = self.model.predict(states)[0]
        return action

    def envReset(self, episode_id):
        """ Reset environment, and re-initialize variables used in
            main loop.
        """
        perturbation_id = 2
        #if episode_id % 10 == 0:
            # if we have to reset simulation
            # perturbation_id = 2
            # perturbation_id = random.randint(0,24)

            # self.simulation_controller.restart(perturbation_id)
            # time.sleep(10) # TODO: Change this for process listener

            # dump data
        try:
            print(self.metrics)
            sio.savemat(
                file_name="{}/episode{}.mat".format(self.directory.rstrip("/"), episode_id),
                mdict=self.metrics,
            )
            print("Saving done")
        except:
            pass
        self.metrics = {}
        self.metrics['map_id'] = perturbation_id

    def discountRewards(self):
        """ Take 1D float array of rewards and compute discounted reward. """
        r = np.array(self.rewards)
        discounted_r = np.zeros_like(r)
        running_add = 0
        # we go from last reward to first one so we don't have to do exponentiations
        for t in reversed(range(0, r.size)):
            running_add = (
                running_add * self.gamma + r[t]
            )  # the point here is to use Horner's method to compute those rewards efficiently
            discounted_r[t] = running_add
        discounted_r += np.array(self.rewards_smooth)
        return discounted_r

    def greedyPolicy(self, action):
        """ Draw a random sample from a gaussian distribution centered
            at 'action', and with stddeviation 'sigmas'.
        """
        return action
        # Si gates passed : update les sigmas
        # sigma / 1.001 ^ nb_run_passed (count du nb de run reussit)
        # Et on passe au sigma suivant
        ret = np.random.normal(
            action, self.sigmas[self.drone_controller.sensors.next_true_gate_id], 4
        )
        bounds = [[-1, 1], [-1, 1], [-1, 1], [-10, 30]]
        for i in range(len(ret)):
            if ret[i] < bounds[i][0]:
                ret[i] = bounds[i][0]
            elif ret[i] > bounds[i][1]:
                ret[i] = bounds[i][1]
        return np.array(ret)

    def run(self):
        """ Main loop used to launch the reinforcement learning. """
        # Setting up our environment
        x_stack, y_stack, reward_stack = None, None, None
        episode_id = 0
        run_id = 0
        starting_action = [0, 0, 0, 10]
        run_data = []

        self.max_reward, self.min_reward = 0, 0
        self.x_train, self.y_train, self.rewards, self.rewards_smooth = [], [], [], []
        self.reward_sum = 0
        self.episode_reward, self.last_episode_reward = 0, 0

        x_dict, y_dict, r_dict = {}, {}, {}

        self.envReset(episode_id=episode_id)
        last_observation = self.drone_controller.reset()
        observation, _, _ = self.drone_controller.step(None)
        action = starting_action
        prec = time.time()
        self.all_observations = []

        print("\n{}Starting episode [{}]{}".format(PURP_DISP, episode_id, RESET_DISP))
        print("\n{}Starting run [{}]{}".format(BLUE_DISP, run_id, RESET_DISP))
        while True:
            prec = time.time()
            # Add current obs to all_obs, make a decision on it
            # 0.001
            curr_state = self.prepro(last_observation, observation)
            # sigma decay if activated
            if (
                self.sigma_decay
                and last_observation["next_true_gate_id"]
                != observation["next_true_gate_id"]
            ):
                self.__updateWeightsGates(last_observation["next_true_gate_id"])
            self.all_observations.append(curr_state)
            # Predict action from state, and transform deterministic to stochastic
            # 0.01
            states = self.pickStates()
            if states is not None:
                action = self.forwardPass(states)
                action = self.greedyPolicy(action)
                action = self.formater.decode(action)

                # Log the input and label to train later
                self.x_train.append(states)
                self.y_train.append(action)
            # Remember current observation, to compute speed in prepro
            last_observation = observation
            # Do one step in the Flightgoggles environment
            # 0.1
            observation, reward, done = self.drone_controller.step(action)

            if states is not None:
                self.rewards.append(reward[0])
                self.rewards_smooth.append(reward[1])
                self.reward_sum += reward[0]
                if self.render:
                    self.drone_controller.render(states[0][-1], action)

            if done:
                # Drone end of run
                if not len(self.x_train):
                    self.all_observations = []
                    observation = self.drone_controller.reset()
                    action = None
                    continue

                # Wait for collision
                self.drone_controller.waitReset()

                # Update sigma
                self.__doneCallbackSigmas(observation)
                self.__saveMetrics(episode_id, run_id, observation)
                run_id += 1
                self.episode_reward += self.reward_sum
                run_data.append((observation['next_true_gate_id'], (self.x_train, self.y_train, self.discountRewards())))
                if self.render:
                    self.drone_controller.renderDiscount(discount_rewards)

                # give the time to reset the drone position
                time.sleep(0.5)

                # Training
                if run_id % 8 == 0:
                    x_stack, y_stack, reward_stack, x_train, y_train, r_train = self.__prepareTraining(x_stack, y_stack, reward_stack, run_data)
                    print(r_train.shape)
                    self.model.fit(
                        x=np.vstack(x_train.tolist()),
                        y=np.vstack(y_train.tolist()),
                        verbose=1,
                        sample_weight=r_train,
                        epochs=episode_id + 1,
                        initial_epoch=episode_id,
                        batch_size=8,
                        callbacks=self.callbacks,
                    )
                    x_stack, y_stack, reward_stack = self.__diluteFrames(x_stack, y_stack, reward_stack)
                    # Save model
                    if self.episode_reward > self.last_episode_reward or episode_id == 0:
                        print("Saving weights")
                        self.model.save(self.directory.rstrip('/') + "/model{}_{}.h5".format(int(episode_id), int(self.episode_reward)))
                        self.last_episode_reward = self.episode_reward
                    self.episode_reward = 0
                    # Save settings of sigma every 10 epochs
                    self.saveSigmas(name=self.experimentname)
                    # Reinitialization
                    self.envReset(episode_id=episode_id)

                    episode_id += 1
                    print("\n{}Starting episode [{}]{}".format(PURP_DISP, episode_id, RESET_DISP))
                    run_data = []
                    if run_id >= 300:
                        exit()

                # reset after training
                self.x_train, self.y_train, self.rewards, self.rewards_smooth = [], [], [], []
                self.reward_sum = 0
                self.all_observations = []
                observation = self.drone_controller.reset()
                action = starting_action
                print("\n{}Starting run [{}]{}".format(BLUE_DISP, run_id, RESET_DISP))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None, help="Model to train")
    parser.add_argument(
        "-s", "--sigma", type=str, default=None, help="Sigmas for training"
    )
    args = parser.parse_args()

    model_name = args.model
    sigma_name = args.sigma

    rospy.init_node("alpha", anonymous=True)

    params = {
        "experimentname": "RL",
        "sigmas": [5e-2 for _ in range(11)],
        "gamma": 0.95,
        "sigma_decay": True,
    }

    rl = ReinforcementLearning(params, model=model_name)

    if sigma_name is not None and params["sigma_decay"]:
        rl.loadSigmas(sigma_name)

    def callback(sig, frame):
        if rl.sigma_decay:
            rl.saveSigmas(name=params["experimentname"])
        sys.exit(0)
        # rospy.signal_shutdown("stop")

    signal.signal(signal.SIGINT, callback)

    while True:
        rl.run()
