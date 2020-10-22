# Basic imports
import math
import numpy as np

# Ros imports
from pyquaternion import Quaternion
from transformations import (
    rotation_matrix,
    angle_between_vectors,
    vector_product,
)

# Local imports
from gate_corner import chollet_gate_fix_missing

gates = [
    'Gate10',
    'Gate21',
    'Gate2',
    'Gate13',
    'Gate9',
    'Gate14',
    'Gate1',
    'Gate22',
    'Gate15',
    'Gate23',
    'Gate6'
]
gates_corners = {
    'Gate10': [2, 1, 3, 4],
    'Gate21': [1, 2, 3, 4],
    'Gate2': [2, 1, 4, 3],
    'Gate13': [2, 1, 4, 3],
    'Gate9': [1, 2, 4, 3],
    'Gate14': [2, 1, 4, 3],
    'Gate1': [1, 2, 3, 4],
    'Gate22': [1, 2, 3, 4],
    'Gate15': [1, 2, 3, 4],
    'Gate23': [2, 1, 3, 4],
    'Gate6': [2, 1, 3, 4],
}

def getnormal(arr):
    p1 = arr[0, :]
    p2 = arr[1, :]
    p3 = arr[2, :]

    v1 = p3 - p1
    v2 = p2 - p1

    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return np.asarray([a, b, c])


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def getRateThrust(line):
    if line['angular_rates'] is None:
        return None
    return [line['angular_rates']['x'], line['angular_rates']['y'], line['angular_rates']['z'], line['thrust']['z']]

def getGatesCorners(line, nb_gates_ir):
    gates = []
    for i in range(nb_gates_ir):
        if len(line["nextGates"]) > i:
            gate = line["markers"][line["nextGates"][i]]
            order = gates_corners[line["nextGates"][i]]
            gate = [
                [gate[str(order[0])][p] / v - 1 for p, v in {"x": 80, "y": 60}.items()]
                if str(order[0]) in gate
                else [],
                [gate[str(order[1])][p] / v - 1 for p, v in {"x": 80, "y": 60}.items()]
                if str(order[1]) in gate
                else [],
                [gate[str(order[2])][p] / v - 1 for p, v in {"x": 80, "y": 60}.items()]
                if str(order[2]) in gate
                else [],
                [gate[str(order[3])][p] / v - 1 for p, v in {"x": 80, "y": 60}.items()]
                if str(order[3]) in gate
                else [],
            ]
            gates.append(gate)
        else:
            gates.append(None)
    return gates

def getDroneSpeed(prec, line):
    delta = abs(line["secs"] - prec["secs"])
    if delta == 0:
        return [0.0,0.0,0.0]
    speed = [(line["trans"][e] - prec["trans"][e])/delta for e in ["x", "y", "z"]]
    #if speed[0] > 3 or speed[1] > 3 or speed[2] > 3:
    #    return None
    speed2 = Quaternion([line["rot"][j] for j in [1,2,3,0]]).inverse.rotate(speed)
    #speed2 = speed
    return speed2

def getDistToNominal(line, nb_gates_nominal, nominal_gates_ref):
    gates = []
    for i in range(nb_gates_nominal):
        if len(line["nextGates"]) > i:
            arr = nominal_gates_ref[line["nextGates"][i]]["nominal_location"]
            center = np.array([np.mean([e[i] for e in arr]) for i in range(3)])
            drone = np.array([line["trans"][e] for e in ["x", "y", "z"]])
            dist = np.linalg.norm(drone - center)
            gates.append(dist)  # TODO: Change this normalisation
        else:
            gates.append(None)
    return gates

def getEulerToDirection(line, nb_gates_nominal, nominal_gates_ref):
    gates = []
    for i in range(nb_gates_nominal):
        if len(line["nextGates"]) > i:
            arr = nominal_gates_ref[line["nextGates"][i]]["nominal_location"]
            center = np.array([np.mean([e[j] for e in arr]) for j in range(3)])
            gate = center - np.array([line["trans"][e] for e in ["x", "y", "z"]])
            drone = np.array(Quaternion([line["rot"][j] for j in [1,2,3,0]]).rotate([1, 0, 0]))
            M = rotation_matrix(
                angle_between_vectors(gate, drone), np.cross(gate, drone)
            )
            gates.append(rotationMatrixToEulerAngles(M).tolist())
        else:
            gates.append(None)
    return gates

def fromBarToEuler(bar, dist_to_nominal, rot):
    #just to be simpler
    bar_y = -bar[1]
    bar_x = bar[0]
    #camera angles
    cam_x = 50
    cam_y = 50
    angle_x = cam_x/2*bar_x
    angle_y = cam_y/2*bar_y
    angle_x = angle_x/180*math.pi
    angle_y = angle_y/180*math.pi
    y = -math.tan(angle_x)*dist_to_nominal
    z = math.tan(angle_y)*dist_to_nominal
    x = dist_to_nominal
    drone_to_true_gate = np.array([x,y,z])
    #drone_dir = np.array(Quaternion([rot[j] for j in [1,2,3,0]]).rotate([1, 0, 0]))
    drone_dir = [1, 0, 0]
    try:
        M = rotation_matrix(
            angle_between_vectors(drone_to_true_gate, drone_dir), vector_product(drone_to_true_gate, drone_dir)
        )
        ret = rotationMatrixToEulerAngles(M).tolist()
    except:
        ret = [0.0,0.0,0.0]
    return ret

class DataFormater(object):
    """docstring for DataFormater."""
    def __init__(self, nb_timesteps=1):
        super(DataFormater, self).__init__()
        # Norme
        self.norm_input = np.array(
            [
                # should put PI there max = PI; min = -PI
                3.14, 3.14, 3.14,    # euler D-N 1er
                3.14, 3.14, 3.14,    # euler D-N 2e
                10,         # dlr   Drone -> [-1:0]
                40, 70,     # dist  D-N
                20, 20, 20, # speed Drone
                1, 1, 1, 1, # ir   Gate1 part1
                1, 1, 1, 1, # ir   Gate1 part2
                1, 1,       # ir   Gate1 center
                1, 1, 1, 1, # ir   Gate1 part1
                1, 1, 1, 1, # ir   Gate1 part2
                1, 1,       # ir   Gate1 center
            ]  # IMU   Gate2
        )
        self.norm_input_add = np.array(
            [
                # should put PI there max = PI; min = -PI
                0, 0, 0,    # euler D-N 1er
                0, 0, 0,    # euler D-N 2e
                0,         # dlr   Drone -> [-1:0]
                0, 0,     # dist  D-N
                0, 0, 0, # speed Drone
                0, 0, 0, 0, # ir   Gate1 part1
                0, 0, 0, 0, # ir   Gate1 part2
                0, 0,       # ir   Gate1 center
                0, 0, 0, 0, # ir   Gate1 part1
                0, 0, 0, 0, # ir   Gate1 part2
                0, 0,       # ir   Gate1 center
            ]  # IMU   Gate2
        )
        self.norm_output = np.array([1, 1, 1, 20])
        self.norm_output_add = np.array([0, 0, 0, -10])
        # setup
        self.nb_timesteps = nb_timesteps
        self.resetFrames()

    def rawToData(self, prec, line, nb_gates_nominal, nb_gates_ir, nominal_gates_ref):
        id = line['nextGates'][0] if len(line['nextGates']) != 0 else 'Gate6' # TODO: this is dirty change it to something nicer
        out = {
            'nextGate': id,
            "run_id": line["run_id"],
            "secs": line["secs"],
            "dlr": line["dlr"],
            "rot": line['rot'],
            "gatesCorners": getGatesCorners(line, nb_gates_ir+1),
            "distDroneNominal": getDistToNominal(line, nb_gates_nominal, nominal_gates_ref),
            "droneSpeed": getDroneSpeed(prec, line),
            "eulerDroneNominal": getEulerToDirection(line, nb_gates_nominal, nominal_gates_ref),
            "rateThrust": getRateThrust(line)
        }
        return out

    def sensorToData(self, prec, line, nb_gates_nominal, nb_gates_ir, nominal_gates_ref):
        line.update({'nextGates':gates[line['next_true_gate_id']:]})
        out = {
            'nextGates': line['nextGates'][0],
            "secs": line["secs"],
            "dlr": line["dlr"],
            "rot": line['rot'],
            "gatesCorners": getGatesCorners(line, nb_gates_ir+1),
            "distDroneNominal": getDistToNominal(line, nb_gates_nominal, nominal_gates_ref),
            "droneSpeed": getDroneSpeed(prec, line),
            "eulerDroneNominal": getEulerToDirection(line, nb_gates_nominal, nominal_gates_ref)
        }
        return out

    ''' Formating part
    '''
    def format_input(self, frame):
        chollet1, n = chollet_gate_fix_missing([frame['gatesCorners'][0]])
        chollet2, n = chollet_gate_fix_missing([frame['gatesCorners'][1]])
        input_array = []
        #if n == 4:
        input_array += frame['eulerDroneNominal'][0]
        #else:
        #    input_array += fromBarToEuler(chollet[-2:], frame['distDroneNominal'][0], frame['rot'])
        if not frame['eulerDroneNominal'][1]: # if it is the last gate
            frame['eulerDroneNominal'][1] = frame['eulerDroneNominal'][0]
        input_array += frame['eulerDroneNominal'][1]
        input_array += [frame['dlr']]
        if not frame['distDroneNominal'][1]:
            frame['distDroneNominal'][1] = frame['distDroneNominal'][0]
        input_array += frame['distDroneNominal']
        input_array += frame['droneSpeed']
        input_array += chollet1
        input_array += chollet2
        return np.array(input_array, dtype=np.float32)

    def format_output(self, frame):
        output_array = frame['rateThrust']
        return np.array(output_array, dtype=np.float32)

    ''' Frame stacking part
    '''
    def resetFrames(self):
        self.frames = []

    def stackFrames(self, data):
        if len(self.frames) == 0:
            self.frames = np.repeat([data], self.nb_timesteps, axis=0)
        else:
            self.frames = np.concatenate((self.frames, [data]))
        data = self.frames.copy()
        self.frames = np.delete(self.frames, 0, 0)
        return data.reshape(1, data.shape[0], data.shape[1])

    ''' Frame encoding/decoding part
    '''
    def encode(self, input, output=None):
        input = input + self.norm_input_add
        input = np.true_divide(input, self.norm_input)
        if output is not None:
            output = output + self.norm_output_add
            output = np.true_divide(output, self.norm_output)
            return input, output
        return input

    def decode(self, output, input=None):
        output = np.multiply(output, self.norm_output)
        output = output - self.norm_output_add
        if input is not None:
            input = np.multiply(input, self.norm_input)
            input = input - self.norm_input_add
            return input, output
        return output
