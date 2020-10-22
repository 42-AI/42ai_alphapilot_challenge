from glob import glob
import numpy as np
import math
import cv2

from utils.data_loader import DataLoader
from utils.gate_corner import chollet_gate_fix_missing
from utils.data_formater import DataFormater

def remove_empty_frames(raw_data):
    # Clean les frames de debut si pas de diff entre (n) et (n+1)
    for key, val in raw_data.items():
        data = raw_data[key] # data est une liste avec toutes les frames
        # clean empty rateThrust
        i = 0
        while i < len(data):
            if not data[i]["rateThrust"]:
                del data[i]
            else:
                i += 1
        # Clean static drone frames
        while len(data) > 1:
            frame = data[1]
            if frame and frame["droneSpeed"][0] == 0 and frame["droneSpeed"][1] == 0 and frame["droneSpeed"][2] == 0:
                del data[0]
            else:
                break
        raw_data[key] = data
    return raw_data

class GroundTruth(object):
    """docstring for GroundTruthUtils."""
    def __init__(self, path=None, json_file=None, yaml_file=None, split=0.1, nb_timesteps=6):
        self.split = split

        # Search json and yaml in path if path not None
        # else use directly json_file and yaml_file
        if path:
            path = path.rstrip('/')
            json_file = [file for file in glob("{}/*.json".format(path))]
            yaml_file = [file for file in glob("{}/*.yaml".format(path))]
            if len(json_file) != 1:
                print("No json or more than one file in the specified path.")
                exit(1)
            if len(yaml_file) != 1:
                print("No yaml or more than one file in the specified path.")
                exit(1)
            json_file = json_file[0]
            yaml_file = yaml_file[0]

        self.path = path
        self.json_file = json_file
        self.yaml_file = yaml_file

        self.data_formater = DataFormater(nb_timesteps=nb_timesteps)
        self.data_loader = DataLoader(json_file=json_file, yaml_file=yaml_file, nb_gates_nominal=2, nb_gates_ir=1)
        data = np.array(self.data_loader.data)
        self.init(data)

    def init(self, data):
        bad_runs = [
            3, 11, 25, 28, 46, 50, 51,
            54, 55, 58, 59, 76, 80,
        ]

        # Clean data and store it in a nice dictionnary
        self.data = {}
        for frame in data:
            if int(frame['run_id']) in bad_runs:
                continue
            if frame['run_id'] not in self.data:
                self.data[frame['run_id']] = []
            id = frame['run_id']
            del frame['run_id']
            self.data[id].append(frame)

        self.data = self.__cleanRuns(self.data)

        # Store informations
        self.nb_runs = len(self.data)
        self.nb_frames = 0 # TODO: Add this

        # Set the idxs for the input generator
        self.idxs = []
        for idx in self.data.keys():
            self.idxs += [(idx, n)for n in range(len(self.data[idx]))]
        self.idxs = np.array(self.idxs)

        self.__split()

    def __formatFrames(self, raw_data):
        for key, val in raw_data.items():
            data = raw_data[key] # data est une liste avec toutes les frames
            data_array = []
            for frame in data:
                # INPUT
                if not frame['eulerDroneNominal'][0]:
                    continue
                input_array = self.data_formater.format_input(frame)
                # OUTPUT
                output_array = self.data_formater.format_output(frame)
                data_array.append([input_array, output_array])
            raw_data[key] = np.array(data_array)
        return raw_data

    def __cleanRuns(self, raw_data):
        raw_data = remove_empty_frames(raw_data)
        raw_data = self.__formatFrames(raw_data)
        return raw_data

    def __split(self):

        self.nb_sample = self.idxs.shape[0]

        size_valid = int(self.split * self.nb_sample)
        size_train = self.nb_sample - size_valid

        self.idxs_train = self.idxs[size_valid:]
        self.idxs_valid = self.idxs[0:size_valid]

if __name__ == "__main__":
    import time
    import pickle
    import cv2

    PATH = "../resources/"
    json_file = PATH + "test_dumps_map1.json"
    yaml_file = PATH + "nominal_gate_locations.yaml"
    gt = GroundTruth(json_file=json_file, yaml_file=yaml_file)

    #with open('data_killian.pkl', 'r') as f:
    #    gt = pickle.load(f)

    data = []
    for k, v in gt.data.items():
        for x in v[:, 0]:
            #print('Euler 1', ["{0:0.2f}".format(i) for i in x[:3]])
            #print('Euler 2', ["{0:0.2f}".format(i) for i in x[3:6]])
            #print('drl', ["{0:0.2f}".format(i) for i in x[6:7]])
            #print('dist', ["{0:0.2f}".format(i) for i in x[7:9]])
            #print('speed', ["{0:0.2f}".format(i) for i in x[9:12]])
            #print('corners', ["{0:0.2f}".format(i) for i in x[12:20]])
            print("Next")
            w, h = 400, 400
            img = np.zeros((w, h, 3), np.uint8)
            bbox = x[12:20]
            xy = bbox.reshape(4, -1) * [w, h]
            xy = xy[:,[1, 0]]
            xy = np.round(xy)
            cv2.polylines(img, np.int32([xy]), True, (0,0,255), thickness=2)
            cv2.imshow("Gate detection", (img))
            cv2.waitKey(0)

            data += [x]
    np.array(data, dtype=np.float32)
