import cPickle
from ground_truth import GroundTruth

PATH = "../resources/"
json_file = PATH + "dumps_map1.json"
yaml_file = PATH + "nominal_gate_locations.yaml"
gt = GroundTruth(json_file=json_file, yaml_file=yaml_file, split=0.1, nb_timesteps=8)

with open("../resources/data_maxime.pkl", "wb") as f:
    cPickle.dump(gt, f)

PATH = "../resources/"
json_file = PATH + "test_dumps_map1.json"
yaml_file = PATH + "nominal_gate_locations.yaml"
gt = GroundTruth(json_file=json_file, yaml_file=yaml_file)

with open("../resources/data_test.pkl", "wb") as f:
    cPickle.dump(gt, f)
