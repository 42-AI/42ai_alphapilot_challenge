# Basic imports
import json
import yaml
from tqdm import tqdm

# Local imports
from utils.data_formater import DataFormater

class DataLoader(object):
    """docstring for DataLoader."""

    def __init__(self, json_file=None, yaml_file=None, nb_gates_nominal=2, nb_gates_ir=1):
        self.data = None
        self.json_file = json_file
        self.yaml_file = yaml_file
        self.nb_gates_nominal = nb_gates_nominal
        self.nb_gates_ir = nb_gates_ir
        self.loadData(json_file=self.json_file, yaml_file=self.yaml_file)
        self.data = self.augmentData()

    def loadData(self, json_file, yaml_file):
        if not json_file or not yaml_file:
            print("Error, no file given in DataLoader.loadData() !")
            exit(1)
        with open(json_file) as f:
            self.raw_data = json.load(f)
        with open(yaml_file) as f:
            self.nominal_gates_ref = yaml.load(f)

    def augmentData(self):
        out = []
        print("Loading data")
        for i in tqdm(range(1, len(self.raw_data))):
            prec = self.raw_data[i - 1]
            line = self.raw_data[i]
            if not prec["run_id"] == line["run_id"]:
                continue
            subout = DataFormater().rawToData(prec, line, self.nb_gates_nominal, self.nb_gates_ir, self.nominal_gates_ref)
            if subout["droneSpeed"] is not None:
                out.append(subout)
        return out

if __name__ == "__main__":
    import matplotlib  # Mac bug fix
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    dl = DataLoader(json_file="../../resources/dumps.json", yaml_file="../../resources/nominal_gate_locations.yaml")
    dl.loadParams()

    x = [e["rateThrust"] for e in dl.data if e["rateThrust"] is not None]
    lineObjects = plt.plot(x)
    plt.legend(iter(lineObjects), ("ax", "ay", "az", "z"))
    plt.show()
