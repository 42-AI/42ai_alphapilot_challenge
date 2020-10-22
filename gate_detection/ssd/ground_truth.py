import numpy
import json
import os
from glob import glob
from PIL import Image
import numpy as np


class GroundTruth(object):
    """GroundTruthUtils for AlphaPilot gate detection module.

    Attributes:
        datapath (str): Description of ``.
        data (list):
        nimg (int):
        nbbox (int):
        avgbbox (float):
    """

    def __init__(self, datapath=None, data=None, split=False):
        super().__init__()
        # class attributes
        # datapath = datapath.rstrip('/')
        self.datapath = []
        self.data = {}
        self.num_objects = 0
        self.num_without_annotation = 0
        if datapath and not split:
            # get JSON filename
            if type(datapath) == list:
                for d in datapath:
                    self.addData(d)
            else:
                self.addData(datapath)

        if data:
            self.data = data

    def addData(self, datapath):
        # get JSON filename
        datapath = datapath.rstrip('/')
        jsons = [file for file in glob("{}/*.json".format(datapath))]
        if len(jsons) != 1:
            print("No json or more than one file in the specified path.")
            exit(1)
        # Load JSON data
        with open(jsons[0]) as f:
            data = json.load(f)
        # data = data.copy()
        self.datapath.append(datapath)

        if not hasattr(self, "size_image"):
            self.size_image = Image.open(datapath+'/'+next(iter(data.keys())).split('/')[-1]).size

        bboxs = []

        for key, value in data.items():
            # If there is no box remove it from data
            if len(value) == 0 or len(value[0]) == 0:
                self.num_without_annotation += 1
                # del self.data[key]
                continue

            img_file = "{}/{}".format(datapath, key.split('/')[-1])
            # If file does not exist do not count it
            if not os.path.exists(img_file):
                del self.data[key]
                continue

            self.num_objects += len(value)
            img_width, img_height = Image.open(img_file).size
            # Extract box
            bboxs = np.array(value, np.float32)
            bboxs = bboxs[:,0:8]
            bboxs[:,1::2] /= float(img_width)
            bboxs[:,0::2] /= float(img_height)
            bboxs = np.concatenate([bboxs, np.ones([bboxs.shape[0],1])], axis=1)
            # Add data to ground truth
            self.data[img_file] = bboxs
        self.num_images = len(self.data)

    def __str__(self):
        s = "GroundTruths\n"
        s += "%s\n" % (25 * "-")
        s += "%-16s %8i\n" % ("images", self.num_images)
        s += "%-16s %8i\n" % ("objects", self.num_objects)
        s += "%-16s %8.2f\n" % ("per image", self.num_objects / self.num_images)
        s += "%-16s %8i\n" % ("no annotation", self.num_without_annotation)
        s += "%-16s %8s\n" % ("size images", self.size_image)
        return s

    def split(self, ratio_valid=0.1):

        size_valid = int(0.1 * self.num_images)
        size_train = self.num_images - size_valid

        data_valid = dict(list(self.data.items())[0:size_valid])
        data_train = dict(list(self.data.items())[size_valid:])

        gt_train = GroundTruth(self.datapath, data=data_train)
        gt_valid = GroundTruth(self.datapath, data=data_valid)

        return gt_train, gt_valid

if __name__ == "__main__":
    datapath = "../generated"
    gt = GroundTruth(datapath)
    gt.split()
    print(gt)
