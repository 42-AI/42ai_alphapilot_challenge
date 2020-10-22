from glob import glob
import cv2
import numpy as np

from ssd.models.ssd_model import *
from ssd.training_manager import TrainingManager
from ssd.formulas import rbox3_to_polygon
from ssd.utils.utils import get_model_by_name

if __name__ == "__main__":

    import os
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    TESTDIR = "/Users/maximechoulika/Documents/Data_LeaderboardTesting"
    WEIGHTPATH = None
    MODEL_NAME = "DSOD300"
    CONFIDENCE = 0.35

    predict = TrainingManager("predict")

    model_name = input("Model to test (default {}): ".format(MODEL_NAME))

    """Setup of model for training"""
    if model_name != '':
        MODEL_NAME = model_name
    model = get_model_by_name(MODEL_NAME)

    print(model_name + " loaded...")

    dir_name = input("Directory to process (default {}): ".format(TESTDIR))

    if dir_name != '':
        TESTDIR = dir_name

    weights = input("Weights file (required): ")
    if weights == '':
        print("This parameter is required")
        exit()
    WEIGHTPATH = weights

    print("#"*30)
    print("Model:    {}".format(model_name))
    print("Testing:  {}".format(TESTDIR))
    print("Weights:  {}".format(WEIGHTPATH))
    print("#"*30)

    predict.setModel(model, weightpath=WEIGHTPATH)

    waiter = True
    files = []

    for e in ['*.JPG', '*.png']:
        files += glob(TESTDIR + "/" + e)

    for file in files:
        print("Processing file {}...".format(file))
        image = cv2.imread(file, 3)  # reads image as BGR
        b, g, r = cv2.split(image)  # get BGR
        image = cv2.merge([r, g, b])  # switch it to RGB

        result = predict.predict(image, conf_thresh=CONFIDENCE)

        img = cv2.imread(file)
        img = cv2.resize(img, (0,0), fx=0.75, fy=0.75)
        img_w, img_h = np.array(img).shape[:2]

        bboxs = result[:,0:4]
        quads = result[:,4:12]
        rboxes = result[:,12:17]

        #boxes = np.asarray([rbox3_to_polygon(r) for r in rboxes])
        #xy = boxes
        #xy = xy * [img_w, img_h]
        #xy = np.round(xy)
        #xy = xy.astype(np.int32)

        boxes = np.asarray([t.reshape(4, -1) * [img_w, img_h] for t in quads])
        xy = boxes
        xy = np.round(xy)
        xy = xy.astype(np.int32)

        cv2.polylines(img, tuple(xy), True, (0,0,255), thickness=2)
        cv2.imshow("Gate detection", (img))
        if waiter:
            cv2.waitKey(0)
            waiter = False
        cv2.waitKey(100)
