import cv2
import json
import numpy
import os.path
# import tqdm

# json_file = '/Users/maximechoulika/Documents/Data_Training/training_GT_labels.json'
json_file = '/Users/maximechoulika/Documents/random_submission.json'
# json_file = '/Users/maximechoulika/Documents/test_submission.json'

with open(json_file) as f:
    img_keys = json.load(f)

#for file, bboxs in tqdm.tqdm(img_keys.items()):
for file, bboxs in img_keys.items():
    # img_file = '/Users/maximechoulika/Documents/Data_Training/'+file
    img_file = '/Users/maximechoulika/Documents/Data_LeaderboardTesting/'+file
    if not os.path.isfile(img_file):
        continue
    img = cv2.imread(img_file)
    img_w, img_h = img.shape[:2]
    bboxs = numpy.array(bboxs)
    img = cv2.resize(img, (0,0), fx=0.5, fy=.5)
    print(img_w, img_h, img_file)
    for bbox in bboxs:
        bbox = bbox[:8]
        xy = bbox.reshape(4, -1) * [0.5, 0.5]
        xy = numpy.round(xy)
        cv2.polylines(img, numpy.int32([xy]), True, (0,0,255), thickness=2)
    cv2.imshow("Gate detection", (img))
    cv2.waitKey(0)
