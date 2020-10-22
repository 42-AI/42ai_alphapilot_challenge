import numpy as np
import cv2
from random import random, shuffle

from .ground_truth import GroundTruth
from .prior_util import PriorUtil
from .models.ssd_model import *
from .utils.geometry import check_coordinates


class InputGenerator(object):
    """docstring for InputGenerator"""

    def __init__(
        self,
        gt_obj,
        prior_util,
        batch_size=32,
        input_size=(640, 480),
        debug=False
    ):
        super().__init__()
        self.gt_obj = gt_obj
        self.prior_util = prior_util
        self.batch_size = batch_size
        self.input_size = input_size
        self.debug = debug

        # metrics
        # TODO: Delete these lines
        # self.resize = np.divide(np.array(self.input_size), np.array(self.gt_obj.size_image))
        # self.resize = np.append(np.tile(self.resize, 4), [1.])
        self.mean = np.array([104, 117, 123])
        # Augmentation
        self.grayscale = 0.0
        self.saturation = 0.0
        self.brightness = 0.0
        self.contrast = 0.0
        self.lighting = 0.0
        self.noise = 0.0
        self.horizontal_flip = 0.0
        self.vertical_flip = 0.0
        self.random_sized_crop = 0.0

    def __str__(self):
        s = "InputGenerator\n"
        s += "%s\n" % (25 * "-")
        s += "%-16s %8.2f\n" % ("grayscale", self.grayscale)
        s += "%-16s %8.2f\n" % ("saturation", self.saturation)
        s += "%-16s %8.2f\n" % ("brightness", self.brightness)
        s += "%-16s %8.2f\n" % ("contrast", self.contrast)
        s += "%-16s %8.2f\n" % ("lighting", self.lighting)
        s += "%-16s %8.2f\n" % ("noise", self.noise)
        s += "%-16s %8.2f\n" % ("horizontal flip", self.horizontal_flip)
        s += "%-16s %8.2f\n" % ("vertical flip", self.vertical_flip)
        s += "%-16s %8.2f\n" % ("random crop", self.random_sized_crop)
        s += "%-16s %8s\n" % ("input size", self.input_size)
        return s

    """Settings fucntions"""

    def setGrayscale(self, n):
        self.grayscale = n

    def setSaturation(self, n):
        self.saturation = n

    def setBrightness(self, n):
        self.brightness = n

    def setContrast(self, n):
        self.contrast = n

    def setLighting(self, n):
        self.lighting = n

    def setNoise(self, n):
        self.noise = n

    def setHorizontalFlip(self, n):
        self.horizontal_flip = n

    def setVerticalFlip(self, n):
        self.vertical_flip = n

    def setRandomSizedCrop(self, n):
        self.random_sized_crop = n

    """Transformation functions"""

    def __doGrayscale(self, img):
        img = img.dot([0.299, 0.587, 0.114])
        img = np.stack((img,)*3, axis=-1)
        return img

    def __doSaturation(self, img):
        saturation_var = 0.5
        gs = self.__doGrayscale(img)
        alpha = 2 * np.random.random() * saturation_var
        alpha += 1 - saturation_var
        img = img * alpha + (1 - alpha) * gs[:, :, None]
        img = np.clip(img, 0, 255)
        return img

    def __doBrightness(self, img):
        brightness_var = 0.5
        saturation_var = 0.5
        alpha = 2 * np.random.random() * brightness_var
        alpha += 1 - saturation_var
        img = img * alpha
        img = np.clip(img, 0, 255)
        return img

    def __doContrast(self, img):
        contrast_var = 0.5
        gs = self.__doGrayscale(img).mean() * np.ones_like(img)
        alpha = 2 * np.random.random() * contrast_var
        alpha += 1 - contrast_var
        img = img * alpha + (1 - alpha) * gs
        img = np.clip(img, 0, 255)
        return img

    def __doLighting(self, img):
        lighting_std = 0.5
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        img = np.clip(img, 0, 255)
        return img

    def __doNoise(self, img):
        img_size = img.shape[:2]
        scale = np.random.randint(16)
        noise = np.array(
            np.random.exponential(scale, img_size), dtype=np.int
        ) * np.random.randint(-1, 2, size=img_size)
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
        img = img + noise
        img = np.clip(img, 0, 255)
        return img

    def __doHorizontalFlip(self, img, bboxs):
        img = img[:, ::-1]
        num_coords = bboxs.shape[1] - 1
        bboxs[:,[0,2,4,6]] = 1 - bboxs[:,[2,0,6,4]]
        bboxs[:,[1,3,5,7]] = bboxs[:,[3,1,7,5]]
        return img, bboxs

    def __doVerticalFlip(self, img, bboxs):
        img = img[::-1]
        num_coords = bboxs.shape[1] - 1
        bboxs[:,[0,2,4,6]] = bboxs[:,[6,4,2,0]]
        bboxs[:,[1,3,5,7]] = 1 - bboxs[:,[7,5,3,1]]
        return img, bboxs

    def __doRandomSizedCrop(self, img, target):
        img_h, img_w = img.shape[:2]
        aspect_ratio_range=[4./3., 3./4.]
        crop_area_range=[0.75, 1.0]
        # make sure that we can preserve the aspect ratio
        ratio_range = aspect_ratio_range
        random_ratio = ratio_range[0] + np.random.random() * (ratio_range[1] - ratio_range[0])
        # a = w/h, w_i-w >= 0, h_i-h >= 0 leads to LP: max. h s.t. h <= w_i/a, h <= h_i
        max_h = min(img_w/random_ratio, img_h)
        max_w = max_h * random_ratio
        # scale the area
        crop_range = crop_area_range
        random_scale = crop_range[0] + np.random.random() * (crop_range[1] - crop_range[0])
        target_area = random_scale * max_w * max_h
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        x = np.random.random() * (img_w - w)
        y = np.random.random() * (img_h - h)
        w_rel = w / img_w
        h_rel = h / img_h
        x_rel = x / img_w
        y_rel = y / img_h
        w, h, x, y = int(w), int(h), int(x), int(y)
        # crop image and transform boxes
        new_img = img[y:y+h, x:x+w]
        num_coords = target.shape[1] - 1
        new_target = []
        for box in target:
            new_box = np.copy(box)
            new_box[0:8:2] -= x_rel
            new_box[0:8:2] /= w_rel
            new_box[1:8:2] -= y_rel
            new_box[1:8:2] /= h_rel
            if (new_box[0] > 0 and new_box[6] > 0 and new_box[2] < 1 and new_box[4] < 1 and
                new_box[1] > 0 and new_box[3] > 0 and new_box[5] < 1 and new_box[7] < 1):
                new_target.append(new_box)
        new_target = np.asarray(new_target)
        return new_img, new_target

    """Processing fucntions"""

    def __resetEpoch(self):
        data = self.gt_obj.data.copy()
        # Randomisation functions
        return data

    def __getBatch(self, data, batch_size):
        batch = []
        keys = list(data.keys())
        shuffle(keys)
        keys = keys[:batch_size]
        for key in keys:
            batch.append((key, data[key]))
            del data[key]
        return data, batch

    def __getInput(self, path):
        name = path
        image = cv2.imread(name, 3)  # reads image as BGR
        b, g, r = cv2.split(image)  # get BGR
        image = cv2.merge([r, g, b])  # switch it to RGB
        image = cv2.resize(image, self.input_size) # TODO: remove this line
        return image

    def __getOutput(self, bboxs):
        bboxs = np.array(bboxs, dtype=np.float32)
        if bboxs.shape[0] != 0:
            pass
            # bboxs = np.multiply(bboxs, self.resize)
        return bboxs

    def __preprocessInputOutput(self, image, bboxs):
        """preprocess Image to get a bit of augmentation
            --- Grayscale Image
            --- Saturation Image
            --- Brightness Image
            --- Contrast Image
            --- Lighting Image
            --- Noise Image
            --- Flip Image
            --- Crop Image
        """
        if random() < self.grayscale:
            image = self.__doGrayscale(image)
        if random() < self.saturation:
            image = self.__doSaturation(image)
        if random() < self.brightness:
            image = self.__doBrightness(image)
        if random() < self.contrast:
            image = self.__doContrast(image)
        if random() < self.lighting:
            image = self.__doLighting(image)
        if random() < self.noise:
            image = self.__doNoise(image)
        if random() < self.horizontal_flip:
            image, bboxs = self.__doHorizontalFlip(image, bboxs)
        if random() < self.vertical_flip:
            image, bboxs = self.__doVerticalFlip(image, bboxs)
        if random() < self.random_sized_crop:
            n_image, n_bboxs = self.__doRandomSizedCrop(image.copy(), bboxs)
            if len(n_bboxs) != 0:
                # if type(check_coordinates(self.input_size * n_bboxs[0,:8].reshape(4, 2), self.input_size)) != "<class 'NoneType'>":
                image, bboxs = n_image, n_bboxs

        # image = cv2.resize(image, self.input_size)

        if self.debug:
            if len(bboxs) == 0:
                print("WTF")
            img = np.uint8(image.copy())
            img_w, img_h, _ = img.shape
            xy = np.asarray([t.reshape(4, -1) * [img_w, img_h] for t in bboxs[:,:8]])
            xy = np.round(xy)
            xy = xy.astype(np.int32)
            print(xy)
            cv2.polylines(img, tuple(xy), True, (0,0,255))
            cv2.imshow("Image", img)
            cv2.waitKey(1000)
            print("Preprocess", image.shape)

        bboxs = self.prior_util.encode(bboxs)

        image = image.astype(np.float32)
        image -= self.mean[np.newaxis, np.newaxis, :]

        return image, bboxs

    """Generation function"""

    def generate(self, debug=False):
        data = self.__resetEpoch()
        batch_size = self.batch_size

        while True:
            # Reset data if needed
            if len(data) < batch_size:
                data = self.__resetEpoch()

            # Get batch data
            data, batch_data = self.__getBatch(data, batch_size)
            batch_input = []
            batch_output = []

            # Read in each input, perform preprocessing and get labels
            for img_path, bboxs in batch_data:
                input = self.__getInput(img_path)
                output = self.__getOutput(bboxs)

                input, output = self.__preprocessInputOutput(image=input, bboxs=output)
                batch_input += [input]
                batch_output += [output]

            # Return a tuple of (input,output) to feed the network
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y


if __name__ == "__main__":
    datapath = "../generated"

    model = SSD300(num_classes=2)
    prior = PriorUtil(model)
    gt = GroundTruth(datapath)

    gen = InputGenerator(gt, prior, batch_size=2, debug=True)

    # OK
    gen.setNoise(0.0)
    gen.setGrayscale(0.0)
    gen.setHorizontalFlip(0.0)
    gen.setVerticalFlip(0.0)
    gen.setRandomSizedCrop(0.0)


    # TODO
    gen.setSaturation(0.0)
    gen.setBrightness(0.0)
    gen.setContrast(0.0)
    gen.setLighting(0.0)


    print(gen)
    for x, y in gen.generate():
        pass
