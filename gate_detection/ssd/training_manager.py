import numpy as np
import cv2

from keras.utils import multi_gpu_model

from .prior_util import PriorUtil
from .ground_truth import GroundTruth
from .input_generator import InputGenerator


class TrainingManager(object):
    """TrainingManager for AlphaPilot gate detection module.

    Attributes:
        experimentname (str): Description of ``.
        weightpath (str): Description of ``.
        model (Keras Object): Description of ``.
        loss (Keras Object): Description of ``.
        checkpoints (Keras Object: list): Description of ``.
    """

    def __init__(self, experimentname, epoch=10, batch_size=32):
        super().__init__()
        # class attributes
        self.experimentname = experimentname
        self.epoch = epoch
        self.batch_size = batch_size

        self.model = None
        self.weightpath = None
        self.loss = None
        self.callbacks = []

    def setModel(self, model, weightpath=None, prior=None, multi_gpu=False):
        self.model = model
        self.weightpath = weightpath
        if weightpath:
            print("Loading weight")
            self.model.load_weights(weightpath, by_name=True)
        a, b, c, d = model.layers[0].input_shape
        self.input_shape = (b, c)
        if not prior:
            self.prior = PriorUtil(self.model)
        else:
            self.prior = prior
        if multi_gpu:
            self.model = multi_gpu_model(self.model, gpus=4)


    def setFreeze(self, freeze):
        self.freeze = freeze
        for layer in self.model.layers:
            layer.trainable = not layer.name in freeze

    def setLoss(self, loss):
        self.loss = loss

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss.compute, metrics=self.loss.metrics
        )

    def setInputGenerator(self, datapath, ratio_valid=0.1):
        if not self.model:
            raise ValueError("could not find model in TrainingManager")
        self.datapath = datapath
        prior = self.prior
        groundtruth = GroundTruth(datapath)
        gt_train, gt_valid = groundtruth.split(ratio_valid=ratio_valid)
        self.train_gen = InputGenerator(
            gt_train,
            prior,
            batch_size=self.batch_size,
            input_size=self.input_shape,
        )
        self.train_gen.setGrayscale(0.25)
        self.train_gen.setNoise(0.25)
        self.train_gen.setHorizontalFlip(0.25)
        self.train_gen.setVerticalFlip(0.25)
        #self.train_gen.setRandomSizedCrop(0.25)
        #self.train_gen.setSaturation(0.25)
        #self.train_gen.setBrightness(0.25)
        #self.train_gen.setContrast(0.25)
        #self.train_gen.setLighting(0.25)
        self.valid_gen = InputGenerator(
            gt_valid,
            prior,
            batch_size=self.batch_size,
            input_size=self.input_shape,
        )
        self.steps_per_epoch = gt_train.num_images / self.batch_size
        self.valid_step = gt_valid.num_images / self.batch_size

    def setCallbacks(self, callbacks):
        self.callbacks.append(callbacks)

    def train(self, epochs, initial_epoch=0):
        history = self.model.fit_generator(
            self.train_gen.generate(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=self.valid_gen.generate(),
            validation_steps=self.valid_step,
            class_weight=None,
            workers=8,
            use_multiprocessing=True,
            initial_epoch=initial_epoch,
        )

    def predict(self, image, confidence_threshold=0.25, keep_top_k=1, pad=False):
        img_w, img_h = np.array(image).shape[:2]
        """if pad:
            if img_w > img_h:
                image = cv2.copyMakeBorder(image, 0, 0, 0, img_w - img_h, cv2.BORDER_REPLICATE)
            else:
                image = cv2.copyMakeBorder(image, 0, img_h - img_w, 0, 0, cv2.BORDER_REPLICATE)"""
        image = cv2.resize(image, self.input_shape)
        # cv2.imwrite('test.jpg', image)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        results = self.model.predict(image)
        results = self.prior.decode(results[0], confidence_threshold=confidence_threshold, keep_top_k=keep_top_k)
        if results.shape[0] > 0:
            results = results[:,4:12]
            results = np.concatenate([results, np.ones([results.shape[0],1])], axis=1) # TODO change confidence setting
            # add = img_w - img_h if img_w > img_h else img_h - img_w
            """if False:
                max_size = np.max([img_w, img_h])
                results = np.multiply(results, np.append(np.tile(np.array([max_size]), 8), [1.]))
                if img_w > img_h:
                    results -= np.append(np.tile(np.array([0, img_w - img_h]), 4), [0.])
                else:
                    results -= np.append(np.tile(np.array([img_h - img_w, 0]), 4), [0.])
            else:"""
            results = np.multiply(results, np.append(np.tile(np.array([img_w, img_h]), 4), [1.]))
        return results

    def evaluate(self, data):
        # Evaluate the model
        pass

    def save(self, filename=None):
        # Save the model
        # optionnal filename, automatically set in function of experiment.
        pass
