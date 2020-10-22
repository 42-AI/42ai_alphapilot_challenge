import numpy as np
from random import random, randint
from PIL import Image, ImageDraw, ImageFilter
import itertools
import sys
import os

from .utils import geometry, light


class Generator(object):
    """Data Generator for AlphaPilot gate detection module.

    Attributes:
        size (int, int): Description of `attr1`.
        backgrounds (list: (str, img)): Description of ``.
        gate (numpy matrix): Description of ``.
    """

    RED_COLOR = (255, 0, 0)

    def __init__(self, inbound=True):
        super().__init__()
        # class attributes
        self.backgrounds = []
        self.gate = []
        self.inbound = inbound

        # Attributes
        self.multigate = 0.0
        self.multigate_nb = 2
        self.burngate = 0.0
        self.lightglobal = 0.0
        self.lightellipse = 0.0
        self.rotation = 0.0
        self.blur = 0.0
        self.obstacle = 0.0

        # computation Attributes
        self._gatesize = (500, 500)
        self._backgroundsize = (1296, 864)
        self._coeffs = None
        self._coords = []
        self._gates = []
        self._image = None

    def setBlur(self, nb):
        self.blur = nb

    def setMultigate(self, nb, mult=3):
        self.multigate = nb
        mult = int(mult) if mult > 0 else 1
        self.multigate_nb = mult

    def setBurnGate(self, nb):
        self.burngate = nb

    def setLightglobal(self, nb):
        self.lightglobal = nb

    def setLightellipse(self, nb):
        self.lightellipse = nb

    def setRotation(self, nb):
        self.rotation = nb

    def setObstacle(self, nb):
        self.obstacle = nb

    def setBackground(self, filename):
        #img = Image.open(filename, "r")
        #img = img.resize(self._backgroundsize, Image.ANTIALIAS)
        #img = img.convert('RGB')
        #self.backgrounds.append([filename, img])
        self.backgrounds.append(filename)

    def setGate(self, filename):
        #img = Image.open(filename, "r")
        #img = img.resize(self._gatesize, Image.ANTIALIAS)
        #img = img.convert('RGB')
        #self.gate.append(img)
        self.gate.append(filename)

    def __computeCoords(self, coords_list, offset, coeffs, draw=False):
        if self.inbound:
            coords = [(66, 66), (437, 66), (437, 438), (66, 438)]
        else:
            coords = [(0, 0), (500, 0), (500, 500), (0, 500)]
        coords = geometry.apply_perspective(coords, coeffs)
        for i in range(4):
            coords[i] = (
                (int)(coords[i][0] * self.scale + offset[0]),
                (int)(coords[i][1] * self.scale + offset[1]),
            )
        # TODO: Fucked brain
        # coords = geometry.normalise_coordinates(coords, self._backgroundsize)
        coords = geometry.check_coordinates(coords, self._backgroundsize)
        if coords:
            coords_list.append(coords + [1.0])
            coords.append(coords[0])
            if draw and False:
                pass
                #draw = ImageDraw.Draw(self._image)
                #draw.line(coords, fill=self.RED_COLOR, width=3)
        return coords_list

    def __processingLight(self, image):
        if random() < self.lightglobal:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            alpha = randint(0, 64)
            light.fake_light(image, color=color, alpha=alpha)

        if random() < self.lightellipse:
            for _ in range(randint(0, 4)):
                light.light_ellipse(image, alpha=True)
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        return image

    def __processingBlur(self, image):
        if random() < self.blur:
            image = image.filter(
                ImageFilter.GaussianBlur(radius=randint(1, 2))
            )
        return image

    def __processingObstacle(self, image):
        if random() < self.obstacle:
            pass
        return image

    def save(self, filename, image):
        image.save(filename)

    def transformGate(self, gate_list, coords_list, gate):
        layer = gate.copy()

        if random() < self.burngate:
            color = (randint(128, 255),)*3
            alpha = randint(25, 132)
            light.fake_light(layer, color=color, alpha=alpha)
            for _ in range(randint(0, 2)):
                light.light_ellipse(layer,  alpha=True, color=False)
            tile = Image.new("RGBA", (372, 372), (0, 0, 0, 0))
            layer.paste(tile, (66,66))

        layer, coeffs = geometry.tranform(layer)

        # Random gate rescale
        self.scale = 1
        def l_resize(layer, n=2.5):
            scale = random() * n + 0.15
            new_size = (
                (int)(layer.size[0] * scale),
                (int)(layer.size[1] * scale),
            )
            self.scale *= scale
            return layer.resize(new_size, Image.ANTIALIAS)

        layer = l_resize(layer)
        layer = layer.filter(ImageFilter.GaussianBlur(radius=randint(1, 2)))

        def size_max(size):
            if size[0] > size[1]:
                return size[0]
            return size[1]

        # offset calculation
        while True:
            width, height = self._backgroundsize
            w, h = layer.size
            width, height = width - w, height - h
            if width < 0 or height < 0:
                layer = l_resize(layer, 1.1)
                continue
            offset = (randint(0, width), randint(0, height))
            break

        i = 0
        while i < len(gate_list):
            if size_max(layer.size) < size_max(gate_list[i][0].size):
                break
            i += 1
        if len(gate_list) == 0:
            gate_list.append([layer, offset])
        else:
            gate_list = gate_list[0:i] + [(layer, offset)] + gate_list[i:]
        # Coords calculation
        coords_list = self.__computeCoords(coords_list, offset, coeffs, draw=False)
        return gate_list, coords_list

    def createImage(self, background, gate_list):
        # merge the data of the gate and the background
        image = Image.new("RGBA", self._backgroundsize, (0, 0, 0, 0))

        image.paste(background, (0, 0))

        while len(gate_list) > 0:
            gate, offset = gate_list[0]
            image.paste(gate, offset, mask=gate)
            del gate_list[0]
        return image

    def generateImage(self, size=10, filename="gates"):
        # Generator part
        for ind in range(size):
            gate_list, coords_list = [], []
            path_to_save = "{}_{}.png".format(filename.split(".")[0], ind)

            # Generate gates data
            nb_gates = randint(1, self.multigate_nb) if random() < self.multigate else 1
            for _ in range(randint(1, nb_gates)):
                gate = self.gate[randint(0, len(self.gate) - 1)]
                gate = Image.open(gate, "r")
                gate = gate.resize(self._gatesize, Image.ANTIALIAS)
                gate_list, coords_list = self.transformGate(gate_list, coords_list, gate)

            # Get a background
            # TODO: Function for loading backgrounds
            background = self.backgrounds[randint(0, len(self.backgrounds) - 1)]
            background = Image.open(background, "r")
            background = background.resize(self._backgroundsize, Image.ANTIALIAS)

            image = self.createImage(background, gate_list)

            # Transformations
            image = self.__processingLight(image)
            image = self.__processingBlur(image)
            image = self.__processingObstacle(image)

            # Save
            self.save(path_to_save, image)

            yield path_to_save, coords_list


    def __mpImage(self, filename, i, debug=False):
        try:
            gate_list, coords_list = [], []
            path_to_save = "{}_{}.png".format(filename.split(".")[0], str(i))

            # Generate gates data
            nb_gates = randint(1, self.multigate_nb) if random() < self.multigate else 1
            for _ in range(randint(1, nb_gates)):
                gate = self.gate[randint(0, len(self.gate) - 1)]
                gate = Image.open(gate, "r")
                gate = gate.resize(self._gatesize, Image.ANTIALIAS)
                gate_list, coords_list = self.transformGate(gate_list, coords_list, gate)

            # Get a background
            # TODO: Function for loading backgrounds
            background = self.backgrounds[randint(0, len(self.backgrounds) - 1)]
            background = Image.open(background, "r")
            background = background.resize(self._backgroundsize, Image.ANTIALIAS)

            image = self.createImage(background, gate_list)

            # Transformations
            image = self.__processingLight(image)
            image = self.__processingBlur(image)
            image = self.__processingObstacle(image)

            if debug:
                draw = ImageDraw.Draw(image)
                for coords in coords_list:
                    draw.line(coords[0:4], fill=(255, 0, 0), width=3)

            # Save
            self.save(path_to_save, image)

            sys.stdout.write('.')
            sys.stdout.flush()
        except:
            os.system('rm {}'.format(background))
            return (path_to_save, [])

        return (path_to_save, coords_list)

    def multiprocessImage(self, nb_process, template, nb=10, debug=False):
        import pathos.multiprocessing as mp

        with mp.Pool(nb_process) as p:
            results = p.starmap(
                self.__mpImage,
                zip(itertools.repeat(template), range(nb), itertools.repeat(debug)),
            )
        return results
