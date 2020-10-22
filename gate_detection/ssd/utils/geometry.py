from PIL import Image
from random import randint, random
import numpy
import math


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def check_coordinates(coords, size):
    for i in range(4):
        x, y = coords[i]
        w = 1 if x < 0 else 0
        e = 1 if x > size[0] else 0
        n = 1 if y < 0 else 0
        s = 1 if y > size[1] else 0
        if s | n | e | w == 1:
            return None
    return coords


def normalise_coordinates(coords, size):
    for i in range(4):
        x, y = coords[i]
        w = 1 if x < 0 else 0
        e = 1 if x > size[0] else 0
        n = 1 if y < 0 else 0
        s = 1 if y > size[1] else 0

        x = 0 if w == 1 else x
        x = size[0] if e == 1 else x
        y = 0 if n == 1 else y
        y = size[0] if s == 1 else y

        if (s | n) & (e | w):
            coords[i] = (x, y)
            continue  # break the current loop
        if w:
            j = 1 if i == 0 else 2
            l1 = [(0, 0), (0, size[1])]
            l2 = [coords[i], coords[j]]
            _, y = line_intersection(l1, l2)
        if e:
            j = 0 if i == 1 else 3
            l1 = [(size[0], 0), (size[0], size[1])]
            l2 = [coords[i], coords[j]]
            _, y = line_intersection(l1, l2)
        if n:
            j = 2 if i == 1 else 3
            l1 = [(0, 0), (size[0], 0)]
            l2 = [coords[i], coords[j]]
            x, _ = line_intersection(l1, l2)
        if s:
            j = 0 if i == 3 else 1
            l1 = [(0, size[1]), (size[0], size[1])]
            l2 = [coords[i], coords[j]]
            x, _ = line_intersection(l1, l2)
        coords[i] = (x, y)
    return coords


def tranform(im):
    def find_coeffs(pa, pb):
        """Find coefficients for perspective transformation."""
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = numpy.matrix(matrix, dtype=numpy.float)
        B = numpy.array(pb).reshape(8)

        res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
        return numpy.array(res).reshape(8)

    def rand_endpoints():
        def seg_endpoints():
            ang = 45
            c = 500
            theta = ang * math.pi / 180
            R = c / (2 * math.sin(theta))
            endpoints = [(0, 0), (500, 0), (500, 500), (0, 500)]
            x = randint(0, int(c/2))
            y = int(math.sqrt(R * R - x * x) - R * math.cos(theta))
            y1, y2 = randint(0, y), randint(0, y)
            x1, x2 = x + randint(0, 10) - 5, x + randint(0, 10) - 5
            def normalise(val, limits):
                if val > limits[1]:
                    val = limits[1]
                if val < limits[0]:
                    val = limits[0]
                return val
            right = True if random() > 0.5 else False
            side = -1 if random() > 0.5 else 1
            c = c/2
            if right:
                x, y = endpoints[1]
                if x - (c + side * x1) < 80: # DEBUG Limit of angle
                    side *= -1
                endpoints[1] = x - (c + side * x1), y + y1
                x, y = endpoints[2]
                endpoints[2] = x - (c + side * x2), y - y2
            else:
                x, y = endpoints[0]
                if x + (c + side * x1) > 420: # DEBUG Limit of angle
                    side *= -1
                endpoints[0] = x + (c + side * x1), y + y1
                x, y = endpoints[3]
                endpoints[3] = x + (c + side * x2), y - y2
            for _i in range(4):
                x, y = endpoints[_i]
                endpoints[_i] = normalise(x + randint(0, 4) - 2, (0, 500)), normalise(y + randint(0, 4) - 2, (0, 500))
            return endpoints

        def rr_endpoints():
            topleft = (randint(0, 125), randint(0, 125))
            topright = (randint(375, 500), randint(0, 125))
            botright = (randint(375, 500), randint(375, 500))
            botleft = (randint(0, 125), randint(375, 500))
            endpoints = [topleft, topright, botright, botleft]
            return endpoints

        # Chance for new vs old gates settings
        if random() < 0.10:
            return rr_endpoints()
        return seg_endpoints()

    startpoints = [(0, 0), (500, 0), (500, 500), (0, 500)]
    endpoints = rand_endpoints()
    width, height = im.size
    coeffs = find_coeffs(endpoints, startpoints)
    inv_coeffs = find_coeffs(startpoints, endpoints)

    im = im.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    return im, inv_coeffs


def apply_perspective(coords, coeffs):
    # coeffs is a 8-tuple (a, b, c, d, e, f, g, h) which contains the
    # coefficients for a perspective transform. For each pixel (x, y)
    # in the output image, the new value is taken from a position
    # (a x + b y + c)/(g x + h y + 1), (d x + e y + f)/(g x + h y + 1)
    # in the input image, rounded to nearest pixel.
    a, b, c, d, e, f, g, h = coeffs

    new_coords = []
    for x, y in coords:
        new_x = (a * x + b * y + c) / (g * x + h * y + 1)
        new_y = (d * x + e * y + f) / (g * x + h * y + 1)
        new_coords.append((new_x, new_y))
    return new_coords
