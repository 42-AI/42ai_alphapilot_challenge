import cv2
import numpy as np
import math

def nb_missing(gate):
    nb = 0
    for corner in gate:
        if len(corner) == 0:
            nb += 1
    return nb

def new_corner(i, gate):
    corner = []
    x, y = gate[:, 0], gate[:, 1]
    if i == 0:
        x, y = np.min(x), np.min(y)
    elif i == 1:
        x, y = np.min(x), np.max(y)
    elif i == 2:
        x, y = np.max(x), np.min(y)
    else:
        x, y = np.max(x), np.max(y)
    return [x, y]

def calculate_last_corner(data):
    '''
    gate = [i for i in data if len(i)]
    mask_all = np.array([[True, True], [True, False], [False, True], [False, False]])
    aset = set([tuple(x) for x in mask_all])
    ngate = np.array(gate, dtype=np.float32)
    center = np.sum(ngate - np.min(ngate, axis=0), axis=0) / ngate.shape[0]
    mask = ngate - np.min(ngate, axis=0) < center
    bset = set([tuple(x) for x in mask])
    result = np.array([x for x in aset ^ bset])[0]
    i = 0
    for x in mask_all:
        if x[0] == result[0] and x[1] == result[1]:
            corner = new_corner(i, np.array(gate))
            break
        i += 1
    '''
    corner = []
    if len(data[2]) == 0:
        corner = [[data[3][0], data[1][1]]]
    elif len(data[3]) == 0:
        corner = [[data[2][0], data[0][1]]]
    elif len(data[0]) == 0:
        corner = [[data[1][0], data[3][1]]]
    elif len(data[1]) == 0:
        corner = [[data[0][0], data[2][1]]]
    else:
        corner = [[-1, -1]]

    return corner

def calculate_2_corners(data):
    corner = []
    # case of top missing
    if len(data[0]) == 0 and len(data[1]) == 0:
        corner = [
            [-1, data[3][1]],
            [-1, data[2][1]],
        ]
    elif len(data[2]) == 0 and len(data[3]) == 0:
        corner = [
            [1, data[1][1]],
            [1, data[0][1]],
        ]
    elif len(data[1]) == 0 and len(data[2]) == 0:
        corner = [
            [data[0][0], 1],
            [data[3][0], 1],
        ]
    elif len(data[0]) == 0 and len(data[3]) == 0:
        corner = [
            [data[1][0], -1],
            [data[2][0], -1],
        ]
    else:
        corner = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
    return corner

def calculate_3_corners(data):
    if len(data[0]) != 0:
        corner = [
            [1, data[0][1]],
            [1, 1],
            [data[0][0], 1],
        ]
    elif len(data[1]) != 0:
        corner = [
            [1, data[1][1]],
            [data[1][0], -1],
            [1, -1],
        ]
    elif len(data[2]) != 0:
        corner = [
            [-1, -1],
            [data[2][0], -1],
            [-1, data[2][1]],
        ]
    elif len(data[3]) != 0:
        corner = [
            [data[3][0], 1],
            [-1, 1],
            [-1, data[3][1]],
        ]
    else:
        corner = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
    return corner

def chollet_gate_fix_missing(frame):
    gate_normalized = []
    for gate in frame:
        if gate is None:
            gate = [[], [], [], []]
        if nb_missing(gate) == 1:
            n = 1
            last_corners = calculate_last_corner(gate)
        elif nb_missing(gate) == 2:
            n = 2
            last_corners = calculate_2_corners(gate)
        elif nb_missing(gate) == 3:
            n = 3
            last_corners = calculate_3_corners(gate)
        elif nb_missing(gate) == 4:
            n = 4
            last_corners = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
        else:
            n = 0
        for corner in gate:
            if len(corner) == 0:
                gate_normalized += [last_corners[0][0], last_corners[0][1]]
                del last_corners[0]
            else:
                gate_normalized += [corner[0], corner[1]]

        center = np.array(gate_normalized, dtype=np.float32).reshape(4, -1)
        gate_normalized += (np.sum(center, axis=0) / 4.).tolist()
    return gate_normalized, n
