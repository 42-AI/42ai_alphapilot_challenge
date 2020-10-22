import os
import json
import numpy as np
import matplotlib # Mac bug fix
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

gates_1 = ['Gate10', 'Gate21', 'Gate2', 'Gate13', 'Gate9', 'Gate14', 'Gate1', 'Gate22', 'Gate15', 'Gate23', 'Gate6']
gates_2_1 = ['Gate2', 'Gate13', 'Gate3', 'Gate22', 'Gate1', 'Gate14', 'Gate9', 'Gate8', 'Gate13', 'Gate23', 'Gate6']
gates_2_2 = ['Gate2', 'Gate13', 'Gate23', 'Gate15', 'Gate22', 'Gate1', 'Gate14', 'Gate9', 'Gate7']
gates = [gates_1, gates_2_1, gates_2_2]
# bad_run = [5,6,23,24,26,31,42,59,74,75,78,79,82]

def merge():
    events = []
    run_id = 0
    tot = 0
    for k, v in [['sensors_logs', 1], ['map1', 1], ['map1_2', 1]]: #,  ['map2', 2], ['map3', 3]]:
        files = os.listdir('train_dumps/'+k)
        for file in files:
            #if run_id in bad_run:
            #    run_id += 1
            #    continue
            subevents = []
            with open('raw_dumps/'+k+'/'+file) as f:
                as_json = json.load(f)
            for key, line in as_json.items():
                tot += 1
                line.update({
                    'nextGates': gates[v-1][line['nextEventId']:],
                    'run_id': run_id,
                    'map': v
                })
                line.pop('nextEventId')
                if line['secs'] is not None:
                    subevents.append(line)
            events[0:0] = sorted(subevents, key=lambda k: float(k['secs']) if k['secs'] else None)
            run_id += 1
    print(len(events))
    with open('dumps_map1.json', 'w+') as f:
        json.dump(events, f)

def showthrustovertime():
    z = []
    ax = []
    ay = []
    az = []
    for _ in range(100):
        z.append([])
        ax.append([])
        ay.append([])
        az.append([])
    with open('dumps.json') as f:
        as_json = json.load(f)
    print(len(as_json))
    return
    for line in as_json[:]:
        if line['angular_rates'] is not None:
            z[line['run_id']].append(line['thrust']['z'])
            ax[line['run_id']].append(line['angular_rates']['x'])
            ay[line['run_id']].append(line['angular_rates']['y'])
            az[line['run_id']].append(line['angular_rates']['z'])
    for i in range(len(ax)):
        plt.subplot(2,2,1)
        plt.plot(ax[i], 'bo')
        plt.subplot(2,2,2)
        plt.plot(ay[i], 'bo')
        plt.subplot(2,2,3)
        plt.plot(az[i], 'bo')
        plt.subplot(2,2,4)
        plt.plot(z[i], 'bo')
        fig = plt.gcf()
        fig.canvas.set_window_title(str(i))
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()
        plt.show()

def showgateinview():
    x = []
    y = []
    with open('dumps.json') as f:
        as_json = json.load(f)
    for line in as_json[:]:
        if len(line['nextGates']) < 1:
            continue
        n = line['nextGates'][0]
        for corner in line['markers'][n].values():
            x.append(corner['x'])
            y.append(corner['y'])
    plt.hist2d(x,y, bins=100)
    plt.show()

def showcornerposovertime():
    x = []
    y = []
    with open('dumps.json') as f:
        as_json = json.load(f)
    for line in as_json[:]:
        if len(line['nextGates']) < 1:
            continue
        for gate in line['markers'].values():
            for corner in gate.values():
                x.append(corner['x'])
                y.append(corner['y'])
    plt.subplot(211)
    plt.plot(x)
    plt.subplot(212)
    plt.plot(y)
    plt.show()

def showlinear():
    x = []
    y = []
    z = []
    for _ in range(100):
        x.append([])
        y.append([])
        z.append([])
    with open('dumps.json') as f:
        as_json = json.load(f)
    for line in as_json[:]:
        if line['linear_acceleration'] is not None:
            x[line['run_id']].append(line['linear_acceleration']['x'])
            y[line['run_id']].append(line['linear_acceleration']['y'])
            z[line['run_id']].append(line['linear_acceleration']['z'])
    for i in range(len(x)):
        plt.subplot(2,2,1)
        plt.plot(x[i], 'bo')
        plt.subplot(2,2,2)
        plt.plot(y[i], 'bo')
        plt.subplot(2,2,3)
        plt.plot(z[i], 'bo')
        fig = plt.gcf()
        fig.canvas.set_window_title(str(i))
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()
        plt.show()

# showlinear()

if __name__ == "__main__":
    merge()
