# Base imports
import rospy
import time
import signal
import sys
import math
import numpy as np
import cv2

# Local imports
from base_controller import BaseController
from gate import Gate

# ROS imports
from mav_msgs import msg as mav_msgs
from sensor_msgs import msg as sensor_msgs
from keyboard import msg as keyboard
from universal_teleop import msg as universal_teleop


class DroneController(BaseController):
    """docstring for DroneController."""

    def __init__(self):
        # Basic init
        super(DroneController, self).__init__()
        self.__initVars()

    def __initVars(self):
        """Subscribers vars initialisation"""
        self.true_gates = self.getTrueGates()
        self.gates = self.getGates()
        self.sensors.add('next_gate_id', 0)
        self.sensors.add('next_true_gate_id', 0)
        self.passed_gate = False
        self.passed_true_gate = False
        self.skipped_true_gate = False
        self.render_bool = False
        self.d_next_gate = 0
        self.count_passed = 0
        self.d_short_path = 0
        self.init_pose = rospy.get_param("/uav/flightgoggles_uav_dynamics/init_pose")[:3]
        self.last_reward_render = np.zeros((800, 800, 3), np.uint8)
        #self.last_discount_render = np.zeros((400, 800, 3), np.uint8)
        self.reward_render = np.zeros((800, 800, 3), np.uint8)

    def getRawSensor(self):
        self.refreshPosition() # time for this part is 0.04
        return self.sensors.get_all()

    def getTrueGates(self):
        events = rospy.get_param("/uav/gate_names", "[]")
        inflation = rospy.get_param("/uav/inflation", 0.1)
        gates = []
        for e in events:
            loc = np.asarray(rospy.get_param("/uav/%s/location" % e, []))
            gates.append(Gate(e, loc, inflation))
        return gates

    def getGates(self):
        events = rospy.get_param("/uav/gate_names", "[]")
        inflation = rospy.get_param("/uav/inflation", 0.1)
        gates = []
        for e in events:
            loc = np.asarray(rospy.get_param("/uav/%s/nominal_location" % e, []))
            gates.append(Gate(e, loc, inflation+2))
        return gates

    def checkGates(self):
        for i, gate in enumerate(self.true_gates[self.sensors.next_true_gate_id:]):
            gate_passed, gate_plane_passed = gate.isEvent(self.sensors.trans, self.old_trans)
            if gate_passed:
                if i > 0:
                    # gates have been skipped
                    pass
                print("{}Gate {} validated{}".format('\033[92m\033[1m', self.true_gates.index(gate), '\033[0m'))
                gate.passed_pos = [self.sensors.trans[e] for e in ['x', 'y' ,'z']]
                self.passed_true_gate = True
                self.sensors.next_true_gate_id += i + 1
                self.passed_true_gate_last_time = self.sensors.secs
            if gate_plane_passed:
                # le plan a ete passe mais la vrai gate non...
                self.skipped_true_gates += 1
                print("{}Gate {} skipped{}".format('\033[91m\033[1m', self.true_gates.index(gate), '\033[0m'))
        for i, gate in enumerate(self.gates[self.sensors.next_gate_id:]):
            gate_passed, _ = gate.isEvent(self.sensors.trans, self.old_trans)
            if gate_passed:
                if i > 0:
                    # gates have been skipped
                    pass
                print("{}Gate {} validated (nominal){}".format('\033[1m', self.gates.index(gate), '\033[0m'))
                gate.passed_pos = [self.sensors.trans[e] for e in ['x', 'y' ,'z']]
                self.passed_gate = True
                self.sensors.next_gate_id += i + 1

    def distanceToShortPath(self):
        def distance(pt_1, pt_2):
            return math.sqrt(
                (pt_1[0] - pt_2[0]) ** 2
                + (pt_1[1] - pt_2[1]) ** 2
                + (pt_1[2] - pt_2[2]) ** 2
            )

        if self.sensors.next_true_gate_id >= len(self.true_gates):
            return 0
        if self.sensors.next_true_gate_id > 0:
            #fromPos = self.true_gates[self.sensors.next_true_gate_id - 1].passed_pos[:3]
            fromPos = self.true_gates[self.sensors.next_true_gate_id - 1].getCenter()
        else:
            fromPos = self.init_pose
        position = [self.sensors.trans[e] for e in ['x', 'y', 'z']]
        toPos = self.true_gates[self.sensors.next_true_gate_id].getCenter()
        fromPos = np.array(fromPos)
        toPos = np.array(toPos)
        position = np.array(position)
        return np.linalg.norm(np.cross(toPos-fromPos, fromPos-position))/np.linalg.norm(toPos-fromPos)

    def distanceToNextGate(self):
        if self.sensors.next_true_gate_id >= len(self.true_gates):
            return 0
        ret = self.true_gates[self.sensors.next_true_gate_id].getDistanceFromPlane(
            [self.sensors.trans[e] for e in ['x', 'y', 'z']]
        )
        return ret

    def reset(self):
        self.updateRatePub()
        self.start_time = self.sensors.secs
        self.passed_true_gate_last_time = self.sensors.secs
        self.collision = False
        self.rate_pub_enabled = False
        self.passed_gate = False
        self.sensors.next_gate_id = 0
        self.sensors.next_true_gate_id = 0
        self.count_passed = 0
        self.passed_true_gate = False
        self.passed_gate = False
        self.skipped_true_gates = 0
        return self.getRawSensor()

    # 0.1
    def step(self, actions):
        if actions is not None:
            self.rate_pub_enabled = True
            self.updateRatePub(*actions) # 0
        # Timeout
        #if self.sensors.secs - self.passed_true_gate_last_time > 20:
            #self.collision_pub.publish()
        observation = self.getRawSensor() # 0.008
        done = self.checkFinished() # 0
        reward = self.getReward()# 0
        return observation, reward, done

    # Check if the drone has crashed or has finished
    def checkFinished(self):
        if self.sensors.next_true_gate_id >= len(self.gates):
            return True
        return self.collision

    def refreshPosition(self):
        (trans, rot) = self.trans_listener.lookupTransform(
            "/world", "/uav/imu", rospy.Time(0)
        )
        #while True:
        #    try:
        #        trans = self.trans_listener.lookup_transform(
        #        #(trans, rot) = self.trans_listener.lookup_transform(
        #            "static_ref", "uav/imu", rospy.Time(0)
        #        )
        #        (trans, rot) = (trans.transform.translation, trans.transform.rotation)
        #    except Exception as e:
        #        print(e)
        #        time.sleep(1)
        #        continue
        #    break
        self.old_trans = self.sensors.trans.copy() if self.sensors.trans else None
        self.sensors.trans = {"x": trans[0], "y": trans[1], "z": trans[2]}
        self.sensors.rot = rot
        #self.sensors.trans = {"x": trans.x, "y": trans.y, "z": trans.z}
        #self.sensors.rot = [rot.x, rot.y, rot.w, rot.z]

        self.checkGates() # 0.04

    def getReward(self, diff=False):
        """ Take 1D float array of rewards and compute discounted reward. This is  """
        elapsed_time = self.sensors.secs - self.start_time
        # Can be changed to diff
        d_next_gate = self.distanceToNextGate()
        d_short_path = self.distanceToShortPath()
        collision = 0
        passed_gate = 0
        skipped_true_gate = 0
        if diff:
            ng, sp = d_next_gate, d_short_path
            d_next_gate -= self.d_next_gate
            d_short_path -= self.d_short_path
            self.d_next_gate = ng
            self.d_short_path = sp
        if self.collision:
            collision = 1
        if self.passed_true_gate:
            self.count_passed += 1
            passed_gate = 1
            self.passed_true_gate = False
        if self.skipped_true_gate:
            skipped_true_gate = 1
            self.skipped_true_gate = False
        w = [0., 0, 0.01, -0.02, 0, -1]
        #w = [0, 0, -0.05, -0.01, 0,2 0]
        #w = [0, 0, -0.01, -0.05, 0, 0]
        dg = (1 - d_next_gate / 30)
        dg = 0 if dg < 0 else dg
        r = [
            passed_gate,
            elapsed_time,
            dg,
            math.sqrt(d_short_path) * dg,
            skipped_true_gate,
            collision * (2 - dg) / 2,
            #collision * (3 + d_next_gate / 30),
        ]
        r = np.multiply(r, w)
        if self.render_bool:
            self.renderReward(r)
        return np.sum(r[[0, 5]]), np.sum(r[1:5])

    def waitReset(self):
        cont = True
        while cont:
            self.refreshPosition()
            position = np.array([self.sensors.trans[i] for i in ['x','y','z']])
            ref = np.array([18.0, -23.0, 5.3])
            cont = not (position==ref).all()

    def renderReward(self, r):
        img = self.last_reward_render
        img = np.roll(img, -1)
        a = 700-sum(r)*5
        color = (155,155,155)
        cv2.polylines(img, np.int32([[[800,a],[800, a]]]), True, color, thickness=3)
        cv2.polylines(img, np.int32([[[200,0],[200, 400]]]), True, (0,0,0), thickness=2)
        self.last_reward_render = img
        #cv2.polylines(self.last_discount_render, np.int32([[[1,0],[1, 400]]]), True, (0,0,0), thickness=2)
        #self.last_discount_render = np.roll(self.last_discount_render, -1)

    def renderDiscount(self, discounted):
        return
        img = self.last_discount_render
        img = np.roll(img, len(discounted))
        for d in discounted:
            img = np.roll(img, -1)
            a = 200-d*20
            cv2.polylines(img, np.int32([[[1,0],[1, 400]]]), True, (0,0,0), thickness=2)
            cv2.polylines(img, np.int32([[[799,a],[799, a]]]), True, (100,100,100), thickness=3)
        self.last_discount_render = img

    def render(self, line, action):
        if self.render_bool == False:
            self.render_bool = True
            cv2.startWindowThread()
            cv2.namedWindow("preview")
        w = 800
        h = 800
        #img = np.concatenate((self.last_reward_render, self.last_discount_render), axis=0)
        img = self.last_reward_render.copy()

        cv2.line(img, (200,0), (200, 800), (255,255,255), thickness=1)
        cv2.line(img, (200,600), (800, 600), (255,255,255), thickness=1)

        # Render gate corners
        bbox = np.array(line[12:20])
        xy = (bbox.reshape(4, -1) + np.array([1,1])) * [300, 300] + [0, 200]
        xy = xy[:,[1, 0]]
        xy = np.round(xy)
        cv2.polylines(img, np.int32([xy]), True, (0,0,255), thickness=2)

        # Render gate distances and dlr
        dist1 = 800-line[7]*800
        cv2.polylines(img, np.int32([[[10,800],[10, dist1]]]), True, (255,0,0), thickness=2)
        dist2 = 800-line[8]*800
        cv2.polylines(img, np.int32([[[20,800],[20, dist2]]]), True, (255,0,0), thickness=2)
        dlr = 800+line[6]*800
        cv2.polylines(img, np.int32([[[790,800],[790, dlr]]]), True, (0,255,0), thickness=2)

        # Render drone speeds
        gray = (100,100,100)
        cv2.arrowedLine(img, (100,780), (100, 620), gray, thickness=1)
        cv2.arrowedLine(img, (20,700), (180, 700), gray, thickness=1)
        cv2.arrowedLine(img, (40,760), (160,640), gray, thickness=1)

        x = 100+line[9]*80
        y = 700-line[9]*80
        cv2.polylines(img, np.int32([[[100,700],[x, y]]]), True, (0,255,0), thickness=2)
        x = 100+line[10]*100
        y = 700
        cv2.polylines(img, np.int32([[[100,700],[x, y]]]), True, (255,0,0), thickness=2)
        x = 100
        y = 700+line[11]*100
        cv2.polylines(img, np.int32([[[100,700],[x, y]]]), True, (0,0,255), thickness=2)

        # Render rateThrust
        ctr = (100,500)
        y = 580-action[3]*10
        cv2.line(img, (10,580), (10, int(y)), (0,255,255), thickness=2)
        a = -90+action[0]*360
        cv2.ellipse(img, ctr, (80,80), 0, 0, 360, gray, thickness=1)
        cv2.ellipse(img, ctr, (80,80), 0, -90, a, (0,255,0), thickness=3)
        a = -90+action[1]*360
        cv2.ellipse(img, ctr, (40,80), 0, 0, 360, gray, thickness=1)
        cv2.ellipse(img, ctr, (40,80), 0, -90, a, (255,0,0), thickness=3)
        a = -90-action[2]*360
        cv2.ellipse(img, ctr, (80,40), 0, 0, 360, gray, thickness=1)
        cv2.ellipse(img, ctr, (80,40), 0, -90, a, (0,0,255), thickness=3)

        # Render
        cv2.imshow("preview", (img))

if __name__ == "__main__":
    import json

    print("Running script {}".format(__name__))
    rospy.init_node("alpha", anonymous=True)

    def callback(sig, frame):
        rospy.signal_shutdown("stop")

    signal.signal(signal.SIGINT, callback)
    drone = DroneController()
    print('go')
    while not rospy.is_shutdown():
        drone.refreshPosition()
        if drone.sensors.next_true_gate_id >= len(drone.gates):
            drone.sensors.endMem(True)
            drone.reset()
            print('go')
        if drone.collision:
            drone.sensors.endMem(False)
            drone.reset()
        drone.sensors.dumpMem()
    drone.endMem(False)
