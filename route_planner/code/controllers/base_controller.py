# Base imports
from threading import Timer
import signal
import numpy as np
#import tf2_ros
import tf
import math

# Local imports
from sensors import Sensors
from gate import Gate

# ROS imports
import rospy
from rosgraph_msgs import msg as rosgraph_msgs
from sensor_msgs import msg as sensor_msgs
from mav_msgs import msg as mav_msgs
from std_msgs import msg as std_msgs
from flightgoggles import msg as flightgoggles_msgs

class BaseController(object):
    """docstring for BaseController."""

    def __init__(self):
        self.__initVars()
        self.__subscribeAll()

    def __initVars(self):
        """Subscribers vars initialisation"""
        self.sensors = Sensors()
        self.trans_listener = tf.TransformListener()
        #self.trans_listener = tf2_ros.Buffer()
        #tf2_ros.TransformListener(self.trans_listener)
        self.sensors.add("trans", None)
        self.old_trans = None
        self.sensors.add("rot", None)
        self.rate_pub_enabled = False
        self.messager = mav_msgs.RateThrust()
        self.collision = False

    def getRatePub(self):
        def callback(data):
            if self.rate_pub_enabled:
                try:
                    self.rate_pub.publish(self.messager)
                except Exception as e:
                    pass
        rospy.Timer(rospy.Duration(0.001), callback)
        return rospy.Publisher(
            "/uav/input/rateThrust", mav_msgs.RateThrust, queue_size=1
        )

    def getCollisionPub(self):
        return rospy.Publisher("/uav/collision", std_msgs.Empty, queue_size=10)

    def updateRatePub(self, ax=0, ay=0, az=0, z=9.81):
        # Tu fait tes bails de securite ici
        self.messager.thrust.z = z
        self.messager.angular_rates.x = ax
        self.messager.angular_rates.y = ay
        self.messager.angular_rates.z = az

    def __subscribeClock(self):
        self.sensors.add("secs", None)
        # self.sensors.add('nsecs', None)
        def callback(data):
            # self.sensors.secs = data.clock.secs
            # self.sensors.nsecs = data.clock.nsecs
            self.sensors.secs = data.clock.secs + data.clock.nsecs * 10e-10

        rospy.Subscriber("/clock", rosgraph_msgs.Clock, callback)

    def __subscribeDlr(self):
        self.sensors.add("dlr", None)
        def callback(data):
            self.sensors.dlr = data.range
        rospy.Subscriber(
            "/uav/sensors/downward_laser_rangefinder", sensor_msgs.Range, callback
        )

    def __subscribeIMU(self):
        self.sensors.add("angular_velocity", None)
        self.sensors.add("linear_acceleration", None)
        def callback(data):
            self.sensors.angular_velocity = data.angular_velocity
            self.sensors.linear_acceleration = data.linear_acceleration
        rospy.Subscriber("/uav/sensors/imu", sensor_msgs.Imu, callback)

    def __subscribeIRMarker(self):
        self.sensors.add("markers", None)
        self.sensors.markers = {}
        for i in range(1, 25):
            name = "Gate" + str(i)
            self.sensors.markers.update({name: {}})
        def callback(data):
            for item in self.sensors.markers.keys():
                self.sensors.markers[item] = {}
            for item in data.markers:
                self.sensors.markers[item.landmarkID.data].update(
                    {item.markerID.data: {"x": item.x, "y": item.y, "z": item.z}}
                )
        rospy.Subscriber(
            "/uav/camera/left/ir_beacons", flightgoggles_msgs.IRMarkerArray, callback
        )

    def __subscribeRateThrust(self):
        self.sensors.add("thrust", None)
        self.sensors.add("angular_rates", None)
        def callback(data):
            self.sensors.thrust = data.thrust
            self.sensors.angular_rates = data.angular_rates
        rospy.Subscriber("/uav/input/rateThrust", mav_msgs.RateThrust, callback)

    def __subscribeCollision(self):
        def callback(data):
            self.collision = True
            self.rate_pub_enabled = False
        rospy.Subscriber("/uav/collision", std_msgs.Empty, callback)

    def __subscribeAll(self):
        self.__subscribeClock()
        self.__subscribeDlr()
        self.__subscribeRateThrust()
        self.__subscribeIMU()
        self.__subscribeIRMarker()
        self.__subscribeCollision()
        self.collision_pub = self.getCollisionPub()
        self.rate_pub = self.getRatePub()
