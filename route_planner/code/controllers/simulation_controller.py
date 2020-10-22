import roslaunch
import rospkg
import multiprocessing as mp
import time
import os
import rospy
import subprocess

class SimulationController(object):
    """docstring for SimulationController."""

    def __init__(self):
        super(SimulationController, self).__init__()
        self.__initVars()

    def __initVars(self):
        self.simulation = None

    def __getRosPkgPath(self, pkg="flightgoggles"):
        rospack = rospkg.RosPack()
        return rospack.get_path(pkg)

    def start(self, launchfile="alpha", perturbation_id=2):
        def callback():
            os.system("roslaunch flightgoggles alpha.launch level:=final gate_locations:="+str(perturbation_id))
        self.ros_process = mp.Process(target=callback)
        self.ros_process.start()
        rospy.init_node("alpha", anonymous=True, log_level=rospy.DEBUG)

    def restart(self, perturbation_id=2):
        os.system('rosparam load /home/ubuntu/catkin_ws/src/flightgoggles/flightgoggles/config/challenges/gate_locations_'+str(perturbation_id)+'.yaml')
        os.system("ps aux | grep Rendere[r] | awk '{print $2}' | xargs sudo kill -9")
        os.system('/home/ubuntu/catkin_ws/devel/lib/flightgoggles/FlightGoggles.x86_64 -screen-quality Fastest -obstacle-perturbation-file /home/ubuntu/catkin_ws/src/flightgoggles/flightgoggles/config/perturbations/perturbed_gates_'+str(perturbation_id)+'.yaml __name:=flightgogglesRenderer &')
        time.sleep(20)

    def stop(self):
        rospy.signal_shutdown("")
        self.ros_process.terminate()
        self.ros_process.join()
        os.system("rosnode kill -a")
        os.system("ps aux | grep ros | awk '{print $2} | xargs sudo kill -9'")
        print('terminated')

    def rosnodeStart(self, package, executable):
        node = roslaunch.core.Node(package, executable)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)

        # print process.is_alive()
        # process.stop()

    def restartUavDynamics(self):
        # should be done if possible without system command
        os.system("rosnode kill /uav/flightgoggles_uav_dynamics")
        self.rosnodeStart("flightgoggles_uav_dynamics", "node")

    """
    roslaunch_file('reporter.launch')
    """


if __name__ == "__main__":
    simu = SimulationController()
    simu.start()

