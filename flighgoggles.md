## Global

If you modify files of flighgoggles nodes, you must again catkin build from the root dir of FG
If you only modify config giles, often .yml, nothing more to do

## Launch FG

There is launchfiles to launch, FG, they are located in ~/catkin/src/flightgoggles/launch
They launch all the node of the simulation and we must modify the one we will submit to also launch our piloting node
They must be started with roslaunch <launchfile>
You can add parameters to roslaunch like this <key>:=<value>

## Change challenge or gates position

The gates location and challenge chosen are parameters in the launchfile.
You can pass challenge through parameters like this challenge:=easy
At first, final challenge cant be loaded from reporter.launch, because they cant code properly, you must edit it !
At first, you can't change gates positions, you must edit it!
But, the output in rviz won't change...

## Change drone speed

Simulation speed can't be changed
Drone time can be mapped to the simulation or not
This is done be commenting or not the config `clockscale` in ~/catkin/src/fligggoggles/config/drone/drone.yaml
You can also change here the drone speed

## Change teleop settings

At first, evveride command must be pressed to enable a command, to change, you must edit it, in ~/catkin/src/universal_teleop/src/teleop.cpp line 202 for example
Command mapping and scales can be edited in ~/catkin/src/universal_teleop/launch/example_input_map.yml
