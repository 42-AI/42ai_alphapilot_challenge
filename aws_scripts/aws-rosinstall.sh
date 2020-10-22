
# https://github.com/mit-fast/FlightGoggles/wiki/installation-local

get_auth () {
    INSTALL=0
    DEFAULT="no"
    echo -n "$1 [yes|no]
[$DEFAULT] >>> "
    read -r ans
    if [[ $ans == "" ]]; then
    	ans=$DEFAULT
    fi
    if [[ ($ans != "yes") && ($ans != "Yes") && ($ans != "YES") && ($ans != "y") && ($ans != "Y")]]; then
    	echo "
Skipping install.
"
    	INSTALL=0
    else
    	INSTALL=1
    fi
}

echo -n "Which terminal are you using on this device?"
$DEFAULT = "bash"
echo -n " [zsh|bash]
[$DEFAULT] >>>"
read -r term
if [[ $term == "zsh" ]]; then
    rc="zshrc"
else
    term=$DEFAULT
    rc="bashrc"
fi

# ROS Préinstalls
get_auth "Download and install ROS préinstalls?"

if [[ $INSTALL == 1 ]]; then

    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
    sudo apt-get update
    sudo apt-get install ros-kinetic-desktop-full
    sudo rosdep init
    rosdep update

    echo "source /opt/ros/kinetic/setup.$term" >> ~/.$rc
    source ~/.$rc
    sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential
    sudo apt-get install python-catkin-tools
    sudo apt install python-wstool
    pip install catkin_pkg

    echo "
Préinstall done...
"

fi

# ROS Préinstalls
get_auth "Download and install ROS?"

if [[ $INSTALL == 1 ]]; then

    # Setup catkin workspace
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/ ; catkin init
    # Add workspace to bashrc.
    echo "source ~/catkin_ws/devel/setup.$term" >> ~/.$rc
    cd ~/catkin_ws/src ; wstool init
    # Install FlightGoggles nodes and deps from rosinstall file
    cd ~/catkin_ws/src ; wstool merge https://raw.githubusercontent.com/mit-fast/FlightGoggles/master/flightgoggles.rosinstall
    cd ~/catkin_ws/src ; wstool update
    # Install required libraries.
    cd ~/catkin_ws/ ; rosdep install --from-paths src --ignore-src --rosdistro kinetic -y
    # Install external libraries for flightgoggles_ros_bridge
    sudo apt install -y libzmqpp-dev libeigen3-dev
    # Install dependencies for flightgoggles renderer
    sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-utils
    # Build nodes
    cd ~/catkin_ws/ ; catkin build
    # Refresh workspace
    source ~/.$rc
    echo "
Install done...
"

fi
