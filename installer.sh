
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
    	echo "Skipping..."
    	INSTALL=0
    else
    	INSTALL=1
    fi
}

dwldata () {
    get_auth "Download data to the current directory?"
    if [[ $INSTALL == 1 ]]; then
        curl -O http://83.169.39.135/ssd_detectors/ssd512_voc_weights_fixed.zip
        curl -O http://83.169.39.135/ssd_detectors/ssd300_voc_weights_fixed.zip
        curl -O https://s3.amazonaws.com/herox-alphapilot/Data_LeaderboardTesting.zip
        curl -O https://s3.amazonaws.com/herox-alphapilot/Data_Training.zip
        curl -O https://d253pvgap36xx8.cloudfront.net/challenges/resources/f1e5a35031ae11e9a27e0242ac110002/training_GT_labels.json
    fi
}

mountdata () {
    get_auth "Mount external disk on /data?"
    if [[ $INSTALL == 1 ]]; then
        NAME=`lsblk | grep 100G | cut -d ' ' -f 1`
        if [ -d "/data" ]; then
            sudo mkdir /data
        fi
        sudo mount /dev/$NAME /data
    fi
}

pyinstall () {
    # Python 3.7
    get_auth "Download and install Python3.7?"
    if [[ $INSTALL == 1 ]]; then
        # Step 1 – Requirements
        sudo apt-get -y install gcc openssl-devel bzip2-devel
        sudo apt-get install libffi-devel
        # Step 2 – Download Python 3.7
        cd /usr/src ; sudo wget https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tgz ; \
        sudo tar xzf Python-3.7.2.tgz
        # Step 3 – Install Python 3.7
        cd /usr/src/Python-3.7.2 ; sudo ./configure --enable-optimizations ; \
        sudo make altinstall
        # Step 4 – Check Python Version
        sudo rm /usr/src/Python-3.7.2.tgz
    fi
    # Install Miniconda3
    get_auth "Download and install Miniconda3?"
    if [[ $INSTALL == 1 ]]; then
        sudo apt-get -y install zip
        # Download and install
        wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p $HOME/anaconda
        # Exports
        export PATH=$HOME/anaconda/bin:$PATH # add to PATH
        echo 'export PATH=$HOME/anaconda/bin:$PATH' >> ~/.bashrc # add to bashrc for future use
        hash -r
        # some configuration to make it easy to install things
        conda config --set always_yes yes --set changeps1 no
        conda update -q conda
        # add channels to look for packages
        conda config --add channels r # for backward compatibility with old r packages
        conda config --add channels defaults
        conda config --add channels conda-forge # additional common tools
        conda config --add channels bioconda # useful bioinformatics
        # Installs of conda
        conda install -n root _license
        # display info
        conda info -a
    fi
}

rosinstall () {
    echo -n "Which terminal are you using on this device?"
    DEFAULT="bash"
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
}

lfsinstall () {
    get_auth "Download and install git-lfs?"
    if [[ $INSTALL == 1 ]]; then
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt-get install git-lfs
        git lfs install
    fi
}

pipinstall () {
    get_auth "Install AlphaPilot requirements?"
    if [[ $INSTALL == 1 ]]; then
        pip install -r requirements.txt
    fi
}

while true;
do
    COLUMNS=0
    PS3='Choose an option: '
    options=("Download data" "Mount" "PYinstall" "ROSinstall" "LFSinstall" "PIPinstall" "Quit")
    select opt in "${options[@]}"
    do
        case $opt in
            "Download data")
                # echo "Download data"
                dwldata; break ;;
            "Mount")
                # echo "Mount disk"
                mountdata; break ;;
            "PYinstall")
                # echo "Python installer"
                pyinstall; break ;;
            "ROSinstall")
                # echo "ROS installer"
                rosinstall; break ;;
            "LFSinstall")
                # echo "Git-LFS installer"
                lfsinstall; break ;;
            "PIPinstall")
                # echo "Pip requirements installer"
                pipinstall; break ;;
            "Quit")
                break 2 ;;
            *) echo "invalid option $REPLY" >&2
        esac
    done

    echo "Are we done?"
    PS3='>>> '
    select opt in "Yes" "No"
    do
        case $REPLY in
            1) break 2 ;;
            2) break ;;
            *) echo "Enter 1 (Yes) OR 2 (No)." >&2
        esac
    done
done
