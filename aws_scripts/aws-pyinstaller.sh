#!/bin/bash -x

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

get_auth "Download and install git-lfs?"

if [[ $INSTALL == 1 ]]; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
fi

get_auth "Install AlphaPilot requirements?"

if [[ $INSTALL == 1 ]]; then
    pip install -r requirements.txt
fi

exit 0
