
****************
Installation
****************




Dependencies
==================

Dependencies:
    
    * Python >= 2.6
    * Python numpy
    * quantities

For testing you also need:
    * Python scipy
    * Pyhton matplotlib

For IO, there are several additional dependencies. If these are not satisfied, neo will still install but the IO module that use them will fail on loading.



Install neo
=======================

First install the dependencies:
    sudo apt-get install python-numpy python-scipy python-matplotlib python-pip
    mkdir quantities-src
    cd quantities-src
    sudo pip install quantities

Next install neo. You have two choices: the stable release, and the real-time release which is updated more frequently with bugfixes and feature additions.
    
    From pypi (last stable release)::
        
        sudo apt-get install python-setuptools
        sudo easy_install neo
    
    From source (real time release)::
        
        sudo apt-get install subversion
        svn co https://neuralensemble.org/svn/neo/
        cd neo/trunk
        python setup.py install



