
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

For IO, there are several depencies that are not imposed but that fail th IO to be  loaded.



Install neo
=======================


2 Possibilities:
    
    From pypi (last stable release)::
        
        sudo apt-get install python-setuptools
        sudo easy_install neo
    
    From source (real time release)::
        
        sudo apt-get install subversion
        svn co https://neuralensemble.org/svn/neo/
        cd neo/trunk
        python setup.py install



