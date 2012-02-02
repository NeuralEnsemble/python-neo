*************************
Download and Installation
*************************

Neo is a pure Python package, so it should be easy to get it running on any
system.

Dependencies
============
  
    * Python_ >= 2.6
    * numpy_ >= 1.3.0  (1.5.0 for Python 3)
    * quantities_ >= 0.9.0

For Debian/Ubuntu, you can install these using::

    $ apt-get install python-numpy python-pip
    $ pip install quantities

You may need to run these as root. For other operating systems, you can
download installers from the links above.

Certain IO modules have additional dependencies. If these are not satisfied,
Neo will still install but the IO module that uses them will fail on loading:

   * scipy >= 0.8 for NeoMatlabIO
   * pytables >= 2.2 for Hdf5IO

For scipy  on Debian testing/Ubuntu, you can install these using::

    $ apt-get install python-scipy


For pytables version 2.2 is a bit more complicated::

    $ apt-get install libhdf5-serial-dev python-numexpr cython
    $ pip install tables



Installing from the Python Package Index
========================================

If you have pip_ installed::

    $ pip install neo
    
Alternatively, if you have setuptools_::
    
    $ easy_install neo
    
Both of these will automatically download and install the latest release (again
you may need to have administrator privileges on the machine you are installing
on).
    
To download and install manually, download:

    http://pypi.python.org/packages/source/N/neo/neo-0.2.0.tar.gz

Then::

    $ tar xzf neo-0.2.0.tar.gz
    $ cd neo-0.2.0
    $ python setup.py install
    
or::

    $ python3 setup.py install
    
depending on which version of Python you are using.

Installing from source
======================

To install the latest version of Neo from the Subversion repository::

    svn co https://neuralensemble.org/svn/neo/trunk neo_trunk
    cd neo_trunk
    python setup.py install


.. _`Python`: http://python.org/
.. _`numpy`: http://numpy.scipy.org/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`pip`: http://pypi.python.org/pypi/pip
.. _`setuptools`: http://pypi.python.org/pypi/setuptools
