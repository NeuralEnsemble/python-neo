************
Installation
************

Neo is a pure Python package, so it should be easy to get it running on any
system.

Dependencies
============
  
    * Python_ >= 2.6
    * numpy_ >= 1.7.1
    * quantities_ >= 0.9.0

For Debian/Ubuntu, you can install these using::

    $ apt-get install python-numpy python-pip
    $ pip install quantities

You may need to run these as root. For other operating systems, you can
download installers from the links above, or use a scientific Python distribution
such as Anaconda_.

Certain IO modules have additional dependencies. If these are not satisfied,
Neo will still install but the IO module that uses them will fail on loading:

   * scipy >= 0.12.0 for NeoMatlabIO
   * h5py >= 2.5 for Hdf5IO, KwikIO
   * igor >= 0.2 for IgorIO
   * nixio >= 1.2 for NixIO
   * stfio for StimfitIO


Installing from the Python Package Index
========================================

.. warning:: alpha and beta releases cannot be installed from PyPI.

If you have pip_ installed::

    $ pip install https://github.com/NeuralEnsemble/python-neo/archive/neo-0.5.0alpha1.zip
    
This will automatically download and install the latest release (again
you may need to have administrator privileges on the machine you are installing
on).
    
To download and install manually, download:

    https://github.com/NeuralEnsemble/python-neo/archive/neo-0.5.0alpha1.zip

Then::

    $ unzip neo-0.5.0alpha1.zip
    $ cd neo-0.5.0alpha1
    $ python setup.py install
    
or::

    $ python3 setup.py install
    
depending on which version of Python you are using.


Installing from source
======================

To install the latest version of Neo from the Git repository::

    $ git clone git://github.com/NeuralEnsemble/python-neo.git
    $ cd python-neo
    $ python setup.py install


Python 3 support
================

:mod:`neo.core` is fully compatible with Python 3, but only some of the IO
modules support it, as shown in the table below:

================== ======== ========
Module             Python 2 Python 3
================== ======== ========
AlphaOmegaIO       Yes      No
AsciiSignalIO      Yes      Yes
AsciiSpikeTrainIO  Yes      Yes
AxonIO             Yes      Yes
BlackrockIO        Yes      No
BrainwareDamIO     Yes      Yes
BrainwareF32IO     Yes      Yes
BrainwareSrcIO     Yes      Yes
ElanIO             Yes      No
IgorIO             Yes      Yes
NeoHdf5IO          Yes      Yes
KlustakwikIO       Yes      No
KwikIO             Yes      Yes
MicromedIO         Yes      No
NeoMatlabIO        Yes      Yes
NeuroExplorerIO    Yes      No
NeuroscopeIO       Yes      Yes
NixIO              Yes      Yes
PickleIO           Yes      Yes
PlexonIO           Yes      No
PyNNIO             Yes      Yes
RawBinarySignalIO  Yes      Yes
Spike2IO           Yes      Yes
TdtIO              Yes      No
WinEdrIO           Yes      Yes
WinWcpIO           Yes      Yes
================== ======== ========




.. _`Python`: http://python.org/
.. _`numpy`: http://numpy.scipy.org/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`pip`: http://pypi.python.org/pypi/pip
.. _`setuptools`: http://pypi.python.org/pypi/setuptools
.. _Anaconda: https://www.continuum.io/downloads
