************
Installation
************

Neo is a pure Python package, so it should be easy to get it running on any
system.

Installing from the Python Package Index
========================================

Dependencies
------------

    * Python_ >= 3.6
    * numpy_ >= 1.13.0
    * quantities_ >= 0.12.1

You can install the latest published version of Neo and its dependencies using::

    $ pip install neo

Certain IO modules have additional dependencies. If these are not satisfied,
Neo will still install but the IO module that uses them will fail on loading:

   * scipy >= 1.0.0 for NeoMatlabIO
   * h5py >= 2.5 for KwikIO
   * klusta for KwikIO
   * igor >= 0.2 for IgorIO
   * nixio >= 1.5 for NixIO
   * stfio for StimfitIO
   * pillow for TiffIO

These dependencies can be installed by specifying a comma-separated list with the
``pip install`` command::

    $ pip install neo[nixio,tiffio]

Or when installing a specific version of neo::

    $ pip install neo[nixio,tiffio]==0.9.0

These additional dependencies for IO modules are available::

  * igorproio
  * kwikio
  * neomatlabio
  * nixio
  * stimfitio
  * tiffio


To download and install the package manually, download:

    |neo_github_url|


Then:

.. parsed-literal::

    $ unzip neo-|release|.zip
    $ cd neo-|release|
    $ python setup.py install


Installing from source
======================

To install the latest version of Neo from the Git repository::

    $ git clone git://github.com/NeuralEnsemble/python-neo.git
    $ cd python-neo
    $ python setup.py install


.. _`Python`: https://www.python.org/
.. _`numpy`: https://numpy.org/
.. _`quantities`: https://pypi.org/project/quantities/
.. _`pip`: https://pypi.org/project/pip/
.. _`setuptools`: http://pypi.python.org/pypi/setuptools
.. _Anaconda: https://www.anaconda.com/distribution/
