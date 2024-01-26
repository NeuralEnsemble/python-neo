========================
Download and install Neo
========================

Neo is a pure Python package, so it should be easy to get it running on any
system.

Installing with pip
===================

You can install the latest published version of Neo and its dependencies using::

    $ pip install neo


Dependencies
------------

    * Python_ >= 3.8
    * numpy_ >= 1.19.5
    * quantities_ >= 0.12.1

Certain IO modules have additional dependencies. If these are not satisfied,
Neo will still install but the IO module that uses them will fail on loading:

   * scipy >= 1.0.0 for NeoMatlabIO
   * h5py >= 2.5 for KwikIO
   * klusta for KwikIO
   * igor2 >= 0.5.2 for IgorIO
   * nixio >= 1.5 for NixIO
   * stfio for StimfitIO
   * pillow for TiffIO

These dependencies can be installed by specifying a comma-separated list with the
``pip install`` command, e.g.::

    $ pip install neo[nixio,tiffio]

Or when installing a specific version of neo::

    $ pip install neo[nixio,tiffio]==0.9.0

The following IO modules have additional dependencies:

   * igorproio
   * kwikio
   * neomatlabio
   * nixio
   * stimfitio
   * tiffio


Installing from source
======================

To download and install the package manually, download:

    |neo_github_url|


Then:

.. parsed-literal::

    $ unzip neo-|release|.zip
    $ cd neo-|release|
    $ pip install .

Alternatively, to install the latest version of Neo from the Git repository::

    $ git clone git://github.com/NeuralEnsemble/python-neo.git
    $ cd python-neo
    $ pip install .


Installing with Conda
=====================

::

    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict
    $ conda install -c conda-forge python-neo


Installing from a package repository
====================================

To install Neo if you're using Fedora_ Linux::

    $ sudo dnf install python-neo

.. NeuroDebian seems out of date - still has Trac as homepage - how to update?

To install Neo if you're using the Spack_ package manager::

    $ spack install py-neo


.. _`Python`: https://www.python.org/
.. _`numpy`: https://numpy.org/
.. _`quantities`: https://pypi.org/project/quantities/
.. _`pip`: https://pypi.org/project/pip/
.. _`setuptools`: http://pypi.python.org/pypi/setuptools
.. _Anaconda: https://www.anaconda.com/distribution/
.. _Fedora: https://src.fedoraproject.org/rpms/python-neo
.. _Spack: https://spack.readthedocs.io/en/latest/package_list.html#py-neo
