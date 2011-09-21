Neo: representing electrophysiology data in Python
==================================================

.. module:: neo

.. image:: images/neologo.png
    :width: 600 px



Neo stands for Neural Ensemble Objects and is a project to provide common class names and concepts
for dealing with electrophysiological (real and/or simulated) data, with the aim of
providing a common basis for OpenElectrophy_, NeuroTools_, G-node_, Helmholtz_ and other projects with similar goals.

Neo provides in particular :
 - a set of classes with precise definitions
 - a IO module that offer a simple API that fit many formats.
 - documentation.
 - a set of examples, including a format convertor.

For Python users, Neo can be taken as a replacement for the Neuroshare_ API. 
Neo natively read: Spike2, NeuroExplorer, AlphaOmega, Axon, Blackrock, Plexon, Tdt, and more.

The people behind the project are very open to discussion. Any feedback is gladly received and highly appreciated!

`NiBabel`_ is a project with similar aims for Neuroimaging file formats.

Contents:

.. toctree::
   :maxdepth: 1
   
   whatisnew
   install
   core
   usecases
   io
   examples
   developers_guide
   io_developers_guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _OpenElectrophy: http://neuralensemble.org/trac/OpenElectrophy
.. _NeuroTools: http://neuralensemble.org/NeuroTools
.. _G-node: http://www.g-node.org/
.. _Neuroshare: http://neuroshare.org/
.. _Helmholtz: https://www.dbunic.cnrs-gif.fr/documentation/helmholtz/
.. _NiBabel: http://nipy.sourceforge.net/nibabel/

