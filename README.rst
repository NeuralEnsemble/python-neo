===
Neo
===

Neo is a package for representing electrophysiology data in Python, together
with support for reading a wide range of neurophysiology file formats, including
Spike2, NeuroExplorer, AlphaOmega, Axon, Blackrock, Plexon, Tdt, and support for
writing to a subset of these formats plus non-proprietary formats including HDF5.

The goal of Neo is to improve interoperability between Python tools for
analyzing, visualizing and generating electrophysiology data (such as
OpenElectrophy, NeuroTools, G-node, Helmholtz, PyNN) by providing a common,
shared object model. In order to be as lightweight a dependency as possible,
Neo is deliberately limited to represention of data, with no functions for data
analysis or visualization.

Neo implements a hierarchical data model well adapted to intracellular and
extracellular electrophysiology and EEG data with support for multi-electrodes
(for example tetrodes). Neo's data objects build on the quantities package,
which in turn builds on NumPy by adding support for physical dimensions. Thus
neo objects behave just like normal NumPy arrays, but with additional metadata,
checks for dimensional consistency and automatic unit conversion.

Code status
-----------

.. image:: https://travis-ci.org/NeuralEnsemble/python-neo.png?branch=master
   :target: https://travis-ci.org/NeuralEnsemble/python-neo
   :alt: Unit Test Status
.. image:: https://coveralls.io/repos/NeuralEnsemble/python-neo/badge.png
   :target: https://coveralls.io/r/NeuralEnsemble/python-neo
   :alt: Unit Test Coverage
.. image:: https://requires.io/github/NeuralEnsemble/python-neo/requirements.png?branch=master
   :target: https://requires.io/github/NeuralEnsemble/python-neo/requirements/?branch=master
   :alt: Requirements Status

More information
----------------

- Home page: http://neuralensemble.org/neo
- Mailing list: https://groups.google.com/forum/?fromgroups#!forum/neuralensemble
- Documentation: http://packages.python.org/neo/
- Bug reports: https://github.com/NeuralEnsemble/python-neo/issues

For installation instructions, see doc/source/install.rst

:copyright: Copyright 2010-2016 by the Neo team, see AUTHORS.
:license: 3-Clause Revised BSD License, see LICENSE.txt for details.
