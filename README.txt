Neo is a package for representing electrophysiology data in Python, together
with support for reading a wide range of neurophysiology file formats, including
Spike2, NeuroExplorer, AlphaOmega, Axon, Blackrock, Plexon, Tdt, and support for
writing to a subset of these formats plus non-proprietary formats including HDF5.

The goal of neo is to improve interoperability between Python tools for
analyzing, visualizing and generating electrophysiology data (such as
OpenElectrophy_, NeuroTools_, G-node_, Helmholtz_, PyNN_) by providing a common,
shared object model. In order to be as lightweight a dependency as possible,
neo is deliberately limited to represention of data, with no functions for data
analysis or visualization.

Neo implements a hierarchical data model well adapted to intracellular and
extracellular electrophysiology and EEG data with support for multi-electrodes
(for example tetrodes). Neo's data objects build on the quantities_ package,
which in turn builds on NumPy by adding support for physical dimensions. Thus
neo objects behave just like normal NumPy arrays, but with additional metadata,
checks for dimensional consistency and automatic unit conversion.

Documentation is available at http://packages.python.org/neo/

The project home page is at http://neuralensemble.org/neo

For installation instructions, see doc/source/install.rst

:copyright: Copyright 2010-2012 by the Neo team, see AUTHORS.
:license: Modified BSD License, see LICENSE.txt for details.