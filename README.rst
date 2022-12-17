===
Neo
===

Neo is a Python package for working with electrophysiology data in Python, together
with support for reading a wide range of neurophysiology file formats, including
Spike2, NeuroExplorer, AlphaOmega, Axon, Blackrock, Plexon, Tdt, and support for
writing to a subset of these formats plus non-proprietary formats including HDF5.

The goal of Neo is to improve interoperability between Python tools for
analyzing, visualizing and generating electrophysiology data by providing a common,
shared object model. In order to be as lightweight a dependency as possible,
Neo is deliberately limited to represention of data, with no functions for data
analysis or visualization.

Neo is used by a number of other software tools, including
SpykeViewer_ (data analysis and visualization), Elephant_ (data analysis),
the G-node_ suite (databasing), PyNN_ (simulations), tridesclous_ (spike sorting)
and ephyviewer_ (data visualization).
OpenElectrophy_ (data analysis and visualization) uses an older version of neo.

Neo implements a hierarchical data model well adapted to intracellular and
extracellular electrophysiology and EEG data with support for multi-electrodes
(for example tetrodes). Neo's data objects build on the quantities package,
which in turn builds on NumPy by adding support for physical dimensions. Thus
Neo objects behave just like normal NumPy arrays, but with additional metadata,
checks for dimensional consistency and automatic unit conversion.

A project with similar aims but for neuroimaging file formats is `NiBabel`_.

Code status
-----------

.. image:: https://github.com/NeuralEnsemble/python-neo/actions/workflows/core-test.yml/badge.svg?event=push&branch=master
   :target: https://github.com/NeuralEnsemble/python-neo/actions?query=event%3Apush+branch%3Amaster
   :alt: Core Test Status (Github Actions)
.. image:: https://github.com/NeuralEnsemble/python-neo/actions/workflows/io-test.yml/badge.svg?event=push&branch=master
   :target: https://github.com/NeuralEnsemble/python-neo/actions?query=event%3Apush+branch%3Amaster
   :alt: IO Test Status (Github Actions)
.. image:: https://coveralls.io/repos/NeuralEnsemble/python-neo/badge.png
   :target: https://coveralls.io/r/NeuralEnsemble/python-neo
   :alt: Unit Test Coverage

More information
----------------

- Home page: http://neuralensemble.org/neo
- Mailing list: https://groups.google.com/forum/?fromgroups#!forum/neuralensemble
- Documentation: http://neo.readthedocs.io/
- Bug reports: https://github.com/NeuralEnsemble/python-neo/issues

For installation instructions, see doc/source/install.rst

To cite Neo in publications, see CITATION.txt

:copyright: Copyright 2010-2022 by the Neo team, see doc/source/authors.rst.
:license: 3-Clause Revised BSD License, see LICENSE.txt for details.

Funding
-------

Development of Neo has been partially funded by the European Union Sixth Framework Program (FP6) under
grant agreement FETPI-015879 (FACETS), by the European Union Seventh Framework Program (FP7/2007­-2013)
under grant agreements no. 269921 (BrainScaleS) and no. 604102 (HBP),
and by the European Union’s Horizon 2020 Framework Programme for
Research and Innovation under the Specific Grant Agreements No. 720270 (Human Brain Project SGA1),
No. 785907 (Human Brain Project SGA2) and No. 945539 (Human Brain Project SGA3).

.. _OpenElectrophy: https://github.com/OpenElectrophy/OpenElectrophy
.. _Elephant: http://neuralensemble.org/elephant
.. _G-node: http://www.g-node.org/
.. _Neuroshare: http://neuroshare.org/
.. _SpykeViewer: https://spyke-viewer.readthedocs.org/en/latest/
.. _NiBabel: http://nipy.sourceforge.net/nibabel/
.. _PyNN: http://neuralensemble.org/PyNN
.. _quantities: http://pypi.python.org/pypi/quantities
.. _`NeuralEnsemble mailing list`: http://groups.google.com/group/neuralensemble
.. _`issue tracker`: https://github.c
.. _tridesclous: https://github.com/tridesclous/tridesclous
.. _ephyviewer: https://github.com/NeuralEnsemble/ephyviewer
