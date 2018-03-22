.. module:: neo

.. image:: images/neologo.png
    :width: 600 px
    
Neo is a Python package for working with electrophysiology data in Python, together
with support for reading a wide range of neurophysiology file formats, including
Spike2, NeuroExplorer, AlphaOmega, Axon, Blackrock, Plexon, Tdt, Igor Pro, and support for
writing to a subset of these formats plus non-proprietary formats including Kwik and HDF5.

The goal of Neo is to improve interoperability between Python tools for
analyzing, visualizing and generating electrophysiology data, by providing a common,
shared object model. In order to be as lightweight a dependency as possible,
Neo is deliberately limited to represention of data, with no functions for data
analysis or visualization.

Neo is used by a number of other software tools, including 
SpykeViewer_ (data analysis and visualization), Elephant_ (data analysis),
the G-node_ suite (databasing), PyNN_ (simulations), tridesclous_ (spike sorting)
and ephyviewer_ (data visualization).
OpenElectrophy_ (data analysis and visualization) used an older version of Neo.


Neo implements a hierarchical data model well adapted to intracellular and
extracellular electrophysiology and EEG data with support for multi-electrodes
(for example tetrodes). Neo's data objects build on the quantities_ package,
which in turn builds on NumPy by adding support for physical dimensions. Thus
Neo objects behave just like normal NumPy arrays, but with additional metadata,
checks for dimensional consistency and automatic unit conversion.

A project with similar aims but for neuroimaging file formats is `NiBabel`_.


Documentation
-------------

.. toctree::
   :maxdepth: 1
   
   install
   core
   usecases
   io
   rawio
   examples
   api_reference
   whatisnew
   developers_guide
   io_developers_guide
   authors


License
-------

Neo is free software, distributed under a 3-clause Revised BSD licence (BSD-3-Clause).


Support
-------

If you have problems installing the software or questions about usage, documentation or anything
else related to Neo, you can post to the `NeuralEnsemble mailing list`_. If you find a bug,
please create a ticket in our `issue tracker`_.


Contributing
------------

Any feedback is gladly received and highly appreciated! Neo is a community project,
and all contributions are welcomed - see the :doc:`developers_guide` for more information.
`Source code <https://github.com/NeuralEnsemble/python-neo>`_ is on GitHub.


Citation
--------

.. include:: ../../CITATION.txt


.. _OpenElectrophy: https://github.com/OpenElectrophy/OpenElectrophy
.. _Elephant: http://neuralensemble.org/elephant
.. _G-node: http://www.g-node.org/
.. _Neuroshare: http://neuroshare.org/
.. _SpykeViewer: https://spyke-viewer.readthedocs.org/en/latest/
.. _NiBabel: http://nipy.sourceforge.net/nibabel/
.. _PyNN: http://neuralensemble.org/PyNN
.. _quantities: http://pypi.python.org/pypi/quantities
.. _`NeuralEnsemble mailing list`: http://groups.google.com/group/neuralensemble
.. _`issue tracker`: https://github.com/NeuralEnsemble/python-neo/issues
.. _tridesclous: https://github.com/tridesclous/tridesclous
.. _ephyviewer: https://github.com/NeuralEnsemble/ephyviewer
