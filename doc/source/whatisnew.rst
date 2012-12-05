*************
Release notes
*************

What's new in version 0.2.1?
----------------------------

 * assorted bug fixes
 * added :func:`time_slice()` method to the :class:`SpikeTrain` and :class:`AnalogSignalArray` classes.
 * improvements to annotation data type handling
 * added PickleIO, allowing saving Neo objects in the Python pickle format.
 * added ElphyIO (see http://www.unic.cnrs-gif.fr/software.html)
 * added BrainVisionIO (see http://www.brainvision.com/)
 * improvements to PlexonIO
 * added :func:`merge()` method to the :class:`Block` and :class:`Segment` classes
 * development was mostly moved to GitHub, although the issue tracker is still at neuralensemble.org/neo


What's new in version 0.2?
--------------------------

New features compared to neo 0.1:
 * new schema more consistent.
 * new objects: RecordingChannelGroup, EventArray, AnalogSignalArray, EpochArray
 * Neuron is now Unit
 * use the quantities_ module for everything that can have units.
 * Some objects directly inherit from Quantity: SpikeTrain, AnalogSignal, AnalogSignalArray, instead of having an attribute for data.
 * Attributes are classifyed in 3 categories: necessary, recommended, free.
 * lazy and cascade keywords are added to all IOs
 * Python 3 support
 * better tests



.. _quantities: http://pypi.python.org/pypi/quantities