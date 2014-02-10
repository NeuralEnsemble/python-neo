*************
Release notes
*************


What's new in version 0.4.0?
----------------------------

  * added StimfitIO

What's new in version 0.3.3?
----------------------------

  * fix a bug in PlexonIO where some EventArrays only load 1 element.
  * fix a bug in BrainwareSrcIo for segments with no spikes.


What's new in version 0.3.2?
----------------------------

  * cleanup ot io test code, with additional helper functions and methods
  * added BrainwareDamIo
  * added BrainwareF32Io
  * added BrainwareSrcIo


What's new in version 0.3.1?
----------------------------

  * lazy/cascading improvement
  * load_lazy_olbject() in neo.io added
  * added NeuroscopeIO


What's new in version 0.3.0?
----------------------------

  * various bug fixes in neo.io
  * added ElphyIO
  * SpikeTrain performence improved
  * An IO class now can return a list of Block (see read_all_blocks in IOs)
  * python3 compatibility improved


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
