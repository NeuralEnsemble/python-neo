**************************
What's new in version 0.2?
**************************

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