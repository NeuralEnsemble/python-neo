# -*- coding: utf-8 -*-
"""
:mod:`neo.core` provides classes for storing common electrophysiological data
types.  Some of these classes contain raw data, such as spike trains or
analog signals, while others are containers to organize other classes
(including both data classes and other container classes).

Classes from :mod:`neo.io` return nested data structures containing one
or more class from this module.

Classes:

.. autoclass:: Block
.. autoclass:: Segment
.. autoclass:: RecordingChannelGroup
.. autoclass:: RecordingChannel
.. autoclass:: Unit

.. autoclass:: AnalogSignal
.. autoclass:: AnalogSignalArray
.. autoclass:: IrregularlySampledSignal

.. autoclass:: Event
.. autoclass:: EventArray
.. autoclass:: Epoch
.. autoclass:: EpochArray

.. autoclass:: Spike
.. autoclass:: SpikeTrain

"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from neo.core.block import Block
from neo.core.segment import Segment
from neo.core.recordingchannelgroup import RecordingChannelGroup
from neo.core.recordingchannel import RecordingChannel
from neo.core.unit import Unit

from neo.core.analogsignal import AnalogSignal
from neo.core.analogsignalarray import AnalogSignalArray
from neo.core.irregularlysampledsignal import IrregularlySampledSignal

from neo.core.event import Event
from neo.core.eventarray import EventArray
from neo.core.epoch import Epoch
from neo.core.epocharray import EpochArray

from neo.core.spike import Spike
from neo.core.spiketrain import SpikeTrain

# Block should always be first in this list
objectlist = [Block, Segment, RecordingChannelGroup, RecordingChannel,
              AnalogSignal, AnalogSignalArray, IrregularlySampledSignal,
              Event, EventArray, Epoch, EpochArray, Unit, Spike, SpikeTrain
              ]

objectnames = [ob.__name__ for ob in objectlist]
class_by_name = dict(zip(objectnames, objectlist))
