"""
Classes:

.. autoclass:: Block
.. autoclass:: Segment
.. autoclass:: RecordingChannelGroup
.. autoclass:: RecordingChannel

.. autoclass:: AnalogSignal
.. autoclass:: AnalogSignalArray
.. autoclass:: IrregularlySampledSignal

.. autoclass:: Event
.. autoclass:: EventArray
.. autoclass:: Epoch
.. autoclass:: EpochArray

.. autoclass:: Unit
.. autoclass:: Spike
.. autoclass:: SpikeTrain



"""

from __future__ import absolute_import

from neo.core.block import Block
from neo.core.segment import Segment
from neo.core.recordingchannelgroup import RecordingChannelGroup
from neo.core.recordingchannel import RecordingChannel


from neo.core.analogsignal import AnalogSignal
from neo.core.analogsignalarray import AnalogSignalArray
from neo.core.irregularlysampledsignal import IrregularlySampledSignal


from neo.core.event import Event
from neo.core.eventarray import EventArray
from neo.core.epoch import Epoch
from neo.core.epocharray import EpochArray

from neo.core.unit import Unit
from neo.core.spike import Spike
from neo.core.spiketrain import SpikeTrain


objectlist = [Block, Segment, RecordingChannelGroup, RecordingChannel,
              AnalogSignal, AnalogSignalArray, IrregularlySampledSignal,
              Event, EventArray, Epoch, EpochArray, Unit, Spike, SpikeTrain
              ]
