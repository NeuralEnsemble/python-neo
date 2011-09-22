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

from .block import Block
from .segment import Segment 
from .recordingchannelgroup import RecordingChannelGroup
from .recordingchannel import RecordingChannel


from .analogsignal import AnalogSignal
from .analogsignalarray import AnalogSignalArray
from .irregularlysampledsignal import IrregularlySampledSignal


from .event import Event
from .eventarray import EventArray
from .epoch import Epoch
from .epocharray import EpochArray

from .unit import Unit
from .spike import Spike
from .spiketrain import SpikeTrain



objectlist = [ Block, Segment, RecordingChannelGroup , RecordingChannel,
        AnalogSignal, AnalogSignalArray, IrregularlySampledSignal,
        Event, EventArray, Epoch, EpochArray,
        Unit, Spike , SpikeTrain
            ]
