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
.. autoclass:: Group

.. autoclass:: AnalogSignal
.. autoclass:: IrregularlySampledSignal

.. autoclass:: ChannelView

.. autoclass:: Event
.. autoclass:: Epoch

.. autoclass:: SpikeTrain
.. autoclass:: ImageSequence

.. autoclass:: RectangularRegionOfInterest
.. autoclass:: CircularRegionOfInterest
.. autoclass:: PolygonRegionOfInterest

"""

from neo.core.block import Block
from neo.core.segment import Segment
from neo.core.analogsignal import AnalogSignal
from neo.core.irregularlysampledsignal import IrregularlySampledSignal

from neo.core.event import Event
from neo.core.epoch import Epoch

from neo.core.spiketrain import SpikeTrain

from neo.core.imagesequence import ImageSequence
from neo.core.regionofinterest import RectangularRegionOfInterest, CircularRegionOfInterest, PolygonRegionOfInterest

from neo.core.view import ChannelView
from neo.core.group import Group

# Block should always be first in this list
objectlist = [Block, Segment,
              AnalogSignal, IrregularlySampledSignal,
              Event, Epoch, SpikeTrain, ImageSequence,
              RectangularRegionOfInterest, CircularRegionOfInterest,
              PolygonRegionOfInterest, ChannelView, Group]

objectnames = [ob.__name__ for ob in objectlist]
class_by_name = dict(zip(objectnames, objectlist))
