# -*- coding: utf-8 -*-
"""
neo.io have been split in 2 level API:
  * neo.io: this API give neo object
  * neo.rawio: this API give raw data as they are in files.

Developper are encourage to use neo.rawio.

When this is done the neo.io is done automagically with
this king of following code.

Author: sgarcia

"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.examplerawio import ExampleRawIO


class ExampleIO(ExampleRawIO, BaseFromRaw):
    name = 'example IO'
    description = "Fake IO"

    # This is an inportant choice when there are several channels.
    #   'split-all' :  1 AnalogSignal each 1 channel
    #   'group-by-same-units' : one 2D AnalogSignal for each group of channel with same units
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename=''):
        ExampleRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
