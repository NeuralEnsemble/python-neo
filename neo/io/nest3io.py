# -*- coding: utf-8 -*-
"""
neo.io have been split in 2 level API:
  * neo.io: this API give neo object
  * neo.rawio: this API give raw data as they are in files.

Developer are encourage to use neo.rawio.

When this is done the neo.io is done automagically with
this king of following code.

Author: Johanna Senk, Julia Sprenger

"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.nest3rawio import Nest3RawIO


class Nest3IO(Nest3RawIO, BaseFromRaw):
    name = 'Nest3IO'
    description = "Fake IO"

    # This is an important choice when there are several channels.
    #   'split-all' :  1 AnalogSignal each 1 channel
    #   'group-by-same-units' : one 2D AnalogSignal for each group of channel with same units
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename=''):
        Nest3RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
