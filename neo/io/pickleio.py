# -*- coding: utf-8 -*-
"""
Module for reading/writing data from/to Python pickle format.


Class:
    PickleIO

Supported: Read/Write

Authors: Andrew Davison
"""

try:
    import cPickle as pickle  # Python 2
except ImportError:
    import pickle  # Python 3

from neo.io.baseio import BaseIO
from neo.core import (Block, Segment,
                      AnalogSignal, AnalogSignalArray, SpikeTrain)


class PickleIO(BaseIO):
    """

    """
    is_readable = True
    is_writable = True
    has_header = False
    is_streameable = False # TODO - correct spelling to "is_streamable"
    supported_objects = [Block, Segment, AnalogSignal, AnalogSignalArray, SpikeTrain] # should extend to Epoch, etc.
    readable_objects = supported_objects
    writeable_objects = supported_objects
    mode = 'file'
    name = "Python pickle file"
    extensions = ['pkl', 'pickle']

    def read_block(self, lazy=False, cascade=True):
        with open(self.filename, "rb") as fp:
            block = pickle.load(fp)
        return block

    def write_block(self, block):
        with open(self.filename, "wb") as fp:
            pickle.dump(block, fp)
