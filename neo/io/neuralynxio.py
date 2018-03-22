# -*- coding: utf-8 -*-
"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV and NSE file formats.

Depends on: numpy

Supported: Read

Author: Julia Sprenger, Carlos Canova
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuralynxrawio import NeuralynxRawIO


class NeuralynxIO(NeuralynxRawIO, BaseFromRaw):
    """
    Class for reading data from Neuralynx files.
    This IO supports NCS, NEV, NSE and NTT file formats.

    NCS contains signals for one channel
    NEV contains events
    NSE contains spikes and waveforms for mono electrodes
    NTT contains spikes and waveforms for tetrodes
    """
    _prefered_signal_group_mode = 'group-by-same-units'
    mode = 'dir'

    def __init__(self, dirname, use_cache=False, cache_path='same_as_resource'):
        NeuralynxRawIO.__init__(self, dirname=dirname,
                                use_cache=use_cache, cache_path=cache_path)
        BaseFromRaw.__init__(self, dirname)
