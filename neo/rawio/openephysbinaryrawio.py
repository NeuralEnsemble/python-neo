"""
This module implement the "new" binary OpenEphys format.
In this format channels are interleaved in one file.


Author: Julia Sprenger and Samuel Garcia
"""


import os
import re

import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)


RECORD_SIZE = 1024
HEADER_SIZE = 1024


class OpenEphysBinaryRawIO(BaseRawIO):
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        pass

    def _segment_t_start(self, block_index, seg_index):
        pass

    def _segment_t_stop(self, block_index, seg_index):
        pass

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        pass

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        pass

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        pass
    
    def _spike_count(self, block_index, seg_index, unit_index):
        pass

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        pass

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        pass

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        pass

    def _event_count(self, block_index, seg_index, event_channel_index):
        pass

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        pass

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        pass

    def _rescale_epoch_duration(self, raw_duration, dtype):
        pass
