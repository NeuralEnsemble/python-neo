# -*- coding: utf-8 -*-
"""
Class for reading data from "Raw" Multi Channel System (MCS) format.
This format is NOT the native MCS format (*.mcd).
This format is a raw format with an internal binary header exported by the
"MC_DataTool binary conversion" with the option header selected.

The internal header contains sampling rate, channel names, gain and units.
Not so bad: everything that Neo needs, so this IO is without parameters.

If some MCS customers read this you should lobby to get the real specification
of the real MCS format (.mcd), then an IO module for the native MCS format
could be written instead of this ersatz.

Author: Samuel Garcia
"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

import os
import sys


class RawMCSRawIO(BaseRawIO):
    extensions = ['raw']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        self._info = info = parse_mcs_raw_header(self.filename)

        self.dtype = 'uint16'
        self.sampling_rate = info['sampling_rate']
        self.nb_channel = len(info['channel_names'])

        self._raw_signals = np.memmap(self.filename, dtype=self.dtype, mode='r',
                                      offset=info['header_size']).reshape(-1, self.nb_channel)

        sig_channels = []
        for c in range(self.nb_channel):
            chan_id = c
            group_id = 0
            sig_channels.append((info['channel_names'][c], chan_id, self.sampling_rate,
                                self.dtype, info['signal_units'], info['signal_gain'],
                                info['signal_offset'], group_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0] / self.sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]

        return raw_signals


def parse_mcs_raw_header(filename):
    """
    This is a from-scratch implementation, with some inspiration
    (but no code) taken from the following files:
    https://github.com/spyking-circus/spyking-circus/blob/master/circus/files/mcs_raw_binary.py
    https://github.com/jeffalstott/Multi-Channel-Systems-Import/blob/master/MCS.py
    """
    MAX_HEADER_SIZE = 5000

    with open(filename, mode='rb') as f:
        raw_header = f.read(MAX_HEADER_SIZE)

        header_size = raw_header.find(b'EOH')
        assert header_size != -1, 'Error in reading raw mcs header'
        header_size = header_size + 5
        raw_header = raw_header[:header_size]
        raw_header = raw_header.replace(b'\r', b'')

        info = {}
        info['header_size'] = header_size

        def parse_line(line, key):
            if key + b' = ' in line:
                v = line.replace(key, b'').replace(b' ', b'').replace(b'=', b'')
                return v

        keys = (b'Sample rate', b'ADC zero', b'ADC zero', b'El', b'Streams')

        for line in raw_header.split(b'\n'):
            for key in keys:
                v = parse_line(line, key)
                if v is None:
                    continue

                if key == b'Sample rate':
                    info['sampling_rate'] = float(v)

                elif key == b'ADC zero':
                    info['adc_zero'] = int(v)

                elif key == b'El':
                    v = v.decode('Windows-1252')
                    v = v.replace('/AD', '')
                    split_pos = 0
                    while v[split_pos] in '1234567890.':
                        split_pos += 1
                        if split_pos == len(v):
                            split_pos = None
                            break
                    assert split_pos is not None, 'Impossible to find units and scaling'
                    info['signal_gain'] = float(v[:split_pos])
                    info['signal_units'] = v[split_pos:].replace(u'µ', u'u')
                    info['signal_offset'] = -info['signal_gain'] * info['adc_zero']

                elif key == b'Streams':
                    info['channel_names'] = v.decode('Windows-1252').split(';')

    return info
