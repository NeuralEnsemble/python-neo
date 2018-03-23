# -*- coding: utf-8 -*-
"""
Class for reading data from WinEdr, a software tool written by
John Dempster.

WinEdr is free:
http://spider.science.strath.ac.uk/sipbs/software.htm

Author: Samuel Garcia

"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

import os
import sys


class WinEdrRawIO(BaseRawIO):
    extensions = ['EDR', 'edr']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        with open(self.filename, 'rb') as fid:
            headertext = fid.read(2048)
            headertext = headertext.decode('ascii')
            header = {}
            for line in headertext.split('\r\n'):
                if '=' not in line:
                    continue
                # print '#' , line , '#'
                key, val = line.split('=')
                if key in ['NC', 'NR', 'NBH', 'NBA', 'NBD', 'ADCMAX', 'NP', 'NZ', 'ADCMAX']:
                    val = int(val)
                elif key in ['AD', 'DT', ]:
                    val = val.replace(',', '.')
                    val = float(val)
                header[key] = val

        self._raw_signals = np.memmap(self.filename, dtype='int16', mode='r',
                                      shape=(header['NP'] // header['NC'], header['NC'],),
                                      offset=header['NBH'])

        DT = header['DT']
        if 'TU' in header:
            if header['TU'] == 'ms':
                DT *= .001
        self._sampling_rate = 1. / DT

        sig_channels = []
        for c in range(header['NC']):
            YCF = float(header['YCF%d' % c].replace(',', '.'))
            YAG = float(header['YAG%d' % c].replace(',', '.'))
            YZ = float(header['YZ%d' % c].replace(',', '.'))
            ADCMAX = header['ADCMAX']
            AD = header['AD']

            name = header['YN%d' % c]
            chan_id = header['YO%d' % c]
            units = header['YU%d' % c]
            gain = AD / (YCF * YAG * (ADCMAX + 1))
            offset = -YZ * gain
            group_id = 0
            sig_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                 units, gain, offset, group_id))

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
        t_stop = self._raw_signals.shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        # WARNING check if id or index for signals (in the old IO it was ids
        # ~ raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        if channel_indexes is None:
            channel_indexes = np.arange(self.header['signal_channels'].size)

        l = self.header['signal_channels']['id'].tolist()
        channel_ids = [l.index(c) for c in channel_indexes]
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_ids]

        return raw_signals
