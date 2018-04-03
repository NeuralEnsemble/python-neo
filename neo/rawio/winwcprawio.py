# -*- coding: utf-8 -*-
"""
Class for reading data from WinWCP, a software tool written by
John Dempster.

WinWCP is free:
http://spider.science.strath.ac.uk/sipbs/software.htm

Author : sgarcia
Author: Samuel Garcia
"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

import os
import sys
import struct


class WinWcpRawIO(BaseRawIO):
    extensions = ['wcp']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        SECTORSIZE = 512

        with open(self.filename, 'rb') as fid:

            headertext = fid.read(1024)
            headertext = headertext.decode('ascii')
            header = {}
            for line in headertext.split('\r\n'):
                if '=' not in line:
                    continue
                # print '#' , line , '#'
                key, val = line.split('=')
                if key in ['NC', 'NR', 'NBH', 'NBA', 'NBD', 'ADCMAX', 'NP', 'NZ', ]:
                    val = int(val)
                elif key in ['AD', 'DT', ]:
                    val = val.replace(',', '.')
                    val = float(val)
                header[key] = val

            nb_segment = header['NR']
            self._raw_signals = {}
            all_sampling_interval = []
            # loop for record number
            for seg_index in range(header['NR']):
                # print 'record ',i
                offset = 1024 + seg_index * (SECTORSIZE * header['NBD'] + 1024)

                # read analysis zone
                analysisHeader = HeaderReader(fid, AnalysisDescription).read_f(offset=offset)

                # read data
                NP = (SECTORSIZE * header['NBD']) // 2
                NP = NP - NP % header['NC']
                NP = NP // header['NC']

                self._raw_signals[seg_index] = np.memmap(self.filename, dtype='int16', mode='r',
                                                         shape=(NP, header['NC'],),
                                                         offset=offset + header['NBA'] * SECTORSIZE)

                all_sampling_interval.append(analysisHeader['SamplingInterval'])

        assert np.unique(all_sampling_interval).size == 1

        self._sampling_rate = 1. / all_sampling_interval[0]

        sig_channels = []
        for c in range(header['NC']):
            YG = float(header['YG%d' % c].replace(',', '.'))
            ADCMAX = header['ADCMAX']
            VMax = analysisHeader['VMax'][c]

            name = header['YN%d' % c]
            chan_id = header['YO%d' % c]
            units = header['YU%d' % c]
            gain = VMax / ADCMAX / YG
            offset = 0.
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
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals[seg_index].shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._raw_signals[seg_index].shape[0]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        # WARNING check if id or index for signals (in the old IO it was ids
        # ~ raw_signals = self._raw_signals[seg_index][slice(i_start, i_stop), channel_indexes]
        if channel_indexes is None:
            channel_indexes = np.arange(self.header['signal_channels'].size)

        l = self.header['signal_channels']['id'].tolist()
        channel_ids = [l.index(c) for c in channel_indexes]
        raw_signals = self._raw_signals[seg_index][slice(i_start, i_stop), channel_ids]
        return raw_signals


AnalysisDescription = [
    ('RecordStatus', '8s'),
    ('RecordType', '4s'),
    ('GroupNumber', 'f'),
    ('TimeRecorded', 'f'),
    ('SamplingInterval', 'f'),
    ('VMax', '8f'),
]


class HeaderReader():
    def __init__(self, fid, description):
        self.fid = fid
        self.description = description

    def read_f(self, offset=0):
        self.fid.seek(offset)
        d = {}
        for key, fmt in self.description:
            val = struct.unpack(fmt, self.fid.read(struct.calcsize(fmt)))
            if len(val) == 1:
                val = val[0]
            else:
                val = list(val)
            d[key] = val
        return d
