# -*- coding: utf-8 -*-
"""
Class for reading data from Elan.

Elan is software for studying time-frequency maps of EEG data.

Elan is developed in Lyon, France, at INSERM U821

https://elan.lyon.inserm.fr

An Elan dataset is separated into 3 files :
 - .eeg          raw data file
 - .eeg.ent      hearder file
 - .eeg.pos      event file

Author: Samuel Garcia

"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

import datetime
import os
import re
import io


class ElanRawIO(BaseRawIO):
    extensions = ['eeg']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _parse_header(self):

        with io.open(self.filename + '.ent', mode='rt', encoding='ascii', newline=None) as f:

            # version
            version = f.readline()[:-1]
            assert version in ['V2', 'V3'], 'Read only V2 or V3 .eeg.ent files. %s given' % version

            # info
            info1 = f.readline()[:-1]
            info2 = f.readline()[:-1]

            # strange 2 line for datetime
            # line1
            l = f.readline()
            r1 = re.findall('(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)', l)
            r2 = re.findall('(\d+):(\d+):(\d+)', l)
            r3 = re.findall('(\d+)-(\d+)-(\d+)', l)
            YY, MM, DD, hh, mm, ss = (None,) * 6
            if len(r1) != 0:
                DD, MM, YY, hh, mm, ss = r1[0]
            elif len(r2) != 0:
                hh, mm, ss = r2[0]
            elif len(r3) != 0:
                DD, MM, YY = r3[0]

            # line2
            l = f.readline()
            r1 = re.findall('(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)', l)
            r2 = re.findall('(\d+):(\d+):(\d+)', l)
            r3 = re.findall('(\d+)-(\d+)-(\d+)', l)
            if len(r1) != 0:
                DD, MM, YY, hh, mm, ss = r1[0]
            elif len(r2) != 0:
                hh, mm, ss = r2[0]
            elif len(r3) != 0:
                DD, MM, YY = r3[0]
            try:
                fulldatetime = datetime.datetime(int(YY), int(MM), int(DD),
                                                 int(hh), int(mm), int(ss))
            except:
                fulldatetime = None

            l = f.readline()
            l = f.readline()
            l = f.readline()

            # sampling rate sample
            l = f.readline()
            self._sampling_rate = 1. / float(l)

            # nb channel
            l = f.readline()
            nb_channel = int(l) - 2

            channel_infos = [{} for c in range(nb_channel + 2)]
            # channel label
            for c in range(nb_channel + 2):
                channel_infos[c]['label'] = f.readline()[:-1]
            # channel kind
            for c in range(nb_channel + 2):
                channel_infos[c]['kind'] = f.readline()[:-1]
            # channel unit
            for c in range(nb_channel + 2):
                channel_infos[c]['units'] = f.readline()[:-1]
            # range for gain and offset
            for c in range(nb_channel + 2):
                channel_infos[c]['min_physic'] = float(f.readline()[:-1])
            for c in range(nb_channel + 2):
                channel_infos[c]['max_physic'] = float(f.readline()[:-1])
            for c in range(nb_channel + 2):
                channel_infos[c]['min_logic'] = float(f.readline()[:-1])
            for c in range(nb_channel + 2):
                channel_infos[c]['max_logic'] = float(f.readline()[:-1])

            # info filter
            info_filter = []
            for c in range(nb_channel + 2):
                channel_infos[c]['info_filter'] = f.readline()[:-1]

        n = int(round(np.log(channel_infos[0]['max_logic'] -
                             channel_infos[0]['min_logic']) / np.log(2)) / 8)
        sig_dtype = np.dtype('>i' + str(n))

        sig_channels = []
        for c, chan_info in enumerate(channel_infos[:-2]):
            chan_name = chan_info['label']
            chan_id = c

            gain = (chan_info['max_physic'] - chan_info['min_physic']) / \
                   (chan_info['max_logic'] - chan_info['min_logic'])
            offset = - chan_info['min_logic'] * gain + chan_info['min_physic']
            gourp_id = 0
            sig_channels.append((chan_name, chan_id, self._sampling_rate, sig_dtype,
                                 chan_info['units'], gain, offset, gourp_id))

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # raw data
        self._raw_signals = np.memmap(self.filename, dtype=sig_dtype, mode='r',
                                      offset=0).reshape(-1, nb_channel + 2)
        self._raw_signals = self._raw_signals[:, :-2]

        # triggers
        with io.open(self.filename + '.pos', mode='rt', encoding='ascii', newline=None) as f:
            self._raw_event_timestamps = []
            self._event_labels = []
            self._reject_codes = []
            for l in f.readlines():
                r = re.findall(' *(\d+) *(\d+) *(\d+) *', l)
                self._raw_event_timestamps.append(int(r[0][0]))
                self._event_labels.append(str(r[0][1]))
                self._reject_codes.append(str(r[0][2]))

        self._raw_event_timestamps = np.array(self._raw_event_timestamps, dtype='int64')
        self._event_labels = np.array(self._event_labels, dtype='U')
        self._reject_codes = np.array(self._reject_codes, dtype='U')

        event_channels = []
        event_channels.append(('Trigger', '', 'event'))
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
        extra_info = dict(rec_datetime=fulldatetime, elan_version=version,
                          info1=info1, info2=info2)
        for obj_name in ('blocks', 'segments'):
            self._raw_annotate(obj_name, **extra_info)
        for c in range(nb_channel):
            d = channel_infos[c]
            self._raw_annotate('signals', chan_index=c, info_filter=d['info_filter'])
            self._raw_annotate('signals', chan_index=c, kind=d['kind'])
        self._raw_annotate('events', chan_index=0, reject_codes=self._reject_codes)

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes=None):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        return 0

    def _event_count(self, block_index, seg_index, event_channel_index):
        return self._raw_event_timestamps.size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        timestamp = self._raw_event_timestamps
        labels = self._event_labels
        durations = None

        if t_start is not None:
            keep = timestamp >= int(t_start * self._sampling_rate)
            timestamp = timestamp[keep]
            labels = labels[keep]

        if t_stop is not None:
            keep = timestamp <= int(t_stop * self._sampling_rate)
            timestamp = timestamp[keep]
            labels = labels[keep]

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype) / self._sampling_rate
        return event_times
