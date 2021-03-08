"""
Class for reading data from WinEdr, a software tool written by
John Dempster.

WinEdr is free:
http://spider.science.strath.ac.uk/sipbs/software.htm

Author: Samuel Garcia

"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype, _common_sig_characteristics)

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

        signal_channels = []
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
            stream_id = '0'
            signal_channels.append((name, str(chan_id), self._sampling_rate, 'int16',
                                 units, gain, offset, stream_id))

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        characteristics = signal_channels[_common_sig_characteristics]
        unique_characteristics = np.unique(characteristics)
        signal_streams = []
        for i in range(unique_characteristics.size):
            mask = unique_characteristics[i] == characteristics
            signal_channels['stream_id'][mask] = str(i)
            signal_streams.append((f'stream {i}', str(i)))
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']
        global_channel_indexes, = np.nonzero(self.header['signal_channels']
                                    ['stream_id'] == stream_id)
        if channel_indexes is None:
            channel_indexes = slice(None)
        global_channel_indexes = global_channel_indexes[channel_indexes]
        raw_signals = self._raw_signals[slice(i_start, i_stop), global_channel_indexes]
        return raw_signals
