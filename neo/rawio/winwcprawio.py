"""
Class for reading data from WinWCP, a software tool written by
John Dempster.

WinWCP is free:
http://spider.science.strath.ac.uk/sipbs/software.htm

Author: Samuel Garcia
"""
from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype, _common_sig_characteristics)

import numpy as np

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

        # only one memmap for all segment to avoid
        # "error: [Errno 24] Too many open files"
        self._memmap = np.memmap(self.filename, dtype='uint8', mode='r')

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
                offset = 1024 + seg_index * (SECTORSIZE * header['NBD'] + 1024)

                # read analysis zone
                analysisHeader = HeaderReader(fid, AnalysisDescription).read_f(offset=offset)

                # read data
                NP = (SECTORSIZE * header['NBD']) // 2
                NP = NP - NP % header['NC']
                NP = NP // header['NC']
                NC = header['NC']
                ind0 = offset + header['NBA'] * SECTORSIZE
                ind1 = ind0 + NP * NC * 2
                sigs = self._memmap[ind0:ind1].view('int16').reshape(NP, NC)
                self._raw_signals[seg_index] = sigs

                all_sampling_interval.append(analysisHeader['SamplingInterval'])

        # sampling interval can be slightly varying due to float precision
        # all_sampling_interval are not always unique
        self._sampling_rate = 1. / np.median(all_sampling_interval)

        signal_channels = []
        for c in range(header['NC']):
            YG = float(header['YG%d' % c].replace(',', '.'))
            ADCMAX = header['ADCMAX']
            VMax = analysisHeader['VMax'][c]

            name = header['YN%d' % c]
            chan_id = header['YO%d' % c]
            units = header['YU%d' % c]
            gain = VMax / ADCMAX / YG
            offset = 0.
            stream_id = '0'
            signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
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
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals[seg_index].shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self._raw_signals[seg_index].shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']
        global_channel_indexes, = np.nonzero(self.header['signal_channels']
                                    ['stream_id'] == stream_id)
        if channel_indexes is None:
            channel_indexes = slice(None)
        inds = global_channel_indexes[channel_indexes]
        raw_signals = self._raw_signals[seg_index][slice(i_start, i_stop), inds]
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
