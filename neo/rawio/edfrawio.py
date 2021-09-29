"""
RawIO for reading EDF and EDF+ files using pyedflib

PyEDFLib
https://pyedflib.readthedocs.io
https://github.com/holgern/pyedflib

EDF Format Specifications: https://www.edfplus.info/

Author: Julia Sprenger
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import numpy as np

try:
    from pyedflib import highlevel
    HAS_PYEDF = True
except ImportError:
    HAS_PYEDF = False


class EDFRawIO(BaseRawIO):
    """
    Class for reading European Data Format files (EDF and EDF+).
    Currently only continuous EDF+ files (EDF+C) and original EDF files (EDF) are supported

    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.EdfRawIO(filename='file.edf')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024, stream_index=0, channel_indexes=range(10))
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
    """

    extensions = ['edf']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        if not HAS_PYEDF:
            raise ValueError('Requires pyedflib')
        BaseRawIO.__init__(self)

        # note that this filename is used in self._source_name
        self.filename = filename

        self.signals = None
        self.signal_headers = []
        self.edf_header = {}

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        # read basic header
        with open(self.filename, 'rb') as f:
            f.seek(192)
            file_version_header = f.read(44).decode('ascii')
            # only accepting basic EDF files (no 'EDF+' in header)
            # or continuous EDF+ files ('EDF+C' in header)
            if ('EDF+' in file_version_header) and ('EDF+C' not in file_version_header):
                raise ValueError('Only continuous EDF+ files are currently supported.')

        # read a edf file content using pyedflib
        self.signals, self.signal_headers, self.edf_header = highlevel.read_edf(self.filename)

        # 1 edf file = 1 stream
        signal_streams = [('edf stream', 0)]
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        signal_channels = []
        for ch_idx, sig_dict in enumerate(self.signal_headers):
            ch_name = sig_dict['label']
            chan_id = ch_idx
            sr = sig_dict['sample_rate']  # Hz
            dtype = self.signals.dtype.str
            units = sig_dict['dimension']
            physical_range = sig_dict['physical_max'] - sig_dict['physical_min']
            digital_range = sig_dict['digital_max'] - sig_dict['digital_min']
            gain = physical_range / digital_range
            offset = -1 * sig_dict['digital_min'] * gain + sig_dict['physical_min']
            stream_id = 0  # file contains only a single stream
            signal_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # no unit/epoch information contained in edf
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]  # we only accept continuous edf files
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

        # Add custom annotations for neo objects
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['name'] = 'EDF Data Block'
        bl_ann.update(self.edf_header)
        seg_ann = bl_ann['segments'][0]
        seg_ann['name'] = 'Seg #0 Block #0'

        # extract keys for array_annotations common to all signals and not already used
        ignore_annotations = ['label', 'dimension', 'sample_rate', 'physical_min', 'physical_max',
                              'digital_min', 'digital_max']
        array_keys = []
        for k in self.signal_headers[0]:
            if k not in ignore_annotations and all([k in h for h in self.signal_headers]):
                array_keys.append(k)

        for array_key in array_keys:
            array_anno = {array_key: [h[array_key] for h in self.signal_headers]}
        seg_ann['signals'].append({'__array_annotations__': array_anno})

    def _segment_t_start(self, block_index, seg_index):
        # no time offset provided by EDF format
        return 0  # in seconds

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self.signals.shape[1] / self.signal_headers[0]['sample_rate']
        # this must return an float scale in second
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self.signals.shape[1]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0  # EDF does not provide temporal offset information

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        # only dealing with single segment, single stream edf files
        assert (block_index, seg_index, stream_index) == (0, 0, 0)

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.signals.shape[1]

        # keep all channels if none are selected
        if channel_indexes is None:
            channel_indexes = slice(None)

        raw_signals = self.signals[channel_indexes, i_start:i_stop]
        return raw_signals.T

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        return None

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index, t_start,
                                 t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return None

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        return None

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        return None

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return None
