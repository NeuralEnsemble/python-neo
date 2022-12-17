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
    from pyedflib import EdfReader

    HAS_PYEDF = True
except ImportError:
    HAS_PYEDF = False


class EDFRawIO(BaseRawIO):
    """
    Class for reading European Data Format files (EDF and EDF+).
    Currently only continuous EDF+ files (EDF+C) and original EDF files (EDF) are supported

    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.EDFRawIO(filename='file.edf')
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

        self.edf_reader = EdfReader(self.filename)
        # load headers, signal information and
        self.edf_header = self.edf_reader.getHeader()
        self.signal_headers = self.edf_reader.getSignalHeaders()

        # add annotations to header
        annotations = self.edf_reader.readAnnotations()
        self.signal_annotations = [[s, d, a] for s, d, a in zip(*annotations)]

        # 1 stream = 1 sampling rate
        stream_characteristics = []
        self.stream_idx_to_chidx = {}

        signal_channels = []
        for ch_idx, sig_dict in enumerate(self.signal_headers):
            ch_name = sig_dict['label']
            chan_id = ch_idx
            sr = sig_dict['sample_rate']  # Hz
            dtype = 'int16'  # assume general int16 based on edf documentation
            units = sig_dict['dimension']
            physical_range = sig_dict['physical_max'] - sig_dict['physical_min']
            # number of digital steps resolved (+1 to account for '0')
            digital_range = sig_dict['digital_max'] - sig_dict['digital_min'] + 1
            gain = physical_range / digital_range
            offset = -1 * sig_dict['digital_min'] * gain + sig_dict['physical_min']

            # identify corresponding stream based on sampling rate
            if (sr,) not in stream_characteristics:
                stream_characteristics += [(sr,)]

            stream_id = stream_characteristics.index((sr,))
            self.stream_idx_to_chidx.setdefault(stream_id, []).append(ch_idx)

            signal_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))

        # convert channel index lists to arrays for indexing
        self.stream_idx_to_chidx = {k: np.array(v) for k, v in self.stream_idx_to_chidx.items()}

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        signal_streams = [(f'stream ({sr} Hz)', i) for i, sr in enumerate(stream_characteristics)]
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # no unit/epoch information contained in edf
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels.append(('Event', 'event_channel', 'event'))
        event_channels.append(('Epoch', 'epoch_channel', 'epoch'))
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

    def _get_stream_channels(self, stream_index):
        return self.header['signal_channels'][self.stream_idx_to_chidx[stream_index]]

    def _segment_t_start(self, block_index, seg_index):
        # no time offset provided by EDF format
        return 0.  # in seconds

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self.edf_reader.datarecord_duration * self.edf_reader.datarecords_in_file
        # this must return an float scale in second
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        chidx = self.stream_idx_to_chidx[stream_index][0]
        # use sample count of first signal in stream
        return self.edf_reader.getNSamples()[chidx]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.  # EDF does not provide temporal offset information

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        # only dealing with single block and segment edf files
        assert (block_index, seg_index) == (0, 0)

        stream_channel_idxs = self.stream_idx_to_chidx[stream_index]

        # keep all channels of the stream if none are selected
        if channel_indexes is None:
            channel_indexes = slice(None)

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.get_signal_size(block_index=block_index, seg_index=seg_index,
                                          stream_index=stream_index)
        n = i_stop - i_start

        # raw_signals = self.edf_reader. am[channel_indexes, i_start:i_stop]
        selected_channel_idxs = stream_channel_idxs[channel_indexes]

        # load data into numpy array buffer
        data = []
        for i, channel_idx in enumerate(selected_channel_idxs):
            # use int32 for compatibility with pyedflib
            buffer = np.empty(n, dtype=np.int32)
            self.edf_reader.read_digital_signal(channel_idx, i_start, n, buffer)
            data.append(buffer)

        # downgrade to int16 as this is what is used in the edf file format
        # use fortran (column major) order to be more efficient after transposing
        data = np.asarray(data, dtype=np.int16, order='F')

        # use dimensions (time, channel)
        data = data.T

        return data

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
        return len(self.edf_reader.readAnnotations()[0])

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # these time should be already in seconds
        timestamps, durations, labels = self.edf_reader.readAnnotations()
        if t_start is None:
            t_start = self.segment_t_start(block_index, seg_index)
        if t_stop is None:
            t_stop = self.segment_t_stop(block_index, seg_index)

        # only consider events and epochs that overlap with t_start t_stop range
        time_mask = ((t_start < timestamps) & (timestamps < t_stop)) | \
                    ((t_start < (timestamps + durations)) & ((timestamps + durations) < t_stop))

        # separate event from epoch times
        event_mask = durations[time_mask] == 0
        if self.header['event_channels']['type'][event_channel_index] == b'epoch':
            event_mask = ~event_mask
            durations = durations[time_mask][event_mask]
        elif self.header['event_channels']['type'][event_channel_index] == b'event':
            durations = None

        times = timestamps[time_mask][event_mask]
        labels = np.asarray(labels[time_mask][event_mask], dtype='U')

        return times, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        return np.asarray(event_timestamps, dtype=dtype)

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return np.asarray(raw_duration, dtype=dtype)

    def __enter__(self):
        return self

    def __del__(self):
        self._close_reader()

    def __exit__(self, exc_type, exc_val, ex_tb):
        self._close_reader()

    def close(self):
        """
        Closes the file handler
        """
        self._close_reader()

    def _close_reader(self):
        if hasattr(self, 'edf_reader'):
            self.edf_reader.close()
