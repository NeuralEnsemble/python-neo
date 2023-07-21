"""
Class for reading data from a 3-brain Biocam system.

See:
https://www.3brain.com/products/single-well/biocam-x

Author : Alessio Buccino
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)

import numpy as np



class BiocamRawIO(BaseRawIO):
    """
    Class for reading data from a Biocam h5 file.

    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.BiocamRawIO(filename='biocam.h5')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                                                 i_start=0, i_stop=1024,
                                                 channel_names=channel_names)
        >>> float_chunk = r.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                                                        channel_indexes=[0, 3, 6])
    """
    extensions = ['h5', 'brw']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        self._header_dict = open_biocam_file_header(self.filename)
        self._num_channels = self._header_dict["num_channels"]
        self._num_frames = self._header_dict["num_frames"]
        self._sampling_rate = self._header_dict["sampling_rate"]
        self._filehandle = self._header_dict["file_handle"]
        self._read_function = self._header_dict["read_function"]
        self._channels = self._header_dict["channels"]
        gain = self._header_dict["gain"]
        offset = self._header_dict["offset"]

        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)

        sig_channels = []
        for c, chan in enumerate(self._channels):
            ch_name = f'ch{chan[0]}-{chan[1]}'
            chan_id = str(c + 1)
            sr = self._sampling_rate  # Hz
            dtype = "uint16"
            units = 'uV'
            gain = gain
            offset = offset
            stream_id = '0'
            sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = sig_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        all_starts = [[0.]]
        return all_starts[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._num_frames / self._sampling_rate
        all_stops = [[t_stop]]
        return all_stops[block_index][seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        assert stream_index == 0
        return self._num_frames

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        assert stream_index == 0
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._num_frames
        if channel_indexes is None:
            channel_indexes = slice(None)
        data = self._read_function(self._filehandle, i_start, i_stop, self._num_channels)
        return data[:, channel_indexes]


def open_biocam_file_header(filename):
    """Open a Biocam hdf5 file, read and return the recording info, pick the correct method to access raw data,
    and return this to the caller."""
    import h5py

    rf = h5py.File(filename, 'r')
    # Read recording variables
    rec_vars = rf.require_group('3BRecInfo/3BRecVars/')
    bit_depth = rec_vars['BitDepth'][0]
    max_uv = rec_vars['MaxVolt'][0]
    min_uv = rec_vars['MinVolt'][0]
    n_frames = rec_vars['NRecFrames'][0]
    sampling_rate = rec_vars['SamplingRate'][0]
    signal_inv = rec_vars['SignalInversion'][0]

    # Get the actual number of channels used in the recording
    file_format = rf['3BData'].attrs.get('Version', None)
    format_100 = False
    if file_format == 100:
        n_channels = len(rf['3BData/Raw'][0])
        format_100 = True
    elif file_format in (101, 102) or file_format is None:
        n_channels = int(rf['3BData/Raw'].shape[0] / n_frames)
    else:
        raise Exception('Unknown data file format.')

    # # get channels
    channels = rf['3BRecInfo/3BMeaStreams/Raw/Chs'][:]

    # determine correct function to read data
    if format_100:
        if signal_inv == 1:
            read_function = readHDF5t_100
        elif signal_inv == 1:
            read_function = readHDF5t_100_i
        else:
            raise Exception("Unknown signal inversion")
    else:
        if signal_inv == 1:
            read_function = readHDF5t_101
        elif signal_inv == 1:
            read_function = readHDF5t_101_i
        else:
            raise Exception("Unknown signal inversion")

    gain = (max_uv - min_uv) / (2 ** bit_depth)
    offset = min_uv

    return dict(file_handle=rf, num_frames=n_frames, sampling_rate=sampling_rate, num_channels=n_channels,
                channels=channels, file_format=file_format, signal_inv=signal_inv,
                read_function=read_function, gain=gain, offset=offset)


def readHDF5t_100(rf, t0, t1, nch):
    return rf['3BData/Raw'][t0:t1]


def readHDF5t_100_i(rf, t0, t1, nch):
    return 4096 - rf['3BData/Raw'][t0:t1]


def readHDF5t_101(rf, t0, t1, nch):
    return rf['3BData/Raw'][nch * t0:nch * t1].reshape((t1 - t0, nch), order='C')


def readHDF5t_101_i(rf, t0, t1, nch):
    return 4096 - rf['3BData/Raw'][nch * t0:nch * t1].reshape((t1 - t0, nch), order='C')
