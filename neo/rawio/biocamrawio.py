"""
Class for reading data from a 3-brain Biocam system.

See:
https://www.3brain.com/products/single-well/biocam-x

Author : Alessio Buccino
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)

import numpy as np
from copy import deepcopy


class BiocamRawIO(BaseRawIO):
    """
    Class for reading data from a Biocam h5 file.

    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.BiocamRawIO(filename='biocam.h5')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
    """
    extensions = ['h5']
    rawmode = 'one-file'

    def __init__(self, filename='', mea_pitch=42, verbose=False):
        BaseRawIO.__init__(self)
        self.filename = filename
        self._mea_pitch = mea_pitch
        self._verbose = verbose

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        try:
            import h5py
            HAVE_H5PY = True
        except ImportError:
            HAVE_H5PY = False
        assert HAVE_H5PY, 'h5py is not installed'
        self._rf, self._num_frames, self._sampling_rate, self._num_chanels, self._ch_indices, \
        self._file_format, self._signal_inv, self._positions, self._read_function = openBiocamFile(
            self.filename, self._mea_pitch, self._verbose)

        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)

        sig_channels = []
        for c in range(self._num_chanels):
            ch_name = 'ch{}'.format(c)
            chan_id = str(c + 1)
            sr = self._sampling_rate  # Hz
            dtype = "uint16"
            units = 'uV'
            gain = 1.  # TODO find gain
            offset = 0.  # TODO find offset
            stream_id = '0'
            sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_spike_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
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

        data = self._read_function(self._rf, i_start, i_stop, self._num_chanels)
        # transform to slice if possible
        if sorted(channel_indexes) == channel_indexes and np.all(np.diff(channel_indexes) == 1):
            channel_ids = slice(channel_indexes[0], channel_indexes[0] + len(channel_indexes))
        return data[:, channel_indexes]


def openBiocamFile(filename, mea_pitch, verbose=False):
    """Open a Biocam hdf5 file, read and return the recording info, pick te correct method to access raw data,
    and return this to the caller."""
    try:
        import h5py
        HAVE_H5PY = True
    except ImportError:
        HAVE_H5PY = False
    assert HAVE_H5PY, 'h5py is not installed'

    rf = h5py.File(filename, 'r')
    # Read recording variables
    recVars = rf.require_group('3BRecInfo/3BRecVars/')
    # bitDepth = recVars['BitDepth'].value[0]
    # maxV = recVars['MaxVolt'].value[0]
    # minV = recVars['MinVolt'].value[0]
    nFrames = recVars['NRecFrames'][0]
    samplingRate = recVars['SamplingRate'][0]
    signalInv = recVars['SignalInversion'][0]
    # Read chip variables
    chipVars = rf.require_group('3BRecInfo/3BMeaChip/')
    nCols = chipVars['NCols'][0]
    # Get the actual number of channels used in the recording
    file_format = rf['3BData'].attrs.get('Version')
    if file_format == 100:
        nRecCh = len(rf['3BData/Raw'][0])
    elif (file_format == 101) or (file_format == 102):
        nRecCh = int(1. * rf['3BData/Raw'].shape[0] / nFrames)
    else:
        raise Exception('Unknown data file format.')

    if verbose:
        print('# 3Brain data format:', file_format, 'signal inversion', signalInv)
        print('#       signal range: ', recVars['MinVolt'][0], '- ', recVars['MaxVolt'][0])
        print('# channels: ', nRecCh)
        print('# frames: ', nFrames)
        print('# sampling rate: ', samplingRate)
    # get channel locations
    r = (rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Row'] - 1) * mea_pitch
    c = (rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Col'] - 1) * mea_pitch
    rawIndices = np.vstack((r, c)).T
    # assign channel numbers
    chIndices = np.array([(x - 1) + (y - 1) * nCols for (y, x) in rawIndices])
    # determine correct function to read data
    if verbose:
        print("# Signal inversion is " + str(signalInv) + ".")
        print("# If your spike sorting results look wrong, invert the signal.")
    if (file_format == 100) & (signalInv == 1):
        read_function = readHDF5t_100
    elif (file_format == 100) & (signalInv == -1):
        read_function = readHDF5t_100_i
    if ((file_format == 101) | (file_format == 102)) & (signalInv == 1):
        read_function = readHDF5t_101
    elif ((file_format == 101) | (file_format == 102)) & (signalInv == -1):
        read_function = readHDF5t_101_i
    else:
        raise RuntimeError("File format unknown.")
    return rf, nFrames, samplingRate, nRecCh, chIndices, file_format, signalInv, rawIndices, read_function


def readHDF5t_100(rf, t0, t1, nch):
    if t0 <= t1:
        return rf['3BData/Raw'][t0:t1]
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')


def readHDF5t_100_i(rf, t0, t1, nch):
    if t0 <= t1:
        return 4096 - rf['3BData/Raw'][t0:t1]
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')


def readHDF5t_101(rf, t0, t1, nch):
    if t0 <= t1:
        return rf['3BData/Raw'][nch * t0:nch * t1].reshape((t1 - t0, nch), order='C')
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')


def readHDF5t_101_i(rf, t0, t1, nch):
    if t0 <= t1:
        return 4096 - rf['3BData/Raw'][nch * t0:nch * t1].reshape((t1 - t0, nch), order='C')
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
