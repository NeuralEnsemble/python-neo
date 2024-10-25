"""
Class for reading data from a 3-brain Biocam system.

See:
https://www.3brain.com/products/single-well/biocam-x

Authors: Alessio Buccino, Robert Wolff
"""

import json
import numpy as np

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from neo.core import NeoReadWriteError


class BiocamRawIO(BaseRawIO):
    """
    Class for reading data from a Biocam h5 file.

    Parameters
    ----------
    filename: str, default: ''
        The *.h5 file to be read

    Examples
    --------
        >>> import neo.rawio
        >>> r = neo.rawio.BiocamRawIO(filename='biocam.h5')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0,
                                                 seg_index=0,
                                                 i_start=0,
                                                 i_stop=1024,
                                                 channel_names=channel_names)
        >>> float_chunk = r.rescale_signal_raw_to_float(raw_chunk,
                                                        dtype='float64',
                                                        channel_indexes=[0, 3, 6])
    """

    extensions = ["h5", "brw"]
    rawmode = "one-file"

    def __init__(self, filename=""):
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

        # buffer concept cannot be used in this reader because of too complicated dtype across version
        signal_buffers = np.array([], dtype=_signal_stream_dtype)
        signal_streams = np.array([("Signals", "0", "")], dtype=_signal_stream_dtype)

        sig_channels = []
        for c, chan in enumerate(self._channels):
            ch_name = f"ch{chan[0]}-{chan[1]}"
            chan_id = str(c + 1)
            sr = self._sampling_rate  # Hz
            dtype = "uint16"
            units = "uV"
            gain = gain
            offset = offset
            stream_id = "0"
            buffer_id = ""
            sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id, buffer_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = sig_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        all_starts = [[0.0]]
        return all_starts[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._num_frames / self._sampling_rate
        all_stops = [[t_stop]]
        return all_stops[block_index][seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("`stream_index` must be 0")
        return self._num_frames

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("`stream_index must be 0")
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._num_frames

        # read functions are different based on the version of biocam
        data = self._read_function(self._filehandle, i_start, i_stop, self._num_channels)

        # older style data returns array of (n_samples, n_channels), should be a view
        # but if memory issues come up we should doublecheck out how the file is being stored
        if data.ndim > 1:
            if channel_indexes is None:
                channel_indexes = slice(None)
            sig_chunk = data[:, channel_indexes]

        # newer style data returns an initial flat array (n_samples * n_channels)
        # we iterate through channels rather than slicing
        # Due to the fact that Neo and SpikeInterface tend to prefer slices we need to add
        # some careful checks around slicing of None in the case we need to iterate through
        # channels. First check if None. Then check if slice and only if slice check that it is slice(None)
        else:
            if channel_indexes is None:
                channel_indexes = [ch for ch in range(self._num_channels)]
            elif isinstance(channel_indexes, slice):
                start = channel_indexes.start or 0
                stop = channel_indexes.stop or self._num_channels
                step = channel_indexes.step or 1
                channel_indexes = [ch for ch in range(start, stop, step)]

            sig_chunk = np.zeros((i_stop - i_start, len(channel_indexes)), dtype=data.dtype)
            # iterate through channels to prevent loading all channels into memory which can cause
            # memory exhaustion. See https://github.com/SpikeInterface/spikeinterface/issues/3303
            for index, channel_index in enumerate(channel_indexes):
                sig_chunk[:, index] = data[channel_index :: self._num_channels]

        return sig_chunk


def open_biocam_file_header(filename) -> dict:
    """Open a Biocam hdf5 file, read and return the recording info, pick the correct method to access raw data,
    and return this to the caller

    Parameters
    ----------
    filename: str
        The file to be parsed

    Returns
    -------
    dict
        The information necessary to read a biocam file (gain, n_samples, n_channels, etc)."""
    import h5py

    rf = h5py.File(filename, "r")

    if "3BRecInfo" in rf.keys():  # brw v3.x
        # Read recording variables
        rec_vars = rf.require_group("3BRecInfo/3BRecVars/")
        bit_depth = rec_vars["BitDepth"][0]
        max_uv = rec_vars["MaxVolt"][0]
        min_uv = rec_vars["MinVolt"][0]
        num_frames = rec_vars["NRecFrames"][0]
        sampling_rate = rec_vars["SamplingRate"][0]
        signal_inv = rec_vars["SignalInversion"][0]

        # Get the actual number of channels used in the recording
        file_format = rf["3BData"].attrs.get("Version", None)
        format_100 = False
        if file_format == 100:
            num_channels = len(rf["3BData/Raw"][0])
            format_100 = True
        elif file_format in (101, 102) or file_format is None:
            num_channels = int(rf["3BData/Raw"].shape[0] / num_frames)
        else:
            raise NeoReadWriteError("Unknown data file format.")

        # get channels
        channels = rf["3BRecInfo/3BMeaStreams/Raw/Chs"][:]

        # determine correct function to read data
        if format_100:
            if signal_inv == 1:
                read_function = readHDF5t_100
            elif signal_inv == -1:
                read_function = readHDF5t_100_i
            else:
                raise NeoReadWriteError("Unknown signal inversion")
        else:
            if signal_inv == 1:
                read_function = readHDF5t_101
            elif signal_inv == -1:
                read_function = readHDF5t_101_i
            else:
                raise NeoReadWriteError("Unknown signal inversion")

        gain = (max_uv - min_uv) / (2**bit_depth)
        offset = min_uv

        return dict(
            file_handle=rf,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            channels=channels,
            file_format=file_format,
            signal_inv=signal_inv,
            read_function=read_function,
            gain=gain,
            offset=offset,
        )
    else:  # brw v4.x
        # Read recording variables
        experiment_settings = json.JSONDecoder().decode(rf["ExperimentSettings"][0].decode())
        max_uv = experiment_settings["ValueConverter"]["MaxAnalogValue"]
        min_uv = experiment_settings["ValueConverter"]["MinAnalogValue"]
        max_digital = experiment_settings["ValueConverter"]["MaxDigitalValue"]
        min_digital = experiment_settings["ValueConverter"]["MinDigitalValue"]
        scale_factor = experiment_settings["ValueConverter"]["ScaleFactor"]
        sampling_rate = experiment_settings["TimeConverter"]["FrameRate"]

        num_channels = None
        for key in rf:
            if key[:5] == "Well_":
                num_channels = len(rf[key]["StoredChIdxs"])
                if len(rf[key]["Raw"]) % num_channels:
                    raise NeoReadWriteError(f"Length of raw data array is not multiple of channel number in {key}")
                num_frames = len(rf[key]["Raw"]) // num_channels
                break

        if num_channels is not None:
            num_channels_x = num_channels_y = int(np.sqrt(num_channels))
        else:
            raise NeoReadWriteError("No Well found in the file")

        if num_channels_x * num_channels_y != num_channels:
            raise NeoReadWriteError(f"Cannot determine structure of the MEA plate with {num_channels} channels")
        channels = 1 + np.concatenate(np.transpose(np.meshgrid(range(num_channels_x), range(num_channels_y))))

        gain = scale_factor * (max_uv - min_uv) / (max_digital - min_digital)
        offset = min_uv
        read_function = readHDF5t_brw4

        return dict(
            file_handle=rf,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            channels=channels,
            read_function=read_function,
            gain=gain,
            offset=offset,
        )


######################################################################
# Helper functions to obtain the raw data split by Biocam version.


# return the full array for the old datasets
def readHDF5t_100(rf, t0, t1, nch):
    return rf["3BData/Raw"][t0:t1]


def readHDF5t_100_i(rf, t0, t1, nch):
    return 4096 - rf["3BData/Raw"][t0:t1]


# return flat array that we will iterate through
def readHDF5t_101(rf, t0, t1, nch):
    return rf["3BData/Raw"][nch * t0 : nch * t1]


def readHDF5t_101_i(rf, t0, t1, nch):
    return 4096 - rf["3BData/Raw"][nch * t0 : nch * t1]


def readHDF5t_brw4(rf, t0, t1, nch):
    for key in rf:
        if key[:5] == "Well_":
            return rf[key]["Raw"][nch * t0 : nch * t1]
