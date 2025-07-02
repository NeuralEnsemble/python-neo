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

import numpy as np
import json
import warnings
from neo.core import NeoReadWriteError


class BiocamRawIO(BaseRawIO):
    """
    Class for reading data from a Biocam h5 file.

    Parameters
    ----------
    filename: str, default: ''
        The *.h5 file to be read
    fill_gaps_strategy: "zeros" | "synthetic_noise" | None, default: None
        The strategy to fill the gaps in the data when using event-based
        compression. If None and the file is event-based compressed,
        you need to specify a fill gaps strategy:

        * "zeros": the gaps are filled with unsigned 0s (2048). This value is the "0" of the unsigned 12 bits
                   representation of the data.
        * "synthetic_noise": the gaps are filled with synthetic noise.

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

    def __init__(self, filename="", fill_gaps_strategy="zeros"):
        BaseRawIO.__init__(self)
        self.filename = filename
        self._fill_gaps_strategy = fill_gaps_strategy

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
        if self._read_function is readHDF5t_brw4_sparse:
            if self._fill_gaps_strategy is None:
                raise ValueError("Please set `fill_gaps_strategy` to 'zeros' or 'synthetic_noise'.")
            if self._fill_gaps_strategy == "synthetic_noise":
                warnings.warn(
                    "Event-based compression : gaps will be filled with synthetic noise. "
                    "Set `fill_gaps_strategy` to 'zeros' to fill gaps with 0s."
                )
                use_synthetic_noise = True
            elif self._fill_gaps_strategy == "zeros":
                use_synthetic_noise = False
            else:
                raise ValueError("`fill_gaps_strategy` must be 'zeros' or 'synthetic_noise'")

            data = self._read_function(
                self._filehandle, i_start, i_stop, self._num_channels, use_synthetic_noise=use_synthetic_noise
            )
        else:
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
        num_frames = rf["TOC"][-1, -1]

        num_channels = None
        well_ID = None
        for well_ID in rf:
            if well_ID.startswith("Well_"):
                num_channels = len(rf[well_ID]["StoredChIdxs"])
                if "Raw" in rf[well_ID]:
                    if len(rf[well_ID]["Raw"]) % num_channels:
                        raise NeoReadWriteError(
                            f"Length of raw data array is not multiple of channel number in {well_ID}"
                        )
                    num_frames = len(rf[well_ID]["Raw"]) // num_channels
                    break
                elif "EventsBasedSparseRaw" in rf[well_ID]:
                    # Not sure how to check for this with sparse data
                    pass

        if num_channels is not None:
            num_channels_x = num_channels_y = int(np.sqrt(num_channels))
        else:
            raise NeoReadWriteError("No Well found in the file")

        if num_channels_x * num_channels_y != num_channels:
            raise NeoReadWriteError(f"Cannot determine structure of the MEA plate with {num_channels} channels")
        channels = 1 + np.concatenate(np.transpose(np.meshgrid(range(num_channels_x), range(num_channels_y))))

        gain = scale_factor * (max_uv - min_uv) / (max_digital - min_digital)
        offset = min_uv
        if "Raw" in rf[well_ID]:
            read_function = readHDF5t_brw4
        elif "EventsBasedSparseRaw" in rf[well_ID]:
            read_function = readHDF5t_brw4_sparse

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
        if key.startswith("Well_"):
            return rf[key]["Raw"][nch * t0 : nch * t1]


def readHDF5t_brw4_sparse(rf, t0, t1, nch, use_synthetic_noise=False):

    # noise_std = None
    start_frame = t0
    num_frames = t1 - t0
    for well_ID in rf:
        if well_ID.startswith("Well_"):
            break
    # initialize an empty (fill with zeros) data collection
    data = np.zeros((nch, num_frames), dtype=np.uint16)
    if not use_synthetic_noise:
        # Will read as 0s after 12 bits signed conversion
        data.fill(2048)
    else:
        # fill the data collection with Gaussian noise if requested
        data = generate_synthetic_noise(rf, data, well_ID, start_frame, num_frames)  # , std=noise_std)
    # fill the data collection with the decoded event based sparse raw data
    data = decode_event_based_raw_data(rf, data, well_ID, start_frame, num_frames)

    return data.T


def decode_event_based_raw_data(rf, data, well_ID, start_frame, num_frames):
    # Source: Documentation by 3Brain
    # https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/src/master/biocam/documentation_brw_4.x_bxr_3.x_bcmp_1.x_in_brainwave_5.x_v1.1.3.pdf
    # collect the TOCs
    toc = np.array(rf["TOC"])
    events_toc = np.array(rf[well_ID]["EventsBasedSparseRawTOC"])
    # from the given start position and duration in frames, localize the corresponding event positions
    # using the TOC
    toc_start_idx = np.searchsorted(toc[:, 1], start_frame)
    toc_end_idx = min(np.searchsorted(toc[:, 1], start_frame + num_frames, side="right") + 1, len(toc) - 1)
    events_start_pos = events_toc[toc_start_idx]
    events_end_pos = events_toc[toc_end_idx]
    # decode all data for the given well ID and time interval
    binary_data = rf[well_ID]["EventsBasedSparseRaw"][events_start_pos:events_end_pos]
    binary_data_length = len(binary_data)
    pos = 0
    while pos < binary_data_length:
        ch_idx = int.from_bytes(binary_data[pos : pos + 4], byteorder="little")
        pos += 4
        ch_data_length = int.from_bytes(binary_data[pos : pos + 4], byteorder="little")
        pos += 4
        ch_data_pos = pos
        while pos < ch_data_pos + ch_data_length:
            from_inclusive = int.from_bytes(binary_data[pos : pos + 8], byteorder="little")
            pos += 8
            to_exclusive = int.from_bytes(binary_data[pos : pos + 8], byteorder="little")
            pos += 8
            range_data_pos = pos
            for j in range(from_inclusive, to_exclusive):
                if j >= start_frame + num_frames:
                    break
                if j >= start_frame:
                    data[ch_idx][j - start_frame] = int.from_bytes(
                        binary_data[range_data_pos : range_data_pos + 2], byteorder="little"
                    )
                range_data_pos += 2
            pos += (to_exclusive - from_inclusive) * 2

    return data


def generate_synthetic_noise(rf, data, well_ID, start_frame, num_frames):
    # Source: Documentation by 3Brain
    # https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/src/master/biocam/documentation_brw_4.x_bxr_3.x_bcmp_1.x_in_brainwave_5.x_v1.1.3.pdf
    # collect the TOCs
    toc = np.array(rf["TOC"])
    noise_toc = np.array(rf[well_ID]["NoiseTOC"])
    # from the given start position in frames, localize the corresponding noise positions
    # using the TOC
    toc_start_idx = np.searchsorted(toc[:, 1], start_frame)
    noise_start_pos = noise_toc[toc_start_idx]
    noise_end_pos = noise_start_pos
    for i in range(toc_start_idx + 1, len(noise_toc)):
        next_pos = noise_toc[i]
        if next_pos > noise_start_pos:
            noise_end_pos = next_pos
        break
    if noise_end_pos == noise_start_pos:
        for i in range(toc_start_idx - 1, 0, -1):
            previous_pos = noise_toc[i]
            if previous_pos < noise_start_pos:
                noise_end_pos = noise_start_pos
                noise_start_pos = previous_pos
                break
    # obtain the noise info at the start position
    noise_ch_idx = rf[well_ID]["NoiseChIdxs"][noise_start_pos:noise_end_pos]
    noise_mean = rf[well_ID]["NoiseMean"][noise_start_pos:noise_end_pos]
    noise_std = rf[well_ID]["NoiseStdDev"][noise_start_pos:noise_end_pos]

    noise_length = noise_end_pos - noise_start_pos
    noise_info = {}
    mean_collection = []
    std_collection = []
    for i in range(1, noise_length):
        noise_info[noise_ch_idx[i]] = [noise_mean[i], noise_std[i]]
        mean_collection.append(noise_mean[i])
        std_collection.append(noise_std[i])
    # calculate the median mean and standard deviation of all channels to be used for
    # invalid channels
    median_mean = np.median(mean_collection)
    median_std = np.median(std_collection)
    # fill with Gaussian noise
    for ch_idx in range(len(data)):
        if ch_idx in noise_info:
            data[ch_idx] = np.array(
                np.random.normal(noise_info[ch_idx][0], noise_info[ch_idx][1], num_frames), dtype=np.uint16
            )
        else:
            data[ch_idx] = np.array(np.random.normal(median_mean, median_std, num_frames), dtype=np.uint16)

    return data
