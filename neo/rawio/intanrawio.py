"""

Support for intan tech rhd and rhs files.

These 2 formats are more or less the same but:
  * some variance in headers.
  * rhs amplifier is more complex because the optional DC channel

RHS supported version 1.0
RHD supported version  1.0 1.1 1.2 1.3 2.0 3.0, 3.1
RHD headerless binary support 3.1

See:
  * http://intantech.com/files/Intan_RHD2000_data_file_formats.pdf
  * http://intantech.com/files/Intan_RHS2000_data_file_formats.pdf

Author: Samuel Garcia (Initial), Zach McKenzie & Heberto Mayorquin (Updates)

"""

from pathlib import Path
import os
from collections import OrderedDict
from packaging.version import Version as V
import warnings

import numpy as np
from neo.core import NeoReadWriteError

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
    _common_sig_characteristics,
)


class IntanRawIO(BaseRawIO):
    """
    Class for reading rhd and rhs Intan data

    Parameters
    ----------
    filename: str, default: ''
       name of the 'rhd' or 'rhs' data file

    Notes
    -----
    * Intan reader can handle two file formats 'rhd' and 'rhs'. It will automatically
    check for the file extension and will gather the header information based on the
    extension. Additionally it functions with RHS v 1.0 and RHD 1.0, 1.1, 1.2, 1.3, 2.0,
    3.0, and 3.1 files.

    * The reader can handle three file formats 'header-attached', 'one-file-per-signal' and
    'one-file-per-channel'.

    * Intan files contain amplifier channels labeled 'A', 'B' 'C' or 'D'
    depending on the port in which they were recorded along with the following
    additional streams.
    0: 'RHD2000' amplifier channel
    1: 'RHD2000 auxiliary input channel',
    2: 'RHD2000 supply voltage channel',
    3: 'USB board ADC input channel',
    4: 'USB board digital input channel',
    5: 'USB board digital output channel'

    * For the "header-attached" and "one-file-per-signal" formats, the structure of the digital input and output channels is
    one long vector, which must be post-processed to extract individual digital channel information. See the intantech website for more information on performing this post-processing.

    Examples
    --------
    >>> import neo.rawio
    >>> reader = neo.rawio.IntanRawIO(filename='data.rhd')
    >>> reader.parse_header()
    >>> raw_chunk = reader.get_analogsignal_chunk(block_index=0,
                                                  seg_index=0
                                                  stream_index=0)
    >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, stream_index=0)

    """

    extensions = ["rhd", "rhs", "dat"]
    rawmode = "one-file"

    def __init__(self, filename=""):

        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        filename = Path(self.filename)

        if not filename.exists() or not filename.is_file():
            raise FileNotFoundError(f"{filename} does not exist")

        if self.filename.endswith(".rhs"):
            if filename.name == "info.rhs":
                if any((filename.parent / file).exists() for file in one_file_per_signal_filenames_rhs):
                    self.file_format = "one-file-per-signal"
                    raw_file_paths_dict = create_one_file_per_signal_dict_rhs(dirname=filename.parent)
                else:
                    self.file_format = "one-file-per-channel"
                    raw_file_paths_dict = create_one_file_per_channel_dict_rhs(dirname=filename.parent)
            else:
                self.file_format = "header-attached"

            (
                self._global_info,
                self._ordered_channels,
                data_dtype,
                header_size,
                self._block_size,
                channel_number_dict,
            ) = read_rhs(self.filename, self.file_format)

        # 3 possibilities for rhd files, one combines the header and the data in the same file with suffix `rhd` while
        # the other two separates the data from the header which is always called `info.rhd`
        # attached to the actual binary file with data
        elif self.filename.endswith(".rhd"):
            if filename.name == "info.rhd":
                # first we have one-file-per-signal which is where one neo stream/file is saved as .dat files
                if any((filename.parent / file).exists() for file in one_file_per_signal_filenames_rhd):
                    self.file_format = "one-file-per-signal"
                    raw_file_paths_dict = create_one_file_per_signal_dict_rhd(dirname=filename.parent)
                # then there is one-file-per-channel where each channel in a neo stream is in its own .dat file
                else:
                    self.file_format = "one-file-per-channel"
                    raw_file_paths_dict = create_one_file_per_channel_dict_rhd(dirname=filename.parent)
            # finally the format with the header-attached to the binary file as one giant file
            else:
                self.file_format = "header-attached"

            (
                self._global_info,
                self._ordered_channels,
                data_dtype,
                header_size,
                self._block_size,
                channel_number_dict,
            ) = read_rhd(self.filename, self.file_format)

        # memmap the raw data for each format type
        # if header-attached there is one giant memory-map
        if self.file_format == "header-attached":
            self._raw_data = np.memmap(self.filename, dtype=data_dtype, mode="r", offset=header_size)

        # for 'one-file-per-signal' we have one memory map / neo stream
        elif self.file_format == "one-file-per-signal":
            self._raw_data = {}
            for stream_index, (stream_index_key, stream_datatype) in enumerate(data_dtype.items()):
                num_channels = channel_number_dict[stream_index_key]
                file_path = raw_file_paths_dict[stream_index_key]
                size_in_bytes = file_path.stat().st_size
                dtype_size = np.dtype(stream_datatype).itemsize
                n_samples = size_in_bytes // (dtype_size * num_channels)
                signal_stream_memmap = np.memmap(
                    file_path, dtype=stream_datatype, mode="r", shape=(num_channels, n_samples)
                ).T
                self._raw_data[stream_index] = signal_stream_memmap

        # for one-file-per-channel we have one memory map / channel stored as a list / neo stream
        elif self.file_format == "one-file-per-channel":
            self._raw_data = {}
            for stream_index, (stream_index_key, stream_datatype) in enumerate(data_dtype.items()):
                self._raw_data[stream_index] = []
                num_channels = channel_number_dict[stream_index_key]
                for channel_index in range(num_channels):
                    file_path = raw_file_paths_dict[stream_index_key][channel_index]
                    channel_memmap = np.memmap(file_path, dtype=stream_datatype, mode="r")
                    self._raw_data[stream_index].append(channel_memmap)

        # check timestamp continuity
        if self.file_format == "header-attached":
            timestamp = self._raw_data["timestamp"].flatten()

        # timestamps are always last stream for headerless binary files
        elif self.file_format == "one-file-per-signal":
            time_stream_index = max(self._raw_data.keys())
            timestamp = self._raw_data[time_stream_index]
        elif self.file_format == "one-file-per-channel":
            time_stream_index = max(self._raw_data.keys())
            timestamp = self._raw_data[time_stream_index][0]

        if not np.all(np.diff(timestamp) == 1):
            raise NeoReadWriteError(
                f"Timestamp have gaps, this could be due to a corrupted file or an inappropriate file merge"
            )

        # signals
        signal_channels = []
        for c, chan_info in enumerate(self._ordered_channels):
            name = chan_info["custom_channel_name"]
            channel_id = chan_info["native_channel_name"]
            sig_dtype = chan_info["dtype"]
            stream_id = str(chan_info["signal_type"])
            signal_channels.append(
                (
                    name,
                    channel_id,
                    chan_info["sampling_rate"],
                    sig_dtype,
                    chan_info["units"],
                    chan_info["gain"],
                    chan_info["offset"],
                    stream_id,
                )
            )
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        stream_ids = np.unique(signal_channels["stream_id"])
        signal_streams = np.zeros(stream_ids.size, dtype=_signal_stream_dtype)

        # we need to sort the data because the string of 10 is mis-sorted.
        stream_ids_sorted = sorted([int(stream_id) for stream_id in stream_ids])
        signal_streams["id"] = [str(stream_id) for stream_id in stream_ids_sorted]

        for stream_index, stream_id in enumerate(stream_ids_sorted):
            if self.filename.endswith(".rhd"):
                signal_streams["name"][stream_index] = stream_type_to_name_rhd.get(int(stream_id), "")
            else:
                signal_streams["name"][stream_index] = stream_type_to_name_rhs.get(int(stream_id), "")

        self._max_sampling_rate = np.max(signal_channels["sampling_rate"])

        # if header is attached we need to incorporate our block size to get signal length
        if self.file_format == "header-attached":
            self._max_sigs_length = self._raw_data.size * self._block_size
        # for one-file-per-signal we just take the size which will give n_samples for each
        # signal stream and then we just take the longest one
        elif self.file_format == "one-file-per-signal":
            self._max_sigs_length = max([raw_data.size for raw_data in self._raw_data.values()])
        # for one-file-per-channel we do the same as for one-file-per-signal, but since they
        # are in a list we just take the first channel in each list of channels
        else:
            self._max_sigs_length = max([raw_data[0].size for raw_data in self._raw_data.values()])
        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fill into header dict
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._max_sigs_length / self._max_sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):

        if self.file_format == "header-attached":
            stream_id = self.header["signal_streams"][stream_index]["id"]
            mask = self.header["signal_channels"]["stream_id"] == stream_id
            signal_channels = self.header["signal_channels"][mask]
            channel_ids = signal_channels["id"]
            channel_id_0 = channel_ids[0]
            size = self._raw_data[channel_id_0].size
        # one-file-per-signal is (n_samples, n_channels)
        elif self.file_format == "one-file-per-signal":
            size = self._raw_data[stream_index].shape[0]
        # one-file-per-channel is (n_samples) so pull from list
        else:
            size = self._raw_data[stream_index][0].size

        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._get_signal_size(block_index, seg_index, stream_index)

        if channel_indexes is None:
            channel_indexes = slice(None)

        if self.file_format == "header-attached":
            sigs_chunk = self._get_analogsignal_chunk_header_attached(
                i_start,
                i_stop,
                stream_index,
                channel_indexes,
            )
        elif self.file_format == "one-file-per-signal":
            sigs_chunk = self._get_analogsignal_chunk_one_file_per_signal(
                i_start,
                i_stop,
                stream_index,
                channel_indexes,
            )
        else:
            sigs_chunk = self._get_analogsignal_chunk_one_file_per_channel(
                i_start,
                i_stop,
                stream_index,
                channel_indexes,
            )

        return sigs_chunk

    def _get_analogsignal_chunk_header_attached(self, i_start, i_stop, stream_index, channel_indexes):

        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream_id
        signal_channels = self.header["signal_channels"][mask]

        channel_ids = signal_channels["id"][channel_indexes]
        channel_id_0 = channel_ids[0]
        shape = self._raw_data[channel_id_0].shape
        dtype = self._raw_data[channel_id_0].dtype
        sigs_chunk = np.zeros((i_stop - i_start, len(channel_ids)), dtype=dtype)

        # This is False for Temperature and timestamps
        multiple_samples_per_block = len(shape) == 2

        # In the header attached case the data for each channel comes interleaved in blocks
        # To avoid unecessary memory access we can calculate the blocks we need to access beforehand:
        if multiple_samples_per_block:
            block_size = shape[1]
            block_start = i_start // block_size
            block_stop = i_stop // block_size + 1
            sl0 = i_start % block_size
            sl1 = sl0 + (i_stop - i_start)

        # raw_data is a structured memmap with a field for each channel_id
        for chunk_index, channel_id in enumerate(channel_ids):
            data_chan = self._raw_data[channel_id]
            if multiple_samples_per_block:
                sigs_chunk[:, chunk_index] = data_chan[block_start:block_stop].flatten()[sl0:sl1]
            else:
                sigs_chunk[:, chunk_index] = data_chan[i_start:i_stop]

        return sigs_chunk

    def _get_analogsignal_chunk_one_file_per_channel(self, i_start, i_stop, stream_index, channel_indexes):

        signal_data_memmap_list = self._raw_data[stream_index]
        channel_indexes_are_slice = isinstance(channel_indexes, slice)
        if channel_indexes_are_slice:
            num_channels = len(signal_data_memmap_list)
            start = channel_indexes.start or 0
            stop = channel_indexes.stop or num_channels
            step = channel_indexes.step or 1
            channel_indexes = range(start, stop, step)

        # We get the dtype from the first channel
        first_channel_index = channel_indexes[0]
        dtype = signal_data_memmap_list[first_channel_index].dtype
        sigs_chunk = np.zeros((i_stop - i_start, len(channel_indexes)), dtype=dtype)

        for chunk_index, channel_index in enumerate(channel_indexes):
            channel_memmap = signal_data_memmap_list[channel_index]
            sigs_chunk[:, chunk_index] = channel_memmap[i_start:i_stop]

        return sigs_chunk

    def _get_analogsignal_chunk_one_file_per_signal(self, i_start, i_stop, stream_index, channel_indexes):

        # One memmap per stream case
        signal_data_memmap = self._raw_data[stream_index]

        return signal_data_memmap[i_start:i_stop, channel_indexes]


def read_qstring(f):
    length = np.fromfile(f, dtype="uint32", count=1)[0]
    if length == 0xFFFFFFFF or length == 0:
        return ""
    txt = f.read(length).decode("utf-16")
    return txt


def read_variable_header(f, header):
    info = {}
    for field_name, field_type in header:
        if field_type == "QString":
            field_value = read_qstring(f)
        else:
            field_value = np.fromfile(f, dtype=field_type, count=1)[0]
        info[field_name] = field_value
    return info


###############
# RHS ZONE

rhs_global_header = [
    ("magic_number", "uint32"),  # 0xD69127AC
    ("major_version", "int16"),
    ("minor_version", "int16"),
    ("sampling_rate", "float32"),
    ("dsp_enabled", "int16"),
    ("actual_dsp_cutoff_frequency", "float32"),
    ("actual_lower_bandwidth", "float32"),
    ("actual_lower_settle_bandwidth", "float32"),
    ("actual_upper_bandwidth", "float32"),
    ("desired_dsp_cutoff_frequency", "float32"),
    ("desired_lower_bandwidth", "float32"),
    ("desired_lower_settle_bandwidth", "float32"),
    ("desired_upper_bandwidth", "float32"),
    ("notch_filter_mode", "int16"),
    ("desired_impedance_test_frequency", "float32"),
    ("actual_impedance_test_frequency", "float32"),
    ("amp_settle_mode", "int16"),
    ("charge_recovery_mode", "int16"),
    ("stim_step_size", "float32"),
    ("recovery_current_limit", "float32"),
    ("recovery_target_voltage", "float32"),
    ("note1", "QString"),
    ("note2", "QString"),
    ("note3", "QString"),
    ("dc_amplifier_data_saved", "int16"),
    ("board_mode", "int16"),
    ("ref_channel_name", "QString"),
    ("nb_signal_group", "int16"),
]

rhs_signal_group_header = [
    ("signal_group_name", "QString"),
    ("signal_group_prefix", "QString"),
    ("signal_group_enabled", "int16"),
    ("channel_num", "int16"),
    ("amplified_channel_num", "int16"),
]

rhs_signal_channel_header = [
    ("native_channel_name", "QString"),
    ("custom_channel_name", "QString"),
    ("native_order", "int16"),
    ("custom_order", "int16"),
    ("signal_type", "int16"),
    ("channel_enabled", "int16"),
    ("chip_channel_num", "int16"),
    ("command_stream", "int16"),
    ("board_stream_num", "int16"),
    ("spike_scope_trigger_mode", "int16"),
    ("spike_scope_voltage_thresh", "int16"),
    ("spike_scope_digital_trigger_channel", "int16"),
    ("spike_scope_digital_edge_polarity", "int16"),
    ("electrode_impedance_magnitude", "float32"),
    ("electrode_impedance_phase", "float32"),
]

stream_type_to_name_rhs = {
    0: "RHS2000 amplifier channel",
    3: "USB board ADC input channel",
    4: "USB board ADC output channel",
    5: "USB board digital input channel",
    6: "USB board digital output channel",
    10: "DC Amplifier channel",
    11: "Stim channel",
}


def read_rhs(filename, file_format: str):
    BLOCK_SIZE = 128  # sample per block

    with open(filename, mode="rb") as f:
        global_info = read_variable_header(f, rhs_global_header)

        # channels_by_type is simpler than data_dtype because 0 contains 0, 10 and 11 internally
        channels_by_type = {k: [] for k in [0, 3, 4, 5, 6]}
        if not file_format == "header-attached":
            # data_dtype for rhs is complicated. There is not 1, 2 (supply and aux),
            # but there are dc-amp (10) and stim (11). we make timestamps (15)
            data_dtype = {k: [] for k in [0, 3, 4, 5, 6, 10, 11, 15]}
        for g in range(global_info["nb_signal_group"]):
            group_info = read_variable_header(f, rhs_signal_group_header)

            if bool(group_info["signal_group_enabled"]):
                for c in range(group_info["channel_num"]):
                    chan_info = read_variable_header(f, rhs_signal_channel_header)
                    if chan_info["signal_type"] in (1, 2):
                        raise NeoReadWriteError("signal_type of 1 or 2 is not yet implemented in Neo")
                    if bool(chan_info["channel_enabled"]):
                        channels_by_type[chan_info["signal_type"]].append(chan_info)

        # useful dictionary for knowing the number of channels for non-header attached formats
        channel_number_dict = {i: len(channels_by_type[i]) for i in [0, 3, 4, 5, 6]}

        header_size = f.tell()

    sr = global_info["sampling_rate"]

    # construct dtype by re-ordering channels by types
    ordered_channels = []
    if file_format == "header-attached":
        data_dtype = [("timestamp", "int32", BLOCK_SIZE)]
    else:
        data_dtype[15] = "int32"
        channel_number_dict[15] = 1

    # 0: RHS2000 amplifier channel.
    for chan_info in channels_by_type[0]:
        chan_info["sampling_rate"] = sr
        chan_info["units"] = "uV"
        chan_info["gain"] = 0.195
        if file_format == "header-attached":
            chan_info["offset"] = -32768 * 0.195
        else:
            chan_info["offset"] = 0.0
        if file_format == "header-attached":
            chan_info["dtype"] = "uint16"
        else:
            chan_info["dtype"] = "int16"
        ordered_channels.append(chan_info)
        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            data_dtype += [(name, "uint16", BLOCK_SIZE)]
        else:
            data_dtype[0] = "int16"

    if bool(global_info["dc_amplifier_data_saved"]):
        # if we have dc amp we need to grab the correct number of channels
        channel_number_dict[10] = channel_number_dict[0]
        for chan_info in channels_by_type[0]:
            chan_info_dc = dict(chan_info)
            name = chan_info["native_channel_name"]
            chan_info_dc["native_channel_name"] = name + "_DC"
            chan_info_dc["sampling_rate"] = sr
            chan_info_dc["units"] = "mV"
            chan_info_dc["gain"] = 19.23
            chan_info_dc["offset"] = -512 * 19.23
            chan_info_dc["signal_type"] = 10  # put it in another group
            chan_info_dc["dtype"] = "uint16"
            ordered_channels.append(chan_info_dc)
            if file_format == "header-attached":
                data_dtype += [(name + "_DC", "uint16", BLOCK_SIZE)]
            else:
                data_dtype[10] = "uint16"
    # I can't seem to get stim files to generate for one-file-per-channel
    # so let's skip for now and can be given on request

    if file_format != "one-file-per-channel":
        channel_number_dict[11] = channel_number_dict[0]  # should be one stim / amplifier channel
        for chan_info in channels_by_type[0]:
            chan_info_stim = dict(chan_info)
            name = chan_info["native_channel_name"]
            chan_info_stim["native_channel_name"] = name + "_STIM"
            chan_info_stim["sampling_rate"] = sr
            # stim channel are complicated because they are coded
            # with bits, they do not fit the gain/offset rawio strategy
            chan_info_stim["units"] = ""
            chan_info_stim["gain"] = 1.0
            chan_info_stim["offset"] = 0.0
            chan_info_stim["signal_type"] = 11  # put it in another group
            chan_info_stim["dtype"] = "uint16"
            ordered_channels.append(chan_info_stim)
            if file_format == "header-attached":
                data_dtype += [(name + "_STIM", "uint16", BLOCK_SIZE)]
            else:
                data_dtype[11] = "uint16"
    else:
        warnings.warn("Stim not implemented for `one-file-per-channel` due to lack of test files")

    # No supply or aux for rhs files (ie no stream 1 and 2)

    # 3: Analog input channel.
    # 4: Analog output channel.
    for sig_type in [3, 4]:
        for chan_info in channels_by_type[sig_type]:
            chan_info["sampling_rate"] = sr
            chan_info["units"] = "V"
            chan_info["gain"] = 0.0003125
            chan_info["offset"] = -32768 * 0.0003125
            chan_info["dtype"] = "uint16"
            ordered_channels.append(chan_info)
            if file_format == "header-attached":
                name = chan_info["native_channel_name"]
                data_dtype += [(name, "uint16", BLOCK_SIZE)]
            else:
                data_dtype[sig_type] = "uint16"

    # 5: Digital input channel.
    # 6: Digital output channel.
    for sig_type in [5, 6]:
        if file_format in ["header-attached", "one-file-per-signal"]:
            if len(channels_by_type[sig_type]) > 0:
                name = {5: "DIGITAL-IN", 6: "DIGITAL-OUT"}[sig_type]
                chan_info = channels_by_type[sig_type][0]
                # So currently until we have get_digitalsignal_chunk we need to do a tiny hack to
                # make this memory map work correctly. So since our digital data is not organized
                # by channel like analog/ADC are we have to overwrite the native name to create
                # a single permanent name that we can find with channel id
                chan_info["native_channel_name"] = name
                chan_info["sampling_rate"] = sr
                chan_info["units"] = "TTL"  # arbitrary units TTL for logic
                chan_info["gain"] = 1.0
                chan_info["offset"] = 0.0
                chan_info["dtype"] = "uint16"
                ordered_channels.append(chan_info)
                if file_format == "header-attached":
                    data_dtype += [(name, "uint16", BLOCK_SIZE)]
                else:
                    data_dtype[sig_type] = "uint16"
        elif file_format == "one-file-per-channel":
            for chan_info in channels_by_type[sig_type]:
                chan_info["sampling_rate"] = sr
                chan_info["units"] = "TTL"
                chan_info["gain"] = 1.0
                chan_info["offset"] = 0.0
                chan_info["dtype"] = "uint16"
                ordered_channels.append(chan_info)
                data_dtype[sig_type] = "uint16"

    if global_info["notch_filter_mode"] == 2 and global_info["major_version"] >= V("3.0"):
        global_info["notch_filter"] = "60Hz"
    elif global_info["notch_filter_mode"] == 1 and global_info["major_version"] >= V("3.0"):
        global_info["notch_filter"] = "50Hz"
    else:
        global_info["notch_filter"] = False

    if not file_format == "header-attached":
        # filter out dtypes without any values
        data_dtype = {k: v for (k, v) in data_dtype.items() if len(v) > 0}
        channel_number_dict = {k: v for (k, v) in channel_number_dict.items() if v > 0}

    return global_info, ordered_channels, data_dtype, header_size, BLOCK_SIZE, channel_number_dict


###############
# RHD ZONE

rhd_global_header_base = [
    ("magic_number", "uint32"),  # 0xC6912702
    ("major_version", "int16"),
    ("minor_version", "int16"),
]

rhd_global_header_part1 = [
    ("sampling_rate", "float32"),
    ("dsp_enabled", "int16"),
    ("actual_dsp_cutoff_frequency", "float32"),
    ("actual_lower_bandwidth", "float32"),
    ("actual_upper_bandwidth", "float32"),
    ("desired_dsp_cutoff_frequency", "float32"),
    ("desired_lower_bandwidth", "float32"),
    ("desired_upper_bandwidth", "float32"),
    ("notch_filter_mode", "int16"),
    ("desired_impedance_test_frequency", "float32"),
    ("actual_impedance_test_frequency", "float32"),
    ("note1", "QString"),
    ("note2", "QString"),
    ("note3", "QString"),
]

rhd_global_header_v11 = [
    ("num_temp_sensor_channels", "int16"),
]

rhd_global_header_v13 = [
    ("eval_board_mode", "int16"),
]

rhd_global_header_v20 = [
    ("reference_channel", "QString"),
]

rhd_global_header_final = [
    ("nb_signal_group", "int16"),
]

rhd_signal_group_header = [
    ("signal_group_name", "QString"),
    ("signal_group_prefix", "QString"),
    ("signal_group_enabled", "int16"),
    ("channel_num", "int16"),
    ("amplified_channel_num", "int16"),
]

rhd_signal_channel_header = [
    ("native_channel_name", "QString"),
    ("custom_channel_name", "QString"),
    ("native_order", "int16"),
    ("custom_order", "int16"),
    ("signal_type", "int16"),
    ("channel_enabled", "int16"),
    ("chip_channel_num", "int16"),
    ("board_stream_num", "int16"),
    ("spike_scope_trigger_mode", "int16"),
    ("spike_scope_voltage_thresh", "int16"),
    ("spike_scope_digital_trigger_channel", "int16"),
    ("spike_scope_digital_edge_polarity", "int16"),
    ("electrode_impedance_magnitude", "float32"),
    ("electrode_impedance_phase", "float32"),
]

stream_type_to_name_rhd = {
    0: "RHD2000 amplifier channel",
    1: "RHD2000 auxiliary input channel",
    2: "RHD2000 supply voltage channel",
    3: "USB board ADC input channel",
    4: "USB board digital input channel",
    5: "USB board digital output channel",
}


def read_rhd(filename, file_format: str):
    """Function for reading the rhd file header

    Parameters
    ----------
    filename: str | Path
        The filename of the *.rhd file
    file_format: 'header-attached' | 'one-file-per-signal' | 'one-file-per-channel'
        Whether the header is included with the rest of the data ('header-attached')
        Or as a standalone file ('one-file-per-signal' or 'one-file-per-channel')
    """
    with open(filename, mode="rb") as f:

        global_info = read_variable_header(f, rhd_global_header_base)

        version = V(f"{global_info['major_version']}.{global_info['minor_version']}")

        # the header size depends on the version :-(
        header = list(rhd_global_header_part1)  # make a copy

        if version >= V("1.1"):
            header = header + rhd_global_header_v11
        else:
            global_info["num_temp_sensor_channels"] = 0

        if version >= V("1.3"):
            header = header + rhd_global_header_v13
        else:
            global_info["eval_board_mode"] = 0

        if version >= V("2.0"):
            header = header + rhd_global_header_v20
        else:
            global_info["reference_channel"] = ""

        header = header + rhd_global_header_final

        global_info.update(read_variable_header(f, header))

        # read channel group and channel header
        channels_by_type = {k: [] for k in [0, 1, 2, 3, 4, 5]}
        if not file_format == "header-attached":
            data_dtype = {k: [] for k in range(7)}  # 5 streams + 6 for timestamps for not header attached
        for g in range(global_info["nb_signal_group"]):
            group_info = read_variable_header(f, rhd_signal_group_header)

            if bool(group_info["signal_group_enabled"]):
                for c in range(group_info["channel_num"]):
                    chan_info = read_variable_header(f, rhd_signal_channel_header)
                    if bool(chan_info["channel_enabled"]):
                        channels_by_type[chan_info["signal_type"]].append(chan_info)

            channel_number_dict = {i: len(channels_by_type[i]) for i in range(6)}

        header_size = f.tell()

    sr = global_info["sampling_rate"]

    # construct the data block dtype and reorder channels
    if version >= V("2.0"):
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 60  # 256 channels

    ordered_channels = []

    if version >= V("1.2"):
        if file_format == "header-attached":
            data_dtype = [("timestamp", "int32", BLOCK_SIZE)]
        else:
            data_dtype[6] = "int32"
            channel_number_dict[6] = 1
    else:
        if file_format == "header-attached":
            data_dtype = [("timestamp", "uint32", BLOCK_SIZE)]
        else:
            data_dtype[6] = "uint32"
            channel_number_dict[6] = 1

    # 0: RHD2000 amplifier channel
    for chan_info in channels_by_type[0]:
        chan_info["sampling_rate"] = sr
        chan_info["units"] = "uV"
        chan_info["gain"] = 0.195
        if file_format == "header-attached":
            chan_info["offset"] = -32768 * 0.195
            chan_info["dtype"] = "uint16"
        else:
            chan_info["offset"] = 0.0
            chan_info["dtype"] = "int16"
        ordered_channels.append(chan_info)

        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            data_dtype += [(name, "uint16", BLOCK_SIZE)]
        else:
            data_dtype[0] = "int16"

    # 1: RHD2000 auxiliary input channel
    for chan_info in channels_by_type[1]:
        chan_info["sampling_rate"] = sr / 4.0
        chan_info["units"] = "V"
        chan_info["gain"] = 0.0000374
        chan_info["offset"] = 0.0
        chan_info["dtype"] = "uint16"
        ordered_channels.append(chan_info)
        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            data_dtype += [(name, "uint16", BLOCK_SIZE // 4)]
        else:
            data_dtype[1] = "uint16"

    # 2: RHD2000 supply voltage channel
    for chan_info in channels_by_type[2]:
        chan_info["sampling_rate"] = sr / BLOCK_SIZE
        chan_info["units"] = "V"
        chan_info["gain"] = 0.0000748
        chan_info["offset"] = 0.0
        chan_info["dtype"] = "uint16"
        ordered_channels.append(chan_info)
        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            data_dtype += [(name, "uint16")]
        else:
            data_dtype[2] = "uint16"

    # temperature is not an official channel in the header
    for i in range(global_info["num_temp_sensor_channels"]):
        name = f"temperature_{i}"
        chan_info = {"native_channel_name": name, "signal_type": 20}
        chan_info["sampling_rate"] = sr / BLOCK_SIZE
        chan_info["units"] = "Celsius"
        chan_info["gain"] = 0.001
        chan_info["offset"] = 0.0
        chan_info["dtype"] = "int16"
        ordered_channels.append(chan_info)
        data_dtype += [(name, "int16")]

    # 3: USB board ADC input channel
    for chan_info in channels_by_type[3]:
        chan_info["sampling_rate"] = sr
        chan_info["units"] = "V"
        if global_info["eval_board_mode"] == 0:
            chan_info["gain"] = 0.000050354
            chan_info["offset"] = 0.0
        elif global_info["eval_board_mode"] == 1:
            chan_info["gain"] = 0.00015259
            chan_info["offset"] = -32768 * 0.00015259
        elif global_info["eval_board_mode"] == 13:
            chan_info["gain"] = 0.0003125
            chan_info["offset"] = -32768 * 0.0003125
        chan_info["dtype"] = "uint16"
        ordered_channels.append(chan_info)
        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            data_dtype += [(name, "uint16", BLOCK_SIZE)]
        else:
            data_dtype[3] = "uint16"

    # 4: USB board digital input channel
    # 5: USB board digital output channel
    for sig_type in [4, 5]:
        if file_format in ["header-attached", "one-file-per-signal"]:
            if len(channels_by_type[sig_type]) > 0:
                name = {4: "DIGITAL-IN", 5: "DIGITAL-OUT"}[sig_type]
                chan_info = channels_by_type[sig_type][0]
                # So currently until we have get_digitalsignal_chunk we need to do a tiny hack to
                # make this memory map work correctly. So since our digital data is not organized
                # by channel like analog/ADC are we have to overwrite the native name to create
                # a single permanent name that we can find with channel id
                chan_info["native_channel_name"] = name
                chan_info["sampling_rate"] = sr
                chan_info["units"] = "TTL"  # arbitrary units TTL for logic
                chan_info["gain"] = 1.0
                chan_info["offset"] = 0.0
                chan_info["dtype"] = "uint16"
                ordered_channels.append(chan_info)
                if file_format == "header-attached":
                    data_dtype += [(name, "uint16", BLOCK_SIZE)]
                else:
                    data_dtype[sig_type] = "uint16"
        elif file_format == "one-file-per-channel":
            for chan_info in channels_by_type[sig_type]:
                chan_info["sampling_rate"] = sr
                chan_info["units"] = "TTL"
                chan_info["gain"] = 1.0
                chan_info["offset"] = 0.0
                chan_info["dtype"] = "uint16"
                ordered_channels.append(chan_info)
                data_dtype[sig_type] = "uint16"

    if global_info["notch_filter_mode"] == 2 and version >= V("3.0"):
        global_info["notch_filter"] = "60Hz"
    elif global_info["notch_filter_mode"] == 1 and version >= V("3.0"):
        global_info["notch_filter"] = "50Hz"
    else:
        global_info["notch_filter"] = False

    if not file_format == "header-attached":
        # filter out dtypes without any values
        data_dtype = {k: v for (k, v) in data_dtype.items() if len(v) > 0}
        channel_number_dict = {k: v for (k, v) in channel_number_dict.items() if v > 0}

    return global_info, ordered_channels, data_dtype, header_size, BLOCK_SIZE, channel_number_dict


##########################################################################
# RHX Zone for Binary Files
# This section provides all possible headerless binary files in both the rhs and rhd
# formats.

# RHD Binary Files for One File Per Signal
one_file_per_signal_filenames_rhd = [
    "amplifier.dat",
    "auxiliary.dat",
    "supply.dat",
    "analogin.dat",
    "digitalin.dat",
    "digitalout.dat",
]


def create_one_file_per_signal_dict_rhd(dirname):
    """Function for One File Per Signal Type

    Parameters
    ----------
    dirname: pathlib.Path
        The folder to explore

    Returns
    -------
    raw_files_paths_dict: dict
        A dict of all the file paths
    """

    raw_file_paths_dict = {}
    for raw_index, raw_file in enumerate(one_file_per_signal_filenames_rhd):
        if Path(dirname / raw_file).is_file():
            raw_file_paths_dict[raw_index] = Path(dirname / raw_file)

    raw_file_paths_dict[6] = Path(dirname / "time.dat")

    return raw_file_paths_dict


# RHS Binary Files for One File Per Signal
one_file_per_signal_filenames_rhs = [
    "amplifier.dat",
    "auxiliary.dat",
    "supply.dat",
    "analogin.dat",
    "analogout.dat",
    "digitalin.dat",
    "digitalout.dat",
]


def create_one_file_per_signal_dict_rhs(dirname):
    """Function for One File Per Signal Type

    Parameters
    ----------
    dirname: pathlib.Path
        The folder to explore

    Returns
    -------
    raw_files_paths_dict: dict
        A dict of all the file paths
    """

    raw_file_paths_dict = {}
    for raw_index, raw_file in enumerate(one_file_per_signal_filenames_rhs):
        if Path(dirname / raw_file).is_file():
            raw_file_paths_dict[raw_index] = Path(dirname / raw_file)

    # we need time to be the last value
    raw_file_paths_dict[15] = Path(dirname / "time.dat")
    # 10 and 11 are hardcoded in the rhs_reader above so hardcoded here too
    if Path(dirname / "dcamplifier.dat").is_file():
        raw_file_paths_dict[10] = Path(dirname / "dcamplifier.dat")
    if Path(dirname / "stim.dat").is_file():
        raw_file_paths_dict[11] = Path(dirname / "stim.dat")

    return raw_file_paths_dict


# RHD Binary Files for One File Per Channel
possible_raw_file_prefixes_rhd = [
    "amp",
    "aux",
    "vdd",
    "board-ANALOG-IN",
    "board-DIGITAL-IN",
    "board-DIGITAL-OUT",
]


def create_one_file_per_channel_dict_rhd(dirname):
    """Utility function for One File Per Channel

    Parameters
    ----------
    dirname: pathlib.Path
        The folder to explore

    Returns
    -------
    raw_files_paths_dict: dict
        A dict of all the file paths
    """

    file_names = dirname.glob("**/*.dat")
    files = [file for file in file_names if file.is_file()]
    raw_file_paths_dict = {}
    for raw_index, prefix in enumerate(possible_raw_file_prefixes_rhd):
        raw_file_paths_dict[raw_index] = [file for file in files if prefix in file.name]

    raw_file_paths_dict[6] = [Path(dirname / "time.dat")]

    return raw_file_paths_dict


# RHS Binary Files for One File Per Channel
possible_raw_file_prefixes_rhs = [
    "amp",
    "aux",
    "vdd",
    "board-ANALOG-IN",
    "board-ANALOG-OUT",
    "board-DIGITAL-IN",
    "board-DIGITAL-OUT",
]


def create_one_file_per_channel_dict_rhs(
    dirname,
):
    """Utility function for One File Per Channel

    Parameters
    ----------
    dirname: pathlib.Path
        The folder to explore

    Returns
    -------
    raw_files_paths_dict: dict
        A dict of all the file paths
    """

    file_names = dirname.glob("**/*.dat")
    files = [file for file in file_names if file.is_file()]
    raw_file_paths_dict = {}
    for raw_index, prefix in enumerate(possible_raw_file_prefixes_rhs):
        raw_file_paths_dict[raw_index] = [file for file in files if prefix in file.name]

    # we need time to be the last value
    raw_file_paths_dict[15] = [Path(dirname / "time.dat")]
    # 10 and 11 are hardcoded in the rhs reader so hardcoded here
    raw_file_paths_dict[10] = [file for file in files if "dc-" in file.name]
    # we can find the files, but I can see how to read them out of header
    # so for now we don't expose the stim files in one-file-per-channel
    raw_file_paths_dict[11] = [file for file in files if "stim-" in file.name]

    return raw_file_paths_dict
