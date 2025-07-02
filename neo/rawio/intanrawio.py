"""
Support for intan tech rhd and rhs files.

These 2 formats are more or less the same but:
  * some variance in headers.
  * rhs amplifier is more complex because the optional DC channel

RHS supported version 1.0
RHD supported version  1.0 1.1 1.2 1.3 2.0 3.0, 3.1
RHD headerless binary support 3.x
RHS headerless binary support 3.x


See:
  * http://intantech.com/files/Intan_RHD2000_data_file_formats.pdf
  * http://intantech.com/files/Intan_RHS2000_data_file_formats.pdf


Author: Samuel Garcia (Initial), Zach McKenzie & Heberto Mayorquin (Updates)

"""

from pathlib import Path
from packaging.version import Version
import warnings

import numpy as np

from neo.core import NeoReadWriteError

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


class IntanRawIO(BaseRawIO):
    """
    Class for reading rhd and rhs Intan data

    Parameters
    ----------
    filename: str, default: ''
         name of the 'rhd' or 'rhs' data/header file
    ignore_integrity_checks: bool, default: False
        If True, data that violates integrity assumptions will be loaded. At the moment the only integrity
        check we perform is that timestamps are continuous. Setting this to True will ignore this check and set
        the attribute `discontinuous_timestamps` to True if the timestamps are not continous. This attribute can be checked
        after parsing the header to see if the timestamps are continuous or not.

    Notes
    -----
    * The Intan reader can handle two file formats 'rhd' and 'rhs'. It will automatically
      check for the file extension and will gather the header information based on the
      extension. Additionally it functions with RHS v 1.0 and v 3.x and RHD 1.0, 1.1, 1.2, 1.3, 2.0,
      3.x files.

    * The Intan reader can also handle the headerless binary formats 'one-file-per-signal' and
      'one-file-per-channel' which have a header file called 'info.rhd' or 'info.rhs' and a series
      of binary files with the '.dat' suffix

    * Intan files contain amplifier channels labeled 'A', 'B' 'C' or 'D' for the 512 recorder
      or 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' for the 1024 recorder system
      depending on the port in which they were recorded along (stored in stream_id '0') with the following
      additional streams.

    0: 'RHD2000 amplifier channel'
    1: 'RHD2000 auxiliary input channel',
    2: 'RHD2000 supply voltage channel',
    3: 'USB board ADC input channel',
    4: 'USB board digital input channel',
    5: 'USB board digital output channel'

    And for RHS:

    0: 'RHS2000 amplfier channel'
    3: 'USB board ADC input channel',
    4: 'USB board ADC output channel',
    5: 'USB board digital input channel',
    6: 'USB board digital output channel',
    10: 'DC Amplifier channel',
    11: 'Stim channel',

    * We currently implement digital data demultiplexing so that if digital streams are requested they are
      returned as arrays of 1s and 0s.

    * We also do stim data decoding which returns the stim data as an int16 of appropriate magnitude. Please
      use `rescale_signal_raw_to_float` to obtain stim data in amperes.


    Examples
    --------
    >>> import neo.rawio
    >>> # for a header-attached file
    >>> reader = neo.rawio.IntanRawIO(filename='data.rhd')
    >>> # for the other formats we point to the info.rhd
    >>> reader = neo.rawioIntanRawIO(filename='info.rhd')
    >>> reader.parse_header()
    >>> raw_chunk = reader.get_analogsignal_chunk(block_index=0,
                                                  seg_index=0
                                                  stream_index=0)
    >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, stream_index=0)

    """

    extensions = ["rhd", "rhs", "dat"]
    rawmode = "one-file"

    def __init__(self, filename="", ignore_integrity_checks=False):

        BaseRawIO.__init__(self)
        self.filename = Path(filename)
        self.ignore_integrity_checks = ignore_integrity_checks
        self.discontinuous_timestamps = False

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        self.filename = Path(self.filename)

        # Input checks
        if not self.filename.is_file():
            raise FileNotFoundError(f"{self.filename} does not exist")

        if not self.filename.suffix in [".rhd", ".rhs"]:
            raise ValueError(f"{self.filename} is not a valid Intan file. Expected .rhd or .rhs extension")

        # see comment below for RHD which explains the division between file types
        if self.filename.suffix == ".rhs":
            if self.filename.name == "info.rhs":
                if any((self.filename.parent / file).exists() for file in stream_name_to_file_name_rhs.values()):
                    self.file_format = "one-file-per-signal"
                    raw_file_paths_dict = create_one_file_per_signal_dict_rhs(dirname=self.filename.parent)
                else:
                    self.file_format = "one-file-per-channel"
                    raw_file_paths_dict = create_one_file_per_channel_dict_rhs(dirname=self.filename.parent)
            else:
                self.file_format = "header-attached"

            (
                self._global_info,
                self._ordered_channel_info,
                memmap_data_dtype,
                header_size,
                self._block_size,
                channel_number_dict,
            ) = read_rhs(self.filename, self.file_format)

        # 3 possibilities for rhd files, one combines the header and the data in the same file with suffix `rhd` while
        # the other two separates the data from the header which is always called `info.rhd`
        # attached to the actual binary file with data
        elif self.filename.suffix == ".rhd":
            if self.filename.name == "info.rhd":
                # first we have one-file-per-signal which is where one neo stream/file is saved as .dat files
                if any((self.filename.parent / file).exists() for file in stream_name_to_file_name_rhd.values()):
                    self.file_format = "one-file-per-signal"
                    raw_file_paths_dict = create_one_file_per_signal_dict_rhd(dirname=self.filename.parent)
                # then there is one-file-per-channel where each channel in a neo stream is in its own .dat file
                else:
                    self.file_format = "one-file-per-channel"
                    raw_file_paths_dict = create_one_file_per_channel_dict_rhd(dirname=self.filename.parent)
            # finally the format with the header-attached to the binary file as one giant file
            else:
                self.file_format = "header-attached"

            (
                self._global_info,
                self._ordered_channel_info,
                memmap_data_dtype,
                header_size,
                self._block_size,
                channel_number_dict,
            ) = read_rhd(self.filename, self.file_format)

        #############################################
        # memmap the raw data for each format type
        #############################################
        # if header-attached there is one giant memory-map
        if self.file_format == "header-attached":
            self._raw_data = np.memmap(self.filename, dtype=memmap_data_dtype, mode="r", offset=header_size)

        # for 'one-file-per-signal' we have one memory map / neo stream
        # based on https://github.com/NeuralEnsemble/python-neo/issues/1556 bug in versions 0.13.1, .2, .3
        elif self.file_format == "one-file-per-signal":
            self._raw_data = {}
            for stream_name, stream_dtype in memmap_data_dtype.items():
                num_channels = channel_number_dict[stream_name]
                file_path = raw_file_paths_dict[stream_name]
                size_in_bytes = file_path.stat().st_size
                dtype_size = np.dtype(stream_dtype).itemsize
                n_samples = size_in_bytes // (dtype_size * num_channels)
                signal_stream_memmap = np.memmap(
                    file_path,
                    dtype=stream_dtype,
                    mode="r",
                    shape=(n_samples, num_channels),
                    order="C",
                )
                self._raw_data[stream_name] = signal_stream_memmap

        # for one-file-per-channel we have one memory map / channel stored as a list / neo stream
        elif self.file_format == "one-file-per-channel":
            self._raw_data = {}
            for stream_name, stream_dtype in memmap_data_dtype.items():
                channel_file_paths = raw_file_paths_dict[stream_name]
                channel_memmap_list = [np.memmap(fp, dtype=stream_dtype, mode="r") for fp in channel_file_paths]
                self._raw_data[stream_name] = channel_memmap_list

        # Data Integrity checks
        # strictness of check is controlled by ignore_integrity_checks
        # which is set at __init__
        self._assert_timestamp_continuity()

        # signals
        signal_channels = []
        # This is used to calculate bit masks
        self.native_channel_order = {}
        for chan_info in self._ordered_channel_info:
            name = chan_info["custom_channel_name"]
            channel_id = chan_info["native_channel_name"]
            sig_dtype = chan_info["dtype"]
            stream_id = str(chan_info["signal_type"])
            # buffer will be handled later bepending the format
            buffer_id = ""
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
                    buffer_id,
                )
            )
            self.native_channel_order[channel_id] = chan_info["native_order"]
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        stream_ids = np.unique(signal_channels["stream_id"])
        signal_streams = np.zeros(stream_ids.size, dtype=_signal_stream_dtype)

        buffer_ids = []
        # we need to sort the data because the string of stream_index 10 is mis-sorted.
        signal_streams["id"] = sorted(stream_ids, key=lambda x: int(x))
        for stream_index, stream_id in enumerate(signal_streams["id"]):
            if self.filename.suffix == ".rhd":
                name = stream_id_to_stream_name_rhd.get(stream_id, "")
            else:
                name = stream_id_to_stream_name_rhs.get(stream_id, "")
            signal_streams["name"][stream_index] = name
            # zach I need you help here
            if self.file_format == "header-attached":
                buffer_id = ""
            elif self.file_format == "one-file-per-signal":
                buffer_id = stream_id
                buffer_ids.append(buffer_id)
            elif self.file_format == "one-file-per-channel":
                buffer_id = ""

            signal_streams["buffer_id"][stream_index] = buffer_id

            # set buffer_id to channels
            if buffer_id != "":
                mask = signal_channels["stream_id"] == stream_id
                signal_channels["buffer_id"][mask] = buffer_id

        # depending the format we can have buffer_id or not
        signal_buffers = np.zeros(len(buffer_ids), dtype=_signal_buffer_dtype)
        signal_buffers["id"] = buffer_ids

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
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # Extract annotations from the format
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations["blocks"][0]
        seg_annotations = bl_annotations["segments"][0]

        for signal_annotation in seg_annotations["signals"]:
            # Add global annotations
            signal_annotation["intan_version"] = (
                f"{self._global_info['major_version']}." f"{self._global_info['minor_version']}"
            )
            global_keys_to_skip = [
                "major_version",
                "minor_version",
                "sampling_rate",
                "magic_number",
                "reference_channel",
            ]
            global_keys_to_annotate = set(self._global_info.keys()) - set(global_keys_to_skip)
            for key in global_keys_to_annotate:
                signal_annotation[key] = self._global_info[key]

            reference_channel = self._global_info.get("reference_channel", None)
            # Following the pdf specification
            reference_channel = "hardware" if reference_channel == "n/a" else reference_channel

            # Add channel annotations
            array_annotations = signal_annotation["__array_annotations__"]
            channel_ids = array_annotations["channel_ids"]

            # TODO refactor ordered channel dict to make this easier
            # Use this to find which elements of the ordered channels correspond to the current signal
            signal_type = int(signal_annotation["stream_id"])
            channel_info = next((info for info in self._ordered_channel_info if info["signal_type"] == signal_type))
            channel_keys_to_skip = [
                "signal_type",
                "custom_channel_name",
                "native_channel_name",
                "gain",
                "offset",
                "channel_enabled",
                "dtype",
                "units",
                "sampling_rate",
            ]

            channel_keys_to_annotate = set(channel_info.keys()) - set(channel_keys_to_skip)
            properties_dict = {key: [] for key in channel_keys_to_annotate}
            for channel_id in channel_ids:
                matching_info = next(
                    info for info in self._ordered_channel_info if info["native_channel_name"] == channel_id
                )
                for key in channel_keys_to_annotate:
                    properties_dict[key].append(matching_info[key])

            for key in channel_keys_to_annotate:
                array_annotations[key] = properties_dict[key]

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._max_sigs_length / self._max_sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):

        stream_name = self.header["signal_streams"][stream_index]["name"][:]

        if self.file_format == "header-attached":
            stream_id = self.header["signal_streams"][stream_index]["id"]
            mask = self.header["signal_channels"]["stream_id"] == stream_id
            signal_channels = self.header["signal_channels"][mask]
            channel_ids = signal_channels["id"]

            stream_is_digital = stream_name in digital_stream_names
            field_name = stream_name if stream_is_digital else channel_ids[0]

            size = self._raw_data[field_name].size

        # one-file-per-signal is (n_samples, n_channels)
        elif self.file_format == "one-file-per-signal":
            size = self._raw_data[stream_name].shape[0]
        # one-file-per-channel is (n_samples) so pull from list
        else:
            size = self._raw_data[stream_name][0].size

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

        stream_name = self.header["signal_streams"][stream_index]["name"][:]
        stream_is_digital = stream_name in digital_stream_names
        stream_is_stim = stream_name == "Stim channel"

        field_name = stream_name if stream_is_digital else channel_ids[0]

        shape = self._raw_data[field_name].shape
        dtype = self._raw_data[field_name].dtype

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

        if not stream_is_digital:
            # For all streams raw_data is a structured memmap with a field for each channel_id
            sigs_chunk = np.zeros((i_stop - i_start, len(channel_ids)), dtype=dtype)
            for chunk_index, channel_id in enumerate(channel_ids):
                data_chan = self._raw_data[channel_id]

                if multiple_samples_per_block:
                    sigs_chunk[:, chunk_index] = data_chan[block_start:block_stop].flatten()[sl0:sl1]
                else:
                    sigs_chunk[:, chunk_index] = data_chan[i_start:i_stop]

            if stream_is_stim:
                sigs_chunk = self._decode_current_from_stim_data(sigs_chunk, 0, sigs_chunk.shape[0])

        else:
            # For digital data the channels come interleaved in a single field so we need to demultiplex
            digital_raw_data = self._raw_data[field_name].flatten()
            sigs_chunk = self._demultiplex_digital_data(digital_raw_data, channel_ids, i_start, i_stop)
        return sigs_chunk

    def _get_analogsignal_chunk_one_file_per_channel(self, i_start, i_stop, stream_index, channel_indexes):

        stream_name = self.header["signal_streams"][stream_index]["name"][:]
        signal_data_memmap_list = self._raw_data[stream_name]
        stream_is_stim = stream_name == "Stim channel"

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

        # If this is stim data, we need to decode the current values
        if stream_is_stim:
            sigs_chunk = self._decode_current_from_stim_data(sigs_chunk, 0, sigs_chunk.shape[0])

        return sigs_chunk

    def _get_analogsignal_chunk_one_file_per_signal(self, i_start, i_stop, stream_index, channel_indexes):

        stream_name = self.header["signal_streams"][stream_index]["name"][:]
        raw_data = self._raw_data[stream_name]

        stream_is_digital = stream_name in digital_stream_names
        stream_is_stim = stream_name == "Stim channel"

        if stream_is_digital:
            stream_id = self.header["signal_streams"][stream_index]["id"]
            mask = self.header["signal_channels"]["stream_id"] == stream_id
            signal_channels = self.header["signal_channels"][mask]
            channel_ids = signal_channels["id"][channel_indexes]

            output = self._demultiplex_digital_data(raw_data, channel_ids, i_start, i_stop)
        elif stream_is_stim:
            output = self._decode_current_from_stim_data(raw_data, i_start, i_stop)
            output = output[:, channel_indexes]
        else:
            output = raw_data[i_start:i_stop, channel_indexes]

        return output

    def _demultiplex_digital_data(self, raw_digital_data, channel_ids, i_start, i_stop):
        """
        Demultiplex digital data by extracting individual channel values from packed 16-bit format.

        According to the Intan format, digital input/output data is stored with all 16 channels
        encoded bit-by-bit in each 16-bit word. This method extracts the specified digital channels
        from the packed format into separate uint16 arrays of 0 and 1.

        Parameters
        ----------
        raw_digital_data : ndarray
            Raw digital data in packed 16-bit format where each bit represents a different channel.
        channel_ids : list or array
            List of channel identifiers to extract. Each channel_id must correspond to a digital
            input or output channel.
        i_start : int
            Starting sample index for demultiplexing.
        i_stop : int
            Ending sample index for demultiplexing (exclusive).

        Returns
        -------
        ndarray
            Demultiplexed digital data with shape (i_stop-i_start, len(channel_ids)),
            containing 0 or 1 values for each requested channel.

        Notes
        -----
        In the Intan format, digital channels are packed into 16-bit words where each bit position
        corresponds to a specific channel number. For example, with digital inputs 0, 4, and 5
        set high and the rest low, the 16-bit word would be 2^0 + 2^4 + 2^5 = 1 + 16 + 32 = 49.

        The native_order property for each channel corresponds to its bit position in the packed word.

        """
        dtype = np.uint16  # We fix this to match the memmap dtype
        output = np.zeros((i_stop - i_start, len(channel_ids)), dtype=dtype)

        for channel_index, channel_id in enumerate(channel_ids):
            native_order = self.native_channel_order[channel_id]
            mask = 1 << native_order
            demultiplex_data = np.bitwise_and(raw_digital_data, mask) > 0
            output[:, channel_index] = demultiplex_data[i_start:i_stop].flatten()

        return output

    def _decode_current_from_stim_data(self, raw_stim_data, i_start, i_stop):
        """
        Demultiplex stimulation data by extracting current values from packed 16-bit format.

        According to the Intan RHS data format, stimulation current is stored in the lower 9 bits
        of each 16-bit word: 8 bits for magnitude and 1 bit for sign. The upper bits contain
        flags for compliance limit, charge recovery, and amplifier settle.

        Parameters
        ----------
        raw_stim_data : ndarray
            Raw stimulation data in packed 16-bit format.
        i_start : int
            Starting sample index for demultiplexing.
        i_stop : int
            Ending sample index for demultiplexing (exclusive).

        Returns
        -------
        ndarray
            Demultiplexed stimulation current values in amperes, preserving the original
            array dimensions. The output values need to be multiplied by the stim_step_size
            parameter (from header) to obtain the actual current in amperes.

        Notes
        -----
        Bit structure of each 16-bit stimulation word:
        - Bits 0-7: Current magnitude
        - Bit 8: Sign bit (1 = negative current)
        - Bits 9-13: Unused (always zero)
        - Bit 14: Amplifier settle flag (1 = activated)
        - Bit 15: Charge recovery flag (1 = activated)
        - Bit 16 (MSB): Compliance limit flag (1 = limit reached)

        The actual current value in amperes is obtained by multiplying the
        output by the 'stim_step_size' parameter from the file header. These scaled values can be
        obtained with the `rescale_signal_raw_to_float` function.
        """
        # Get the relevant portion of the data
        data = raw_stim_data[i_start:i_stop]

        # Extract current value (bits 0-8)
        magnitude = np.bitwise_and(data, 0xFF)
        sign_bit = np.bitwise_and(np.right_shift(data, 8), 0x01)  # Shift right by 8 bits to get the sign bit

        # Apply sign to current values
        # We need to cast to int16 to handle negative values correctly
        # The max value of 8 bits is 255 so the casting is safe as there are non-negative values
        magnitude = magnitude.astype(np.int16)
        current = np.where(sign_bit == 1, -magnitude, magnitude)

        # Note: If needed, other flag bits could be extracted as follows:
        # compliance_flag = np.bitwise_and(np.right_shift(data, 15), 0x01).astype(bool)  # Bit 16 (MSB)
        # charge_recovery_flag = np.bitwise_and(np.right_shift(data, 14), 0x01).astype(bool)  # Bit 15
        # amp_settle_flag = np.bitwise_and(np.right_shift(data, 13), 0x01).astype(bool)  # Bit 14
        # These could be returned as a structured array or dictionary if needed

        return current

    def get_intan_timestamps(self, i_start=None, i_stop=None):
        """
        Retrieves the sample indices from the Intan raw data within a specified range.

        Note that sample indices are called timestamps in the Intan format but they are
        in fact just sample indices. This function extracts the sample index timestamps
        from Intan files, which represent  relative time points in sample units (not absolute time).
        These indices can be  particularly useful when working with recordings that have discontinuities.

        Parameters
        ----------
        i_start : int, optional
            The starting index from which to retrieve sample indices. If None, starts from 0.
        i_stop : int, optional
            The stopping index up to which to retrieve sample indices (exclusive).
            If None, retrieves all available indices from i_start onward.

        Returns
        -------
        timestamps : ndarray
            The flattened array of sample indices within the specified range.

        Notes
        -----
        - Sample indices can be converted to seconds by dividing by the sampling rate of the amplifier stream.
        - The function automatically handles different file formats:
        * header-attached: Timestamps are extracted directly from the timestamp field
        * one-file-per-signal: Timestamps are read from the timestamp stream
        * one-file-per-channel: Timestamps are read from the first channel in the timestamp stream
        - When recordings have discontinuities (indicated by the `discontinuous_timestamps`
        attribute being True), these indices allow for proper temporal alignment of the data.
        """
        if i_start is None:
            i_start = 0

        # Get the timestamps based on file format
        if self.file_format == "header-attached":
            timestamps = self._raw_data["timestamp"]
        elif self.file_format == "one-file-per-signal":
            timestamps = self._raw_data["timestamp"]
        elif self.file_format == "one-file-per-channel":
            timestamps = self._raw_data["timestamp"][0]

        # TODO if possible ensure that timestamps memmaps are always of correct shape to avoid memory copy here.
        timestamps = timestamps.flatten() if timestamps.ndim > 1 else timestamps

        if i_stop is None:
            return timestamps[i_start:]
        else:
            return timestamps[i_start:i_stop]

    def _assert_timestamp_continuity(self):
        """
        Asserts the continuity of timestamps in the data.

        This method verifies that the timestamps in the raw data are sequential,
        indicating a continuous recording. If discontinuities are found, a flag
        is set to indicate potential data integrity issues, and an error is raised
        unless `ignore_integrity_checks` is True.

        Raises
        ------
        NeoReadWriteError
            If timestamps are not continuous and `ignore_integrity_checks` is False.
            The error message includes a table detailing the discontinuities found.
        """
        # check timestamp continuity
        timestamps = self.get_intan_timestamps()

        discontinuous_timestamps = np.diff(timestamps) != 1
        timestamps_are_not_contiguous = np.any(discontinuous_timestamps)
        if timestamps_are_not_contiguous:
            # Mark a flag that can be checked after parsing the header to see if the timestamps are continuous or not
            self.discontinuous_timestamps = True
            if not self.ignore_integrity_checks:
                error_msg = (
                    "\nTimestamps are not continuous, likely due to a corrupted file or inappropriate file merge.\n"
                    "To open the file anyway, initialize the reader with `ignore_integrity_checks=True`.\n\n"
                    "Discontinuities Found:\n"
                    "+-----------------+-----------------+-----------------+-----------------------+\n"
                    "| Discontinuity   | Previous        | Next            | Time Difference       |\n"
                    "| Index           | (Frames)        | (Frames)        | (Seconds)             |\n"
                    "+-----------------+-----------------+-----------------+-----------------------+\n"
                )

                amplifier_sampling_rate = self._global_info["sampling_rate"]
                for discontinuity_index in np.where(discontinuous_timestamps)[0]:
                    prev_ts = timestamps[discontinuity_index]
                    next_ts = timestamps[discontinuity_index + 1]
                    time_diff = (next_ts - prev_ts) / amplifier_sampling_rate

                    error_msg += (
                        f"| {discontinuity_index + 1:>15,} | {prev_ts:>15,} | {next_ts:>15,} | {time_diff:>21.6f} |\n"
                    )

                error_msg += "+-----------------+-----------------+-----------------+-----------------------+\n"

                raise NeoReadWriteError(error_msg)


###################################
# Header reading helper functions


def read_qstring(f):
    """
    Reads the optional notes included in the Intan RHX software

    Parameters
    ----------
    f: BinaryIO
        The file object

    Returns
    -------
    txt: str
        The string
    """
    length = np.fromfile(f, dtype="uint32", count=1)[0]
    if length == 0xFFFFFFFF or length == 0:
        return ""
    txt = f.read(length).decode("utf-16")
    return txt


def read_variable_header(f, header: list):
    """
    Reads the information from the binary file for the header info dict

    Parameters
    ----------
    f: BinaryIO
        The file object
    header: list[tuple]
        The list of header sections along with their associated dtype

    Returns
    -------
    info: dict
        The dictionary containing the information contained in the header
    """
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

stream_id_to_stream_name_rhs = {
    "0": "RHS2000 amplifier channel",
    "3": "USB board ADC input channel",
    "4": "USB board ADC output channel",
    "5": "USB board digital input channel",
    "6": "USB board digital output channel",
    "10": "DC Amplifier channel",
    "11": "Stim channel",
}


def read_rhs(filename, file_format: str):
    BLOCK_SIZE = 128  # sample per block

    with open(filename, mode="rb") as f:
        global_info = read_variable_header(f, rhs_global_header)

        # We use signal_type in the header as stream_id in neo with the following
        # complications: for rhs files in the header-attaached stream_id 0 also
        # contains information for stream_id 10 and stream_id 11 so we need to
        # break these up. See notes throughout code base; for timestamps we always
        # force them to be the last stream_id.
        stream_names = stream_id_to_stream_name_rhs.values()
        stream_name_to_channel_info_list = {name: [] for name in stream_names}
        if not file_format == "header-attached":
            memmap_data_dtype = {name: [] for name in stream_names}
        for g in range(global_info["nb_signal_group"]):
            group_info = read_variable_header(f, rhs_signal_group_header)

            if bool(group_info["signal_group_enabled"]):
                for c in range(group_info["channel_num"]):
                    chan_info = read_variable_header(f, rhs_signal_channel_header)
                    if chan_info["signal_type"] in (1, 2):
                        error_msg = (
                            "signal_type of 1 or 2 does not exist for rhs files. If you have an rhs file "
                            "with these formats open an issue on the python-neo github page"
                        )
                        raise NeoReadWriteError(error_msg)
                    if bool(chan_info["channel_enabled"]):
                        stream_id = str(chan_info["signal_type"])
                        stream_name = stream_id_to_stream_name_rhs[stream_id]
                        stream_name_to_channel_info_list[stream_name].append(chan_info)

        # Build a dictionary with channel count
        special_cases_for_counting = [
            "DC Amplifier channel",
            "Stim channel",
        ]
        names_to_count = [name for name in stream_names if name not in special_cases_for_counting]
        channel_number_dict = {name: len(stream_name_to_channel_info_list[name]) for name in names_to_count}

        # Each DC amplifier channel has a corresponding RHS2000 amplifier channel
        channel_number_dict["DC Amplifier channel"] = channel_number_dict["RHS2000 amplifier channel"]

        if file_format == "one-file-per-channel":
            # There is a way to shut off saving amplifier data and only keeping the DC amplifier or shutting off all amplifier file saving,
            # so we need to count the number of files we find instead of relying on the header.
            raw_file_paths_dict = create_one_file_per_channel_dict_rhs(dirname=filename.parent)
            channel_number_dict["Stim channel"] = len(raw_file_paths_dict["Stim channel"])
            # Moreover, even if the amplifier channels are in the header their files are dropped
            channel_number_dict["RHS2000 amplifier channel"] = len(raw_file_paths_dict["RHS2000 amplifier channel"])
        else:
            channel_number_dict["Stim channel"] = channel_number_dict["RHS2000 amplifier channel"]

        header_size = f.tell()

    sr = global_info["sampling_rate"]

    # construct dtype by re-ordering channels by types
    ordered_channel_info = []
    if file_format == "header-attached":
        memmap_data_dtype = [("timestamp", "int32", BLOCK_SIZE)]
    else:
        memmap_data_dtype["timestamp"] = "int32"
        channel_number_dict["timestamp"] = 1

    if channel_number_dict["RHS2000 amplifier channel"] > 0:
        for chan_info in stream_name_to_channel_info_list["RHS2000 amplifier channel"]:
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
            ordered_channel_info.append(chan_info)
            if file_format == "header-attached":
                name = chan_info["native_channel_name"]
                memmap_data_dtype += [(name, "uint16", BLOCK_SIZE)]
            else:
                memmap_data_dtype["RHS2000 amplifier channel"] = "int16"

    if bool(global_info["dc_amplifier_data_saved"]):
        # if we have dc amp we need to grab the correct number of channels
        # the channel number is the same as the count for amplifier data
        for chan_info in stream_name_to_channel_info_list["RHS2000 amplifier channel"]:
            chan_info_dc = dict(chan_info)
            name = chan_info["native_channel_name"]
            chan_info_dc["native_channel_name"] = name + "_DC"
            chan_info_dc["custom_channel_name"] = chan_info_dc["custom_channel_name"] + "_DC"
            chan_info_dc["sampling_rate"] = sr
            chan_info_dc["units"] = "mV"
            chan_info_dc["gain"] = 19.23
            chan_info_dc["offset"] = -512 * 19.23
            chan_info_dc["signal_type"] = 10  # put it in another group
            chan_info_dc["dtype"] = "uint16"
            ordered_channel_info.append(chan_info_dc)
            if file_format == "header-attached":
                memmap_data_dtype += [(name + "_DC", "uint16", BLOCK_SIZE)]
            else:
                memmap_data_dtype["DC Amplifier channel"] = "uint16"

    # I can't seem to get stim files to generate for one-file-per-channel
    # so ideally at some point we need test data to confirm this is true
    # based on what Heberto and I read in the docs

    # Add stim channels
    for chan_info in stream_name_to_channel_info_list["RHS2000 amplifier channel"]:
        # stim channel presence is not indicated in the header so for some formats each amplifier channel has a stim channel, but for other formats this isn't the case.
        if file_format == "one-file-per-channel":
            # Some amplifier channels don't have a corresponding stim channel,
            # so we need to make sure we don't add channel info for stim channels that don't exist.
            # In this case, if the stim channel has no data, there won't be a file for it.
            stim_file_paths = raw_file_paths_dict["Stim channel"]
            amplifier_native_name = chan_info["native_channel_name"]
            stim_file_exists = any([amplifier_native_name in stim_file.stem for stim_file in stim_file_paths])
            if not stim_file_exists:
                continue

        chan_info_stim = dict(chan_info)
        name = chan_info["native_channel_name"]
        chan_info_stim["native_channel_name"] = name + "_STIM"
        chan_info_stim["custom_channel_name"] = chan_info_stim["custom_channel_name"] + "_STIM"
        chan_info_stim["sampling_rate"] = sr
        chan_info_stim["units"] = "A"  # Amps
        chan_info_stim["gain"] = global_info["stim_step_size"]
        chan_info_stim["offset"] = 0.0
        chan_info_stim["signal_type"] = 11  # put it in another group
        chan_info_stim["dtype"] = "int16"  # this change is due to bit decoding see note below
        ordered_channel_info.append(chan_info_stim)
        # Note that the data on disk is uint16 but the data is
        # then decoded as int16 so the chan_info is int16
        if file_format == "header-attached":
            memmap_data_dtype += [(name + "_STIM", "uint16", BLOCK_SIZE)]
        else:
            memmap_data_dtype["Stim channel"] = "uint16"

    # No supply or aux for rhs files (ie no stream_id 1 and 2)
    # We have an error above that requests test files to help if the spec is changed
    # in the future.

    for stream_name in ["USB board ADC input channel", "USB board ADC output channel"]:
        for chan_info in stream_name_to_channel_info_list[stream_name]:
            chan_info["sampling_rate"] = sr
            chan_info["units"] = "V"
            chan_info["gain"] = 0.0003125
            chan_info["offset"] = -32768 * 0.0003125
            chan_info["dtype"] = "uint16"
            ordered_channel_info.append(chan_info)
            if file_format == "header-attached":
                name = chan_info["native_channel_name"]
                memmap_data_dtype += [(name, "uint16", BLOCK_SIZE)]
            else:
                memmap_data_dtype[stream_name] = "uint16"

    for stream_name in ["USB board digital input channel", "USB board digital output channel"]:
        for chan_info in stream_name_to_channel_info_list[stream_name]:
            chan_info["sampling_rate"] = sr
            # arbitrary units are used to indicate that Intan does not
            # store raw voltages but only the boolean TTL state
            chan_info["units"] = "a.u."
            chan_info["gain"] = 1.0
            chan_info["offset"] = 0.0
            chan_info["dtype"] = "uint16"
            ordered_channel_info.append(chan_info)

        # Note that all the channels are packed in one buffer, so the data type only needs to be added once
        if len(stream_name_to_channel_info_list[stream_name]) > 0:
            if file_format == "header-attached":
                memmap_data_dtype += [(stream_name, "uint16", BLOCK_SIZE)]
            elif file_format == "one-file-per-channel":
                memmap_data_dtype[stream_name] = "uint16"
            elif file_format == "one-file-per-signal":
                memmap_data_dtype[stream_name] = "uint16"

    # per discussion with Intan developers before version 3 of their software the 'notch_filter_mode'
    # was a request for postprocessing to be done in one of their scripts. From version 3+ the notch
    # filter is now applied to the data in realtime and only the post notched amplifier data is
    # saved.
    if global_info["notch_filter_mode"] == 2 and global_info["major_version"] >= 3:
        global_info["notch_filter"] = "60Hz"
    elif global_info["notch_filter_mode"] == 1 and global_info["major_version"] >= 3:
        global_info["notch_filter"] = "50Hz"
    else:
        global_info["notch_filter"] = False

    if not file_format == "header-attached":
        # filter out dtypes without any values
        memmap_data_dtype = {k: v for (k, v) in memmap_data_dtype.items() if len(v) > 0}
        channel_number_dict = {k: v for (k, v) in channel_number_dict.items() if v > 0}

    return global_info, ordered_channel_info, memmap_data_dtype, header_size, BLOCK_SIZE, channel_number_dict


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

stream_id_to_stream_name_rhd = {
    "0": "RHD2000 amplifier channel",
    "1": "RHD2000 auxiliary input channel",
    "2": "RHD2000 supply voltage channel",
    "3": "USB board ADC input channel",
    "4": "USB board digital input channel",
    "5": "USB board digital output channel",
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

        version = Version(f"{global_info['major_version']}.{global_info['minor_version']}")

        # the header size depends on the version :-(
        header = list(rhd_global_header_part1)  # make a copy

        if version >= Version("1.1"):
            header = header + rhd_global_header_v11
        else:
            global_info["num_temp_sensor_channels"] = 0

        if version >= Version("1.3"):
            header = header + rhd_global_header_v13
        else:
            global_info["eval_board_mode"] = 0

        if version >= Version("2.0"):
            header = header + rhd_global_header_v20
        else:
            global_info["reference_channel"] = ""

        header = header + rhd_global_header_final

        global_info.update(read_variable_header(f, header))

        # read channel group and channel header
        stream_names = stream_id_to_stream_name_rhd.values()
        stream_name_to_channel_info_list = {name: [] for name in stream_names}
        if not file_format == "header-attached":
            memmap_data_dtype = {name: [] for name in stream_names}
            memmap_data_dtype["timestamp"] = []
        for g in range(global_info["nb_signal_group"]):
            group_info = read_variable_header(f, rhd_signal_group_header)

            if bool(group_info["signal_group_enabled"]):
                for c in range(group_info["channel_num"]):
                    chan_info = read_variable_header(f, rhd_signal_channel_header)
                    if bool(chan_info["channel_enabled"]):
                        stream_id = str(chan_info["signal_type"])
                        stream_name = stream_id_to_stream_name_rhd[stream_id]
                        stream_name_to_channel_info_list[stream_name].append(chan_info)

            channel_number_dict = {name: len(stream_name_to_channel_info_list[name]) for name in stream_names}

        header_size = f.tell()

    sr = global_info["sampling_rate"]

    # construct the data block dtype and reorder channels
    if version >= Version("2.0"):
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 60  # 256 channels

    ordered_channel_info = []

    if version >= Version("1.2"):
        if file_format == "header-attached":
            memmap_data_dtype = [("timestamp", "int32", BLOCK_SIZE)]
        else:
            memmap_data_dtype["timestamp"] = "int32"
            channel_number_dict["timestamp"] = 1
    else:
        if file_format == "header-attached":
            memmap_data_dtype = [("timestamp", "uint32", BLOCK_SIZE)]
        else:
            memmap_data_dtype["timestamp"] = "uint32"
            channel_number_dict["timestamp"] = 1

    for chan_info in stream_name_to_channel_info_list["RHD2000 amplifier channel"]:
        chan_info["sampling_rate"] = sr
        chan_info["units"] = "uV"
        chan_info["gain"] = 0.195
        if file_format == "header-attached":
            chan_info["offset"] = -32768 * 0.195
            chan_info["dtype"] = "uint16"
        else:
            chan_info["offset"] = 0.0
            chan_info["dtype"] = "int16"
        ordered_channel_info.append(chan_info)

        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            memmap_data_dtype += [(name, "uint16", BLOCK_SIZE)]
        else:
            memmap_data_dtype["RHD2000 amplifier channel"] = "int16"

    for chan_info in stream_name_to_channel_info_list["RHD2000 auxiliary input channel"]:
        chan_info["sampling_rate"] = sr / 4.0
        chan_info["units"] = "V"
        chan_info["gain"] = 0.0000374
        chan_info["offset"] = 0.0
        chan_info["dtype"] = "uint16"
        ordered_channel_info.append(chan_info)
        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            memmap_data_dtype += [(name, "uint16", BLOCK_SIZE // 4)]
        else:
            memmap_data_dtype["RHD2000 auxiliary input channel"] = "uint16"

    for chan_info in stream_name_to_channel_info_list["RHD2000 supply voltage channel"]:
        chan_info["sampling_rate"] = sr / BLOCK_SIZE
        chan_info["units"] = "V"
        chan_info["gain"] = 0.0000748
        chan_info["offset"] = 0.0
        chan_info["dtype"] = "uint16"
        ordered_channel_info.append(chan_info)
        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            memmap_data_dtype += [(name, "uint16")]
        else:
            memmap_data_dtype["RHD2000 supply voltage channel"] = "uint16"

    # temperature is not an official channel in the header
    for i in range(global_info["num_temp_sensor_channels"]):
        name = f"temperature_{i}"
        chan_info = {"native_channel_name": name, "signal_type": 20}
        chan_info["sampling_rate"] = sr / BLOCK_SIZE
        chan_info["units"] = "Celsius"
        chan_info["gain"] = 0.001
        chan_info["offset"] = 0.0
        chan_info["dtype"] = "int16"
        ordered_channel_info.append(chan_info)
        memmap_data_dtype += [(name, "int16")]

    # 3: USB board ADC input channel
    for chan_info in stream_name_to_channel_info_list["USB board ADC input channel"]:
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
        ordered_channel_info.append(chan_info)
        if file_format == "header-attached":
            name = chan_info["native_channel_name"]
            memmap_data_dtype += [(name, "uint16", BLOCK_SIZE)]
        else:
            memmap_data_dtype["USB board ADC input channel"] = "uint16"

    for stream_name in ["USB board digital input channel", "USB board digital output channel"]:
        for chan_info in stream_name_to_channel_info_list[stream_name]:
            chan_info["sampling_rate"] = sr
            # arbitrary units are used to indicate that Intan does not
            # store raw voltages but only the boolean TTL state
            chan_info["units"] = "a.u."
            chan_info["gain"] = 1.0
            chan_info["offset"] = 0.0
            chan_info["dtype"] = "uint16"
            ordered_channel_info.append(chan_info)

        # Note that all the channels are packed in one buffer, so the data type only needs to be added once
        if len(stream_name_to_channel_info_list[stream_name]) > 0:
            if file_format == "header-attached":
                memmap_data_dtype += [(stream_name, "uint16", BLOCK_SIZE)]
            elif file_format == "one-file-per-channel":
                memmap_data_dtype[stream_name] = "uint16"
            elif file_format == "one-file-per-signal":
                memmap_data_dtype[stream_name] = "uint16"

    # per discussion with Intan developers before version 3 of their software the 'notch_filter_mode'
    # was a request for postprocessing to be done in one of their scripts. From version 3+ the notch
    # filter is now applied to the data in realtime and only the post notched amplifier data is
    # saved.
    if global_info["notch_filter_mode"] == 2 and version >= Version("3.0"):
        global_info["notch_filter"] = "60Hz"
    elif global_info["notch_filter_mode"] == 1 and version >= Version("3.0"):
        global_info["notch_filter"] = "50Hz"
    else:
        global_info["notch_filter"] = False

    if not file_format == "header-attached":
        # filter out dtypes without any values
        memmap_data_dtype = {k: v for (k, v) in memmap_data_dtype.items() if len(v) > 0}
        channel_number_dict = {k: v for (k, v) in channel_number_dict.items() if v > 0}

    return global_info, ordered_channel_info, memmap_data_dtype, header_size, BLOCK_SIZE, channel_number_dict


##########################################################################
# RHX Zone for Binary Files (note this is for version 3+ of Intan software)
# This section provides all possible headerless binary files in both the rhs and rhd
# formats.

digital_stream_names = ["USB board digital input channel", "USB board digital output channel"]


stream_name_to_file_name_rhd = {
    "RHD2000 amplifier channel": "amplifier.dat",
    "RHD2000 auxiliary input channel": "auxiliary.dat",
    "RHD2000 supply voltage channel": "supply.dat",
    "USB board ADC input channel": "analogin.dat",
    "USB board digital input channel": "digitalin.dat",
    "USB board digital output channel": "digitalout.dat",
}


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
    for stream_name, file_name in stream_name_to_file_name_rhd.items():
        if Path(dirname / file_name).is_file():
            raw_file_paths_dict[stream_name] = Path(dirname / file_name)

    raw_file_paths_dict["timestamp"] = Path(dirname / "time.dat")

    return raw_file_paths_dict


stream_name_to_file_name_rhs = {
    "RHS2000 amplifier channel": "amplifier.dat",
    "RHS2000 auxiliary input channel": "auxiliary.dat",
    "RHS2000 supply voltage channel": "supply.dat",
    "USB board ADC input channel": "analogin.dat",
    "USB board ADC output channel": "analogout.dat",
    "USB board digital input channel": "digitalin.dat",
    "USB board digital output channel": "digitalout.dat",
}


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
    for stream_name, file_name in stream_name_to_file_name_rhs.items():
        if Path(dirname / file_name).is_file():
            raw_file_paths_dict[stream_name] = Path(dirname / file_name)

    raw_file_paths_dict["timestamp"] = Path(dirname / "time.dat")
    if Path(dirname / "dcamplifier.dat").is_file():
        raw_file_paths_dict["DC Amplifier channel"] = Path(dirname / "dcamplifier.dat")
    if Path(dirname / "stim.dat").is_file():
        raw_file_paths_dict["Stim channel"] = Path(dirname / "stim.dat")

    return raw_file_paths_dict


# RHD Binary Files for One File Per Channel
stream_name_to_file_prefix_rhd = {
    "RHD2000 amplifier channel": "amp",
    "RHD2000 auxiliary input channel": "aux",
    "RHD2000 supply voltage channel": "vdd",
    "USB board ADC input channel": "board-ANALOG-IN",
    "USB board digital input channel": "board-DIGITAL-IN",
    "USB board digital output channel": "board-DIGITAL-OUT",
}


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
    # Developer note, at the moment, the channels require to be n in order
    # See https://github.com/NeuralEnsemble/python-neo/issues/1599 fo

    file_names = dirname.glob("**/*.dat")
    files = [file for file in file_names if file.is_file()]
    raw_file_paths_dict = {}
    for stream_name, file_prefix in stream_name_to_file_prefix_rhd.items():
        stream_files = [file for file in files if file_prefix in file.name]
        # Map file name to channel name amp-A-000.dat -> A-000  amp-B-006.dat -> B-006  etc
        file_path_to_channel_name = lambda x: "-".join(x.stem.split("-")[1:])
        sorted_stream_files = sorted(stream_files, key=file_path_to_channel_name)
        raw_file_paths_dict[stream_name] = sorted_stream_files

    raw_file_paths_dict["timestamp"] = [Path(dirname / "time.dat")]
    return raw_file_paths_dict


# RHS Binary Files for One File Per Channel
stream_name_to_file_prefix_rhs = {
    "RHS2000 amplifier channel": "amp",
    "RHS2000 auxiliary input channel": "aux",
    "RHS2000 supply voltage channel": "vdd",
    "USB board ADC input channel": "board-ANALOG-IN",
    "USB board ADC output channel": "board-ANALOG-OUT",
    "USB board digital input channel": "board-DIGITAL-IN",
    "USB board digital output channel": "board-DIGITAL-OUT",
}


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
    for stream_name, file_prefix in stream_name_to_file_prefix_rhs.items():
        stream_files = [file for file in files if file_prefix in file.name]
        # Map file name to channel name amp-A-000.dat -> A-000  amp-B-006.dat -> B-006  etc
        file_path_to_channel_name = lambda x: "-".join(x.stem.split("-")[1:])
        sorted_stream_files = sorted(stream_files, key=file_path_to_channel_name)
        raw_file_paths_dict[stream_name] = sorted_stream_files

    raw_file_paths_dict["timestamp"] = [Path(dirname / "time.dat")]
    raw_file_paths_dict["DC Amplifier channel"] = [file for file in files if "dc-" in file.name]
    # we can find the files, but I can see how to read them out of header
    # so for now we don't expose the stim files in one-file-per-channel
    raw_file_paths_dict["Stim channel"] = [file for file in files if "stim-" in file.name]

    return raw_file_paths_dict
