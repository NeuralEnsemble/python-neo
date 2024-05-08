"""
Class for reading spikegadgets files.
Only continuous signals are supported at the moment.

https://spikegadgets.com/spike-products/

Documentation of the format:
https://bitbucket.org/mkarlsso/trodes/wiki/Configuration

Note :
  * this file format have multiple version. news version include the gain for scaling.
     The actual implementation do not contain this feature because we don't have
     files to test this. So now the gain is "hardcoded" to 1. and so units
     is not handled correctly.

The ".rec" file format contains:
  * a first  text part with information in an XML structure
  * a second part for the binary buffer

Author: Samuel Garcia
"""
import functools
from xml.etree import ElementTree

import numpy as np
from neo.rawio.baserawio import (  # TODO the import location was updated for this notebook
    BaseRawIO,
    _event_channel_dtype,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
)
from scipy.stats import linregress

INT_16_CONVERSION = 256


class SpikeGadgetsRawIO(BaseRawIO):
    extensions = ["rec"]
    rawmode = "one-file"

    def __init__(
        self, filename="", selected_streams=None, interpolate_dropped_packets=False
    ):
        """
        Class for reading spikegadgets files.
        Only continuous signals are supported at the moment.

        Initialize a SpikeGadgetsRawIO for a single ".rec" file.

        Args:
            filename: str
                The filename
            selected_streams: None, list, str
                sublist of streams to load/expose to API
                useful for spikeextractor when one stream only is needed.
                For instance streams = ['ECU', 'trodes']
                'trodes' is name for ephy channel (ntrodes)
            interpolate_dropped_packets: bool
                If True, interpolates single dropped packets in the analog data.
        """
        BaseRawIO.__init__(self)
        self.filename = filename
        self.selected_streams = selected_streams
        self.interpolate_dropped_packets = interpolate_dropped_packets

    def _source_name(self):
        return self.filename

    def _produce_ephys_channel_ids(self, n_total_channels, n_channels_per_chip):
        """Compute the channel ID labels
        The ephys channels in the .rec file are stored in the following order:
        hwChan ID of channel 0 of first chip, hwChan ID of channel 0 of second chip, ..., hwChan ID of channel 0 of Nth chip,
        hwChan ID of channel 1 of first chip, hwChan ID of channel 1 of second chip, ..., hwChan ID of channel 1 of Nth chip,
        ...
        So if there are 32 channels per chip and 128 channels (4 chips), then the channel IDs are:
        0, 32, 64, 96, 1, 33, 65, 97, ..., 128
        See also: https://github.com/NeuralEnsemble/python-neo/issues/1215
        """
        x = []
        for k in range(n_channels_per_chip):
            x.append(
                [
                    k + i * n_channels_per_chip
                    for i in range(int(n_total_channels / n_channels_per_chip))
                ]
            )
        return [item for sublist in x for item in sublist]

    def _parse_header(self):
        # parse file until "</Configuration>"
        header_size = None
        with open(self.filename, mode="rb") as f:
            while True:
                line = f.readline()
                if b"</Configuration>" in line:
                    header_size = f.tell()
                    break

            if header_size is None:
                ValueError(
                    "SpikeGadgets: the xml header does not contain '</Configuration>'"
                )

            f.seek(0)
            header_txt = f.read(header_size).decode("utf8")

        # explore xml header
        root = ElementTree.fromstring(header_txt)
        gconf = root.find("GlobalConfiguration")
        hconf = root.find("HardwareConfiguration")
        sconf = root.find("SpikeConfiguration")

        # unix time in milliseconds at creation
        self.system_time_at_creation = gconf.attrib["systemTimeAtCreation"].strip()
        self.timestamp_at_creation = gconf.attrib["timestampAtCreation"].strip()
        # convert to python datetime object
        # dt = datetime.datetime.fromtimestamp(int(self.system_time_at_creation) / 1000.0)

        self._sampling_rate = float(hconf.attrib["samplingRate"])
        num_ephy_channels = int(hconf.attrib["numChannels"])
        try:
            num_chan_per_chip = int(sconf.attrib["chanPerChip"])
        except KeyError:
            num_chan_per_chip = 32  # default value

        # explore sub stream and count packet size
        # first bytes is 0x55
        packet_size = 1
        device_bytes = {}
        for device in hconf:
            device_name = device.attrib["name"]
            num_bytes = int(device.attrib["numBytes"])
            device_bytes[device_name] = packet_size
            packet_size += num_bytes
        self.sysClock_byte = (
            device_bytes["SysClock"] if "SysClock" in device_bytes else False
        )

        # timestamps 4 uint32
        self._timestamp_byte = packet_size
        packet_size += 4
        assert (
            "sysTimeIncluded" not in hconf.attrib
        ), "sysTimeIncluded not supported yet"
        # if sysTimeIncluded, then 8-byte system clock is included after timestamp

        packet_size += 2 * num_ephy_channels

        # read the binary part lazily
        raw_memmap = np.memmap(self.filename, mode="r", offset=header_size, dtype="<u1")

        num_packet = raw_memmap.size // packet_size
        raw_memmap = raw_memmap[: num_packet * packet_size]
        self._raw_memmap = raw_memmap.reshape(-1, packet_size)

        # create signal channels - parallel lists
        stream_ids = []
        signal_streams = []
        signal_channels = []

        self._mask_channels_ids = {}
        self._mask_channels_bytes = {}
        self._mask_channels_bits = {}  # for digital data

        self.multiplexed_channel_xml = {}  # dictionary from id to channel xml
        if "Multiplexed" in device_bytes:
            self._multiplexed_byte_start = device_bytes["Multiplexed"]
        elif "headstageSensor" in device_bytes:
            self._multiplexed_byte_start = device_bytes["headstageSensor"]

        # walk through xml devices
        for device in hconf:
            device_name = device.attrib["name"]
            for channel in device:
                if (
                    device.attrib["name"] in ["Multiplexed", "headstageSensor"]
                    and channel.attrib["dataType"] == "analog"
                ):
                    # the multiplexed analog device has interleaved data from multiple sources
                    # that are sampled at a lower rate.
                    # for each packet,
                    # the interleavedDataIDByte and the interleavedDataIDBit indicate which
                    # channel has an updated value.
                    # the startByte contains the int16 updated value.
                    # if there was no update, use the last value received.
                    # thus, there is a value at every timestamp, but usually it will be the same
                    # as the previous value.
                    # it is assumed that for a given startByte, only one of the
                    # interleavedDataIDByte and interleavedDataIDBit combinations that
                    # use that startByte is active at any given timestamp,
                    # i.e. there should be at most one 1 in the interleavedDataIDByte value
                    # at each timestamp.

                    # the typical mask approach will not work, so store the channel specs
                    # and use them to read the analog data on demand.
                    self.multiplexed_channel_xml[channel.attrib["id"]] = channel
                    continue

                # one device can have streams with different data types,
                # so create a stream_id that differentiates them.
                # users need to be aware of this when using the API
                stream_id = device_name + "_" + channel.attrib["dataType"]

                if "interleavedDataIDByte" in channel.attrib:
                    # TODO LATER: deal with "headstageSensor" which have interleaved
                    continue

                if channel.attrib["dataType"] == "analog":
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                        self._mask_channels_ids[stream_id] = []
                        self._mask_channels_bytes[stream_id] = []
                        self._mask_channels_bits[stream_id] = []

                    name = channel.attrib["id"]
                    chan_id = channel.attrib["id"]
                    dtype = "int16"
                    # TODO LATER : handle gain correctly according the file version
                    units = ""
                    gain = 1.0
                    offset = 0.0
                    signal_channels.append(
                        (
                            name,
                            chan_id,
                            self._sampling_rate,
                            dtype,
                            units,
                            gain,
                            offset,
                            stream_id,
                        )
                    )

                    self._mask_channels_ids[stream_id].append(channel.attrib["id"])

                    num_bytes = device_bytes[device_name] + int(
                        channel.attrib["startByte"]
                    )
                    chan_mask_bytes = np.zeros(packet_size, dtype="bool")
                    chan_mask_bytes[num_bytes] = True
                    chan_mask_bytes[num_bytes + 1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask_bytes)
                    chan_mask_bits = np.zeros(packet_size * 8, dtype="bool")  # TODO
                    self._mask_channels_bits[stream_id].append(chan_mask_bits)

                elif channel.attrib["dataType"] == "digital":  # handle DIO
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                        self._mask_channels_ids[stream_id] = []
                        self._mask_channels_bytes[stream_id] = []
                        self._mask_channels_bits[stream_id] = []

                    # NOTE store data in signal_channels to make neo happy
                    name = channel.attrib["id"]
                    chan_id = channel.attrib["id"]
                    dtype = "int8"
                    units = ""
                    gain = 1.0
                    offset = 0.0

                    signal_channels.append(
                        (
                            name,
                            chan_id,
                            self._sampling_rate,
                            dtype,
                            units,
                            gain,
                            offset,
                            stream_id,
                        )
                    )

                    self._mask_channels_ids[stream_id].append(channel.attrib["id"])

                    # to handle digital data, need to split the data by bits
                    num_bytes = device_bytes[device_name] + int(
                        channel.attrib["startByte"]
                    )
                    chan_byte_mask = np.zeros(packet_size, dtype="bool")
                    chan_byte_mask[num_bytes] = True
                    self._mask_channels_bytes[stream_id].append(chan_byte_mask)

                    # within the concatenated, masked bytes, mask the bit (flipped order)
                    chan_bit_mask = np.zeros(8 * 1, dtype="bool")
                    chan_bit_mask[int(channel.attrib["bit"])] = True
                    chan_bit_mask = np.flip(chan_bit_mask)
                    self._mask_channels_bits[stream_id].append(chan_bit_mask)

                    # NOTE: _mask_channels_ids, _mask_channels_bytes, and
                    # _mask_channels_bits are parallel lists

        if num_ephy_channels > 0:
            stream_id = "trodes"
            stream_name = stream_id
            signal_streams.append((stream_name, stream_id))
            self._mask_channels_bytes[stream_id] = []

            channel_ids = self._produce_ephys_channel_ids(
                num_ephy_channels, num_chan_per_chip
            )

            chan_ind = 0
            for trode in sconf:
                for schan in trode:
                    chan_id = str(channel_ids[chan_ind])
                    name = "hwChan " + chan_id

                    # TODO LATER : handle gain correctly according the file version
                    units = ""
                    gain = 1.0
                    offset = 0.0
                    signal_channels.append(
                        (
                            name,
                            chan_id,
                            self._sampling_rate,
                            "int16",
                            units,
                            gain,
                            offset,
                            stream_id,
                        )
                    )

                    chan_mask = np.zeros(packet_size, dtype="bool")
                    num_bytes = packet_size - 2 * num_ephy_channels + 2 * chan_ind
                    chan_mask[num_bytes] = True
                    chan_mask[num_bytes + 1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask)

                    chan_ind += 1

        # make mask as array (used in _get_analogsignal_chunk(...))
        self._mask_streams = {}
        for stream_id, l in self._mask_channels_bytes.items():
            mask = np.array(l)
            self._mask_channels_bytes[stream_id] = mask
            self._mask_streams[stream_id] = np.any(mask, axis=0)

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # remove some stream if no wanted
        if self.selected_streams is not None:
            if isinstance(self.selected_streams, str):
                self.selected_streams = [self.selected_streams]
            assert isinstance(self.selected_streams, list)

            keep = np.isin(signal_streams["id"], self.selected_streams)
            signal_streams = signal_streams[keep]

            keep = np.isin(signal_channels["stream_id"], self.selected_streams)
            signal_channels = signal_channels[keep]

        # No events channels
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes  channels
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # initialize interpolate index as none so can check if it has been set in a trodes timestamps call
        self.interpolate_index = None

        self._generate_minimal_annotations()
        # info from GlobalConfiguration in xml are copied to block and seg annotations
        bl_ann = self.raw_annotations["blocks"][0]
        seg_ann = self.raw_annotations["blocks"][0]["segments"][0]
        for ann in (bl_ann, seg_ann):
            ann.update(gconf.attrib)

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        size = self._raw_memmap.shape[0]
        t_stop = size / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        if self.interpolate_dropped_packets and self.interpolate_index is None:
            raise ValueError("interpolate_index must be set before calling this")
        size = self._raw_memmap.shape[0]
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_chunk(
        self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes
    ):
        stream_id = self.header["signal_streams"][stream_index]["id"]

        raw_unit8 = self._raw_memmap[i_start:i_stop]

        num_chan = len(self._mask_channels_bytes[stream_id])
        re_order = None
        if channel_indexes is None:
            # no loop : entire stream mask
            stream_mask = self._mask_streams[stream_id]
        else:
            # accumulate mask
            if isinstance(channel_indexes, slice):
                chan_inds = np.arange(num_chan)[channel_indexes]
            else:
                chan_inds = channel_indexes

                if np.any(np.diff(channel_indexes) < 0):
                    # handle channel are not ordered
                    sorted_channel_indexes = np.sort(channel_indexes)
                    re_order = np.array(
                        [
                            list(sorted_channel_indexes).index(ch)
                            for ch in channel_indexes
                        ]
                    )

            stream_mask = np.zeros(raw_unit8.shape[1], dtype="bool")
            for chan_ind in chan_inds:
                chan_mask = self._mask_channels_bytes[stream_id][chan_ind]
                stream_mask |= chan_mask

        # this copies the data from the memmap into memory
        raw_unit8_mask = raw_unit8[:, stream_mask]
        shape = raw_unit8_mask.shape
        shape = (shape[0], shape[1] // 2)
        # reshape the and retype by view
        raw_unit16 = raw_unit8_mask.reshape(-1).view("int16").reshape(shape)

        if re_order is not None:
            raw_unit16 = raw_unit16[:, re_order]

        if stream_id == "ECU_analog":
            # automatically include the interleaved analog signals:
            analog_multiplexed_data = self.get_analogsignal_multiplexed()[
                i_start:i_stop, :
            ]
            raw_unit16 = np.concatenate((raw_unit16, analog_multiplexed_data), axis=1)

        return raw_unit16

    def get_analogsignal_timestamps(self, i_start, i_stop):
        if not self.interpolate_dropped_packets:
            # no interpolation
            raw_uint8 = self._raw_memmap[
                i_start:i_stop, self._timestamp_byte : self._timestamp_byte + 4
            ]
            raw_uint32 = (
                raw_uint8.view("uint8").reshape(-1, 4).view("uint32").reshape(-1)
            )
            return raw_uint32

        if self.interpolate_dropped_packets and self.interpolate_index is None:
            # first call in a interpolation iterator, needs to find the dropped packets
            # has to run through the entire file to find missing packets
            raw_uint8 = self._raw_memmap[
                :, self._timestamp_byte : self._timestamp_byte + 4
            ]
            raw_uint32 = (
                raw_uint8.view("uint8").reshape(-1, 4).view("uint32").reshape(-1)
            )
            self.interpolate_index = np.where(np.diff(raw_uint32) == 2)[
                0
            ]  # find locations of single dropped packets
            self._interpolate_raw_memmap()  # interpolates in the memmap

        # subsequent calls in a interpolation iterator don't remake the interpolated memmap, start here
        if i_stop is None:
            i_stop = self._raw_memmap.shape[0]
        raw_uint8 = self._raw_memmap[
            i_start:i_stop, self._timestamp_byte : self._timestamp_byte + 4
        ]
        raw_uint32 = raw_uint8.view("uint8").reshape(-1, 4).view("uint32").reshape(-1)
        # add +1 to the inserted locations
        inserted_locations = np.array(self._raw_memmap.inserted_locations) - i_start + 1
        inserted_locations = inserted_locations[
            (inserted_locations >= 0) & (inserted_locations < i_stop - i_start)
        ]
        if not len(inserted_locations) == 0:
            raw_uint32[inserted_locations] += 1
        return raw_uint32

    def get_sys_clock(self, i_start, i_stop):
        if not self.sysClock_byte:
            raise ValueError("sysClock not available")
        if i_stop is None:
            i_stop = self._raw_memmap.shape[0]
        raw_uint8 = self._raw_memmap[
            i_start:i_stop, self.sysClock_byte : self.sysClock_byte + 8
        ]
        raw_uint64 = raw_uint8.view(dtype=np.int64).reshape(-1)
        return raw_uint64

    @functools.lru_cache(maxsize=2)
    def get_analogsignal_multiplexed(self, channel_names=None) -> np.ndarray:
        print("compute multiplex cache", self.filename)
        if channel_names is None:
            # read all multiplexed channels
            channel_names = list(self.multiplexed_channel_xml.keys())
        else:
            for ch_name in channel_names:
                if ch_name not in self.multiplexed_channel_xml:
                    raise ValueError(f"Channel name '{ch_name}' not found in file.")

        # because of the encoding scheme, it is easiest to read all the data in sequence
        # one packet at a time
        num_packet = self._raw_memmap.shape[0]
        analog_multiplexed_data = np.empty(
            (num_packet, len(channel_names)), dtype=np.int16
        )

        # precompute the static data offsets
        data_offsets = np.empty((len(channel_names), 3), dtype=int)
        for j, ch_name in enumerate(channel_names):
            ch_xml = self.multiplexed_channel_xml[ch_name]
            data_offsets[j, 0] = int(
                self._multiplexed_byte_start + int(ch_xml.attrib["startByte"])
            )
            data_offsets[j, 1] = int(ch_xml.attrib["interleavedDataIDByte"])
            data_offsets[j, 2] = int(ch_xml.attrib["interleavedDataIDBit"])
        interleaved_data_id_byte_values = self._raw_memmap[:, data_offsets[:, 1]]
        interleaved_data_id_bit_values = (
            interleaved_data_id_byte_values >> data_offsets[:, 2]
        ) & 1
        # calculate which packets encode for which channel
        initialize_stream_mask = np.logical_or(
            (np.arange(num_packet) == 0)[:, None], interleaved_data_id_bit_values == 1
        )
        # read the data into int16
        data = (
            self._raw_memmap[:, data_offsets[:, 0]]
            + self._raw_memmap[:, data_offsets[:, 0] + 1] * INT_16_CONVERSION
        )
        # initialize the first row
        analog_multiplexed_data[0] = data[0]
        # for packets that do not have an update for a channel, use the previous value
        for i in range(1, num_packet):
            analog_multiplexed_data[i] = np.where(
                initialize_stream_mask[i], data[i], analog_multiplexed_data[i - 1]
            )
        return analog_multiplexed_data

    def get_analogsignal_multiplexed_partial(
        self,
        i_start: int,
        i_stop: int,
        channel_names: list = None,
        padding: int = 30000,
    ) -> np.ndarray:
        """Alternative method to access part of the multiplexed data.
        Not memory efficient for many calls because it reads a buffer chunk before the requested data.
        Better than get_analogsignal_multiplexed when need one call to specific time region

        Parameters
        ----------
        i_start : int
            index start
        i_stop : int
            index stop
        channel_names : list[str], optional
            channels to get, by default None will get all multiplex channels
        padding : int, optional
            how many packets before the desired series to load to ensure every channel receives update before requested,
            by default 30000

        Returns
        -------
        np.ndarray
            multiplex data

        Raises
        ------
        ValueError
            _description_
        """
        print("compute multiplex cache", self.filename)
        if channel_names is None:
            # read all multiplexed channels
            channel_names = list(self.multiplexed_channel_xml.keys())
        else:
            for ch_name in channel_names:
                if ch_name not in self.multiplexed_channel_xml:
                    raise ValueError(f"Channel name '{ch_name}' not found in file.")
        # determine which packets to get from data
        padding = min(padding, i_start)
        i_start = i_start - padding
        if i_stop is None:
            i_stop = self._raw_memmap.shape[0]

        # Make object to hold data
        num_packet = i_stop - i_start
        analog_multiplexed_data = np.empty(
            (num_packet, len(channel_names)), dtype=np.int16
        )

        # precompute the static data offsets
        data_offsets = np.empty((len(channel_names), 3), dtype=int)
        for j, ch_name in enumerate(channel_names):
            ch_xml = self.multiplexed_channel_xml[ch_name]
            data_offsets[j, 0] = int(
                self._multiplexed_byte_start + int(ch_xml.attrib["startByte"])
            )
            data_offsets[j, 1] = int(ch_xml.attrib["interleavedDataIDByte"])
            data_offsets[j, 2] = int(ch_xml.attrib["interleavedDataIDBit"])
        interleaved_data_id_byte_values = self._raw_memmap[
            i_start:i_stop, data_offsets[:, 1]
        ]
        interleaved_data_id_bit_values = (
            interleaved_data_id_byte_values >> data_offsets[:, 2]
        ) & 1
        # calculate which packets encode for which channel
        initialize_stream_mask = np.logical_or(
            (np.arange(num_packet) == 0)[:, None], interleaved_data_id_bit_values == 1
        )
        # read the data into int16
        data = (
            self._raw_memmap[i_start:i_stop, data_offsets[:, 0]]
            + self._raw_memmap[i_start:i_stop, data_offsets[:, 0] + 1]
            * INT_16_CONVERSION
        )
        # initialize the first row
        analog_multiplexed_data[0] = data[0]
        # for packets that do not have an update for a channel, use the previous value
        # this method assumes that every channel has an update within the buffer
        for i in range(1, num_packet):
            analog_multiplexed_data[i] = np.where(
                initialize_stream_mask[i], data[i], analog_multiplexed_data[i - 1]
            )
        return analog_multiplexed_data[padding:]

    def get_digitalsignal(self, stream_id, channel_id):
        # stream_id = self.header["signal_streams"][stream_index]["id"]

        # for now, allow only reading the entire dataset
        i_start = 0
        i_stop = None

        channel_index = -1
        for i, chan_id in enumerate(self._mask_channels_ids[stream_id]):
            if chan_id == channel_id:
                channel_index = i
                break
        assert (
            channel_index >= 0
        ), f"channel_id {channel_id} not found in stream {stream_id}"

        # num_chan = len(self._mask_channels_bytes[stream_id])
        # re_order = None
        # if channel_indexes is None:
        #     # no loop : entire stream mask
        #     stream_mask = self._mask_streams[stream_id]
        # else:
        #     # accumulate mask
        #     if isinstance(channel_indexes, slice):
        #         chan_inds = np.arange(num_chan)[channel_indexes]
        #     else:
        #         chan_inds = channel_indexes

        #         if np.any(np.diff(channel_indexes) < 0):
        #             # handle channel are not ordered
        #             sorted_channel_indexes = np.sort(channel_indexes)
        #             re_order = np.array(
        #                 [
        #                     list(sorted_channel_indexes).index(ch)
        #                     for ch in channel_indexes
        #                 ]
        #             )

        #     stream_mask = np.zeros(raw_packets.shape[1], dtype="bool")
        #     for chan_ind in chan_inds:
        #         chan_mask = self._mask_channels_bytes[stream_id][chan_ind]
        #         stream_mask |= chan_mask

        # this copies the data from the memmap into memory
        byte_mask = self._mask_channels_bytes[stream_id][channel_index]
        raw_packets_masked = self._raw_memmap[i_start:i_stop, byte_mask]

        bit_mask = self._mask_channels_bits[stream_id][channel_index]
        continuous_dio = np.unpackbits(raw_packets_masked, axis=1)[:, bit_mask].reshape(
            -1
        )
        change_dir = np.diff(continuous_dio).astype(
            np.int8
        )  # possible values: [-1, 0, 1]
        change_dir_trim = change_dir[change_dir != 0]  # keeps -1 and 1
        change_dir_trim[change_dir_trim == -1] = 0  # change -1 to 0
        # resulting array has 1 when there is a change from 0 to 1,
        # 0 when there is change from 1 to 0

        # track the timestamps when there is a change from 0 to 1 or 1 to 0
        if self.sysClock_byte:
            timestamps = self.get_regressed_systime(i_start, i_stop)
        else:
            timestamps = self.get_systime_from_trodes_timestamps(i_start, i_stop)
        dio_change_times = timestamps[np.where(change_dir)[0] + 1]

        # insert the first timestamp with the first value
        dio_change_times = np.insert(dio_change_times, 0, timestamps[0])
        change_dir_trim = np.insert(change_dir_trim, 0, continuous_dio[0])

        change_dir_trim = change_dir_trim.astype(np.uint8)

        # if re_order is not None:
        #     raw_unit16 = raw_unit16[:, re_order]

        return dio_change_times, change_dir_trim

    @functools.lru_cache(maxsize=1)
    def get_regressed_systime(self, i_start, i_stop=None):
        NANOSECONDS_PER_SECOND = 1e9
        # get values
        trodestime = self.get_analogsignal_timestamps(i_start, i_stop)
        systime_seconds = self.get_sys_clock(i_start, i_stop)
        # Convert
        trodestime_index = np.asarray(trodestime, dtype=np.float64)
        # regress
        slope, intercept, _, _, _ = linregress(trodestime_index, systime_seconds)
        adjusted_timestamps = intercept + slope * trodestime_index
        return (adjusted_timestamps) / NANOSECONDS_PER_SECOND

    @functools.lru_cache(maxsize=1)
    def get_systime_from_trodes_timestamps(self, i_start, i_stop=None):
        MILLISECONDS_PER_SECOND = 1e3
        # get values
        trodestime = self.get_analogsignal_timestamps(i_start, i_stop)
        initial_time = self.get_analogsignal_timestamps(0, 1)[0]
        return (trodestime - initial_time) * (1.0 / self._sampling_rate) + int(
            self.system_time_at_creation
        ) / MILLISECONDS_PER_SECOND

    def _interpolate_raw_memmap(
        self,
    ):
        # """Interpolates single dropped packets in the analog data."""
        print("Interpolate memmap: ", self.filename)
        self._raw_memmap = InsertedMemmap(self._raw_memmap, self.interpolate_index)


class InsertedMemmap:
    """
    class to return slices into an interpolated memmap
    Avoids loading data into memory during np.insert
    """

    def __init__(self, _raw_memmap, inserted_index=[]) -> None:
        self._raw_memmap = _raw_memmap
        self.mapped_index = np.arange(self._raw_memmap.shape[0])
        self.mapped_index = np.insert(
            self.mapped_index, inserted_index, self.mapped_index[inserted_index]
        )
        self.inserted_locations = inserted_index + np.arange(len(inserted_index))
        self.shape = (self.mapped_index.size, self._raw_memmap.shape[1])

    def __getitem__(self, index):
        # request a slice in both time and channel
        if isinstance(index, tuple):
            index_chan = index[1]
            return self._raw_memmap[self.access_coordinates(index[0]), index_chan]
        # request a slice in time
        return self._raw_memmap[self.access_coordinates(index)]

    def access_coordinates(self, index):
        if isinstance(index, int):
            return self.mapped_index[index]
        # if slice object
        elif isinstance(index, slice):
            # see if slice contains inserted values
            if (
                (
                    (not index.start is None)
                    and (not index.stop is None)
                    and np.any(
                        (self.inserted_locations >= index.start)
                        & (self.inserted_locations < index.stop)
                    )
                )
                | (
                    (index.start is None)
                    and (not index.stop is None)
                    and np.any(self.inserted_locations < index.stop)
                )
                | (
                    index.stop is None
                    and (not index.start is None)
                    and np.any(self.inserted_locations > index.start)
                )
                | (
                    index.start is None
                    and index.stop is None
                    and len(self.inserted_locations) > 0
                )
            ):
                # if so, need to use advanced indexing. return list of indeces
                return self.mapped_index[index]
            # if not, return slice object with coordinates adjusted
            else:
                return slice(
                    index.start
                    - np.searchsorted(self.inserted_locations, index.start, "right"),
                    index.stop
                    - np.searchsorted(self.inserted_locations, index.stop, "right"),
                    index.step,
                )
        # if list of indeces
        else:
            return self.mapped_index[index]
