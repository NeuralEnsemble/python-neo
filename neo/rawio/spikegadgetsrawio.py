"""
Class for reading spikegadgets files.
Only continuous signals are supported at the moment.

https://spikegadgets.com/spike-products/

Documentation of the format:
https://bitbucket.org/mkarlsso/trodes/wiki/Configuration

Note :
  * this file format have multiple version. news version include the gain for scaling.
     The actual implementation does not contain this feature because we don't have
     files to test this. So now the gain is "hardcoded" to 1. and so units are
      not handled correctly.

The ".rec" file format contains:
  * a first text part with information in an XML structure
  * a second part for the binary buffer

Author: Samuel Garcia
"""

import numpy as np

from xml.etree import ElementTree

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from neo.core import NeoReadWriteError


class SpikeGadgetsRawIO(BaseRawIO):
    extensions = ["rec"]
    rawmode = "one-file"

    def __init__(self, filename="", selected_streams=None):
        """
        Class for reading spikegadgets files.
        Only continuous signals are supported at the moment.

        Initialize a SpikeGadgetsRawIO for a single ".rec" file.

        Parameters
        ----------
        filename: str, default: ''
            The *.rec file to be loaded
        selected_streams: str | list | None, default: None
            sublist of streams to load/expose to API, e.g., ['ECU', 'trodes']
            'trodes' is name for ephy channel (ntrodes)
            None will keep all streams

        Notes
        -----
        This file format has multiple versions:
            - Newer versions include the gain for scaling to microvolts [uV].
            - If the scaling is not found in the header, the gain will be "hardcoded" to 1,
              in which case the units are not handled correctly.
        This will not affect functions that do not rely on the data having physical units,
            e.g., _get_analogsignal_chunk, but functions such as rescale_signal_raw_to_float
            will be inaccurate.

        Examples
        --------
        >>> import neo.rawio
        >>> reader = neo.rawio.SpikeGadgetRawIO(filename='data.rec') # all streams
        # just the electrode channels
        >>> reader_trodes = neo.rawio.SpikeGadgetRawIO(filename='data.rec', selected_streams='trodes')


        """
        BaseRawIO.__init__(self)
        self.filename = filename
        self.selected_streams = selected_streams

    def _source_name(self):
        return self.filename

    def _produce_ephys_channel_ids(self, n_total_channels, n_channels_per_chip, missing_hw_chans):
        """Compute the channel ID labels for subset of spikegadgets recordings
        The ephys channels in the .rec file are stored in the following order:
        hwChan ID of channel 0 of first chip, hwChan ID of channel 0 of second chip, ..., hwChan ID of channel 0 of Nth chip,
        hwChan ID of channel 1 of first chip, hwChan ID of channel 1 of second chip, ..., hwChan ID of channel 1 of Nth chip,
        ...
        So if there are 32 channels per chip and 128 channels (4 chips), then the channel IDs are:
        0, 32, 64, 96, 1, 33, 65, 97, ..., 128
        See also: https://github.com/NeuralEnsemble/python-neo/issues/1215

        This doesn't work for all types of spikegadgets
        see: https://github.com/NeuralEnsemble/python-neo/issues/1517

        If there are any missing hardware channels, they must be specified in missing_hw_chans.
        See: https://github.com/NeuralEnsemble/python-neo/issues/1592
        """
        ephys_channel_ids_list = []
        for local_hw_channel in range(n_channels_per_chip):
            n_chips = int(n_total_channels / n_channels_per_chip)
            for chip in range(n_chips):
                global_hw_chan = local_hw_channel + chip * n_channels_per_chip
                if global_hw_chan in missing_hw_chans:
                    continue
                ephys_channel_ids_list.append(local_hw_channel + chip * n_channels_per_chip)
        return ephys_channel_ids_list

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
                ValueError("SpikeGadgets: the xml header does not contain '</Configuration>'")

            f.seek(0)
            header_txt = f.read(header_size).decode("utf8")

        # explore xml header
        root = ElementTree.fromstring(header_txt)
        gconf = sr = root.find("GlobalConfiguration")
        hconf = root.find("HardwareConfiguration")
        sconf = root.find("SpikeConfiguration")

        self._sampling_rate = float(hconf.attrib["samplingRate"])
        num_ephy_channels_xml = int(hconf.attrib["numChannels"])
        num_ephy_channels = num_ephy_channels_xml

        # check for agreement with number of channels in xml
        sconf_channels = np.sum([len(x) for x in sconf])
        if sconf_channels < num_ephy_channels:
            num_ephy_channels = sconf_channels
        if sconf_channels > num_ephy_channels:
            raise NeoReadWriteError(
                "SpikeGadgets: the number of channels in the spike configuration is larger "
                "than the number of channels in the hardware configuration"
            )

        # as spikegadgets change we should follow this
        try:
            num_chan_per_chip = int(sconf.attrib["chanPerChip"])
        except KeyError:
            num_chan_per_chip = 32  # default value for Intan chips

        # explore sub stream and count packet size
        # first bytes is 0x55
        packet_size = 1
        stream_bytes = {}
        for device in hconf:
            stream_id = device.attrib["name"]
            if "numBytes" in device.attrib.keys():
                num_bytes = int(device.attrib["numBytes"])
                stream_bytes[stream_id] = packet_size
                packet_size += num_bytes

        # timestamps 4 uint32
        self._timestamp_byte = packet_size
        packet_size += 4

        packet_size += 2 * num_ephy_channels

        # read the binary part lazily
        raw_memmap = np.memmap(self.filename, mode="r", offset=header_size, dtype="<u1")

        num_packet = raw_memmap.size // packet_size
        raw_memmap = raw_memmap[: num_packet * packet_size]
        self._raw_memmap = raw_memmap.reshape(-1, packet_size)

        # create signal channels
        stream_ids = []
        signal_streams = []
        signal_channels = []

        # walk in xml device and keep only "analog" one
        self._mask_channels_bytes = {}
        for device in hconf:
            stream_id = device.attrib["name"]
            for channel in device:

                if "interleavedDataIDByte" in channel.attrib:
                    # TODO LATER: deal with "headstageSensor" which have interleaved
                    continue

                if ("dataType" in channel.attrib.keys()) and (channel.attrib["dataType"] == "analog"):

                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        buffer_id = ""
                        signal_streams.append((stream_name, stream_id, buffer_id))
                        self._mask_channels_bytes[stream_id] = []

                    name = channel.attrib["id"]
                    chan_id = channel.attrib["id"]
                    dtype = "int16"
                    # TODO LATER : handle gain correctly according the file version
                    units = ""
                    gain = 1.0
                    offset = 0.0
                    buffer_id = ""
                    signal_channels.append(
                        (name, chan_id, self._sampling_rate, "int16", units, gain, offset, stream_id, buffer_id)
                    )

                    num_bytes = stream_bytes[stream_id] + int(channel.attrib["startByte"])
                    chan_mask = np.zeros(packet_size, dtype="bool")
                    chan_mask[num_bytes] = True
                    chan_mask[num_bytes + 1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask)

        if num_ephy_channels > 0:
            stream_id = "trodes"
            stream_name = stream_id
            buffer_id = ""
            signal_streams.append((stream_name, stream_id, buffer_id))
            self._mask_channels_bytes[stream_id] = []

            # we can only produce these channels for a subset of spikegadgets setup. If this criteria isn't
            # true then we should just use the raw_channel_ids and let the end user sort everything out
            if num_ephy_channels % num_chan_per_chip == 0:
                all_hw_chans = [int(schan.attrib["hwChan"]) for trode in sconf for schan in trode]
                missing_hw_chans = set(range(num_ephy_channels)) - set(all_hw_chans)
                channel_ids = self._produce_ephys_channel_ids(
                    num_ephy_channels_xml, num_chan_per_chip, missing_hw_chans
                )
                raw_channel_ids = False
            else:
                raw_channel_ids = True

            chan_ind = 0
            self.is_scaleable = all("spikeScalingToUv" in trode.attrib for trode in sconf)
            if not self.is_scaleable:
                self.logger.warning(
                    "Unable to read channel gain scaling (to uV) from .rec header. Data has no physical units!"
                )

            for trode in sconf:
                if "spikeScalingToUv" in trode.attrib:
                    gain = float(trode.attrib["spikeScalingToUv"])
                    units = "uV"
                else:
                    gain = 1  # revert to hardcoded gain
                    units = ""

                for schan in trode:
                    # Here we use raw ids if necessary for parsing (for some neuropixel recordings)
                    # otherwise we default back to the raw hwChan IDs
                    if raw_channel_ids:
                        name = "trode" + trode.attrib["id"] + "chan" + schan.attrib["hwChan"]
                        chan_id = schan.attrib["hwChan"]
                    else:
                        chan_id = str(channel_ids[chan_ind])
                        name = "hwChan" + chan_id

                    offset = 0.0
                    buffer_id = ""
                    signal_channels.append(
                        (name, chan_id, self._sampling_rate, "int16", units, gain, offset, stream_id, buffer_id)
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

        # no buffer concept here data are too fragmented
        signal_buffers = np.array([], dtype=_signal_buffer_dtype)

        # remove some stream if not wanted
        if self.selected_streams is not None:
            if isinstance(self.selected_streams, str):
                self.selected_streams = [self.selected_streams]
            if not isinstance(self.selected_streams, list):
                raise TypeError(
                    f"`selected_streams` must be of type str or list not of type {type(self.selected_streams)}"
                )

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
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

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
        size = self._raw_memmap.shape[0]
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
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
                    re_order = np.array([list(sorted_channel_indexes).index(ch) for ch in channel_indexes])

            stream_mask = np.zeros(raw_unit8.shape[1], dtype="bool")
            for chan_ind in chan_inds:
                chan_mask = self._mask_channels_bytes[stream_id][chan_ind]
                stream_mask |= chan_mask

        # this copies the data from the memmap into memory
        raw_unit8_mask = raw_unit8[:, stream_mask]
        shape = raw_unit8_mask.shape
        shape = (shape[0], shape[1] // 2)
        # reshape the and retype by view
        raw_unit16 = raw_unit8_mask.flatten().view("int16").reshape(shape)

        if re_order is not None:
            raw_unit16 = raw_unit16[:, re_order]

        return raw_unit16
