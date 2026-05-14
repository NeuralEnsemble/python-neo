"""
Class for reading spikegadgets files.
Only continuous signals are supported at the moment.

https://spikegadgets.com/spike-products/

Documentation of the format:
https://bitbucket.org/mkarlsso/trodes/wiki/Configuration

Note :
  * this file format has multiple versions. When the SpikeConfiguration entries in the
    XML header carry a `spikeScalingToUv` attribute (newer versions), the reader uses
    it to populate per-channel `gain` and sets `units="uV"`. When the attribute is
    absent (older files), the reader falls back to `gain=1` with empty `units` and
    emits a warning that the data has no physical units.

The ".rec" file format contains two parts:

    +--------------------------------------------+
    |  XML configuration                         |
    |    GlobalConfiguration                     |
    |    HardwareConfiguration                   |
    |    SpikeConfiguration                      |
    |    ...                                     |
    |  terminated by "</Configuration>"          |
    +--------------------------------------------+
    |                                            |
    |  Binary section                            |
    |    A stream of fixed-size packets, one     |
    |    per sample tick. packet_size is         |
    |    constant for a given file, computed     |
    |    at parse time from the XML.             |
    |                                            |
    |    packet 0                                |
    |    packet 1                                |
    |    packet 2                                |
    |    ...                                     |
    +--------------------------------------------+

Each packet has the following structure (packet_size bytes total):

    +------+-----------+-----------+ ... +-----------+--------+----- ... -----+
    | 0x55 | device A  | device B  |     | device K  | tstamp | ephys region  |
    | 1 B  | numBytes  | numBytes  |     | numBytes  |  4 B   | 2*N_ephy B    |
    +------+-----------+-----------+ ... +-----------+--------+----- ... -----+
       ^         ^                                       ^           ^
       |         |                                       |           |
       packet    one block per hardware device           uint32      one int16 per
       marker    with a numBytes attribute in            sample      ephys channel;
                 HardwareConfiguration (e.g.             tick        layout of this
                 Controller_DIO, Multiplexed,                        region depends
                 SysClock, ECU)                                      on the device,
                                                                     see below.

The ephys region is laid out differently depending on the acquisition hardware,
which is declared by SpikeConfiguration.device:

  * Intan recordings (`device="intan"` or absent on legacy files):
    chip-interleaved order. The SpikeGadgets MCU (main control unit) clocks all
    attached Intan chips in parallel over SPI (serial peripheral interface), so
    samples are written in the sequence
    [chip 0 ch 0, chip 1 ch 0, ..., chip N-1 ch 0,
     chip 0 ch 1, chip 1 ch 1, ..., chip N-1 ch 1, ...].

    Example with chanPerChip=32 and 4 chips, every packet's ephys region:

        slot     0    1    2    3  |  4    5    6    7  | ... | 124  125  126  127
        hwChan   0   32   64   96  |  1   33   65   97  | ... |  31   63   95  127

    Formula: hwChan at slot i = (i % n_chips) * chanPerChip + (i // n_chips).
    The XML's <SpikeChannel> hwChan attributes are *not* in slot order; they are
    listed in user-defined SpikeNTrode (tetrode) groupings, so the reader cannot
    use XML document position as a proxy and must reproduce the chip-interleaved
    sequence directly (see `_intan_hwchans_in_binary_order`).

  * Neuropixels recordings (`device="neuropixels1"` or `"neuropixels2"`):
    hwChan ascending order. The SpikeGadgets MCU (main control unit) firmware
    emits Neuropixels samples in hwChan ascending order: byte pair i of each
    packet holds the sample from the electrode whose hwChan = i. The XML's
    <SpikeChannel> elements may be listed in any order Trodes wrote them
    (typically not hwChan-ascending), but that order does not constrain the
    binary; the binary always follows hwChan ordering.

    Example, every packet's ephys region:

        slot     0    1    2    3    4    5  ... 380  381  382  383
        hwChan   0    1    2    3    4    5  ... 380  381  382  383

    The reader walks the XML to recover per-trode metadata (gain, anatomical
    coordinates) for each hwChan, but the column-to-byte-position mapping is
    the identity: column i reads byte pair i, with hwChan i's data.

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

    def _intan_hwchans_in_binary_order(self, sconf, num_ephy_channels, num_ephy_channels_xml):
        """Return the hwChan-at-each-byte-position sequence for an Intan recording.

        The SpikeGadgets MCU (main control unit) multiplexes Intan chips in chip-interleaved
        order: local-channel outer, chip inner. Every chip contributes its local channel 0
        first, then every chip contributes its local channel 1, and so on. So byte pair i of
        each packet holds the sample for the hwChan returned at index i.

        Example: chanPerChip=32 and 4 chips (128 channels total). Returns
        [0, 32, 64, 96, 1, 33, 65, 97, ..., 31, 63, 95, 127].

        If the channel count does not divide evenly into chanPerChip we fall back to XML
        document order; in that case the chip-interleaved bridging assumption does not
        apply and labels reflect the XML's hwChan attributes directly.

        See also:
            - https://github.com/NeuralEnsemble/python-neo/issues/1215 (origin of this logic)
            - https://github.com/NeuralEnsemble/python-neo/issues/1517 (doesn't fit all setups)
            - https://github.com/NeuralEnsemble/python-neo/issues/1592 (missing channels)
        """
        intan_chans_per_chip = int(sconf.attrib.get("chanPerChip", 32))  # RHD2132 default for legacy files
        hw_chans_in_xml = [int(schan.attrib["hwChan"]) for trode in sconf for schan in trode]

        channels_fit_chip_layout = (
            intan_chans_per_chip > 0
            and num_ephy_channels % intan_chans_per_chip == 0
        )
        if not channels_fit_chip_layout:
            return hw_chans_in_xml

        # Reproduce the chip-interleaved hwChan sequence (local-channel outer, chip inner)
        # so that hwchans_in_binary_order[i] is the hwChan whose data lives at byte pair i.
        # Any hwChans absent from the SpikeConfiguration are skipped.
        missing_hw_chans = set(range(num_ephy_channels)) - set(hw_chans_in_xml)
        n_chips = num_ephy_channels_xml // intan_chans_per_chip
        hwchans_in_binary_order = []
        for local_hw_channel in range(intan_chans_per_chip):
            for chip in range(n_chips):
                hw_chan = local_hw_channel + chip * intan_chans_per_chip
                if hw_chan in missing_hw_chans:
                    continue
                hwchans_in_binary_order.append(hw_chan)
        return hwchans_in_binary_order

    def _parse_header(self):
        # parse file until "</Configuration>"
        with open(self.filename, mode="rb") as f:
            for line in f:
                if b"</Configuration>" in line:
                    header_size = f.tell()
                    break
            else:
                raise ValueError("SpikeGadgets: the xml header does not contain '</Configuration>'")

            f.seek(0)
            header_txt = f.read(header_size).decode("utf8")

        # explore xml header
        root = ElementTree.fromstring(header_txt)
        gconf = root.find("GlobalConfiguration")
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
            signal_streams.append((stream_id, stream_id, ""))
            self._mask_channels_bytes[stream_id] = []

            self.is_scaleable = all("spikeScalingToUv" in trode.attrib for trode in sconf)
            if not self.is_scaleable:
                self.logger.warning(
                    "Unable to read channel gain scaling (to uV) from .rec header. Data has no physical units!"
                )

            # Compute the hwChan-at-each-byte-position sequence based on the hardware kind
            # declared in SpikeConfiguration.device.
            # - Intan recordings are chip-interleaved by the MCU (main control unit), so we
            #   reproduce that ordering from the chip layout.
            # - Neuropixels recordings are emitted in hwChan ascending order: byte pair i
            #   holds the sample from the electrode whose hwChan = i.
            spike_device = sconf.attrib.get("device")
            if spike_device in (None, "intan"):
                hwchans_in_binary_order = self._intan_hwchans_in_binary_order(
                    sconf, num_ephy_channels, num_ephy_channels_xml
                )
            elif spike_device in ("neuropixels1", "neuropixels2"):
                xml_hwchans = {int(schan.attrib["hwChan"]) for trode in sconf for schan in trode}
                expected = set(range(num_ephy_channels))
                if xml_hwchans != expected:
                    raise NeoReadWriteError(
                        "SpikeGadgets Neuropixels: hwChan values in SpikeConfiguration do not "
                        f"cover [0, {num_ephy_channels}). The reader assumes the firmware emits "
                        "samples in hwChan ascending order over a contiguous range; this file "
                        f"has missing or out-of-range hwChans (missing: {sorted(expected - xml_hwchans)[:5]}..., "
                        f"extra: {sorted(xml_hwchans - expected)[:5]}...)."
                    )
                hwchans_in_binary_order = list(range(num_ephy_channels))
            else:
                raise NeoReadWriteError(
                    f"SpikeGadgets: unsupported SpikeConfiguration.device {spike_device!r}. "
                    "Add a dedicated branch for this hardware in SpikeGadgetsRawIO._parse_header."
                )

            # Walk binary order, using hwChan as the explicit bridge to per-trode metadata.
            trode_by_hwchan = {int(schan.attrib["hwChan"]): trode for trode in sconf for schan in trode}
            for binary_index, hw_chan in enumerate(hwchans_in_binary_order):
                parent_trode = trode_by_hwchan[hw_chan]
                if "spikeScalingToUv" in parent_trode.attrib:
                    gain = float(parent_trode.attrib["spikeScalingToUv"])
                    units = "uV"
                else:
                    gain = 1.0
                    units = ""

                chan_id = str(hw_chan)
                name = f"trode{parent_trode.attrib['id']}chan{hw_chan}"
                signal_channels.append(
                    (name, chan_id, self._sampling_rate, "int16", units, gain, 0.0, stream_id, "")
                )

                num_bytes = packet_size - 2 * num_ephy_channels + 2 * binary_index
                chan_mask = np.zeros(packet_size, dtype="bool")
                chan_mask[num_bytes] = True
                chan_mask[num_bytes + 1] = True
                self._mask_channels_bytes[stream_id].append(chan_mask)

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
