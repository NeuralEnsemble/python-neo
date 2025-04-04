"""
Class for reading the old data format from Plexon
acquisition system (.plx)

Note that Plexon now use a new format PL2 which is NOT
supported by this IO.

Compatible with versions 100 to 106.
Other versions have not been tested.

This IO is developed thanks to the header file downloadable from:
http://www.plexon.com/software-downloads

This IO was rewritten in 2017 and this was a huge pain because
the underlying file format is really inefficient.
The rewrite is now based on numpy dtype and not on Python struct.
This should be faster.
If one day, somebody use it, consider to offer me a beer.


Author: Samuel Garcia

"""

import datetime
from collections import OrderedDict
import re

import numpy as np

try:
    from tqdm import tqdm, trange

    HAVE_TQDM = True
except:
    HAVE_TQDM = False

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)

from neo.core.baseneo import NeoReadWriteError


class PlexonRawIO(BaseRawIO):
    extensions = ["plx"]
    rawmode = "one-file"

    def __init__(self, filename="", progress_bar=True):
        """

        Class for reading non-pl2 plexon files

        Parameters
        ----------
        filename: str, default: ''
            The *.plx file to be loaded
        progress_bar: bool, default True
            Display progress bar using tqdm (if installed) when parsing the file.

        Notes
        -----
        * Compatible with versions 100 to 106. Other versions have not been tested.
        * Note that Plexon now use a new format PL2 which is NOT supported by this IO.

        Examples
        --------
        >>> import neo.rawio
        >>> r = neo.rawio.PlexonRawIO(filename='data.plx')
        >>> r.parse_header()
        >>> print(r)

        """
        BaseRawIO.__init__(self)
        self.filename = filename
        self.progress_bar = HAVE_TQDM and progress_bar

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        # global header
        with open(self.filename, "rb") as fid:
            global_header = read_as_dict(fid, GlobalHeader)

        rec_datetime = datetime.datetime(
            global_header["Year"],
            global_header["Month"],
            global_header["Day"],
            global_header["Hour"],
            global_header["Minute"],
            global_header["Second"],
        )

        # dsp channels header = spikes and waveforms
        nb_unit_chan = int(global_header["NumDSPChannels"])
        offset1 = np.dtype(GlobalHeader).itemsize
        dspChannelHeaders = np.memmap(
            self.filename, dtype=DspChannelHeader, mode="r", offset=offset1, shape=(nb_unit_chan,)
        )

        # event channel header
        nb_event_chan = int(global_header["NumEventChannels"])
        offset2 = offset1 + np.dtype(DspChannelHeader).itemsize * nb_unit_chan
        eventHeaders = np.memmap(
            self.filename,
            dtype=EventChannelHeader,
            mode="r",
            offset=offset2,
            shape=(nb_event_chan,),
        )

        # slow channel header = signal
        nb_sig_chan = int(global_header["NumSlowChannels"])
        offset3 = offset2 + np.dtype(EventChannelHeader).itemsize * nb_event_chan
        slowChannelHeaders = np.memmap(
            self.filename, dtype=SlowChannelHeader, mode="r", offset=offset3, shape=(nb_sig_chan,)
        )

        offset4 = offset3 + np.dtype(SlowChannelHeader).itemsize * nb_sig_chan

        # locate data blocks and group them by type and channel
        block_pos = {
            1: {c: [] for c in dspChannelHeaders["Channel"]},
            4: {c: [] for c in eventHeaders["Channel"]},
            5: {c: [] for c in slowChannelHeaders["Channel"]},
        }
        data = self._memmap = np.memmap(self.filename, dtype="u1", offset=0, mode="r")
        pos = offset4

        # Create a tqdm object with a total of len(data) and an initial value of 0 for offset
        if self.progress_bar:
            progress_bar = tqdm(total=len(data), initial=0, desc="Parsing data blocks", leave=True)

        while pos < data.size:
            bl_header = data[pos : pos + 16].view(DataBlockHeader)[0]
            number_of_waveforms = int(bl_header["NumberOfWaveforms"])
            number_of_words_in_waveform = int(bl_header["NumberOfWordsInWaveform"])
            length = (number_of_waveforms * number_of_words_in_waveform * 2) + 16
            bl_type = int(bl_header["Type"])
            chan_id = int(bl_header["Channel"])
            block_pos[bl_type][chan_id].append(pos)
            pos += length

            # Update tqdm with the number of bytes processed in this iteration
            if self.progress_bar:
                progress_bar.update(length)  # This was clever, Sam : )

        if self.progress_bar:
            progress_bar.close()

        upper_byte_of_5_byte_timestamp = int(bl_header["UpperByteOf5ByteTimestamp"])
        bl_header_timestamp = int(bl_header["TimeStamp"])
        self._last_timestamps = upper_byte_of_5_byte_timestamp * 2**32 + bl_header_timestamp

        # ... and finalize them in self._data_blocks
        # for a faster access depending on type (1, 4, 5)
        self._data_blocks = {}
        dt_base = [("pos", "int64"), ("timestamp", "int64"), ("size", "int64")]
        dtype_by_bltype = {
            # Spikes and waveforms
            1: np.dtype(
                dt_base
                + [
                    ("unit_id", "uint16"),
                    ("n1", "uint16"),
                    ("n2", "uint16"),
                ]
            ),
            # Events
            4: np.dtype(
                dt_base
                + [
                    ("label", "uint16"),
                ]
            ),
            # Signals
            5: np.dtype(
                dt_base
                + [
                    ("cumsum", "int64"),
                ]
            ),
        }
        if self.progress_bar:
            bl_loop = tqdm(block_pos, desc="Finalizing data blocks", leave=True)
        else:
            bl_loop = block_pos
        for bl_type in bl_loop:
            self._data_blocks[bl_type] = {}
            if self.progress_bar:
                chan_loop = tqdm(
                    block_pos[bl_type],
                    desc=f"Finalizing data blocks for type {bl_type}",
                    leave=True,
                )
            else:
                chan_loop = block_pos[bl_type]
            for chan_id in chan_loop:
                positions = block_pos[bl_type][chan_id]
                dt = dtype_by_bltype[bl_type]
                data_block = np.empty((len(positions)), dtype=dt)
                for index, pos in enumerate(positions):
                    bl_header = data[pos : pos + 16].view(DataBlockHeader)[0]

                    # To avoid overflow errors when doing arithmetic operations on numpy scalars
                    np_scalar_to_python_scalar = lambda x: x.item() if isinstance(x, np.generic) else x
                    bl_header = {key: np_scalar_to_python_scalar(bl_header[key]) for key in bl_header.dtype.names}

                    current_upper_byte_of_5_byte_timestamp = int(bl_header["UpperByteOf5ByteTimestamp"])
                    current_bl_timestamp = int(bl_header["TimeStamp"])
                    timestamp = current_upper_byte_of_5_byte_timestamp * 2**32 + current_bl_timestamp
                    n1 = bl_header["NumberOfWaveforms"]
                    n2 = bl_header["NumberOfWordsInWaveform"]
                    sample_count = n1 * n2

                    data_block["pos"][index] = pos + 16
                    data_block["timestamp"][index] = timestamp
                    data_block["size"][index] = sample_count * 2

                    if bl_type == 1:  # Spikes and waveforms
                        data_block["unit_id"][index] = bl_header["Unit"]
                        data_block["n1"][index] = n1
                        data_block["n2"][index] = n2
                    elif bl_type == 4:  # Events
                        data_block["label"][index] = bl_header["Unit"]
                    elif bl_type == 5:  # Signals
                        if data_block.size > 0:
                            # cumulative sum of sample index for fast access to chunks
                            if index == 0:
                                data_block["cumsum"][index] = 0
                            else:
                                data_block["cumsum"][index] = data_block["cumsum"][index - 1] + sample_count

                self._data_blocks[bl_type][chan_id] = data_block

        # signals channels
        source_id = []

        # Scanning sources and populating signal channels at the same time. Sources have to have
        # same sampling rate and number of samples to belong to one stream.
        signal_channels = []
        channel_num_samples = []

        # We will build the stream ids based on the channel prefixes
        # The channel prefixes are the first characters of the channel names which have the following format:
        # WB{number}, FPX{number}, SPKCX{number}, AI{number}, etc
        # We will extract the prefix and use it as stream id
        regex_prefix_pattern = r"^\D+"  # Match any non-digit character at the beginning of the string

        if self.progress_bar:
            chan_loop = trange(nb_sig_chan, desc="Parsing signal channels", leave=True)
        else:
            chan_loop = range(nb_sig_chan)
        for chan_index in chan_loop:
            slow_channel_headers = slowChannelHeaders[chan_index]

            # To avoid overflow errors when doing arithmetic operations on numpy scalars
            np_scalar_to_python_scalar = lambda x: x.item() if isinstance(x, np.generic) else x
            slow_channel_headers = {
                key: np_scalar_to_python_scalar(slow_channel_headers[key]) for key in slow_channel_headers.dtype.names
            }

            name = slow_channel_headers["Name"].decode("utf8")
            chan_id = slow_channel_headers["Channel"]
            length = self._data_blocks[5][chan_id]["size"].sum() // 2
            if length == 0:
                continue  # channel not added
            source_id.append(slow_channel_headers["SrcId"])
            channel_num_samples.append(length)
            sampling_rate = float(slow_channel_headers["ADFreq"])
            sig_dtype = "int16"
            units = ""  # I don't know units
            if global_header["Version"] in [100, 101]:
                gain = 5000.0 / (2048 * slow_channel_headers["Gain"] * 1000.0)
            elif global_header["Version"] in [102]:
                gain = 5000.0 / (2048 * slow_channel_headers["Gain"] * slow_channel_headers["PreampGain"])
            elif global_header["Version"] >= 103:
                gain = global_header["SlowMaxMagnitudeMV"] / (
                    0.5
                    * (2 ** global_header["BitsPerSpikeSample"])
                    * slow_channel_headers["Gain"]
                    * slow_channel_headers["PreampGain"]
                )
            offset = 0.0

            channel_prefix = re.match(regex_prefix_pattern, name).group(0)
            stream_id = channel_prefix
            buffer_id = ""
            signal_channels.append(
                (name, str(chan_id), sampling_rate, sig_dtype, units, gain, offset, stream_id, buffer_id)
            )

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # no buffer here because block are splitted by channel
        signal_buffers = np.array([], dtype=_signal_buffer_dtype)

        if signal_channels.size == 0:
            signal_streams = np.array([], dtype=_signal_stream_dtype)

        else:
            # Detect streams
            channel_num_samples = np.asarray(channel_num_samples)
            # We are using channel prefixes as stream_ids
            # The meaning of the channel prefixes was provided by a Plexon Engineer, see here:
            # https://github.com/NeuralEnsemble/python-neo/pull/1495#issuecomment-2184256894
            stream_id_to_stream_name = {
                "WB": "WB-Wideband",
                "FP": "FPl-Low Pass Filtered",
                "SP": "SPKC-High Pass Filtered",
                "AI": "AI-Auxiliary Input",
                "AIF": "AIF-Auxiliary Input Filtered",
            }

            unique_stream_ids = np.unique(signal_channels["stream_id"])
            signal_streams = []
            for stream_id in unique_stream_ids:
                # We are using the channel prefixes as ids
                # The users of plexon can modify the prefix of the channel names (e.g. `my_prefix` instead of `WB`).
                # In that case we use the channel prefix both as stream id and name
                buffer_id = ""
                stream_name = stream_id_to_stream_name.get(stream_id, stream_id)
                buffer_id = ""
                signal_streams.append((stream_name, stream_id, buffer_id))

            signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

            self._stream_id_samples = {}
            self._stream_id_sampling_frequency = {}
            self._stream_index_to_stream_id = {}
            for stream_index, stream_id in enumerate(signal_streams["id"]):
                # Keep a mapping from stream_index to stream_id
                self._stream_index_to_stream_id[stream_index] = stream_id

                mask = signal_channels["stream_id"] == stream_id

                signal_num_samples = np.unique(channel_num_samples[mask])
                if signal_num_samples.size > 1:
                    raise NeoReadWriteError(f"Channels in stream {stream_id} don't have the same number of samples")
                self._stream_id_samples[stream_id] = signal_num_samples[0]

                signal_sampling_frequency = np.unique(signal_channels[mask]["sampling_rate"])
                if signal_sampling_frequency.size > 1:
                    raise NeoReadWriteError(f"Channels in stream {stream_id} don't have the same sampling frequency")
                self._stream_id_sampling_frequency[stream_id] = signal_sampling_frequency[0]

        self._global_ssampling_rate = global_header["ADFrequency"]

        # Determine number of units per channels
        self.internal_unit_ids = []
        for chan_id, data_clock in self._data_blocks[1].items():
            unit_ids = np.unique(data_clock["unit_id"])
            for unit_id in unit_ids:
                self.internal_unit_ids.append((chan_id, unit_id))

        # Spikes channels
        spike_channels = []
        if self.progress_bar:
            unit_loop = tqdm(
                enumerate(self.internal_unit_ids),
                desc="Parsing spike channels",
                leave=True,
            )
        else:
            unit_loop = enumerate(self.internal_unit_ids)

        for unit_index, (chan_id, unit_id) in unit_loop:
            channel_index = np.nonzero(dspChannelHeaders["Channel"] == chan_id)[0][0]
            dsp_channel_headers = dspChannelHeaders[channel_index]

            name = dsp_channel_headers["Name"].decode("utf8")
            _id = f"ch{chan_id}#{unit_id}"
            wf_units = ""
            if global_header["Version"] < 103:
                wf_gain = 3000.0 / (2048 * dsp_channel_headers["Gain"] * 1000.0)
            elif 103 <= global_header["Version"] < 105:
                wf_gain = global_header["SpikeMaxMagnitudeMV"] / (
                    0.5 * 2.0 ** (global_header["BitsPerSpikeSample"]) * dsp_channel_headers["Gain"] * 1000.0
                )
            elif global_header["Version"] >= 105:
                wf_gain = global_header["SpikeMaxMagnitudeMV"] / (
                    0.5
                    * 2.0 ** (global_header["BitsPerSpikeSample"])
                    * dsp_channel_headers["Gain"]
                    * global_header["SpikePreAmpGain"]
                )
            wf_offset = 0.0
            wf_left_sweep = -1  # DONT KNOWN
            wf_sampling_rate = global_header["WaveformFreq"]
            spike_channels.append((name, _id, wf_units, wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate))
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # Event channels
        event_channels = []
        if self.progress_bar:
            chan_loop = trange(nb_event_chan, desc="Parsing event channels", leave=True)
        else:
            chan_loop = range(nb_event_chan)
        for chan_index in chan_loop:
            h = eventHeaders[chan_index]
            chan_id = h["Channel"]
            name = h["Name"].decode("utf8")
            event_channels.append((name, chan_id, "event"))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fill into header dict
        self.header = {
            "nb_block": 1,
            "nb_segment": [1],
            "signal_buffers": signal_buffers,
            "signal_streams": signal_streams,
            "signal_channels": signal_channels,
            "spike_channels": spike_channels,
            "event_channels": event_channels,
        }

        # Annotations
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations["blocks"][0]
        seg_annotations = bl_annotations["segments"][0]
        for d in (bl_annotations, seg_annotations):
            d["rec_datetime"] = rec_datetime
            d["plexon_version"] = global_header["Version"]

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = float(self._last_timestamps) / self._global_ssampling_rate
        if hasattr(self, "__stream_id_samples"):
            for stream_id in self._stream_id_samples.keys():
                t_stop_sig = self._stream_id_samples[stream_id] / self._stream_id_sampling_frequency[stream_id]
                t_stop = max(t_stop, t_stop_sig)
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self._stream_index_to_stream_id[stream_index]
        return self._stream_id_samples[stream_id]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        signal_channels = self.header["signal_channels"]
        signal_streams = self.header["signal_streams"]
        stream_id = signal_streams[stream_index]["id"]

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._stream_id_samples[stream_id]

        mask = signal_channels["stream_id"] == stream_id
        signal_channels = signal_channels[mask]
        if channel_indexes is not None:
            signal_channels = signal_channels[channel_indexes]
        channel_ids = signal_channels["id"]

        raw_signals = np.zeros((i_stop - i_start, channel_ids.size), dtype="int16")
        for c, channel_id in enumerate(channel_ids):
            chan_id = np.int32(channel_id)

            data_blocks = self._data_blocks[5][chan_id]

            # loop over data blocks and get chunks
            bl0 = np.searchsorted(data_blocks["cumsum"], i_start, side="right") - 1
            bl1 = np.searchsorted(data_blocks["cumsum"], i_stop, side="right")
            ind = 0
            for bl in range(bl0, bl1):
                ind0 = data_blocks[bl]["pos"]
                ind1 = data_blocks[bl]["size"] + ind0
                data = self._memmap[ind0:ind1].view("int16")
                if bl == bl1 - 1:
                    # right border
                    # be careful that bl could be both bl0 and bl1!!
                    border = data.size - (i_stop - data_blocks[bl]["cumsum"])
                    data = data[:-border]
                if bl == bl0:
                    # left border
                    border = i_start - data_blocks[bl]["cumsum"]
                    data = data[border:]
                raw_signals[ind : data.size + ind, c] = data
                ind += data.size

        return raw_signals

    def _get_internal_mask(self, data_block, t_start, t_stop):
        timestamps = data_block["timestamp"]

        if t_start is None:
            lim0 = 0
        else:
            lim0 = int(t_start * self._global_ssampling_rate)

        if t_stop is None:
            lim1 = self._last_timestamps
        else:
            lim1 = int(t_stop * self._global_ssampling_rate)

        keep = (timestamps >= lim0) & (timestamps <= lim1)

        return keep

    def _spike_count(self, block_index, seg_index, unit_index):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data_block = self._data_blocks[1][chan_id]
        nb_spike = np.sum(data_block["unit_id"] == unit_id)
        return nb_spike

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data_block = self._data_blocks[1][chan_id]

        keep = self._get_internal_mask(data_block, t_start, t_stop)
        keep &= data_block["unit_id"] == unit_id
        spike_timestamps = data_block[keep]["timestamp"]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._global_ssampling_rate
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data_block = self._data_blocks[1][chan_id]

        n1 = data_block["n1"][0]
        n2 = data_block["n2"][0]

        keep = self._get_internal_mask(data_block, t_start, t_stop)
        keep &= data_block["unit_id"] == unit_id

        data_block = data_block[keep]
        nb_spike = data_block.size

        waveforms = np.zeros((nb_spike, n1, n2), dtype="int16")
        for i, db in enumerate(data_block):
            ind0 = db["pos"]
            ind1 = db["size"] + ind0
            data = self._memmap[ind0:ind1].view("int16").reshape(n1, n2)
            waveforms[i, :, :] = data

        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        chan_id = int(self.header["event_channels"][event_channel_index]["id"])
        nb_event = self._data_blocks[4][chan_id].size
        return nb_event

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        chan_id = int(self.header["event_channels"][event_channel_index]["id"])
        data_block = self._data_blocks[4][chan_id]
        keep = self._get_internal_mask(data_block, t_start, t_stop)

        db = data_block[keep]
        timestamps = db["timestamp"]
        labels = db["label"].astype("U")
        durations = None

        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_times = event_timestamps.astype(dtype)
        event_times /= self._global_ssampling_rate
        return event_times


def read_as_dict(fid, dtype, offset=None):
    """
    Given a file descriptor
    and a numpy.dtype of the binary struct return a dict.
    Make conversion for strings.
    """
    if offset is not None:
        fid.seek(offset)
    dt = np.dtype(dtype)
    h = np.frombuffer(fid.read(dt.itemsize), dt)[0]
    info = OrderedDict()
    for k in dt.names:
        v = h[k]

        if dt[k].kind == "S":
            v = v.decode("utf8")
            v = v.replace("\x03", "")
            v = v.replace("\x00", "")

        info[k] = v.item() if isinstance(v, np.generic) else v
    return info


GlobalHeader = [
    ("MagicNumber", "uint32"),
    ("Version", "int32"),
    ("Comment", "S128"),
    ("ADFrequency", "int32"),
    ("NumDSPChannels", "int32"),
    ("NumEventChannels", "int32"),
    ("NumSlowChannels", "int32"),
    ("NumPointsWave", "int32"),
    ("NumPointsPreThr", "int32"),
    ("Year", "int32"),
    ("Month", "int32"),
    ("Day", "int32"),
    ("Hour", "int32"),
    ("Minute", "int32"),
    ("Second", "int32"),
    ("FastRead", "int32"),
    ("WaveformFreq", "int32"),
    ("LastTimestamp", "float64"),
    # version >103
    ("Trodalness", "uint8"),
    ("DataTrodalness", "uint8"),
    ("BitsPerSpikeSample", "uint8"),
    ("BitsPerSlowSample", "uint8"),
    ("SpikeMaxMagnitudeMV", "uint16"),
    ("SlowMaxMagnitudeMV", "uint16"),
    # version 105
    ("SpikePreAmpGain", "uint16"),
    # version 106
    ("AcquiringSoftware", "S18"),
    ("ProcessingSoftware", "S18"),
    ("Padding", "S10"),
    # all version
    ("TSCounts", "int32", (650,)),
    ("WFCounts", "int32", (650,)),
    ("EVCounts", "int32", (512,)),
]

DspChannelHeader = [
    ("Name", "S32"),
    ("SIGName", "S32"),
    ("Channel", "int32"),
    ("WFRate", "int32"),
    ("SIG", "int32"),
    ("Ref", "int32"),
    ("Gain", "int32"),
    ("Filter", "int32"),
    ("Threshold", "int32"),
    ("Method", "int32"),
    ("NUnits", "int32"),
    ("Template", "uint16", (320,)),
    ("Fit", "int32", (5,)),
    ("SortWidth", "int32"),
    ("Boxes", "uint16", (40,)),
    ("SortBeg", "int32"),
    # version 105
    ("Comment", "S128"),
    # version 106
    ("SrcId", "uint8"),
    ("reserved", "uint8"),
    ("ChanId", "uint16"),
    ("Padding", "int32", (10,)),
]

EventChannelHeader = [
    ("Name", "S32"),
    ("Channel", "int32"),
    # version 105
    ("Comment", "S128"),
    # version 106
    ("SrcId", "uint8"),
    ("reserved", "uint8"),
    ("ChanId", "uint16"),
    ("Padding", "int32", (32,)),
]

SlowChannelHeader = [
    ("Name", "S32"),
    ("Channel", "int32"),
    ("ADFreq", "int32"),
    ("Gain", "int32"),
    ("Enabled", "int32"),
    ("PreampGain", "int32"),
    # version 104
    ("SpikeChannel", "int32"),
    # version 105
    ("Comment", "S128"),
    # version 106
    ("SrcId", "uint8"),
    ("reserved", "uint8"),
    ("ChanId", "uint16"),
    ("Padding", "int32", (27,)),
]

DataBlockHeader = [
    ("Type", "uint16"),
    ("UpperByteOf5ByteTimestamp", "uint16"),
    ("TimeStamp", "int32"),
    ("Channel", "uint16"),
    ("Unit", "uint16"),
    ("NumberOfWaveforms", "uint16"),
    ("NumberOfWordsInWaveform", "uint16"),
]  # 16 bytes
