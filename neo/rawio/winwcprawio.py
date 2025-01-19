"""
Class for reading data from WinWCP, a software tool written by
John Dempster.

WinWCP is free:
http://spider.science.strath.ac.uk/sipbs/software.htm

Author: Samuel Garcia
"""

import struct

import numpy as np

from .baserawio import (
    BaseRawWithBufferApiIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
    _common_sig_characteristics,
)


class WinWcpRawIO(BaseRawWithBufferApiIO):
    """
    Class for reading WinWCP data

    Parameters
    ----------
    filename: str, default: ''
        The .wcp file to load

    """

    extensions = ["wcp"]
    rawmode = "one-file"

    def __init__(self, filename=""):
        BaseRawWithBufferApiIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        SECTORSIZE = 512

        # one unique block with several segments
        # one unique buffer splited in several streams
        self._buffer_descriptions = {0: {}}

        with open(self.filename, "rb") as fid:

            headertext = fid.read(1024)
            headertext = headertext.decode("ascii")
            header = {}
            for line in headertext.split("\r\n"):
                if "=" not in line:
                    continue
                key, val = line.split("=")
                if key in [
                    "NC",
                    "NR",
                    "NBH",
                    "NBA",
                    "NBD",
                    "ADCMAX",
                    "NP",
                    "NZ",
                ]:
                    val = int(val)
                elif key in [
                    "AD",
                    "DT",
                ]:
                    val = val.replace(",", ".")
                    val = float(val)
                header[key] = val

            nb_segment = header["NR"]
            all_sampling_interval = []
            # loop for record number
            for seg_index in range(header["NR"]):
                offset = 1024 + seg_index * (SECTORSIZE * header["NBD"] + 1024)

                # read analysis zone
                analysisHeader = HeaderReader(fid, AnalysisDescription).read_f(offset=offset)

                # read data
                NP = (SECTORSIZE * header["NBD"]) // 2
                NP = NP - NP % header["NC"]
                NP = NP // header["NC"]
                NC = header["NC"]
                ind0 = offset + header["NBA"] * SECTORSIZE
                buffer_id = "0"
                self._buffer_descriptions[0][seg_index] = {}
                self._buffer_descriptions[0][seg_index][buffer_id] = {
                    "type": "raw",
                    "file_path": str(self.filename),
                    "dtype": "int16",
                    "order": "C",
                    "file_offset": ind0,
                    "shape": (NP, NC),
                }

                all_sampling_interval.append(analysisHeader["SamplingInterval"])

        # sampling interval can be slightly varying due to float precision
        # all_sampling_interval are not always unique
        self._sampling_rate = 1.0 / np.median(all_sampling_interval)

        signal_channels = []
        for c in range(header["NC"]):
            YG = float(header[f"YG{c}"].replace(",", "."))
            ADCMAX = header["ADCMAX"]
            VMax = analysisHeader["VMax"][c]

            name = header[f"YN{c}"]
            chan_id = header[f"YO{c}"]
            units = header[f"YU{c}"]
            gain = VMax / ADCMAX / YG
            offset = 0.0
            stream_id = "0"
            buffer_id = "0"
            signal_channels.append(
                (name, chan_id, self._sampling_rate, "int16", units, gain, offset, stream_id, buffer_id)
            )

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        characteristics = signal_channels[_common_sig_characteristics]
        unique_characteristics = np.unique(characteristics)
        signal_streams = []
        self._stream_buffer_slice = {}
        for i in range(unique_characteristics.size):
            mask = unique_characteristics[i] == characteristics
            signal_channels["stream_id"][mask] = str(i)
            # unique buffer for all streams
            buffer_id = "0"
            stream_id = str(i)
            signal_streams.append((f"stream {i}", stream_id, buffer_id))
            self._stream_buffer_slice[stream_id] = np.flatnonzero(mask)
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # all stream are in the same unique buffer : memmap
        signal_buffers = np.array([("", "0")], dtype=_signal_buffer_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [nb_segment]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        sig_size = self.get_signal_size(block_index, seg_index, 0)
        t_stop = sig_size / self._sampling_rate
        return t_stop

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self._buffer_descriptions[block_index][seg_index][buffer_id]


AnalysisDescription = [
    ("RecordStatus", "8s"),
    ("RecordType", "4s"),
    ("GroupNumber", "f"),
    ("TimeRecorded", "f"),
    ("SamplingInterval", "f"),
    ("VMax", "8f"),
]


class HeaderReader:
    def __init__(self, fid, description):
        self.fid = fid
        self.description = description

    def read_f(self, offset=0):
        self.fid.seek(offset)
        d = {}
        for key, fmt in self.description:
            val = struct.unpack(fmt, self.fid.read(struct.calcsize(fmt)))
            if len(val) == 1:
                val = val[0]
            else:
                val = list(val)
            d[key] = val
        return d
