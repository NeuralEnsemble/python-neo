"""
Class for reading data from WinEdr, a software tool written by
John Dempster.

WinEdr is free:
http://spider.science.strath.ac.uk/sipbs/software.htm

Author: Samuel Garcia

"""

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


class WinEdrRawIO(BaseRawWithBufferApiIO):
    extensions = ["EDR", "edr"]
    rawmode = "one-file"

    def __init__(self, filename=""):
        """
        Class for reading WinEdr data

        Parameters
        ----------
        filename: str, default: ''
            The *.edr file to be loaded

        """
        BaseRawWithBufferApiIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        with open(self.filename, "rb") as fid:
            headertext = fid.read(2048)
            headertext = headertext.decode("ascii")
            header = {}
            for line in headertext.split("\r\n"):
                if "=" not in line:
                    continue
                # print '#' , line , '#'
                key, val = line.split("=")
                if key in ["NC", "NR", "NBH", "NBA", "NBD", "ADCMAX", "NP", "NZ", "ADCMAX"]:
                    val = int(val)
                elif key in [
                    "AD",
                    "DT",
                ]:
                    val = val.replace(",", ".")
                    val = float(val)
                header[key] = val

        # one unique block with one unique segment
        # one unique buffer splited in several streams
        buffer_id = "0"
        self._buffer_descriptions = {0: {0: {}}}
        self._buffer_descriptions[0][0][buffer_id] = {
            "type": "raw",
            "file_path": str(self.filename),
            "dtype": "int16",
            "order": "C",
            "file_offset": int(header["NBH"]),
            "shape": (header["NP"] // header["NC"], header["NC"]),
        }

        DT = header["DT"]
        if "TU" in header:
            if header["TU"] == "ms":
                DT *= 0.001
        self._sampling_rate = 1.0 / DT

        signal_channels = []
        for c in range(header["NC"]):
            YCF = float(header[f"YCF{c}"].replace(",", "."))
            YAG = float(header[f"YAG{c}"].replace(",", "."))
            YZ = float(header[f"YZ{c}"].replace(",", "."))
            ADCMAX = header["ADCMAX"]
            AD = header["AD"]

            name = header[f"YN{c}"]
            chan_id = header[f"YO{c}"]
            units = header[f"YU{c}"]
            gain = AD / (YCF * YAG * (ADCMAX + 1))
            offset = -YZ * gain
            stream_id = "0"
            buffer_id = "0"
            signal_channels.append(
                (name, str(chan_id), self._sampling_rate, "int16", units, gain, offset, stream_id, buffer_id)
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
            signal_streams.append((f"stream {i}", str(i), buffer_id))
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
        self.header["nb_segment"] = [1]
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
