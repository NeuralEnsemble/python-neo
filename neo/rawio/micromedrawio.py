"""
Class for reading/writing data from micromed (.trc).
Inspired by the Matlab code for EEGLAB from Rami K. Niazy.

Completed with matlab Guillaume BECQ code.

Author: Samuel Garcia
"""

import datetime
import struct
import io

import numpy as np

from .baserawio import (
    BaseRawWithBufferApiIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from .utils import get_memmap_shape

from neo.core import NeoReadWriteError


class StructFile(io.BufferedReader):
    def read_f(self, fmt, offset=None):
        if offset is not None:
            self.seek(offset)
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))


class MicromedRawIO(BaseRawWithBufferApiIO):
    """
    Class for reading  data from micromed (.trc).

    Parameters
    ----------
    filename: str, default: None
        The *.trc file to be loaded
    """

    extensions = ["trc", "TRC"]
    rawmode = "one-file"

    def __init__(self, filename=""):
        BaseRawWithBufferApiIO.__init__(self)
        self.filename = filename

    def _parse_header(self):

        with open(self.filename, "rb") as fid:
            f = StructFile(fid)

            # Name
            f.seek(64)
            surname = f.read(22).strip(b" ")
            firstname = f.read(20).strip(b" ")

            # Date
            day, month, year, hour, minute, sec = f.read_f("bbbbbb", offset=128)
            rec_datetime = datetime.datetime(year + 1900, month, day, hour, minute, sec)

            Data_Start_Offset, Num_Chan, Multiplexer, Rate_Min, Bytes = f.read_f("IHHHH", offset=138)
            sig_dtype = "u" + str(Bytes)

            # header version
            (header_version,) = f.read_f("b", offset=175)
            if header_version != 4:
                raise NotImplementedError(f"`header_version {header_version} is not implemented in neo yet")

            # area
            f.seek(176)
            zone_names = [
                "ORDER",
                "LABCOD",
                "NOTE",
                "FLAGS",
                "TRONCA",
                "IMPED_B",
                "IMPED_E",
                "MONTAGE",
                "COMPRESS",
                "AVERAGE",
                "HISTORY",
                "DVIDEO",
                "EVENT A",
                "EVENT B",
                "TRIGGER",
            ]
            zones = {}
            for zname in zone_names:
                zname2, pos, length = f.read_f("8sII")
                zones[zname] = zname2, pos, length
                if zname != zname2.decode("ascii").strip(" "):
                    raise NeoReadWriteError("expected the zone name to match")

            # "TRONCA" zone define segments
            zname2, pos, length = zones["TRONCA"]
            f.seek(pos)
            # this number avoid a infinite loop in case of corrupted  TRONCA zone (seg_start!=0 and trace_offset!=0)
            max_segments = 100
            self.info_segments = []
            for i in range(max_segments):
                # 4 bytes u4 each
                seg_start = int(np.frombuffer(f.read(4), dtype="u4")[0])
                trace_offset = int(np.frombuffer(f.read(4), dtype="u4")[0])
                if seg_start == 0 and trace_offset == 0:
                    break
                else:
                    self.info_segments.append((seg_start, trace_offset))

            if len(self.info_segments) == 0:
                # one unique segment = general case
                self.info_segments.append((0, 0))

            nb_segment = len(self.info_segments)

            # Reading Code Info
            zname2, pos, length = zones["ORDER"]
            f.seek(pos)
            code = np.frombuffer(f.read(Num_Chan * 2), dtype="u2")

            # unique stream and buffer
            buffer_id = "0"
            stream_id = "0"

            units_code = {-1: "nV", 0: "uV", 1: "mV", 2: 1, 100: "percent", 101: "dimensionless", 102: "dimensionless"}
            signal_channels = []
            sig_grounds = []
            for c in range(Num_Chan):
                zname2, pos, length = zones["LABCOD"]
                f.seek(pos + code[c] * 128 + 2, 0)

                chan_name = f.read(6).strip(b"\x00").decode("ascii")
                ground = f.read(6).strip(b"\x00").decode("ascii")
                sig_grounds.append(ground)
                logical_min, logical_max, logical_ground, physical_min, physical_max = f.read_f("iiiii")
                (k,) = f.read_f("h")
                units = units_code.get(k, "uV")

                factor = float(physical_max - physical_min) / float(logical_max - logical_min + 1)
                gain = factor
                offset = -logical_ground * factor

                f.seek(8, 1)
                (sampling_rate,) = f.read_f("H")
                sampling_rate *= Rate_Min
                chan_id = str(c)
                signal_channels.append(
                    (chan_name, chan_id, sampling_rate, sig_dtype, units, gain, offset, stream_id, buffer_id)
                )

            signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

            self._stream_buffer_slice = {"0": slice(None)}
            signal_buffers = np.array([("Signals", buffer_id)], dtype=_signal_buffer_dtype)
            signal_streams = np.array([("Signals", stream_id, buffer_id)], dtype=_signal_stream_dtype)

            if np.unique(signal_channels["sampling_rate"]).size != 1:
                raise NeoReadWriteError("The sampling rates must be the same across signal channels")
            self._sampling_rate = float(np.unique(signal_channels["sampling_rate"])[0])

            # memmap traces buffer
            full_signal_shape = get_memmap_shape(
                self.filename, sig_dtype, num_channels=Num_Chan, offset=Data_Start_Offset
            )
            seg_limits = [trace_offset for seg_start, trace_offset in self.info_segments] + [full_signal_shape[0]]
            self._t_starts = []
            self._buffer_descriptions = {0: {}}
            for seg_index in range(nb_segment):
                seg_start, trace_offset = self.info_segments[seg_index]
                self._t_starts.append(seg_start / self._sampling_rate)

                start = seg_limits[seg_index]
                stop = seg_limits[seg_index + 1]

                shape = (stop - start, Num_Chan)
                file_offset = Data_Start_Offset + (start * np.dtype(sig_dtype).itemsize * Num_Chan)
                self._buffer_descriptions[0][seg_index] = {}
                self._buffer_descriptions[0][seg_index][buffer_id] = {
                    "type": "raw",
                    "file_path": str(self.filename),
                    "dtype": sig_dtype,
                    "order": "C",
                    "file_offset": file_offset,
                    "shape": shape,
                }

            # Event channels
            event_channels = []
            event_channels.append(("Trigger", "", "event"))
            event_channels.append(("Note", "", "event"))
            event_channels.append(("Event A", "", "epoch"))
            event_channels.append(("Event B", "", "epoch"))
            event_channels = np.array(event_channels, dtype=_event_channel_dtype)

            # Read trigger and notes
            self._raw_events = []
            ev_dtypes = [
                ("TRIGGER", [("start", "u4"), ("label", "u2")]),
                ("NOTE", [("start", "u4"), ("label", "S40")]),
                ("EVENT A", [("label", "u4"), ("start", "u4"), ("stop", "u4")]),
                ("EVENT B", [("label", "u4"), ("start", "u4"), ("stop", "u4")]),
            ]
            for zname, ev_dtype in ev_dtypes:
                zname2, pos, length = zones[zname]
                dtype = np.dtype(ev_dtype)
                rawevent = np.memmap(self.filename, dtype=dtype, mode="r", offset=pos, shape=length // dtype.itemsize)

                # important : all events timing are related to the first segment t_start
                self._raw_events.append([])
                for seg_index in range(nb_segment):
                    left_lim = seg_limits[seg_index]
                    right_lim = seg_limits[seg_index + 1]
                    keep = (rawevent["start"] >= left_lim) & (rawevent["start"] < right_lim) & (rawevent["start"] != 0)
                    self._raw_events[-1].append(rawevent[keep])

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
            bl_annotations = self.raw_annotations["blocks"][0]
            seg_annotations = bl_annotations["segments"][0]

            for d in (bl_annotations, seg_annotations):
                d["rec_datetime"] = rec_datetime
                d["firstname"] = firstname
                d["surname"] = surname
                d["header_version"] = header_version

            sig_annotations = self.raw_annotations["blocks"][0]["segments"][0]["signals"][0]
            sig_annotations["__array_annotations__"]["ground"] = np.array(sig_grounds)

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return self._t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        duration = self.get_signal_size(block_index, seg_index, stream_index=0) / self._sampling_rate
        return duration + self.segment_t_start(block_index, seg_index)

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        assert stream_index == 0
        return self._t_starts[seg_index]

    def _spike_count(self, block_index, seg_index, unit_index):
        return 0

    def _event_count(self, block_index, seg_index, event_channel_index):
        n = self._raw_events[event_channel_index][seg_index].size
        return n

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):

        raw_event = self._raw_events[event_channel_index][seg_index]

        # important : all events timing are related to the first segment t_start
        seg_start0, _ = self.info_segments[0]

        if t_start is not None:
            keep = raw_event["start"] + seg_start0 >= int(t_start * self._sampling_rate)
            raw_event = raw_event[keep]

        if t_stop is not None:
            keep = raw_event["start"] + seg_start0 <= int(t_stop * self._sampling_rate)
            raw_event = raw_event[keep]

        timestamp = raw_event["start"] + seg_start0

        if event_channel_index < 2:
            durations = None
        else:
            durations = raw_event["stop"] - raw_event["start"]

        try:
            labels = raw_event["label"].astype("U")
        except UnicodeDecodeError:
            # sometimes the conversion do not work : here a simple fix
            labels = np.array([e.decode("cp1252") for e in raw_event["label"]], dtype="U")

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_times = event_timestamps.astype(dtype) / self._sampling_rate
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        durations = raw_duration.astype(dtype) / self._sampling_rate
        return durations

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self._buffer_descriptions[block_index][seg_index][buffer_id]
