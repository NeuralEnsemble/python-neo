"""
This module implement the "old" OpenEphys format.
In this format channels are split into several files

https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Open-Ephys-format.html


Author: Samuel Garcia
"""

import re
from pathlib import Path

import numpy as np

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from neo.core import NeoReadWriteError


RECORD_SIZE = 1024
HEADER_SIZE = 1024


class OpenEphysRawIO(BaseRawIO):
    """
    Class for reading openephys data

    Parameters
    ----------
    dirname: str
        The directory where the files are stored.
    ignore_timestamps_errors: bool
        (deprecated) This parameter is not used anymore.
    fill_gap_value: int
        When gaps are detected in continuous files, the gap is filled with this value.
        Default is 0.

    Notes
    -----
    OpenEphys GUI software offers several data formats, see
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491632/Data+format

    This class implements the legacy OpenEphys format here
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/65667092/Open+Ephys+format

    The OpenEphys group already proposes some tools here:
    https://github.com/open-ephys/analysis-tools/blob/master/OpenEphys.py
    but (i) there is no package at PyPI and (ii) those tools read everything in memory.

    The format is directory based with several files:
        * .continuous
        * .events
        * .spikes

    This implementation is based on:
      * this code https://github.com/open-ephys/analysis-tools/blob/master/Python3/OpenEphys.py
        written by Dan Denman and Josh Siegle
      * a previous PR by Cristian Tatarau at Charité Berlin

    In contrast to previous code for reading this format, here all data use memmap so it should
    be super fast and light compared to legacy code.

    When the acquisition is stopped and restarted then files are named ``*_2``, ``*_3``.
    In that case this class creates a new Segment. Note that timestamps are reset in this
    situation.

    Limitation :
      * Works only if all continuous channels have the same sampling rate, which is a reasonable
        hypothesis.
      * A recording can contain gaps due to USB stream loss when high CPU load when recording.
        These gaps are checked channel per channel which makes the parse_header() slow.
        If gaps are detected then they are filled with zeros but then the reading will be much slower for getting signals.

    """

    # file formats used by openephys
    extensions = ["continuous", "openephys", "spikes", "events", "xml"]
    rawmode = "one-dir"

    def __init__(self, dirname="", ignore_timestamps_errors=None, fill_gap_value=0):
        BaseRawIO.__init__(self)
        self.dirname = dirname
        self.fill_gap_value = int(fill_gap_value)
        if ignore_timestamps_errors is not None:
            self.logger.warning("OpenEphysRawIO ignore_timestamps_errors=True/False is not used anymore")

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        info = self._info = explore_folder(self.dirname)
        nb_segment = info["nb_segment"]

        # scan for continuous files
        self._sigs_memmap = {}
        self._sig_length = {}
        self._sig_timestamp0 = {}
        self._sig_has_gap = {}
        self._gap_mode = False
        signal_channels = []
        oe_indices = sorted(list(info["continuous"].keys()))
        for seg_index, oe_index in enumerate(oe_indices):
            self._sigs_memmap[seg_index] = {}
            self._sig_has_gap[seg_index] = {}

            all_first_timestamps = []
            all_last_timestamps = []
            all_samplerate = []
            for chan_index, continuous_filename in enumerate(info["continuous"][oe_index]):
                chan_info = read_file_header(continuous_filename)

                s = continuous_filename.stem.split("_")
                # Formats are ['processor_id', 'ch_name'] or  ['processor_id', 'name', 'ch_name']
                if len(s) == 2:
                    processor_id, ch_name = s[0], s[1]
                    chan_str = re.split(r"(\d+)", s[1])[0]
                else:
                    processor_id, ch_name = s[0], s[2]
                    chan_str = re.split(r"(\d+)", s[2])[0]
                # note that chan_id is not unique in case of CH + AUX
                chan_id = int(ch_name.replace(chan_str, ""))

                filesize = continuous_filename.stat().st_size
                size = (filesize - HEADER_SIZE) // np.dtype(continuous_dtype).itemsize
                data_chan = np.memmap(
                    continuous_filename, mode="r", offset=HEADER_SIZE, dtype=continuous_dtype, shape=(size,)
                )
                self._sigs_memmap[seg_index][chan_index] = data_chan

                all_first_timestamps.append(data_chan[0]["timestamp"])
                all_last_timestamps.append(data_chan[-1]["timestamp"] + RECORD_SIZE)
                all_samplerate.append(chan_info["sampleRate"])

                # check for continuity (no gaps)
                diff = np.diff(data_chan["timestamp"])
                channel_has_gaps = not np.all(diff == RECORD_SIZE)
                self._sig_has_gap[seg_index][chan_index] = channel_has_gaps

                if channel_has_gaps:
                    # protect against strange timestamp block like in file 'OpenEphys_SampleData_3' CH32
                    if not np.median(diff) == RECORD_SIZE:
                        raise NeoReadWriteError(
                            f"This file has a non valid data block size for channel {chan_id}, this case cannot be handled"
                        )

                channel_has_corrupted_timestamps = np.any(diff <= 0)
                if channel_has_corrupted_timestamps:
                    # protect against corrupted timestamp in channel
                    raise ValueError(
                        f"{ch_name} has timestamps with zero values or negative differences between consecutive timestamps, this file ({continuous_filename}) with corrupted timestamps needs to be moved away from the folder."
                    )

                if seg_index == 0:
                    # add in channel list
                    if ch_name[:2].upper() == "CH":
                        units = "uV"
                    else:
                        units = "V"
                    stream_id = processor_id
                    buffer_id = ""
                    signal_channels.append(
                        (
                            ch_name,
                            chan_id,
                            chan_info["sampleRate"],
                            "int16",
                            units,
                            chan_info["bitVolts"],
                            0.0,
                            stream_id,
                            buffer_id,
                        )
                    )

            if any(self._sig_has_gap[seg_index].values()):
                channel_with_gaps = list(self._sig_has_gap[seg_index].keys())
                self.logger.warning(
                    f"This OpenEphys dataset contains gaps for some channels {channel_with_gaps} in segment {seg_index} the read will be slow"
                )
                self._gap_mode = True

            if not all(all_first_timestamps[0] == e for e in all_first_timestamps) or not all(
                all_last_timestamps[0] == e for e in all_last_timestamps
            ):
                # In some cases, continuous do not have the same length because
                # we need to clip
                self.logger.warning(
                    "Continuous files do not have aligned timestamps; " "clipping to make them aligned."
                )

                first = max(all_first_timestamps)
                last = min(all_last_timestamps)
                for chan_index in self._sigs_memmap[seg_index]:
                    data_chan = self._sigs_memmap[seg_index][chan_index]
                    keep = (data_chan["timestamp"] >= first) & (data_chan["timestamp"] < last)
                    data_chan = data_chan[keep]
                    self._sigs_memmap[seg_index][chan_index] = data_chan
            else:
                # no clip
                first = all_first_timestamps[0]
                last = all_last_timestamps[0]

            # check unique sampling rate
            if not all(all_samplerate[0] == e for e in all_samplerate):
                raise NeoReadWriteError("Not all signals have the same sample rate")

            self._sig_length[seg_index] = last - first
            self._sig_timestamp0[seg_index] = first

        if len(signal_channels) > 0:
            signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
            self._sig_sampling_rate = signal_channels["sampling_rate"][0]  # unique for channel

            # split channels in stream depending the name CHxxx ADCxxx
            chan_stream_ids = [name[:2] if name.startswith("CH") else name[:3] for name in signal_channels["name"]]
            signal_channels["stream_id"] = chan_stream_ids

            # and create streams channels (keep natural order 'CH' first)
            stream_ids, order = np.unique(chan_stream_ids, return_index=True)
            stream_ids = stream_ids[np.argsort(order)]
            signal_streams = [(f"Signals {stream_id}", f"{stream_id}", "") for stream_id in stream_ids]
        else:
            signal_streams = []
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        # no buffer handling in this format because one channel per file
        signal_buffers = np.array([], dtype=_signal_buffer_dtype)

        # scan for spikes files
        spike_channels = []

        if len(info["spikes"]) > 0:
            self._first_spk_timestamps = []
            self._last_spk_timestamps = []
            self._spikes_memmap = {}
            oe_indices_spk = sorted(list(info["spikes"].keys()))
            for seg_index, oe_index in enumerate(oe_indices_spk):
                self._spikes_memmap[seg_index] = {}
                for spike_filename in info["spikes"][oe_index]:
                    spike_info = read_file_header(spike_filename)
                    spikes_dtype = make_spikes_dtype(spike_filename)

                    # "STp106.0n0_2.spikes" to "STp106.0n0"
                    name = spike_filename.stem
                    if seg_index > 0:
                        name = name.replace("_" + str(seg_index + 1), "")

                    data_spike = np.memmap(spike_filename, mode="r", offset=HEADER_SIZE, dtype=spikes_dtype)
                    self._spikes_memmap[seg_index][name] = data_spike

                    self._first_spk_timestamps.append(data_spike[0]["timestamp"])
                    self._last_spk_timestamps.append(data_spike[-1]["timestamp"])

            # In each file 'sorted_id' indicate the number of cluster so number of units
            # so need to scan file for all segment to get units
            self._spike_sampling_rate = None
            for spike_filename_seg0 in info["spikes"][0]:
                name = spike_filename_seg0.stem

                spike_info = read_file_header(spike_filename_seg0)
                if self._spike_sampling_rate is None:
                    self._spike_sampling_rate = spike_info["sampleRate"]
                else:
                    if self._spike_sampling_rate != spike_info["sampleRate"]:
                        raise ValueError("There is a mismatch in spike sampling rate")

                # scan all to detect several all unique(sorted_ids)
                all_sorted_ids = []
                for seg_index in range(nb_segment):
                    data_spike = self._spikes_memmap[seg_index][name]
                    all_sorted_ids += np.unique(data_spike["sorted_id"]).tolist()
                all_sorted_ids = np.unique(all_sorted_ids)

                # suppose all channels have the same gain
                wf_units = "uV"
                wf_gain = 1000.0 / data_spike[0]["gains"][0]
                wf_offset = -(2**15) * wf_gain
                wf_left_sweep = 0
                wf_sampling_rate = spike_info["sampleRate"]

                # each sorted_id is one channel
                for sorted_id in all_sorted_ids:
                    unit_name = f"{name}#{sorted_id}"
                    unit_id = f"{name}#{sorted_id}"
                    spike_channels.append(
                        (unit_name, unit_id, wf_units, wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate)
                    )

        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # event file are:
        #    * all_channel.events (header + binary)  -->  event 0
        #    * n_RhythmData-a.events (header + binary) --> event 0 (maybe a new naming convention? )
        # and message.events (text based)      --> event 1 not implemented yet
        event_channels = []
        self._events_memmap = {}
        event_files = list(
            [
                event_file
                for event_file in Path(self.dirname).glob("**/*.events")
                if event_file.name != "messages.events"
            ]
        )
        event_files.sort()  # sort should put the xx.events first followed by xx_x.events
        # only run if we have actual potential event files
        if len(event_files) > 0:
            event_file_name_0 = event_files[0].stem  # this should always be the file without a '_n' appended
            for seg_index, oe_index in enumerate(oe_indices):
                if oe_index == 0:
                    event_filename = Path(self.dirname) / (event_file_name_0 + ".events")
                else:
                    event_filename = Path(self.dirname) / (event_file_name_0 + f"_{oe_index + 1}.events")

                event_info = read_file_header(event_filename)
                # event files can exist, but just not have data
                try:
                    self._event_sampling_rate = event_info["sampleRate"]
                except KeyError:
                    break
                data_event = np.memmap(event_filename, mode="r", offset=HEADER_SIZE, dtype=events_dtype)
                self._events_memmap[seg_index] = data_event
        # only append event channels if they actually exist & have data
        if len(self._events_memmap.keys()) > 0:
            event_channels.append((event_filename.stem, "", "event"))
        # event_channels.append(('message', '', 'event')) # not implemented
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # main header
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [nb_segment]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # Annotate some objects from continuous files
        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations["blocks"][0]
        for seg_index, oe_index in enumerate(oe_indices):
            seg_ann = bl_ann["segments"][seg_index]
            if len(info["continuous"]) > 0:
                fullname = info["continuous"][oe_index][0]
                chan_info = read_file_header(fullname)
                seg_ann["openephys_version"] = chan_info["version"]
                bl_ann["openephys_version"] = chan_info["version"]
                seg_ann["date_created"] = chan_info["date_created"]
                seg_ann["openephys_segment_index"] = oe_index + 1

    def _segment_t_start(self, block_index, seg_index):
        # segment start/stop are defined by continuous channels
        return self._sig_timestamp0[seg_index] / self._sig_sampling_rate

    def _segment_t_stop(self, block_index, seg_index):
        return (self._sig_timestamp0[seg_index] + self._sig_length[seg_index]) / self._sig_sampling_rate

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self._sig_length[seg_index]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._sig_timestamp0[seg_index] / self._sig_sampling_rate

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._sig_length[seg_index]

        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"]
        (global_channel_indexes,) = np.nonzero(mask == stream_id)
        if channel_indexes is None:
            channel_indexes = slice(None)
        global_channel_indexes = global_channel_indexes[channel_indexes]

        if not self._gap_mode:
            sigs_chunk = np.zeros((i_stop - i_start, len(global_channel_indexes)), dtype="int16")
            # previous behavior block index are linear
            block_start = i_start // RECORD_SIZE
            block_stop = i_stop // RECORD_SIZE + 1
            sl0 = i_start % RECORD_SIZE
            sl1 = sl0 + (i_stop - i_start)

            for i, global_chan_index in enumerate(global_channel_indexes):
                data = self._sigs_memmap[seg_index][global_chan_index]
                sub = data[block_start:block_stop]
                sigs_chunk[:, i] = sub["samples"].flatten()[sl0:sl1]
        else:
            sigs_chunk = np.full(
                shape=(i_stop - i_start, len(global_channel_indexes)), fill_value=self.fill_gap_value, dtype="int16"
            )
            # slow mode
            for i, global_chan_index in enumerate(global_channel_indexes):
                data = self._sigs_memmap[seg_index][global_chan_index]
                timestamp0 = data[0]["timestamp"]

                # find first block
                block0 = np.searchsorted(data["timestamp"], timestamp0 + i_start, side="right") - 1
                block0_pos = data[block0]["timestamp"] - timestamp0

                if i_start - block0_pos > RECORD_SIZE:
                    # the block has gap!!
                    pos = -((i_start - block0_pos) % RECORD_SIZE)
                    block_index = block0 + 1
                else:
                    # the first block do not have gaps
                    shift0 = i_start - block0_pos

                    if shift0 + (i_stop - i_start) < RECORD_SIZE:
                        # protect when only one small block
                        pos = i_stop - i_start
                        sigs_chunk[:, i][:pos] = data[block0]["samples"][shift0 : shift0 + pos]
                    else:

                        pos = RECORD_SIZE - shift0
                        sigs_chunk[:, i][:pos] = data[block0]["samples"][shift0:]
                    block_index = block0 + 1

                # full block
                while block_index < data.size and data[block_index]["timestamp"] - timestamp0 < i_stop - RECORD_SIZE:
                    diff = data[block_index]["timestamp"] - data[block_index - 1]["timestamp"]
                    if diff > RECORD_SIZE:
                        # gap detected need jump
                        pos += diff - RECORD_SIZE

                    sigs_chunk[:, i][pos : pos + RECORD_SIZE] = data[block_index]["samples"][:]
                    pos += RECORD_SIZE
                    block_index += 1

                # last block
                if pos < i_stop - i_start:
                    diff = data[block_index]["timestamp"] - data[block_index - 1]["timestamp"]
                    if diff == RECORD_SIZE:
                        # ensure no gaps for last block
                        sigs_chunk[:, i][pos:] = data[block_index]["samples"][: i_stop - i_start - pos]

        return sigs_chunk

    def _get_spike_slice(self, seg_index, unit_index, t_start, t_stop):
        name, sorted_id = self.header["spike_channels"][unit_index]["name"].split("#")
        sorted_id = int(sorted_id)
        data_spike = self._spikes_memmap[seg_index][name]

        if t_start is None:
            t_start = self._first_spk_timestamps[seg_index]
        if t_stop is None:
            t_stop = self._last_spk_timestamps[seg_index]
        ts0 = int(t_start * self._spike_sampling_rate)
        ts1 = int(t_stop * self._spike_sampling_rate)

        ts = data_spike["timestamp"]
        keep = (data_spike["sorted_id"] == sorted_id) & (ts >= ts0) & (ts <= ts1)
        return data_spike, keep

    def _spike_count(self, block_index, seg_index, unit_index):
        data_spike, keep = self._get_spike_slice(seg_index, unit_index, None, None)
        return np.sum(keep)

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        data_spike, keep = self._get_spike_slice(seg_index, unit_index, t_start, t_stop)
        return data_spike["timestamp"][keep]

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype) / self._spike_sampling_rate
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        data_spike, keep = self._get_spike_slice(seg_index, unit_index, t_start, t_stop)
        nb_chan = data_spike[0]["nb_channel"]
        nb = np.sum(keep)
        waveforms = data_spike[keep]["samples"].flatten()
        waveforms = waveforms.reshape(nb, nb_chan, -1)
        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        # assert event_channel_index==0
        return self._events_memmap[seg_index].size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # assert event_channel_index==0

        if t_start is None:
            t_start = self._segment_t_start(block_index, seg_index)
        if t_stop is None:
            t_stop = self._segment_t_stop(block_index, seg_index)
        ts0 = int(t_start * self._event_sampling_rate)
        ts1 = int(t_stop * self._event_sampling_rate)
        ts = self._events_memmap[seg_index]["timestamp"]
        keep = (ts >= ts0) & (ts <= ts1)

        subdata = self._events_memmap[seg_index][keep]
        timestamps = subdata["timestamp"]
        # question what is the label????
        # here I put a combination
        labels = np.array(
            [f'{int(d["event_type"])}#{int(d["processor_id"])}#{int(d["chan_id"])}' for d in subdata],
            dtype="U",
        )
        durations = None

        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_times = event_timestamps.astype(dtype) / self._event_sampling_rate
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return None


continuous_dtype = [
    ("timestamp", "int64"),
    ("nb_sample", "uint16"),
    ("rec_num", "uint16"),
    ("samples", ">i2", RECORD_SIZE),
    ("markers", "uint8", 10),
]

events_dtype = [
    ("timestamp", "int64"),
    ("sample_pos", "int16"),
    ("event_type", "uint8"),
    ("processor_id", "uint8"),
    ("event_id", "uint8"),
    ("chan_id", "uint8"),
    ("record_num", "uint16"),
]

# the dtype is dynamic and depend on nb_channel and nb_sample
_base_spikes_dtype = [
    ("event_stype", "uint8"),
    ("timestamp", "int64"),
    ("software_timestamp", "int64"),
    ("source_id", "uint16"),
    ("nb_channel", "uint16"),
    ("nb_sample", "uint16"),
    ("sorted_id", "uint16"),
    ("electrode_id", "uint16"),
    ("within_chan_index", "uint16"),
    ("color", "uint8", 3),
    ("pca", "float32", 2),
    ("sampling_rate", "uint16"),
    ("samples", "uint16", None),
    ("gains", "float32", None),
    ("thresholds", "uint16", None),
    ("rec_num", "uint16"),
]


def make_spikes_dtype(filename):
    """
    Given the spike file make the appropriate dtype that depends on:
      * N - number of channels
      * M - samples per spike
    See documentation of file format.
    """

    # strangely the header do not have the sample size
    # So this do not work (too bad):
    # spike_info = read_file_header(filename)
    # N = spike_info['num_channels']
    # M =????

    # so we need to read the very first spike
    # but it will fail when 0 spikes (too bad)
    filesize = filename.stat().st_size
    if filesize >= (HEADER_SIZE + 23):
        with open(filename, mode="rb") as f:
            # M and N is at 1024 + 19 bytes
            f.seek(HEADER_SIZE + 19)
            N = np.fromfile(f, np.dtype("<u2"), 1)[0]
            M = np.fromfile(f, np.dtype("<u2"), 1)[0]
    else:
        spike_info = read_file_header(filename)
        N = spike_info["num_channels"]
        M = 40  # this is in the original code from openephys

    # make a copy
    spikes_dtype = [e for e in _base_spikes_dtype]
    spikes_dtype[12] = ("samples", "uint16", N * M)
    spikes_dtype[13] = ("gains", "float32", N)
    spikes_dtype[14] = ("thresholds", "uint16", N)

    return spikes_dtype


def explore_folder(dirname):
    """
    This explores a folder and dispatch continuous, event and spikes
    files by segment (aka recording session).

    The number of segments is checked with these rules
    "100_CH0.continuous" ---> seg_index 0
    "100_CH0_2.continuous" ---> seg_index 1
    "100_CH0_N.continuous" ---> seg_index N-1

    Newer formats follow similar rules but have an addition
    "100_RhythmData-A_CH0.continuous" ----> seg_index 0
    """
    filenames = [filename for filename in Path(dirname).glob("**/*") if filename.is_file()]
    filenames.sort()

    info = {}
    info["nb_segment"] = 0
    info["continuous"] = {}
    info["spikes"] = {}
    for filename in filenames:
        if filename.suffix == ".continuous":
            s = filename.stem.split("_")
            # For continuous files we check if the last value is an int indicating that a new segment should be
            # generated and if it is not an int then this must be same segment
            try:
                seg_index = int(s[-1]) - 1
            except ValueError:
                seg_index = 0
            if seg_index not in info["continuous"].keys():
                info["continuous"][seg_index] = []
            info["continuous"][seg_index].append(filename)
            if (seg_index + 1) > info["nb_segment"]:
                info["nb_segment"] += 1
        elif filename.suffix == ".spikes":
            s = re.findall(r"(_\d+)$", filename.stem)
            if s:
                seg_index = int(s[0][1:]) - 1
            else:
                seg_index = 0
            if seg_index not in info["spikes"].keys():
                info["spikes"][seg_index] = []
            info["spikes"][seg_index].append(filename)
            if (seg_index + 1) > info["nb_segment"]:
                info["nb_segment"] += 1

    # order continuous file by channel number within segment
    # order "CH before "ADC"
    for seg_index, continuous_filenames in info["continuous"].items():
        chan_ids_by_type = {}
        filenames_by_type = {}
        for continuous_filename in continuous_filenames:
            s = continuous_filename.stem.split("_")
            # new format includes putting a name between e.g. ['124', 'RhythmData', 'CH1']
            # old format would just be ['124', 'CH1']
            if len(s) == 2:
                processor_id, ch_name = s[0], s[1]
                chan_type = re.split(r"(\d+)", s[1])[0]
            else:
                processor_id, ch_name = s[0], s[2]
                chan_type = re.split(r"(\d+)", s[2])[0]
            chan_id = int(ch_name.replace(chan_type, ""))
            if chan_type in chan_ids_by_type.keys():
                chan_ids_by_type[chan_type].append(chan_id)
                filenames_by_type[chan_type].append(continuous_filename)
            else:
                chan_ids_by_type[chan_type] = [chan_id]
                filenames_by_type[chan_type] = [continuous_filename]
        chan_types = list(chan_ids_by_type.keys())

        if "CH" in chan_types:
            # force CH at beginning
            chan_types.remove("CH")
            chan_types = ["CH"] + chan_types

        ordered_continuous_filenames = []
        for chan_type in chan_types:
            local_order = np.argsort(chan_ids_by_type[chan_type])
            local_filenames = np.array(filenames_by_type[chan_type])[local_order]
            ordered_continuous_filenames.extend(local_filenames)
        info["continuous"][seg_index] = ordered_continuous_filenames

    # order spike files within segment
    for seg_index, spike_filenames in info["spikes"].items():
        names = []
        for spike_filename in spike_filenames:
            name = spike_filename.stem
            if seg_index > 0:
                name = name.replace("_" + str(seg_index + 1), "")
            names.append(name)
        order = np.argsort(names)
        spike_filenames = [spike_filenames[i] for i in order]
        info["spikes"][seg_index] = spike_filenames

    return info


def read_file_header(filename):
    """Read header information from the first 1024 bytes of an OpenEphys file.
    See docs.
    """
    header = {}
    with open(filename, mode="rb") as f:
        # Read the data as a string
        # Remove newlines and redundant "header." prefixes
        # The result should be a series of "key = value" strings, separated
        # by semicolons.
        header_string = f.read(HEADER_SIZE).replace(b"\n", b"").replace(b"header.", b"")

    # Parse each key = value string separately
    for pair in header_string.split(b";"):
        if b"=" in pair:
            key, value = pair.split(b" = ")
            key = key.strip().decode("ascii")
            value = value.strip()

            # Convert some values to numeric
            if key in ["bitVolts", "sampleRate"]:
                header[key] = float(value)
            elif key in ["blockLength", "bufferSize", "header_bytes", "num_channels"]:
                header[key] = int(value)
            else:
                # Keep as string
                header[key] = value.decode("ascii")

    return header
