"""
Class for reading data from a MEArec simulated data.

See:
https://mearec.readthedocs.io/en/latest/
https://github.com/alejoe91/MEArec
https://link.springer.com/article/10.1007/s12021-020-09467-7

Author : Alessio Buccino
"""

from copy import deepcopy

import numpy as np

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


class MEArecRawIO(BaseRawIO):
    """
    Class for "reading" simulated data from a MEArec file.

    Parameters
    ----------
    filename : str, default: ''
        The filename of the MEArec file to read.
    load_spiketrains : bool, default: True
        Whether or not to load spike train data.
    load_analogsignal : bool, default: True
        Whether or not to load continuous recording data.


    Examples
    --------
    >>> import neo.rawio
    >>> r = neo.rawio.MEArecRawIO(filename='mearec.h5', load_spiketrains=True)
    >>> r.parse_header()
    >>> print(r)
    >>> raw_chunk = r.get_analogsignal_chunk(block_index=0,
                                             seg_index=0,
                                             i_start=0,
                                             i_stop=1024,
                                             channel_names=channel_names)
    >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk,
                                                        dtype='float64',
                                                        channel_indexes=[0, 3, 6])
    >>> spike_timestamp = reader.spike_timestamps(unit_index=0, t_start=None, t_stop=None)
    >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')

    """

    extensions = ["h5"]
    rawmode = "one-file"

    def __init__(self, filename="", load_spiketrains=True, load_analogsignal=True):
        BaseRawIO.__init__(self)
        self.filename = filename
        self.load_spiketrains = load_spiketrains
        self.load_analogsignal = load_analogsignal

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        load = ["channel_positions"]
        if self.load_analogsignal:
            load.append("recordings")
        if self.load_spiketrains:
            load.append("spiketrains")

        import MEArec as mr

        self._recgen = mr.load_recordings(
            recordings=self.filename, return_h5_objects=True, check_suffix=False, load=load, load_waveforms=False
        )

        self.info_dict = deepcopy(self._recgen.info)
        self.channel_positions = self._recgen.channel_positions
        if self.load_analogsignal:
            self._recordings = self._recgen.recordings
        if self.load_spiketrains:
            self._spiketrains = self._recgen.spiketrains

        self._sampling_rate = self.info_dict["recordings"]["fs"]
        self.duration_seconds = self.info_dict["recordings"]["duration"]
        self._num_frames = int(self._sampling_rate * self.duration_seconds)
        self._num_channels = self.channel_positions.shape[0]
        self._dtype = self.info_dict["recordings"]["dtype"]

        signal_buffers = [("Signals", "0")] if self.load_analogsignal else []
        signal_streams = [("Signals", "0", "0")] if self.load_analogsignal else []
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_buffers = np.array(signal_buffers, dtype=_signal_buffer_dtype)

        sig_channels = []
        if self.load_analogsignal:
            for c in range(self._num_channels):
                ch_name = f"ch{c}"
                chan_id = str(c + 1)
                sr = self._sampling_rate  # Hz
                dtype = self._dtype
                units = "uV"
                gain = 1.0
                offset = 0.0
                stream_id = "0"
                buffer_id = "0"
                sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id, buffer_id))

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # creating units channels
        spike_channels = []
        if self.load_spiketrains:
            for c in range(len(self._spiketrains)):
                unit_name = f"unit{c}"
                unit_id = f"#{c}"
                # if spiketrains[c].waveforms is not None:
                wf_units = ""
                wf_gain = 1.0
                wf_offset = 0.0
                wf_left_sweep = 0
                wf_sampling_rate = self._sampling_rate
                spike_channels.append(
                    (unit_name, unit_id, wf_units, wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate)
                )

        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = sig_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()
        for block_index in range(1):
            bl_ann = self.raw_annotations["blocks"][block_index]
            bl_ann["mearec_info"] = self.info_dict

    def _segment_t_start(self, block_index, seg_index):
        all_starts = [[0.0]]
        return all_starts[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._num_frames / self._sampling_rate
        all_stops = [[t_stop]]
        return all_stops[block_index][seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("`stream_index` must be 0")
        return self._num_frames

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("`stream_index` must be 0")
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):

        if not self.load_analogsignal:
            raise AttributeError("Recordings not loaded. Set load_analogsignal=True in MEArecRawIO constructor")

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._num_frames

        if channel_indexes is None:
            channel_indexes = slice(self._num_channels)
        if isinstance(channel_indexes, slice):
            raw_signals = self._recordings[i_start:i_stop, channel_indexes]
        else:
            # sort channels because h5py neeeds sorted indexes
            if np.any(np.diff(channel_indexes) < 0):
                sorted_channel_indexes = np.sort(channel_indexes)
                sorted_idx = np.array([list(sorted_channel_indexes).index(ch) for ch in channel_indexes])
                raw_signals = self._recordings[i_start:i_stop, sorted_channel_indexes]
                raw_signals = raw_signals[:, sorted_idx]
            else:
                raw_signals = self._recordings[i_start:i_stop, channel_indexes]
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):

        return len(self._spiketrains[unit_index])

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):

        spike_timestamps = self._spiketrains[unit_index].times.magnitude
        if t_start is None:
            t_start = self._segment_t_start(block_index, seg_index)
        if t_stop is None:
            t_stop = self._segment_t_stop(block_index, seg_index)
        timestamp_idxs = np.where((spike_timestamps >= t_start) & (spike_timestamps < t_stop))

        return spike_timestamps[timestamp_idxs]

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return spike_timestamps.astype(dtype)

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        return None
