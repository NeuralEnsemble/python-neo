"""
Class for reading MED (Multiscale Electrophysiology Data) Format.

Uses the dhn-med-py python package, created by Dark Horse Neuro, Inc.

Authors: Dan Crepeau, Matt Stead
"""

import numpy as np

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


class MedRawIO(BaseRawIO):
    """
    Class for reading MED (Multiscale Electrophysiology Data) Format.

    Uses the dhn-med-py MED python package (version >= 1.0.0), created by
    Dark Horse Neuro, Inc. and medformat.org.

    Parameters
    ----------
    dirname: str | Path | None, default: None
        The folder containing the data files to load
    password: str | None, default: None
        The password for the Med session
    keep_original_times: bool, default: False
        If True UTC timestamps are used and returned as seconds referenced
        to midnight 1 Jan 1970
        If False timestamps are referenced to the beginning of the session
        with the beginning being 0

    Notes
    -----
    Currently reads the entire MED session.  Every discontinuity is considered
    to be a new segment.  Channels are grouped by sampling frequency, to
    create streams.  In MED all channels will line up time-wise, so streams
    will span the entire recording, and continuous sections of those streams
    are divided up into segments.


    """

    extensions = ["medd", "rdat", "ridx"]
    rawmode = "one-dir"

    def __init__(self, dirname=None, password=None, keep_original_times=False, **kargs):
        BaseRawIO.__init__(self, **kargs)

        import dhn_med_py
        from dhn_med_py import MedSession

        self.dirname = str(dirname)
        self.password = password
        self.keep_original_times = keep_original_times

    def _source_name(self):
        return self.dirname

    def _parse_header(self):

        import dhn_med_py
        from dhn_med_py import MedSession

        # Set a default password to improve compatibility and ease-of-use.
        # This password will be ignored if an unencrypted MED dataset is being used.
        if self.password is None:
            self.password = "L2_password"

        # Open the MED session (open data file pointers and read metadata files)
        self.sess = MedSession(self.dirname, self.password)

        # set the matrix calls to be "sample_major" as opposed to "channel_major"
        self.sess.set_major_dimension("sample")

        # find number of segments
        sess_contigua = self.sess.session_info["contigua"]
        self._nb_segment = len(sess_contigua)

        # find overall session start time.
        self._session_start_time = sess_contigua[0]["start_time"]

        # keep track of offset from metadata, if we are keeping original times.
        if not self.keep_original_times:
            self._session_time_offset = 0 - self._session_start_time
        else:
            self._session_time_offset = self.sess.session_info["metadata"]["recording_time_offset"]

        # find start/stop times of each segment
        self._seg_t_starts = []
        self._seg_t_stops = []
        for seg_idx in range(self._nb_segment):
            self._seg_t_starts.append(sess_contigua[seg_idx]["start_time"])
            self._seg_t_stops.append(sess_contigua[seg_idx]["end_time"])

        # find number of streams per segment
        self._stream_info = []
        self._num_stream_info = 0
        for chan_idx, chan_info in enumerate(self.sess.session_info["channels"]):
            chan_freq = chan_info["metadata"]["sampling_frequency"]

            # set MED session reference channel to be this channel, so the correct contigua is returned
            self.sess.set_reference_channel(chan_info["metadata"]["channel_name"])
            contigua = self.sess.find_discontinuities()

            # find total number of samples in this channel
            chan_num_samples = 0
            for seg_idx in range(len(contigua)):
                chan_num_samples += (contigua[seg_idx]["end_index"] - contigua[seg_idx]["start_index"]) + 1

            # see if we need a new stream, or add channel to existing stream
            add_to_existing_stream_info = False
            for stream_info in self._stream_info:
                if chan_freq == stream_info["sampling_frequency"] and chan_num_samples == stream_info["num_samples"]:
                    # found a match, so add it!
                    add_to_existing_stream_info = True
                    stream_info["chan_list"].append((chan_idx, chan_info["metadata"]["channel_name"]))
                    stream_info["raw_chans"].append(chan_info["metadata"]["channel_name"])
                    break

            if not add_to_existing_stream_info:
                self._num_stream_info += 1

                new_stream_info = {
                    "sampling_frequency": chan_info["metadata"]["sampling_frequency"],
                    "chan_list": [(chan_idx, chan_info["metadata"]["channel_name"])],
                    "contigua": contigua,
                    "raw_chans": [chan_info["metadata"]["channel_name"]],
                    "num_samples": chan_num_samples,
                }

                self._stream_info.append(new_stream_info)

        self.num_channels_in_session = len(self.sess.session_info["channels"])
        self.num_streams_in_session = self._num_stream_info

        signal_streams = []
        signal_channels = []

        # fill in signal_streams and signal_channels info
        for signal_stream_counter, stream_info in enumerate(self._stream_info):

            # get the stream start time, which is the start time of the first continuous section
            stream_start_time = (stream_info["contigua"][0]["start_time"] + self._session_time_offset) / 1e6

            # create stream name/id with info that we now have
            name = f'stream (rate,#sample,t0): ({stream_info["sampling_frequency"]}, {stream_info["num_samples"]}, {stream_start_time})'
            stream_id = signal_stream_counter
            buffer_id = ""
            signal_streams.append((name, stream_id, buffer_id))

            # add entry for signal_channels for each channel in a stream
            for chan in stream_info["chan_list"]:
                signal_channels.append(
                    (chan[1], chan[0], stream_info["sampling_frequency"], "int32", "uV", 1, 0, stream_id, buffer_id)
                )

        # the MED format is one dir per channel and so no buffer concept
        signal_buffers = np.array([], dtype=_signal_buffer_dtype)
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # no unit/epoch information contained in MED
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # Events
        event_channels = []
        event_channels.append(("Event", "event_channel", "event"))
        event_channels.append(("Epoch", "epoch_channel", "epoch"))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # Create Neo header structure
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [self._nb_segment]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # `_generate_minimal_annotations()` must be called to generate the nested
        # dict of annotations/array_annotations
        self._generate_minimal_annotations()

        # Add custom annotations for neo objects
        bl_ann = self.raw_annotations["blocks"][0]
        bl_ann["name"] = "MED Data Block"
        # The following adds all of the MED session_info to the block annotations,
        # which includes features like patient name, recording location, etc.
        bl_ann.update(self.sess.session_info)

        # Give segments unique names
        for i in range(self._nb_segment):
            seg_ann = bl_ann["segments"][i]
            seg_ann["name"] = "Seg #" + str(i) + " Block #0"

        # this pprint lines really help for understand the nested (and complicated sometimes) dict
        # from pprint import pprint
        # pprint(self.raw_annotations)

    def _segment_t_start(self, block_index, seg_index):
        return (self._seg_t_starts[seg_index] + self._session_time_offset) / 1e6

    def _segment_t_stop(self, block_index, seg_index):
        return (self._seg_t_stops[seg_index] + self._session_time_offset) / 1e6

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_segment_contigua = self._stream_info[stream_index]["contigua"]
        return (stream_segment_contigua[seg_index]["end_index"] - stream_segment_contigua[seg_index]["start_index"]) + 1

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return (self._seg_t_starts[seg_index] + self._session_time_offset) / 1e6

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):

        import dhn_med_py
        from dhn_med_py import MedSession

        # Correct for None start/stop inputs
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.get_signal_size(block_index=block_index, seg_index=seg_index, stream_index=stream_index)

        # Check for invalid start/stop inputs
        if i_start < 0 or i_stop > self.get_signal_size(
            block_index=block_index, seg_index=seg_index, stream_index=stream_index
        ):
            raise IndexError("MED read error: Too many samples requested!")
        if i_start > i_stop:
            raise IndexError("MED read error: i_start (" + i_start + ") is greater than i_stop (" + i_stop + ")")

        num_channels = 0
        if channel_indexes is None:
            self.sess.set_channel_inactive("all")
            self.sess.set_channel_active(self._stream_info[stream_index]["raw_chans"])
            num_channels = len(self._stream_info[stream_index]["raw_chans"])
            self.sess.set_reference_channel(self._stream_info[stream_index]["raw_chans"][0])

        # in the case we have a slice or we give an ArrayLike we need to iterate through the channels
        # in order to activate them.
        else:
            if isinstance(channel_indexes, slice):
                start = channel_indexes.start or 0
                stop = channel_indexes.stop or len(self._stream_info[stream_index]["raw_chans"])
                step = channel_indexes.step or 1
                channel_indexes = [ch for ch in range(start, stop, step)]
            else:
                if any(channel_indexes < 0):
                    raise IndexError(f"Can not index negative channels: {channel_indexes}")
            # Set all channels to be inactive, then selectively set some of them to be active
            self.sess.set_channel_inactive("all")
            for i, channel_idx in enumerate(channel_indexes):
                num_channels += 1
                self.sess.set_channel_active(self._stream_info[stream_index]["raw_chans"][channel_idx])
            self.sess.set_reference_channel(self._stream_info[stream_index]["raw_chans"][channel_indexes[0]])

        # Return empty dataset if start/stop samples are equal
        if i_start == i_stop:
            raw_signals = np.zeros((0, num_channels), dtype="int32")
            return raw_signals

        # Otherwise, return the matrix 2D array returned by the MED library
        start_sample_offset = self._stream_info[stream_index]["contigua"][seg_index]["start_index"]
        self.sess.read_by_index(i_start + start_sample_offset, i_stop + start_sample_offset)

        raw_signals = np.empty((i_stop - i_start, num_channels), dtype=np.int32)
        for i, chan in enumerate(self.sess.data["channels"]):
            raw_signals[:, i] = chan["data"]

        return raw_signals

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        return None

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):

        # there are no epochs to consider for MED in this interface
        if self.header["event_channels"]["type"][event_channel_index] == b"epoch":
            return 0

        records = self.sess.get_session_records()

        # Find segment boundaries
        ts0 = (self.segment_t_start(block_index, seg_index) * 1e6) - self._session_time_offset
        ts1 = (self.segment_t_stop(block_index, seg_index) * 1e6) - self._session_time_offset

        # Only count Note and Neuralynx type records.
        count = 0
        for record in records:
            if (record["type_string"] == "Note" or record["type_string"] == "NlxP") and (
                record["start_time"] >= ts0 and record["start_time"] < ts1
            ):
                count += 1

        return count

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):

        # There are no epochs to consider for MED in this interface,
        # so just bail out if that's what's being asked for
        if self.header["event_channels"]["type"][event_channel_index] == b"epoch":
            return np.array([]), np.array([]), np.array([], dtype="U")

        if t_start is not None:
            start_time = (t_start * 1e6) - self._session_time_offset
        else:
            start_time = (self.segment_t_start(block_index, seg_index) * 1e6) - self._session_time_offset

        if t_stop is not None:
            end_time = (t_stop * 1e6) - self._session_time_offset
        else:
            end_time = (self.segment_t_stop(block_index, seg_index) * 1e6) - self._session_time_offset

        # Ask MED session for a list of events that match time parameters
        records = self.sess.get_session_records(start_time, end_time)

        # create a subset of only Note and Neuralynx type records
        records_subset = []
        for record in records:
            if record["type_string"] == "Note" or record["type_string"] == "NlxP":
                records_subset.append(record)
        records = records_subset

        # if no records match our criteria, then we are done, output empty arrays
        if len(records) == 0:
            return np.array([]), np.array([]), np.array([], dtype="U")

        # inialize output arrays
        times = np.empty(shape=[len(records)])
        durations = None
        labels = []

        # populate output arrays of times and labels
        for i, record in enumerate(records):
            times[i] = (record["start_time"] + self._session_time_offset) / 1e6
            if record["type_string"] == "Note":
                labels.append(record["text"])
            elif record["type_string"] == "NlxP":
                labels.append("NlxP subport: " + str(record["subport"]) + " value: " + str(record["value"]))

        labels = np.asarray(labels, dtype="U")

        return times, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        return np.asarray(event_timestamps, dtype=dtype)

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return np.asarray(raw_duration, dtype=dtype)

    def __del__(self):
        try:
            # Important to make sure the session is closed, since the MED library only allows
            # one session to be open at a time.
            self.sess.close()
            del self.sess
        except Exception:
            pass

    def close(self):
        try:
            # Important to make sure the session is closed, since the MED library only allows
            # one session to be open at a time.
            self.sess.close()
        except Exception:
            pass
