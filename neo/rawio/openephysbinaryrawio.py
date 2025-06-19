"""
This module implements the "new" binary OpenEphys format.
In this format channels are interleaved in one file.


See
https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html

Author: Julia Sprenger, Samuel Garcia, and Alessio Buccino
"""

import os
import json
from pathlib import Path
from warnings import warn

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


class OpenEphysBinaryRawIO(BaseRawWithBufferApiIO):
    """
    Handle several Blocks and several Segments.

    Parameters
    ----------
    dirname : str
        Path to Open Ephys directory
    load_sync_channel : bool
        If False (default) and a SYNC channel is present (e.g. Neuropixels), this is not loaded.
        If True, the SYNC channel is loaded and can be accessed in the analog signals.
    experiment_names : str or list or None
        If multiple experiments are available, this argument allows users to select one
        or more experiments. If None, all experiements are loaded as blocks.
        E.g. `experiment_names="experiment2"`, `experiment_names=["experiment1", "experiment2"]`

    Note
    ----
    For multi-experiment datasets, the streams need to be consistent across experiments.
    If this is not the case, you can select a subset of experiments with the `experiment_names`
    argument.

    # Correspondencies
    Neo          OpenEphys
    block[n-1]   experiment[n]    New device start/stop
    segment[s-1] recording[s]     New recording start/stop

    This IO handles several signal streams.
    Special event (npy) data are represented as array_annotations.
    The current implementation does not handle spiking data, this will be added upon user request

    """

    extensions = ["xml", "oebin", "txt", "dat", "npy"]
    rawmode = "one-dir"

    def __init__(self, dirname="", load_sync_channel=False, experiment_names=None):
        BaseRawWithBufferApiIO.__init__(self)
        self.dirname = dirname
        if experiment_names is not None:
            if isinstance(experiment_names, str):
                experiment_names = [experiment_names]
        self.experiment_names = experiment_names
        self.load_sync_channel = load_sync_channel
        if load_sync_channel:
            warn(
                "The load_sync_channel=True option is deprecated and will be removed in version 0.15. "
                "Use load_sync_channel=False instead, which will add sync channels as separate streams.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.folder_structure = None
        self._use_direct_evt_timestamps = None

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        folder_structure, all_streams, nb_block, nb_segment_per_block, possible_experiments = explore_folder(
            self.dirname, self.experiment_names
        )
        check_folder_consistency(folder_structure, possible_experiments)
        self.folder_structure = folder_structure

        # all streams are consistent across blocks and segments.
        # also checks that 'continuous' and 'events' folder are present
        if "continuous" in all_streams[0][0]:
            sig_stream_names = sorted(list(all_streams[0][0]["continuous"].keys()))
        else:
            sig_stream_names = []
        if "events" in all_streams[0][0]:
            event_stream_names = sorted(list(all_streams[0][0]["events"].keys()))
        else:
            event_stream_names = []

        self._num_of_signal_streams = len(sig_stream_names)

        # first loop to reassign stream by "stream_index" instead of "stream_name"
        self._sig_streams = {}
        self._evt_streams = {}
        for block_index in range(nb_block):
            self._sig_streams[block_index] = {}
            self._evt_streams[block_index] = {}
            for seg_index in range(nb_segment_per_block[block_index]):
                self._sig_streams[block_index][seg_index] = {}
                self._evt_streams[block_index][seg_index] = {}
                for stream_index, stream_name in enumerate(sig_stream_names):
                    info_cnt = all_streams[block_index][seg_index]["continuous"][stream_name]
                    info_cnt["stream_name"] = stream_name
                    self._sig_streams[block_index][seg_index][stream_index] = info_cnt

                    # check for SYNC channel for Neuropixels streams
                    has_sync_trace = any(["SYNC" in ch["channel_name"] for ch in info_cnt["channels"]])
                    self._sig_streams[block_index][seg_index][stream_index]["has_sync_trace"] = has_sync_trace
                for i, stream_name in enumerate(event_stream_names):
                    info_evt = all_streams[block_index][seg_index]["events"][stream_name]
                    info_evt["stream_name"] = stream_name
                    self._evt_streams[block_index][seg_index][i] = info_evt

        # signals zone
        # create signals channel map: several channel per stream
        signal_channels = []
        sync_stream_id_to_buffer_id = {}
        normal_stream_id_to_sync_stream_id = {}
        for stream_index, stream_name in enumerate(sig_stream_names):
            # stream_index is the index in vector stream names
            stream_id = str(stream_index)
            buffer_id = stream_id
            info = self._sig_streams[0][0][stream_index]
            new_channels = []
            for chan_info in info["channels"]:
                chan_id = chan_info["channel_name"]

                units = chan_info["units"]
                channel_stream_id = stream_id
                if units == "":
                    # When units are not provided they are microvolts for neural channels and volts for ADC channels
                    # See https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html#continuous
                    units = "uV" if "ADC" not in chan_id else "V"

                # Special cases for stream
                if "SYNC" in chan_id and not self.load_sync_channel:
                    # Every stream sync channel is added as its own stream
                    sync_stream_id = f"{stream_name}SYNC"
                    sync_stream_id_to_buffer_id[sync_stream_id] = buffer_id

                    # We save this mapping for the buffer description protocol
                    normal_stream_id_to_sync_stream_id[stream_id] = sync_stream_id
                    # We then set the stream_id to the sync stream id
                    channel_stream_id = sync_stream_id

                if "ADC" in chan_id:
                    # These are non-neural channels and their stream should be separated
                    # We defined their stream_id as the stream_index of neural data plus the number of neural streams
                    # This is to not break backwards compatbility with the stream_id numbering
                    channel_stream_id = str(stream_index + len(sig_stream_names))

                gain = float(chan_info["bit_volts"])
                sampling_rate = float(info["sample_rate"])
                offset = 0.0
                new_channels.append(
                    (
                        chan_info["channel_name"],
                        chan_id,
                        sampling_rate,
                        info["dtype"],
                        units,
                        gain,
                        offset,
                        channel_stream_id,
                        buffer_id,
                    )
                )
            signal_channels.extend(new_channels)

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        signal_streams = []
        signal_buffers = []

        unique_streams_ids = np.unique(signal_channels["stream_id"])

        # This is getting too complicated, we probably should just have a table which would be easier to read
        # And for users to understand
        for stream_id in unique_streams_ids:

            # Handle sync channel on a special way
            if "SYNC" in stream_id:
                # This is a sync channel and should not be added to the signal streams
                buffer_id = sync_stream_id_to_buffer_id[stream_id]
                stream_name = stream_id
                signal_streams.append((stream_name, stream_id, buffer_id))
                continue

            # Neural signal
            stream_index = int(stream_id)
            if stream_index < self._num_of_signal_streams:
                stream_name = sig_stream_names[stream_index]
                buffer_id = stream_id
                # We add the buffers here as both the neural and the ADC channels are in the same buffer
                signal_buffers.append((stream_name, buffer_id))
            else:  # This names the ADC streams
                neural_stream_index = stream_index - self._num_of_signal_streams
                neural_stream_name = sig_stream_names[neural_stream_index]
                stream_name = f"{neural_stream_name}_ADC"
                buffer_id = str(neural_stream_index)
            signal_streams.append((stream_name, stream_id, buffer_id))

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_buffers = np.array(signal_buffers, dtype=_signal_buffer_dtype)

        # create memmap for signals
        self._buffer_descriptions = {}
        self._stream_buffer_slice = {}
        for block_index in range(nb_block):
            self._buffer_descriptions[block_index] = {}
            for seg_index in range(nb_segment_per_block[block_index]):
                self._buffer_descriptions[block_index][seg_index] = {}
                for stream_index, info in self._sig_streams[block_index][seg_index].items():
                    num_channels = len(info["channels"])
                    stream_id = str(stream_index)
                    buffer_id = str(stream_index)
                    shape = get_memmap_shape(info["raw_filename"], info["dtype"], num_channels=num_channels, offset=0)
                    self._buffer_descriptions[block_index][seg_index][buffer_id] = {
                        "type": "raw",
                        "file_path": str(info["raw_filename"]),
                        "dtype": info["dtype"],
                        "order": "C",
                        "file_offset": 0,
                        "shape": shape,
                    }

                    has_sync_trace = self._sig_streams[block_index][seg_index][stream_index]["has_sync_trace"]

                    # check sync channel validity (only for AP and LF)
                    if not has_sync_trace and self.load_sync_channel and "NI-DAQ" not in info["stream_name"]:
                        raise ValueError(
                            "SYNC channel is not present in the recording. " "Set load_sync_channel to False"
                        )

                    # Check if ADC and non-ADC channels are contiguous
                    is_channel_adc = ["ADC" in ch["channel_name"] for ch in info["channels"]]
                    if any(is_channel_adc):
                        first_adc_index = is_channel_adc.index(True)
                        non_adc_channels_after_adc_channels = [
                            not is_adc for is_adc in is_channel_adc[first_adc_index:]
                        ]
                        if any(non_adc_channels_after_adc_channels):
                            raise ValueError(
                                "Interleaved ADC and non-ADC channels are not supported. "
                                "ADC channels must be contiguous. Open an issue in python-neo to request this feature."
                            )

                    # Find sync channel and verify it's the last channel
                    sync_index = next(
                        (index for index, ch in enumerate(info["channels"]) if ch["channel_name"].endswith("_SYNC")),
                        None,
                    )
                    if sync_index is not None and sync_index != num_channels - 1:
                        raise ValueError(
                            "SYNC channel must be the last channel in the buffer. Open an issue in python-neo to request this feature."
                        )

                    neural_channels = [ch for ch in info["channels"] if "ADC" not in ch["channel_name"]]
                    adc_channels = [ch for ch in info["channels"] if "ADC" in ch["channel_name"]]
                    num_neural_channels = len(neural_channels)
                    num_adc_channels = len(adc_channels)

                    if num_adc_channels == 0:
                        if has_sync_trace and not self.load_sync_channel:
                            # Exclude the sync channel from the main stream
                            self._stream_buffer_slice[stream_id] = slice(None, -1)

                            # Add a buffer slice for the sync channel
                            sync_stream_id = normal_stream_id_to_sync_stream_id[stream_id]
                            self._stream_buffer_slice[sync_stream_id] = slice(-1, None)
                        else:
                            self._stream_buffer_slice[stream_id] = None
                    else:
                        stream_id_neural = stream_id
                        stream_id_non_neural = str(int(stream_id) + self._num_of_signal_streams)

                        self._stream_buffer_slice[stream_id_neural] = slice(0, num_neural_channels)

                        if has_sync_trace and not self.load_sync_channel:
                            # Exclude the sync channel from the non-neural stream
                            self._stream_buffer_slice[stream_id_non_neural] = slice(num_neural_channels, -1)

                            # Add a buffer slice for the sync channel
                            sync_stream_id = normal_stream_id_to_sync_stream_id[stream_id]
                            self._stream_buffer_slice[sync_stream_id] = slice(-1, None)
                        else:
                            self._stream_buffer_slice[stream_id_non_neural] = slice(num_neural_channels, None)

        # events zone
        # channel map: one channel one stream
        event_channels = []
        for stream_ind, stream_name in enumerate(event_stream_names):
            info = self._evt_streams[0][0][stream_ind]
            if "states" in info or "channel_states" in info:
                evt_channel_type = "epoch"
            else:
                evt_channel_type = "event"
            event_channels.append((info["channel_name"], info["channel_name"], evt_channel_type))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # create memmap for events
        for block_index in range(nb_block):
            for seg_index in range(nb_segment_per_block[block_index]):
                for stream_index, info in self._evt_streams[block_index][seg_index].items():
                    for name in _possible_event_stream_names:
                        if name + "_npy" in info:
                            data = np.load(info[name + "_npy"], mmap_mode="r")
                            info[name] = data
                    # check that events have timestamps
                    assert "timestamps" in info, "Event stream does not have timestamps!"
                    # Updates for OpenEphys v0.6:
                    # In new vesion (>=0.6) timestamps.npy is now called sample_numbers.npy
                    # The timestamps are already in seconds, so that event times don't require scaling
                    # see https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html#events
                    if "sample_numbers" in info:
                        self._use_direct_evt_timestamps = True
                    else:
                        self._use_direct_evt_timestamps = False

                    # for event the neo "label" will change depending the nature
                    #  of event (ttl, text, binary)
                    # and this is transform into unicode
                    # all theses data are put in event array annotations
                    if "text" in info:
                        # text case
                        info["labels"] = info["text"].astype("U")
                    elif "metadata" in info:
                        # binary case
                        info["labels"] = info["channels"].astype("U")
                    elif "channels" in info:
                        # ttl case use channels
                        info["labels"] = info["channels"].astype("U")
                    elif "states" in info:
                        # ttl case use states
                        info["labels"] = info["states"].astype("U")
                    else:
                        raise ValueError(f"There is no possible labels for this event!")

                    # # If available, use 'states' to compute event duration
                    info["durations"] = None
                    # 'states' was introduced in OpenEphys v0.6. For previous versions, events used 'channel_states'
                    if "states" in info or "channel_states" in info:
                        states = info["channel_states"] if "channel_states" in info else info["states"]

                        if states.size > 0:
                            timestamps = info["timestamps"]
                            labels = info["labels"]

                            # Identify unique channels based on state values
                            channels = np.unique(np.abs(states))

                            rising_indices = []
                            falling_indices = []

                            # all channels are packed into the same `states` array.
                            # So the states array includes positive and negative values for each channel:
                            #  for example channel one rising would be +1 and channel one falling would be -1,
                            # channel two rising would be +2 and channel two falling would be -2, etc.
                            # This is the case for sure for version >= 0.6.x.
                            for channel in channels:
                                # Find rising and falling edges for each channel
                                rising = np.where(states == channel)[0]
                                falling = np.where(states == -channel)[0]

                                # Ensure each rising has a corresponding falling
                                if rising.size > 0 and falling.size > 0:
                                    if rising[0] > falling[0]:
                                        falling = falling[1:]
                                    if rising.size > falling.size:
                                        rising = rising[:-1]

                                    # ensure that the number of rising and falling edges are the same:
                                    if len(rising) != len(falling):
                                        warn(
                                            f"Channel {channel} has {len(rising)} rising edges and "
                                            f"{len(falling)} falling edges. The number of rising and "
                                            f"falling edges should be equal. Skipping events from this channel."
                                        )
                                        continue

                                    rising_indices.extend(rising)
                                    falling_indices.extend(falling)

                            rising_indices = np.array(rising_indices, dtype=np.int64)
                            falling_indices = np.array(falling_indices, dtype=np.int64)

                            # Sort the indices to maintain chronological order
                            sorted_order = np.argsort(rising_indices)
                            rising_indices = rising_indices[sorted_order]
                            falling_indices = falling_indices[sorted_order]

                            durations = timestamps[falling_indices] - timestamps[rising_indices]
                            if not self._use_direct_evt_timestamps:
                                timestamps = timestamps / info["sample_rate"]
                                durations = durations / info["sample_rate"]

                            info["rising"] = rising_indices
                            info["timestamps"] = timestamps[rising_indices]
                            info["labels"] = labels[rising_indices]
                            info["durations"] = durations

        # no spike read yet
        # can be implemented on user demand
        spike_channels = np.array([], dtype=_spike_channel_dtype)

        # loop for t_start/t_stop on segment browse all object
        self._t_start_segments = {}
        self._t_stop_segments = {}
        for block_index in range(nb_block):
            self._t_start_segments[block_index] = {}
            self._t_stop_segments[block_index] = {}
            for seg_index in range(nb_segment_per_block[block_index]):
                global_t_start = None
                global_t_stop = None

                # loop over signals
                for stream_index, info in self._sig_streams[block_index][seg_index].items():
                    t_start = info["t_start"]
                    stream_id = str(stream_index)
                    buffer_id = str(stream_index)
                    sig_size = self._buffer_descriptions[block_index][seg_index][buffer_id]["shape"][0]
                    dur = sig_size / float(info["sample_rate"])
                    t_stop = t_start + dur
                    if global_t_start is None or global_t_start > t_start:
                        global_t_start = t_start
                    if global_t_stop is None or global_t_stop < t_stop:
                        global_t_stop = t_stop

                # loop over events
                for stream_index, stream_name in enumerate(event_stream_names):
                    info = self._evt_streams[block_index][seg_index][stream_index]
                    if info["timestamps"].size == 0:
                        continue
                    t_start = info["timestamps"][0]
                    t_stop = info["timestamps"][-1]

                    if not self._use_direct_evt_timestamps:
                        t_start /= info["sample_rate"]
                        t_stop /= info["sample_rate"]

                    if global_t_start is None or global_t_start > t_start:
                        global_t_start = t_start
                    if global_t_stop is None or global_t_stop < t_stop:
                        global_t_stop = t_stop

                self._t_start_segments[block_index][seg_index] = global_t_start
                self._t_stop_segments[block_index][seg_index] = global_t_stop

        # main header
        self.header = {}
        self.header["nb_block"] = nb_block
        self.header["nb_segment"] = nb_segment_per_block
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # Annotate some objects from continuous files
        self._generate_minimal_annotations()
        for block_index in range(nb_block):
            bl_ann = self.raw_annotations["blocks"][block_index]
            for seg_index in range(nb_segment_per_block[block_index]):
                seg_ann = bl_ann["segments"][seg_index]

                # array annotations for signal channels
                for stream_index, stream_name in enumerate(self.header["signal_streams"]["name"]):
                    sig_ann = seg_ann["signals"][stream_index]
                    if stream_index < self._num_of_signal_streams:
                        _sig_stream_index = stream_index
                        is_neural_stream = True
                    else:
                        _sig_stream_index = stream_index - self._num_of_signal_streams
                        is_neural_stream = False
                    info = self._sig_streams[block_index][seg_index][_sig_stream_index]
                    has_sync_trace = self._sig_streams[block_index][seg_index][_sig_stream_index]["has_sync_trace"]

                    for key in ("identifier", "history", "source_processor_index", "recorded_processor_index"):
                        if key in info["channels"][0]:
                            values = np.array([chan_info[key] for chan_info in info["channels"]])

                            if has_sync_trace:
                                values = values[:-1]

                            neural_channels = [ch for ch in info["channels"] if "ADC" not in ch["channel_name"]]
                            num_neural_channels = len(neural_channels)
                            if is_neural_stream:
                                values = values[:num_neural_channels]
                            else:
                                values = values[num_neural_channels:]

                            sig_ann["__array_annotations__"][key] = values

                # array annotations for event channels
                # use other possible data in _possible_event_stream_names
                for stream_index, stream_name in enumerate(event_stream_names):
                    ev_ann = seg_ann["events"][stream_index]
                    info = self._evt_streams[0][0][stream_index]
                    if "rising" in info:
                        selected_indices = info["rising"]
                    else:
                        selected_indices = None
                    for k in _possible_event_stream_names:
                        if k in ("timestamps", "rising"):
                            continue
                        if k in info:
                            # split custom dtypes into separate annotations
                            if info[k].dtype.names:
                                for name in info[k].dtype.names:
                                    arr_ann = info[k][name].flatten()
                                    if selected_indices is not None:
                                        arr_ann = arr_ann[selected_indices]
                                    ev_ann["__array_annotations__"][name] = arr_ann
                            else:
                                arr_ann = info[k]
                                if selected_indices is not None:
                                    arr_ann = arr_ann[selected_indices]
                                ev_ann["__array_annotations__"][k] = arr_ann

    def _segment_t_start(self, block_index, seg_index):
        return self._t_start_segments[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stop_segments[block_index][seg_index]

    def _channels_to_group_id(self, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        channels = self.header["signal_channels"]
        group_ids = channels[channel_indexes]["group_id"]
        assert np.unique(group_ids).size == 1
        group_id = group_ids[0]
        return group_id

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        if stream_index < self._num_of_signal_streams:
            _sig_stream_index = stream_index
        else:
            _sig_stream_index = stream_index - self._num_of_signal_streams

        t_start = self._sig_streams[block_index][seg_index][_sig_stream_index]["t_start"]
        return t_start

    def _spike_count(self, block_index, seg_index, unit_index):
        pass

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        pass

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        pass

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        pass

    def _event_count(self, block_index, seg_index, event_channel_index):
        timestamps, _, _ = self._get_event_timestamps(block_index, seg_index, event_channel_index, None, None)
        return timestamps.size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        info = self._evt_streams[block_index][seg_index][event_channel_index]
        timestamps = info["timestamps"]
        durations = info["durations"]
        labels = info["labels"]

        # slice it if needed
        if t_start is not None:
            if not self._use_direct_evt_timestamps:
                ind_start = int(t_start * info["sample_rate"])
                mask = timestamps >= ind_start
            else:
                mask = timestamps >= t_start
            timestamps = timestamps[mask]
            labels = labels[mask]
        if t_stop is not None:
            if not self._use_direct_evt_timestamps:
                ind_stop = int(t_stop * info["sample_rate"])
                mask = timestamps < ind_stop
            else:
                mask = timestamps < t_stop
            timestamps = timestamps[mask]
            labels = labels[mask]
        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        info = self._evt_streams[0][0][event_channel_index]
        if not self._use_direct_evt_timestamps:
            event_times = event_timestamps.astype(dtype) / float(info["sample_rate"])
        else:
            event_times = event_timestamps.astype(dtype)
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        info = self._evt_streams[0][0][event_channel_index]
        if not self._use_direct_evt_timestamps:
            durations = raw_duration.astype(dtype) / float(info["sample_rate"])
        else:
            durations = raw_duration.astype(dtype)
        return durations

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self._buffer_descriptions[block_index][seg_index][buffer_id]


_possible_event_stream_names = (
    "timestamps",
    "sample_numbers",
    "channels",
    "text",
    "states",
    "full_word",
    "channel_states",
    "data_array",
    "metadata",
)


def explore_folder(dirname, experiment_names=None):
    """
    Exploring the OpenEphys folder structure, by looping through the
    folder to find recordings.

    Parameters
    ----------
    dirname (str): Root folder of the dataset

    Returns
    -------
    folder_structure: dict
        The folder_structure is dictionary that describes the Open Ephys folder.
        Dictionary structure:
        [node_name]["experiments"][exp_id]["recordings"][rec_id][stream_type][stream_information]
    all_streams: dict
        From the folder_structure, the another dictionary is reorganized with NEO-like
        indexing: block_index (experiments) and seg_index (recordings):
        Dictionary structure:
        [block_index][seg_index][stream_type][stream_information]
        where
        - node_name is the open ephys node id
        - block_index is the neo Block index
        - segment_index is the neo Segment index
        - stream_type can be 'continuous'/'events'/'spikes'
        - stream_information is a dictionary containing e.g. the sampling rate
    nb_block : int
        Number of blocks (experiments) loaded
    nb_segment_per_block : dict
        Dictionary with number of segment per block.
        Keys are block indices, values are number of segments
    possible_experiment_names : list
        List of all available experiments in the Open Ephys folder
    """
    # folder with nodes, experiments, setting files, recordings, and streams
    folder_structure = {}
    possible_experiment_names = []

    for root, dirs, files in os.walk(dirname):
        for file in files:
            if not file == "structure.oebin":
                continue
            root = Path(root)

            node_folder = root.parents[1]
            node_name = node_folder.stem
            if not node_name.startswith("Record"):
                # before version 5.x.x there was not multi Node recording
                # so no node_name
                node_name = ""

            if node_name not in folder_structure:
                folder_structure[node_name] = {}
                folder_structure[node_name]["experiments"] = {}

            # here we skip if experiment_names is not None
            experiment_folder = root.parents[0]
            experiment_name = experiment_folder.stem
            experiment_id = int(experiment_name.replace("experiment", ""))
            if experiment_name not in possible_experiment_names:
                possible_experiment_names.append(experiment_name)
            if experiment_names is not None and experiment_name not in experiment_names:
                continue
            if experiment_id not in folder_structure[node_name]["experiments"]:
                experiment = {}
                experiment["name"] = experiment_name
                if experiment_name == "experiment1":
                    settings_file = node_folder / "settings.xml"
                else:
                    settings_file = node_folder / f"settings_{experiment_id}.xml"
                experiment["settings_file"] = settings_file
                experiment["recordings"] = {}
                folder_structure[node_name]["experiments"][experiment_id] = experiment

            recording_folder = root
            recording_name = root.stem
            recording_id = int(recording_name.replace("recording", ""))
            # add recording
            recording = {}
            recording["name"] = recording_name
            recording["streams"] = {}

            # metadata
            with open(recording_folder / "structure.oebin", encoding="utf8", mode="r") as f:
                rec_structure = json.load(f)

            if (recording_folder / "continuous").exists() and len(rec_structure["continuous"]) > 0:
                recording["streams"]["continuous"] = {}
                for info in rec_structure["continuous"]:
                    # when multi Record Node the stream name also contains
                    # the node name to make it unique
                    oe_stream_name = info["folder_name"].split("/")[0]  # remove trailing slash
                    if len(node_name) > 0:
                        stream_name = node_name + "#" + oe_stream_name
                    else:
                        stream_name = oe_stream_name

                    # skip streams if folder is on oebin, but doesn't exist
                    if not (recording_folder / "continuous" / info["folder_name"]).is_dir():
                        warn(
                            f"For {recording_folder} the folder continuous/{info['folder_name']} is missing. "
                            f"Skipping {stream_name} continuous stream."
                        )
                        continue

                    raw_filename = recording_folder / "continuous" / info["folder_name"] / "continuous.dat"

                    # Updates for OpenEphys v0.6:
                    # In new vesion (>=0.6) timestamps.npy is now called sample_numbers.npy
                    # see https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html#continuous
                    sample_numbers = recording_folder / "continuous" / info["folder_name"] / "sample_numbers.npy"
                    if sample_numbers.is_file():
                        timestamp_file = sample_numbers
                    else:
                        timestamp_file = recording_folder / "continuous" / info["folder_name"] / "timestamps.npy"
                    timestamps = np.load(str(timestamp_file), mmap_mode="r")
                    timestamp0 = timestamps[0]
                    t_start = timestamp0 / info["sample_rate"]

                    # TODO for later : gap checking
                    signal_stream = info.copy()
                    signal_stream["raw_filename"] = str(raw_filename)
                    signal_stream["dtype"] = "int16"
                    signal_stream["timestamp0"] = timestamp0
                    signal_stream["t_start"] = t_start

                    recording["streams"]["continuous"][stream_name] = signal_stream

            if (root / "events").exists() and len(rec_structure["events"]) > 0:
                recording["streams"]["events"] = {}
                for info in rec_structure["events"]:
                    # when multi Record Node the stream name also contains
                    # the node name to make it unique
                    oe_stream_name = info["folder_name"].split("/")[0]  # remove trailing slash
                    if len(node_name) > 0:
                        stream_name = node_name + "#" + oe_stream_name
                    else:
                        stream_name = oe_stream_name

                    # skip streams if folder is on oebin, but doesn't exist
                    if not (recording_folder / "events" / info["folder_name"]).is_dir():
                        warn(
                            f"For {recording_folder} the folder events/{info['folder_name']} is missing. "
                            f"Skipping {stream_name} event stream."
                        )
                        continue

                    event_stream = info.copy()
                    for name in _possible_event_stream_names:
                        npy_filename = root / "events" / info["folder_name"] / f"{name}.npy"
                        if npy_filename.is_file():
                            event_stream[f"{name}_npy"] = str(npy_filename)

                    recording["streams"]["events"][stream_name] = event_stream

            folder_structure[node_name]["experiments"][experiment_id]["recordings"][recording_id] = recording

    # now create all_streams, nb_block, nb_segment_per_block
    # nested dictionary: block_index > seg_index > data_type > stream_name
    all_streams = {}
    nb_segment_per_block = {}
    record_node_names = list(folder_structure.keys())
    if len(record_node_names) == 0:
        raise ValueError(
            f"{dirname} is not a valid Open Ephys binary folder. No 'structure.oebin' "
            f"files were found in sub-folders."
        )
    recording_node = folder_structure[record_node_names[0]]

    # nb_block needs to be consistent across record nodes. Use the first one
    nb_block = len(recording_node["experiments"])

    for node_id, recording_node in folder_structure.items():
        exp_ids_sorted = sorted(list(recording_node["experiments"].keys()))
        for block_index, exp_id in enumerate(exp_ids_sorted):
            experiment = recording_node["experiments"][exp_id]
            nb_segment_per_block[block_index] = len(experiment["recordings"])
            if block_index not in all_streams:
                all_streams[block_index] = {}

            rec_ids_sorted = sorted(list(experiment["recordings"].keys()))
            for seg_index, rec_id in enumerate(rec_ids_sorted):
                recording = experiment["recordings"][rec_id]
                if seg_index not in all_streams[block_index]:
                    all_streams[block_index][seg_index] = {}
                for stream_type in recording["streams"]:
                    if stream_type not in all_streams[block_index][seg_index]:
                        all_streams[block_index][seg_index][stream_type] = {}
                    for stream_name, signal_stream in recording["streams"][stream_type].items():
                        all_streams[block_index][seg_index][stream_type][stream_name] = signal_stream

    # natural sort possible experiment names
    experiment_order = np.argsort([int(exp.replace("experiment", "")) for exp in possible_experiment_names])
    possible_experiment_names = list(np.array(possible_experiment_names)[experiment_order])

    return folder_structure, all_streams, nb_block, nb_segment_per_block, possible_experiment_names


def check_folder_consistency(folder_structure, possible_experiment_names=None):
    # check that experiment names are the same for differend record nodes
    if len(folder_structure) > 1:
        experiments = None
        for node in folder_structure.values():
            experiments_node = node["experiments"]
            if experiments is None:
                experiments = experiments_node
            experiment_names = [e["name"] for e_id, e in experiments.items()]
            assert all(
                ename["name"] in experiment_names for ename in experiments_node.values()
            ), "Inconsistent experiments across recording nodes!"

    # check that "continuous" streams are the same across multiple segments (recordings)
    record_node_names = list(folder_structure.keys())
    experiments = folder_structure[record_node_names[0]]["experiments"]
    for exp_id, experiment in experiments.items():
        segment_stream_names = None
        if len(experiment["recordings"]) > 1:
            for rec_id, recording in experiment["recordings"].items():
                stream_names = sorted(list(recording["streams"]["continuous"].keys()))
                if segment_stream_names is None:
                    segment_stream_names = stream_names
                assert segment_stream_names == stream_names, (
                    "Inconsistent continuous streams across segments! Streams for different "
                    "segments in the same experiment must be the same. Check your open ephys "
                    "folder."
                )

    # check that "continuous" streams across blocks (experiments)
    block_stream_names = None
    if len(experiments) > 1:
        for exp_id, experiment in experiments.items():
            # use 1st segment
            rec_ids = list(experiment["recordings"])
            stream_names = list(experiment["recordings"][rec_ids[0]]["streams"]["continuous"].keys())
            stream_names = sorted(stream_names)
            if block_stream_names is None:
                block_stream_names = stream_names
            assert block_stream_names == stream_names, (
                f"Inconsistent continuous streams across blocks (experiments)! Streams for "
                f"different experiments in the same folder must be the same. You can load a "
                f"subset of experiments with the 'experiment_names' argument: "
                f"{possible_experiment_names}"
            )
