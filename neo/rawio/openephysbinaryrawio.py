"""
This module implements the "new" binary OpenEphys format.
In this format channels are interleaved in one file.


See
https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html

Author: Julia Sprenger, Samuel Garcia, and Alessio Buccino
"""


import os
import re
import json

from pathlib import Path

import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
    _spike_channel_dtype, _event_channel_dtype)


class OpenEphysBinaryRawIO(BaseRawIO):
    """
    Handle several Blocks and several Segments.

    Parameters
    ----------
    dirname : str
        Path to Open Ephys directory
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
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname='', experiment_names=None):
        BaseRawIO.__init__(self)
        self.dirname = dirname
        if experiment_names is not None:
            if isinstance(experiment_names, str):
                experiment_names = [experiment_names]
        self.experiment_names = experiment_names
        self.folder_structure = None

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        folder_structure, all_streams, nb_block, nb_segment_per_block, possible_experiments = \
            explore_folder(self.dirname, self.experiment_names)
        check_folder_consistency(folder_structure, possible_experiments)
        self.folder_structure = folder_structure

        # all streams are consistent across blocks and segments
        sig_stream_names = sorted(list(all_streams[0][0]['continuous'].keys()))
        event_stream_names = sorted(list(all_streams[0][0]['events'].keys()))

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
                    d = all_streams[block_index][seg_index]['continuous'][stream_name]
                    d['stream_name'] = stream_name
                    self._sig_streams[block_index][seg_index][stream_index] = d
                for i, stream_name in enumerate(event_stream_names):
                    d = all_streams[block_index][seg_index]['events'][stream_name]
                    d['stream_name'] = stream_name
                    self._evt_streams[block_index][seg_index][i] = d

        # signals zone
        # create signals channel map: several channel per stream
        signal_channels = []
        for stream_index, stream_name in enumerate(sig_stream_names):
            # stream_index is the index in vector sytream names
            stream_id = str(stream_index)
            d = self._sig_streams[0][0][stream_index]
            new_channels = []
            for chan_info in d['channels']:
                chan_id = chan_info['channel_name']
                if chan_info["units"] == "":
                    # in some cases for some OE version the unit is "", but the gain is to "uV"
                    units = "uV"
                else:
                    units = chan_info["units"]
                new_channels.append((chan_info['channel_name'],
                    chan_id, float(d['sample_rate']), d['dtype'], units,
                    chan_info['bit_volts'], 0., stream_id))
            signal_channels.extend(new_channels)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        signal_streams = []
        for stream_index, stream_name in enumerate(sig_stream_names):
            stream_id = str(stream_index)
            signal_streams.append((stream_name, stream_id))
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # create memmap for signals
        for block_index in range(nb_block):
            for seg_index in range(nb_segment_per_block[block_index]):
                for stream_index, d in self._sig_streams[block_index][seg_index].items():
                    num_channels = len(d['channels'])
                    memmap_sigs = np.memmap(d['raw_filename'], d['dtype'],
                                 order='C', mode='r').reshape(-1, num_channels)
                    d['memmap'] = memmap_sigs

        # events zone
        # channel map: one channel one stream
        event_channels = []
        for stream_ind, stream_name in enumerate(event_stream_names):
            d = self._evt_streams[0][0][stream_ind]
            event_channels.append((d['channel_name'], stream_ind, 'event'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # create memmap
        for stream_ind, stream_name in enumerate(event_stream_names):
            # inject memmap loaded into main dict structure
            d = self._evt_streams[0][0][stream_ind]

            for name in _possible_event_stream_names:
                if name + '_npy' in d:
                    data = np.load(d[name + '_npy'], mmap_mode='r')
                    d[name] = data

            # check that events have timestamps
            assert 'timestamps' in d

            # for event the neo "label" will change depending the nature
            #  of event (ttl, text, binary)
            # and this is transform into unicode
            # all theses data are put in event array annotations
            if 'text' in d:
                # text case
                d['labels'] = d['text'].astype('U')
            elif 'metadata' in d:
                # binary case
                d['labels'] = d['channels'].astype('U')
            elif 'channels' in d:
                # ttl case use channels
                d['labels'] = d['channels'].astype('U')
            elif 'states' in d:
                # ttl case use states
                d['labels'] = d['states'].astype('U')
            else:
                raise ValueError(f'There is no possible labels for this event: {stream_name}')

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
                for stream_index, d in self._sig_streams[block_index][seg_index].items():
                    t_start = d['t_start']
                    dur = d['memmap'].shape[0] / float(d['sample_rate'])
                    t_stop = t_start + dur
                    if global_t_start is None or global_t_start > t_start:
                        global_t_start = t_start
                    if global_t_stop is None or global_t_stop < t_stop:
                        global_t_stop = t_stop

                # loop over events
                for stream_index, stream_name in enumerate(event_stream_names):
                    d = self._evt_streams[0][0][stream_index]
                    if d['timestamps'].size == 0:
                        continue
                    t_start = d['timestamps'][0] / d['sample_rate']
                    t_stop = d['timestamps'][-1] / d['sample_rate']
                    if global_t_start is None or global_t_start > t_start:
                        global_t_start = t_start
                    if global_t_stop is None or global_t_stop < t_stop:
                        global_t_stop = t_stop

                self._t_start_segments[block_index][seg_index] = global_t_start
                self._t_stop_segments[block_index][seg_index] = global_t_stop

        # main header
        self.header = {}
        self.header['nb_block'] = nb_block
        self.header['nb_segment'] = nb_segment_per_block
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # Annotate some objects from continuous files
        self._generate_minimal_annotations()
        for block_index in range(nb_block):
            bl_ann = self.raw_annotations['blocks'][block_index]
            for seg_index in range(nb_segment_per_block[block_index]):
                seg_ann = bl_ann['segments'][seg_index]

                # array annotations for signal channels
                for stream_index, stream_name in enumerate(sig_stream_names):
                    sig_ann = seg_ann['signals'][stream_index]
                    d = self._sig_streams[0][0][stream_index]
                    for k in ('identifier', 'history', 'source_processor_index',
                              'recorded_processor_index'):
                        if k in d['channels'][0]:
                            values = np.array([chan_info[k] for chan_info in d['channels']])
                            sig_ann['__array_annotations__'][k] = values

                # array annotations for event channels
                # use other possible data in _possible_event_stream_names
                for stream_index, stream_name in enumerate(event_stream_names):
                    ev_ann = seg_ann['events'][stream_index]
                    d = self._evt_streams[0][0][stream_index]
                    for k in _possible_event_stream_names:
                        if k in ('timestamps', ):
                            continue
                        if k in d:
                            # split custom dtypes into separate annotations
                            if d[k].dtype.names:
                                for name in d[k].dtype.names:
                                    ev_ann['__array_annotations__'][name] = d[k][name].flatten()
                            else:
                                ev_ann['__array_annotations__'][k] = d[k]

    def _segment_t_start(self, block_index, seg_index):
        return self._t_start_segments[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stop_segments[block_index][seg_index]

    def _channels_to_group_id(self, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        channels = self.header['signal_channels']
        group_ids = channels[channel_indexes]['group_id']
        assert np.unique(group_ids).size == 1
        group_id = group_ids[0]
        return group_id

    def _get_signal_size(self, block_index, seg_index, stream_index):
        sigs = self._sig_streams[block_index][seg_index][stream_index]['memmap']
        return sigs.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        t_start = self._sig_streams[block_index][seg_index][stream_index]['t_start']
        return t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        sigs = self._sig_streams[block_index][seg_index][stream_index]['memmap']
        sigs = sigs[i_start:i_stop, :]
        if channel_indexes is not None:
            sigs = sigs[:, channel_indexes]
        return sigs

    def _spike_count(self, block_index, seg_index, unit_index):
        pass

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        pass

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        pass

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        pass

    def _event_count(self, block_index, seg_index, event_channel_index):
        d = self._evt_streams[0][0][event_channel_index]
        return d['timestamps'].size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        d = self._evt_streams[0][0][event_channel_index]
        timestamps = d['timestamps']
        durations = None
        labels = d['labels']

        # slice it if needed
        if t_start is not None:
            ind_start = int(t_start * d['sample_rate'])
            mask = timestamps >= ind_start
            timestamps = timestamps[mask]
            labels = labels[mask]
        if t_stop is not None:
            ind_stop = int(t_stop * d['sample_rate'])
            mask = timestamps < ind_stop
            timestamps = timestamps[mask]
            labels = labels[mask]
        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        d = self._evt_streams[0][0][event_channel_index]
        event_times = event_timestamps.astype(dtype) / float(d['sample_rate'])
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype):
        pass


_possible_event_stream_names = ('timestamps', 'channels', 'text', 'states',
                                'full_word', 'channel_states', 'data_array', 'metadata')


def explore_folder(dirname, experiment_names=None):
    """
    Exploring the OpenEphys folder structure and structure.oebin

    Returns nested dictionary structure:
    [block_index][seg_index][stream_type][stream_information]
    where
    - node_name is the open ephys node id
    - block_index is the neo Block index
    - segment_index is the neo Segment index
    - stream_type can be 'continuous'/'events'/'spikes'
    - stream_information is a dictionary containing e.g. the sampling rate

    Parameters
    ----------
    dirname (str): Root folder of the dataset

    Returns
    -------
    nested dictionaries containing structure and stream information
    """
    nb_block = 0
    nb_segment_per_block = {}
    # nested dictionary: block_index > seg_index > data_type > stream_name
    all_streams = {}
    possible_experiment_names = []

    # folder with nodes, experiments, setting files, recordings, and streams
    folder_structure = {}

    for root, dirs, files in os.walk(dirname):
        for file in files:
            if not file == 'structure.oebin':
                continue
            root = Path(root)

            node_folder = root.parents[1]
            node_name = node_folder.stem
            if not node_name.startswith('Record'):
                # before version 5.x.x there was not multi Node recording
                # so no node_name
                node_name = ''

            if node_name not in folder_structure:
                folder_structure[node_name] = {}
                folder_structure[node_name]['experiments'] = []

            # here we skip if experiment_names is not None
            experiment_folder = root.parents[0]
            experiment_name = experiment_folder.stem
            possible_experiment_names.append(experiment_name)
            if experiment_names is not None and experiment_name not in experiment_names:
                continue
            if experiment_name not in [e['name'] for e in folder_structure[node_name]['experiments']]:
                experiment = {}
                experiment['name'] = experiment_name
                if experiment_name == 'experiment1':
                    settings_file = node_folder / "settings.xml"
                else:
                    settings_file = node_folder / f"settings_{experiment_folder.stem.replace('experiment', '')}.xml"
                experiment['settings'] = settings_file
                experiment['recordings'] = []
                folder_structure[node_name]['experiments'].append(experiment)

            recording_folder = root
            recording_name = root.stem
            if recording_name not in [r['name'] for r in folder_structure[node_name]['experiments'][-1]['recordings']]:
                recording = {}
                recording['name'] = recording_name
                recording['streams'] = {}

                # metadata
                with open(recording_folder / 'structure.oebin', encoding='utf8', mode='r') as f:
                    rec_structure = json.load(f)

                if (recording_folder / 'continuous').exists() and len(rec_structure['continuous']) > 0:
                    recording['streams']['continuous'] = {}
                    for d in rec_structure['continuous']:
                        # when multi Record Node the stream name also contains
                        # the node name to make it unique
                        oe_stream_name = Path(d["folder_name"]).name # remove trailing slash
                        stream_name = node_name + '#' + oe_stream_name
                        raw_filename = recording_folder / 'continuous' / d['folder_name'] / 'continuous.dat'

                        # Updates for OpenEphys v0.6:
                        # In new vesion (>=0.6) timestamps.npy is now called sample_numbers.npy
                        # see https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html#continuous
                        if (recording_folder / 'continuous' / d['folder_name'] / 'sample_numbers.npy').is_file():
                            timestamp_file = recording_folder / 'continuous' / d['folder_name'] / \
                                'sample_numbers.npy'
                        else:
                            timestamp_file = recording_folder / 'continuous' / d['folder_name'] / 'timestamps.npy'
                        timestamps = np.load(str(timestamp_file), mmap_mode='r')
                        timestamp0 = timestamps[0]
                        t_start = timestamp0 / d['sample_rate']

                        # TODO for later : gap checking
                        signal_stream = d.copy()
                        signal_stream['raw_filename'] = str(raw_filename)
                        signal_stream['dtype'] = 'int16'
                        signal_stream['timestamp0'] = timestamp0
                        signal_stream['t_start'] = t_start

                        recording['streams']['continuous'][stream_name] = signal_stream

                if (root / 'events').exists() and len(rec_structure['events']) > 0:
                    recording['streams']['events'] = {}
                    for d in rec_structure['events']:
                        oe_stream_name = Path(d["folder_name"]).name # remove trailing slash
                        stream_name = node_name + '#' + oe_stream_name

                        event_stream = d.copy()
                        for name in _possible_event_stream_names:
                            npy_filename = root / 'events' / d['folder_name'] / f'{name}.npy'
                            if npy_filename.is_file():
                                event_stream[f'{name}_npy'] = str(npy_filename)

                        recording['streams']['events'][stream_name] = event_stream

                folder_structure[node_name]['experiments'][-1]['recordings'].append(recording)

    # now create all_streams, nb_block, nb_segment_per_block (from first recording Node)
    recording_node = folder_structure[list(folder_structure.keys())[0]]
    nb_block = len(recording_node['experiments'])
    for block_index, experiment in enumerate(recording_node['experiments']):
        nb_segment_per_block[block_index] = len(experiment['recordings'])
        all_streams[block_index] = {}
        for seg_index, recording in enumerate(experiment['recordings']):
            all_streams[block_index][seg_index] = {}
            for stream_type in recording['streams']:
                all_streams[block_index][seg_index][stream_type] = {}
                for stream_name, signal_stream in recording['streams'][stream_type].items():
                    all_streams[block_index][seg_index][stream_type][stream_name] = signal_stream

    return folder_structure, all_streams, nb_block, nb_segment_per_block, possible_experiment_names


def check_folder_consistency(folder_structure, possible_experiment_names=None):
    # experiments across nodes
    if len(folder_structure) > 1:
        experiments = None
        for node in folder_structure.values():
            experiments_node = node['experiments']
            if experiments is None:
                experiments = experiments_node
            experiment_names = [e['name'] for e in experiments]
            assert all(ename['name'] in experiment_names for ename in experiments_node), \
                ("Inconsistent experiments across recording nodes!")

    # "continuous" streams across segments
    experiments = folder_structure[list(folder_structure.keys())[0]]['experiments']
    for experiment in experiments:
        segment_stream_names = None
        if len(experiment['recordings']) > 1:
            for recording in experiment['recordings']:
                stream_names = sorted(list(recording['streams'].keys()))
                if segment_stream_names is None:
                    segment_stream_names = stream_names
                assert segment_stream_names == stream_names, \
                    ("Inconsistent continuous streams across segments! Streams for different "
                     "segments in the same experiment must be the same. Check your open ephys "
                     "folder.")

    # "continuous" streams across blocks
    block_stream_names = None
    if len(experiments) > 1:
        for experiment in experiments:
            # use 1st segment
            stream_names = sorted(list(experiment['recordings'][0]['streams'].keys()))
            if block_stream_names is None:
                block_stream_names = stream_names
            assert block_stream_names == stream_names, \
                (f"Inconsistent continuous streams across blocks (experiments)! Streams for "
                 f"different experiments in the same folder must be the same. You can load a subset "
                 f"of experiments with the 'experiment_names' argument: {possible_experiment_names}")
