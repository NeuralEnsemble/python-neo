"""
This module implement the "new" binary OpenEphys format.
In this format channels are interleaved in one file.


See 
https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html

Author: Julia Sprenger and Samuel Garcia
"""


import os
import re
import json

from pathlib import Path

import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)


class OpenEphysBinaryRawIO(BaseRawIO):
    """
    Handle several Block and several Segment.


    # Correspondencies
    Neo          OpenEphys
    block[n-1]   experiment[n]    New device start/stop
    segment[s-1] recording[s]     New recording start/stop
    """
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        all_streams, nb_block, nb_segment_per_block = explore_folder(self.dirname)

        source_names = []
        sig_channels = []
        self._memmap_sigs = {}
        self._t_start_sigs = {}
        self._t_start_segments = {}
        self._t_stop_segments = {}
        for block_index in range(nb_block):
            self._memmap_sigs[block_index] = {}
            self._t_start_sigs[block_index] = {}
            self._t_start_segments[block_index] = {}
            self._t_stop_segments[block_index] = {}
            for seg_index in range(nb_segment_per_block[block_index]):
                self._memmap_sigs[block_index][seg_index] = []
                self._t_start_sigs[block_index][seg_index] = []
                self._t_start_segments[block_index][seg_index] = None
                self._t_stop_segments[block_index][seg_index] = None
                
        for node_name in sorted(list((all_streams.keys()))):
            for block_index in all_streams[node_name]:
                for seg_index in all_streams[node_name][block_index]:
                    for d in all_streams[node_name][block_index][seg_index]['continuous']:
                        source_name = node_name + '#' + d['name']
                        if source_name not in source_names:
                            
                            source_names.append(source_name)
                            group_id = source_names.index(source_name)

                            num_channels = len(d['channels'])
                            new_channels = []
                            for chan_info in d['channels']:
                                # using 0 as default, generate final ids later
                                chan_id = 0
                                new_channels.append((chan_info['channel_name'],
                                    chan_id, float(d['sample_rate']), d['dtype'], chan_info['units'],
                                    chan_info['bit_volts'], 0., group_id))
                            sig_channels.extend(new_channels)
                        memmap_sigs = np.memmap(d['raw_filename'], d['dtype'], order='C', mode='r').reshape(-1, num_channels)
                        self._memmap_sigs[block_index][seg_index].append(memmap_sigs)
                        self._t_start_sigs[block_index][seg_index].append(d['t_start'])

                        if self._t_start_segments[block_index][seg_index] is None or\
                                self._t_start_segments[block_index][seg_index] > d['t_start']:
                            self._t_start_segments[block_index][seg_index] = d['t_start']
                        
                        t_stop = d['t_start']  + memmap_sigs.shape[0] / float(d['sample_rate'])
                        if self._t_stop_segments[block_index][seg_index] is None or\
                                self._t_stop_segments[block_index][seg_index] < t_stop:
                            self._t_stop_segments[block_index][seg_index] = t_stop

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        sig_channels['id'] = np.arange(sig_channels.size, dtype='int')
        
        # make an array global to local channel
        self._global_channel_to_local_channel = np.zeros(sig_channels.size, dtype='int64')
        for group_id in np.unique(sig_channels['id']):
            loc_chans, = np.nonzero(sig_channels['group_id'] == group_id)
            self._global_channel_to_local_channel[loc_chans] = np.arange(loc_chans.size)

        #
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # no spike read yet
        # can be implemented on user demand
        unit_channels = np.array([], dtype=_unit_channel_dtype)

        # handle segment t_start
        # update t_start/t_stop with events
        # for block_index in range(nb_block):
        #     for seg_index in range(nb_segment_per_block[block_index]):
        #         pass
        
        # main header
        self.header = {}
        self.header['nb_block'] = nb_block
        self.header['nb_segment'] = nb_segment_per_block
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # Annotate some objects from continuous files
        self._generate_minimal_annotations()
        # TODO signals annotaions name / stream_name / ...
        '''
            "channel_name": "CH0",
            "description": "Headstage data channel",
            "identifier": "genericdata.continuous",
            "history": "File Reader -> Splitter -> Bandpass Filter -> Record Node",
            "bit_volts": 0.050000000745058059692,
            "units": "uV",
            "source_processor_index": 0,
            "recorded_processor_index": 0
        '''



    def _segment_t_start(self, block_index, seg_index):
        return self._t_start_segments[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stop_segments[block_index][seg_index]

    def _channels_to_group_id(self, channel_indexes):
        channels = self.header['signal_channels']
        group_ids = channels[channel_indexes]['group_id']
        assert np.unique(group_ids).size == 1
        group_id = group_ids[0]
        return group_id

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        group_id = self._channels_to_group_id(channel_indexes)
        sigs = self._memmap_sigs[block_index][seg_index][group_id]
        return sigs.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        group_id = self._channels_to_group_id(channel_indexes)
        t_start = self._t_start_sigs[block_index][seg_index][group_id]
        return t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        group_id = self._channels_to_group_id(channel_indexes)
        sigs = self._memmap_sigs[block_index][seg_index][group_id]
        
        sigs = sigs[i_start:i_stop, :]
        
        if channel_indexes is not None:
            local_chans = self._global_channel_to_local_channel[channel_indexes]
            sigs = sigs[:, local_chans]

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
        pass

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        pass

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        pass

    def _rescale_epoch_duration(self, raw_duration, dtype):
        pass


def explore_folder(dirname):
    """
    Exploring the OpenEphys folder structure and structure.oebin
    
    Returns nested dictionary structure:
    [node_name][block_index][seg_index][stream_type][stream_information]
    where
    - node_name is the open ephys node id
    - block_index is the neo Block index
    - segment_index is the neo Segment index
    - stream_type can be 'continuous'/'events'/'spikes'
    - stream_information is a dictionionary containing e.g. the sampling rate
    
    Parmeters
    ---------
    dirname (str): Root folder of the dataset
    
    Returns
    -------
    nested dictionaries containing structure and stream information:
   
    """
    
    # TODO make code stronger in case of experimentX and recordingY
    # are not consecutive numbers (one missing)

    nb_block = 0
    nb_segment_per_block = []
    # nested node_name / seg_index
    all_streams = {}
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if not file == 'structure.oebin':
                continue
            root = Path(root)
            
            node_name = root.parents[1].stem

            if node_name not in all_streams:
                all_streams[node_name] = {}
            
            block_index = int(root.parents[0].stem.replace('experiment', '')) - 1
            if block_index not in all_streams[node_name]:
                all_streams[node_name][block_index] = {}
                if block_index >= nb_block:
                    nb_block = block_index + 1
                    nb_segment_per_block.append(0)

            
            seg_index = int(root.stem.replace('recording', '')) -1
            if seg_index not in all_streams[node_name][block_index]:
                all_streams[node_name][block_index][seg_index] = {}
                if seg_index >= nb_segment_per_block[block_index]:
                    nb_segment_per_block[block_index] = seg_index + 1
            
            # metadata
            with open(root / 'structure.oebin', encoding='utf8', mode='r') as f:
                structure = json.load(f)

            if (root / 'continuous').exists() and len(structure['continuous']) > 0:
                all_streams[node_name][block_index][seg_index]['continuous'] = []
                for d in structure['continuous']:
                    raw_filename = root / 'continuous' / d['folder_name'] / 'continuous.dat'
                    
                    timestamp_file = root / 'continuous' / d['folder_name'] / 'timestamps.npy'
                    timestamps = np.load(str(timestamp_file), mmap_mode='r')
                    timestamp0 =  timestamps[0]
                    t_start = timestamp0 / d['sample_rate']

                    # sync_timestamp is -1 for all elements in our dataset
                    # sync_timestamp_file = root / 'continuous' / d['folder_name'] / 'synchronized_timestamps.npy'
                    # sync_timestamps = np.load(str(sync_timestamp_file), mmap_mode='r')
                    # t_start =  sync_timestamps[0]
                    
                    # TODO gap checking
                    signal_stream = d.copy()
                    signal_stream['raw_filename'] =  str(raw_filename)
                    signal_stream['name'] = raw_filename.parents[0].stem
                    signal_stream['dtype'] = 'int16'
                    signal_stream['timestamp0'] = timestamp0
                    signal_stream['t_start'] = t_start

                    all_streams[node_name][block_index][seg_index]['continuous'].append(signal_stream)
            
            if (root / 'events').exists() and len(structure['events']) > 0:
                all_streams[node_name][block_index][seg_index]['events'] = []
                for d in structure['events']:
                    
                    text_npy = root / 'events' / d['folder_name'] / 'text.npy'
                    timestamps_npy = root / 'events' / d['folder_name'] / 'timestamps.npy'
                    if text_npy.is_file():
                        # case with text event
                        event_stream = dict(
                            text_npy=str(text_npy),
                            timestamps_npy=str(timestamps_npy),
                        )
                        all_streams[node_name][block_index][seg_index]['events'].append(signal_stream)
                    
                    full_word_npy = root / 'events' / d['folder_name'] / 'full_word.npy'
                    if full_word_npy.is_file():
                        print('TTL is ignored')
    
    return all_streams, nb_block, nb_segment_per_block
    