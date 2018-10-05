# -*- coding: utf-8 -*-
"""

"""
from __future__ import unicode_literals, print_function, division, absolute_import

import os

import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)


RECORD_SIZE = 1024
HEADER_SIZE = 1024

class OpenEphysRawIO(BaseRawIO):
    """
    OpenEphys GUI software offer several data format see
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491632/Data+format
    
    This class implement the legacy OpenEphys format here
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/65667092/Open+Ephys+format
    
    OpenEphy group already propose some tools here:
    https://github.com/open-ephys/analysis-tools/blob/master/OpenEphys.py
    but there is no package at pypi.
    
    So this implementation
    
    
    Its directory based with several files :
        * .continuous
        * .events
        * .spikes
    
    This class is based on:
      * this code https://github.com/open-ephys/analysis-tools/blob/master/Python3/OpenEphys.py
        done by Dan Denman and Josh Siegle
      * a previous PR done by Cristian Tatarau and Charite Berlin
    
    
    Limitation : 
      * work only if all continuous channels have the same samplerate, first timestamp and length

    """
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        info = self._info = explore_folder(self.dirname)
        
        # TODO verify consistency across file type : maybe ???
        nb_segment = len(info['continuous'])
        
        # scan for continuous files
        self._sigs_memmap = {}
        self._sig_length = {}
        self._sig_timestamp0 ={}
        sig_channels = []
        for seg_index in range(nb_segment):
            self._sigs_memmap[seg_index] = {}
        
            all_sigs_length = []
            all_timestamp0 = []
            all_samplerate = []
            for continuous_filename in info['continuous'][seg_index]:
                fullname = os.path.join(self.dirname, continuous_filename)
                chan_info = read_file_header(fullname)
                
                s = continuous_filename.replace('.continuous', '').split('_')
                processor_id, ch_name = s[0], s[1]
                chan_id = int(ch_name.replace('CH', '')) # FIXME: this is wrong when several processor ids

                data_chan = np.memmap(fullname, mode='r', offset=HEADER_SIZE,
                                        dtype=continuous_dtype)
                self._sigs_memmap[seg_index][chan_id] = data_chan

                all_sigs_length.append(data_chan.size*RECORD_SIZE)
                all_timestamp0.append(data_chan[0]['timestamp'])
                all_samplerate.append(chan_info['sampleRate'])

                # check for continuity (no gaps)
                diff = np.diff(data_chan['timestamp'])
                assert np.all(diff==RECORD_SIZE), 'Not continuous timestamps for {}'.format(continuous_filename)

                if seg_index == 0:
                    # add in channel list
                    sig_channels.append((ch_name, chan_id, chan_info['sampleRate'],
                                'int16', 'V', chan_info['bitVolts'], 0., int(processor_id)))
                else:
                    # check that it exist in channel list
                    pass #TODO
            
            # chech that all signals have the same lentgh and timestamp0 for this segment
            assert all(all_sigs_length[0] == e for e in all_sigs_length), 'All signals do not have the same lentgh'
            assert all(all_timestamp0[0] == e for e in all_timestamp0), 'All signals do not have the same first timestamp'
            assert all(all_samplerate[0] == e for e in all_samplerate), 'All signals do not have the same sample rate'
            
            self._sig_length[seg_index] = all_sigs_length[0]
            self._sig_timestamp0[seg_index] = all_timestamp0[0]

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        self._sig_sampling_rate = sig_channels['sampling_rate'][0] # unique for channel
        
        
        # scan for spikes files
        #TODO
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        
        
        # event file are:
        #    * all_channel.events (header + binray)  -->  event 0
        # and message.events (text based)      --> event 1 not implemented yet
        event_channels = []
        self._events_memmap = {}
        for seg_index in range(nb_segment):
            if seg_index == 0:
                event_filename = 'all_channels.events'
            else:
                event_filename = 'all_channels_{}.events'.format(seg_index+1)

            fullname = os.path.join(self.dirname, event_filename)
            event_info = read_file_header(fullname)
            self._event_sampling_rate = event_info['sampleRate']
            data_event = np.memmap(fullname, mode='r', offset=HEADER_SIZE,
                                    dtype=events_dtype)
            self._events_memmap[seg_index] = data_event
        
        event_channels.append(('all_channels', '', 'event'))
        # event_channels.append(('message', '', 'event')) # not implemented
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        

        # main header
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels


        # get info from one channel and make then as global info
        if len(info['continuous'][0])>0:
            fullname = os.path.join(self.dirname, info['continuous'][0][0])
            chan_info = read_file_header(fullname)
            openephys_version = chan_info['version']
            # TODO date_created
        else:
            openephys_version = None

        # Annotate some objects
        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['openephys_version'] = openephys_version

    def _segment_t_start(self, block_index, seg_index):
        # segment start/stop are difine by  continuous channels
        return self._sig_timestamp0[seg_index] / self._sig_sampling_rate

    def _segment_t_stop(self, block_index, seg_index):
        return (self._sig_timestamp0[seg_index] + self._sig_length[seg_index]) / self._sig_sampling_rate

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return self._sig_length[seg_index]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return self._sig_timestamp0[seg_index] / self._sig_sampling_rate

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._sig_length[seg_index]

        block_start = i_start // RECORD_SIZE
        block_stop = i_stop // RECORD_SIZE + 1
        sl0 = i_start % RECORD_SIZE
        sl1 = sl0 + (i_stop - i_start)

        if channel_indexes is None:
            channel_indexes = slice(None)
        channel_ids = self.header['signal_channels'][channel_indexes]['id']

        sigs_chunk = np.zeros((i_stop - i_start, len(channel_ids)), dtype='int16')
        for i, chan_id in enumerate(channel_ids):
            data = self._sigs_memmap[seg_index][chan_id]
            sub = data[block_start:block_stop]
            sigs_chunk[:, i] = sub['samples'].flatten()[sl0:sl1]

        return sigs_chunk

        
        

    def _spike_count(self, block_index, seg_index, unit_index):
        return 0

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        if event_channel_index==0:
            return self._events_memmap[seg_index].size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        if event_channel_index==0:
            if t_start is None:
                t_start = self._segment_t_start(block_index, seg_index)
            if t_stop is None:
                t_stop = self._segment_t_stop(block_index, seg_index)
            ts0 = int(t_start * self._event_sampling_rate)
            ts1 = int(t_stop * self._event_sampling_rate)
            ts = self._events_memmap[seg_index]['timestamp']
            keep = (ts>=ts0) & (ts<=ts1)
            
            subdata = self._events_memmap[seg_index][keep]
            timestamps = subdata['timestamp']
            # question what is the label????
            # here I put a combinaison
            labels = np.array(['{}#{}#{}'.format(int(d['event_type']), int(d['processor_id']), int(d['chan_id']) ) for d in subdata])
            durations = None
            
            return timestamps, durations, labels


    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype) / self._event_sampling_rate
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None




continuous_dtype = [('timestamp', 'int64'), ('nb_sample', 'uint16'),
    ('rec_num', 'uint16'), ('samples', 'int16', RECORD_SIZE),
    ('markers', 'uint8', 10)]

events_dtype = [('timestamp', 'int64'), ('sample_pos', 'int16'), 
    ('event_type', 'uint8'), ('processor_id', 'uint8'), 
    ('event_id', 'uint8'), ('chan_id', 'uint8'),
    ('record_num', 'uint16')]



def explore_folder(dirname):
    """
    This explore a folder and disptach coninuous, event and spikes
    files by segment (aka recording session).
    
    The nb of segment is check with this rules
    "100_CH0.continuous" ---> seg_index 0
    "100_CH0_2.continuous" ---> seg_index 1
    "100_CH0_N.continuous" ---> seg_index N-1

    """
    filenames = os.listdir(dirname)
    
    info = {}
    
    info['continuous'] = {}
    #~ info['events'] = {}
    info['spikes'] = {}
    for filename in filenames:
        if filename.endswith('.continuous'):
            s = filename.replace('.continuous', '').split('_')
            if len(s) ==2:
                seg_index = 0
            else:
                seg_index = int(s[2]) -1
            if seg_index not in info['continuous'].keys():
                info['continuous'][seg_index] = []
            info['continuous'][seg_index].append(filename)
        #~ elif filename.endswith('.events'):
            #~ # here ticky case because of iconsistency of naming and the use of '_'
            #~ s = filename.replace('.events', '').replace('all_channels', 'allChannel').split('_')
            #~ if len(s) ==1:
                #~ seg_index = 0
            #~ else:
                #~ seg_index = int(s[1]) -1
            #~ if seg_index not in info['events'].keys():
                #~ info['events'][seg_index] = []
            #~ info['events'][seg_index].append(filename)
        elif filename.endswith('.spikes'):
            s = filename.replace('.spikes', '').split('_')
            if len(s) ==1:
                seg_index = 0
            else:
                seg_index = int(s[1]) -1
            if seg_index not in info['spikes'].keys():
                info['spikes'][seg_index] = []
            info['spikes'][seg_index].append(filename)

    # TODO sort by channel number for continuous

    return info
    






def read_file_header(filename):
    """Read header information from the first 1024 bytes of an OpenEphys file.
    
    Args:
        f: An open file handle to an OpenEphys file
    
    Returns: dict with the following keys.
        - bitVolts : float, scaling factor, microvolts per bit
        - blockLength : int, e.g. 1024, length of each record (see 
            loadContinuous)
        - bufferSize : int, e.g. 1024
        - channel : the channel, eg "'CH1'"
        - channelType : eg "'Continuous'"
        - date_created : eg "'15-Jun-2016 21212'" (What are these numbers?)
        - description : description of the file format
        - format : "'Open Ephys Data Format'"
        - header_bytes : int, e.g. 1024
        - sampleRate : float, e.g. 30000.
        - version: eg '0.4'
        Note that every value is a string, even numeric data like bitVolts.
        Some strings have extra, redundant single apostrophes.
    """
    header = {}
    with open(filename, mode='rb') as f:
        # Read the data as a string
        # Remove newlines and redundant "header." prefixes
        # The result should be a series of "key = value" strings, separated
        # by semicolons.
        header_string = f.read(HEADER_SIZE).replace(b'\n',b'').replace(b'header.',b'')
    #~ print(header_string)

    # Parse each key = value string separately
    for pair in header_string.split(b';'):
        if b'=' in pair:
            key, value = pair.split(b' = ')
            key = key.strip().decode('ascii')
            value = value.strip()
            
            # Convert some values to numeric
            if key in ['bitVolts', 'sampleRate']:
                header[key] = float(value)
            elif key in ['blockLength', 'bufferSize', 'header_bytes']:
                header[key] = int(value)
            else:
                # Keep as string
                header[key] = value.decode('ascii')

    return header
    