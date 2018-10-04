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
        print(info)
        
        if len(info['continuous_filenames'])>0:
            fullname = os.path.join(self.dirname, info['continuous_filenames'][0])
            chan_info = read_file_header(fullname)
            openephys_version = chan_info['version']
        else:
            openephys_version = None
        
        all_sigs_length = []
        all_timestamp0 = []
        all_samplerate = []
        self._sigs_memmap = {}
        sig_channels = []
        for continuous_filename in info['continuous_filenames']:
            fullname = os.path.join(self.dirname, continuous_filename)
            chan_info = read_file_header(fullname)

            processor_id, ch_name = continuous_filename.replace('.continuous', '').split('_')
            chan_id = int(ch_name.replace('CH', '')) # FIXME: this is wrong when several processor ids

            data_chan = np.memmap(fullname, mode='r', offset=HEADER_SIZE,
                                    dtype=continuous_dtype)
            self._sigs_memmap[chan_id] = data_chan

            all_sigs_length.append(data_chan.size*RECORD_SIZE)
            all_timestamp0.append(data_chan[0]['timestamp'])
            all_samplerate.append(chan_info['sampleRate'])

            # check for continuity (no gaps)
            diff = np.diff(data_chan['timestamp'])
            assert np.all(diff==RECORD_SIZE), 'Not continuous timestamps for {}'.format(continuous_filename)

            sig_channels.append((ch_name, chan_id, chan_info['sampleRate'],
                        'int16', 'V', chan_info['bitVolts'], 0., int(processor_id)))

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
            
        
        
        # chech that all signals have the same lentgh and timestamp0
        assert all(all_sigs_length[0] == e for e in all_sigs_length), 'All signals do not have the same lentgh'
        assert all(all_timestamp0[0] == e for e in all_timestamp0), 'All signals do not have the same first timestamp'
        assert all(all_samplerate[0] == e for e in all_samplerate), 'All signals do not have the same sample rate'
        
        self._sig_length = all_sigs_length[0]
        self._sig_timestamp0 = all_timestamp0[0]
        self._sig_sample_rate = all_samplerate[0]
        
        unit_channels = []
        #~ for c in range(3):
            #~ unit_name = 'unit{}'.format(c)
            #~ unit_id = '#{}'.format(c)
            #~ wf_units = 'uV'
            #~ wf_gain = 1000. / 2 ** 16
            #~ wf_offset = 0.
            #~ wf_left_sweep = 20
            #~ wf_sampling_rate = 10000.
            #~ unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  #~ wf_offset, wf_left_sweep, wf_sampling_rate))
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        event_channels = []
        #~ event_channels.append(('Some events', 'ev_0', 'event'))
        #~ event_channels.append(('Some epochs', 'ep_1', 'epoch'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # Annotate some objects
        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['openephys_version'] = openephys_version

    def _segment_t_start(self, block_index, seg_index):
        # segment start/stop are difine by  continuous channels
        return self._sig_timestamp0 / self._sig_sample_rate

    def _segment_t_stop(self, block_index, seg_index):
        return (self._sig_timestamp0 + self._sig_length) / self._sig_sample_rate

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return self._sig_length

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return self._sig_timestamp0 / self._sig_sample_rate

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._sig_length

        block_start = i_start // RECORD_SIZE
        block_stop = i_stop // RECORD_SIZE + 1
        sl0 = i_start % RECORD_SIZE
        sl1 = sl0 + (i_stop - i_start)

        if channel_indexes is None:
            channel_indexes = slice(None)
        channel_ids = self.header['signal_channels'][channel_indexes]['id']

        sigs_chunk = np.zeros((i_stop - i_start, len(channel_ids)), dtype='int16')
        for i, chan_id in enumerate(channel_ids):
            data = self._sigs_memmap[chan_id]
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
        return 0

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        return None, None, None

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        return None

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None


def explore_folder(dirname):
    info = {}
    
    info['continuous_filenames'] = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        if filename.endswith('.continuous'):
            info['continuous_filenames'].append(filename)
            
            
    
    
    return info
    


continuous_dtype = [('timestamp', 'int64'), ('nb_sample', 'uint16'),
            ('rec_num', 'uint16'), ('samples', 'int16', RECORD_SIZE),
            ('markers', 'uint8', 10)]




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
    