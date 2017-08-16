# -*- coding: utf-8 -*-
"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV, NSE and NTT file formats.


NCS contains signals for one channel
NEV contains events
NSE contains spikes and waveforms
NTT contains


NCS can contains gaps that can be detected in inregularity in timestamps
of datat block. Each gap lead to one new segment.
NCVS files need to be read entirly to detect that gaps.... too bad....



Author: Julia Sprenger, Carlos Canova, Samuel Garcia

"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, 
        _event_channel_dtype)

import numpy as np
import os
import re
import distutils.version
import datetime
from collections import OrderedDict


BLOCK_SIZE = 512 #nb sample per signal block

class NeuralynxRawIO(BaseRawIO):
    extensions = ['nse', 'ncs', 'nev', 'ntt']
    rawmode = 'one-dir'
    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname
    
    def _source_name(self):
        return self.dirname
    
    def _parse_header(self):
        
        sig_channels = []
        unit_channels = []#NOT DONE
        event_channels = []#NOT DONE
        
        self._sigs_sampling_rate = None
        self._sigs_t_start = {}#key is seg_index
        self._sigs_t_stop = {}#key is seg_index
        self._raw_sigs_by_chans = {}#key is seg_index then chan_id
        
        
        # explore the directory looking for ncs, nev, nse and ntt
        for filename in os.listdir(self.dirname):
            filename = os.path.join(self.dirname, filename)
            
            _, ext = os.path.splitext(filename)
            ext = ext[1:]#remove dot
            if ext not in self.extensions:
                continue
            
            #header
            info = read_txt_header(filename)
            
            if ext=='ncs':
                # a signal channels
                chan_name = info['channel_name']
                chan_id = info['channel_id']
                units = 'mV'
                gain = info['bit_to_microVolt']
                if info['input_inverted']:
                    gain *= -1
                offset = 0.
                sig_channels.append((chan_name, chan_id, units, gain,offset))
                if self._sigs_sampling_rate is None:
                    self._sigs_sampling_rate = info['sampling_rate']
                else:
                    assert self._sigs_sampling_rate == info['sampling_rate'], 'Sampling is not the same across NCS'
                
                data = np.memmap(filename, dtype=ncs_dtype, mode='r', offset=2**14)
                #detect gaps
                good_delta = int(BLOCK_SIZE*1e6/self._sigs_sampling_rate)
                timestamps = data['timestamp']
                deltas = np.diff(timestamps)
                gap_indexes, = np.nonzero(deltas!=good_delta)
                gap_bounds = [0] + (gap_indexes+1).tolist() + [data.size]
                
                #create segment with subdata block/t_start/t_stop
                for seg_index in range(len(gap_bounds)-1):
                    i0 = gap_bounds[seg_index]
                    i1 = gap_bounds[seg_index+1]
                    if seg_index not in self._raw_sigs_by_chans:
                        self._raw_sigs_by_chans[seg_index] = {}
                    subdata = data[i0:i1]
                    self._raw_sigs_by_chans[seg_index][chan_id] = subdata

                    t_start = subdata[0]['timestamp']/1e6
                    if seg_index not in self._sigs_t_start:
                        self._sigs_t_start[seg_index] = t_start
                    else:
                        assert self._sigs_t_start[seg_index] == t_start

                    t_stop = subdata[-1]['timestamp']/1e6 + BLOCK_SIZE/self._sigs_sampling_rate
                    if seg_index not in self._sigs_t_stop:
                        self._sigs_t_stop[seg_index] = t_stop
                    else:
                        assert self._sigs_t_stop[seg_index] == t_stop
            
            elif ext=='nse':
                pass

            elif ext=='nev':
                pass

            elif ext=='ntt':
                pass
            
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        
        
        if len(sig_channels)>0:
            nb_segment = len(self._raw_sigs_by_chans)
            #test if all segment length are the same for all channels
            for seg_index in range(nb_segment):
                datalength = [ data.size for data in self._raw_sigs_by_chans[seg_index].values()]
                assert np.unique(datalength).size<=1
            
            self._sigs_length = []
            for seg_index in range(nb_segment):
                length = list(self._raw_sigs_by_chans[seg_index].values())[0].size * BLOCK_SIZE
                self._sigs_length.append(length)
            
        else:
            nb_segment = 1
        
        
        #TODO global t_start/t_stop that include event and spikes
        self.global_t_start = self._sigs_t_start
        self.global_t_stop = self._sigs_t_stop
        
        #fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels
        
        #TODO
        # Annotations
        self._generate_minimal_annotations()
        #~ bl_annotations = self.raw_annotations['blocks'][0]
        #~ seg_annotations = bl_annotations['segments'][0]

    def _block_count(self):
        return 1
    
    def _segment_count(self, block_index):
        return self.header['nb_segment'][block_index]
    
    def _segment_t_start(self, block_index, seg_index):
        return self.global_t_start[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        return self.global_t_stop[seg_index]
    

    def _analogsignal_shape(self, block_index, seg_index):
        return (self._sigs_length[seg_index], len(self.header['signal_channels']))
    
    def _analogsignal_sampling_rate(self):
        return self._sigs_sampling_rate

    def _get_analogsignal_chunk(self, block_index, seg_index,  i_start, i_stop, channel_indexes):
        if i_start is None: i_start=0
        if i_stop is None: i_stop=self._sigs_length[seg_index]
        
        block_start = i_start//BLOCK_SIZE
        block_stop = i_stop//BLOCK_SIZE+1
        sl0 = i_start % 512
        sl1 = sl0 + (i_stop-i_start)
        
        channel_ids = self.header['signal_channels'][channel_indexes]['id']
        
        sigs_chunk = np.zeros((i_stop-i_start, len(channel_indexes)), dtype='int16')
        for i, chan_id in enumerate(channel_ids):
            data = self._raw_sigs_by_chans[seg_index][chan_id]
            sub = data[block_start:block_stop]
            sigs_chunk[:, i] = sub['samples'].flatten()[sl0:sl1]
        
        return sigs_chunk
    
    ###
    # spiketrain and unit zone
    def _spike_count(self,  block_index, seg_index, unit_index):
        raise(NotImplementedError)
    
    def _spike_timestamps(self,  block_index, seg_index, unit_index, t_start, t_stop):
        raise(NotImplementedError)
    
    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        raise(NotImplementedError)

    ###
    # spike waveforms zone
    def _spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        raise(NotImplementedError)
    
    ###
    # event and epoch zone
    def _event_count(self, block_index, seg_index, event_channel_index):
        raise(NotImplementedError)
    
    def _event_timestamps(self,  block_index, seg_index, event_channel_index, t_start, t_stop):
        raise(NotImplementedError)
    
    def _rescale_event_timestamp(self, event_timestamps, dtype):
        raise(NotImplementedError)
    
    def _rescale_epoch_duration(self, raw_duration, dtype):
        raise(NotImplementedError)




# keys in 
txt_header_keys = [
    ('AcqEntName', 'channel_name', None),#used
    ('FileType', '', None),
    ('FileVersion', '', None),
    ('RecordSize', '', None),
    ('HardwareSubSystemName', '', None),
    ('HardwareSubSystemType', '', None),
    ('SamplingFrequency', 'sampling_rate', float),#used
    ('ADMaxValue', '', None),
    ('ADBitVolts', 'bit_to_microVolt', float),#used
    ('NumADChannels', '', None),
    ('ADChannel', 'channel_id', int),#used
    ('InputRange', '', None),
    ('InputInverted', 'input_inverted', bool),#used
    ('DSPLowCutFilterEnabled', '', None),
    ('DspLowCutFrequency', '', None),
    ('DspLowCutNumTaps', '', None),
    ('DspLowCutFilterType', '', None),
    ('DSPHighCutFilterEnabled', '', None),
    ('DspHighCutFrequency', '', None),
    ('DspHighCutNumTaps', '', None),
    ('DspHighCutFilterType', '', None),
    ('DspDelayCompensation', '', None),
    ('DspFilterDelay_\xb5s', '', None),
    ('DisabledSubChannels', '', None),
    ('WaveformLength', '', None),
    ('AlignmentPt', '', None),
    ('ThreshVal', '', None),
    ('MinRetriggerSamples', '', None),
    ('SpikeRetriggerTime', '', None),
    ('DualThresholding', '', None),
    ('Feature Peak 0', '', None),
    ('Feature Valley 1', '', None),
    ('Feature Energy 2', '', None),
    ('Feature Height 3', '', None),
    ('Feature NthSample 4', '', None),
    ('Feature NthSample 5', '', None),
    ('Feature NthSample 6', '', None),
    ('Feature NthSample 7', '', None),
    ('SessionUUID', '', None),
    ('FileUUID', '', None),
    ('CheetahRev', 'version', None),#used  possibilty 1 for version
    ('ProbeName', '', None),
    ('OriginalFileName', '', None),
    ('TimeCreated', '', None),
    ('TimeClosed', '', None),
    ('ApplicationName Cheetah', 'version', None),#used  possibilty 2 for version
    ('AcquisitionSystem', '', None),
    ('ReferenceChannel',  '', None),
]


def read_txt_header(filename):
    """
    All file in neuralynx contains a 16kB hedaer in txt
    format.
    This function parse it to create info dict.
    This include datetime
    """
    with  open(filename, 'rb') as f:
        txt_header = f.read(2**14)
    txt_header = txt_header.strip(b'\x00').decode('latin-1')
    
    # find keys
    #info = {}
    info = OrderedDict()
    for k1, k2, type_ in txt_header_keys:
        pattern = '-'+k1+' (\S+)'
        r = re.findall(pattern, txt_header)
        if len(r) == 1:
            if k2 =='':
                k2=k1
            info[k2] = r[0]
            if type_ is not None:
                info[k2] = type_(info[k2])
    
    #some conversions
    if 'bit_to_microVolt' in info:
        info['bit_to_microVolt'] = info['bit_to_microVolt']*1e6
    if 'version' in info:
        version = info['version'].replace('"', '')
        info['version'] = distutils.version.LooseVersion(version)
    
    # filename and datetime
    if info['version']<='5.6.4':
        datetime1_regex = '## Time Opened \(m/d/y\): (?P<date>\S+)  \(h:m:s\.ms\) (?P<time>\S+)'
        datetime2_regex = '## Time Closed \(m/d/y\): (?P<date>\S+)  \(h:m:s\.ms\) (?P<time>\S+)'
        filename_regex = '## File Name (?P<filename>\S+)'
        datetimeformat = '%m/%d/%Y %H:%M:%S.%f'        
    else:
        datetime1_regex = '-TimeCreated (?P<date>\S+) (?P<time>\S+)'
        datetime2_regex = '-TimeClosed (?P<date>\S+) (?P<time>\S+)'
        filename_regex = '-OriginalFileName "?(?P<filename>\S+)"?'
        datetimeformat = '%Y/%m/%d %H:%M:%S'
    filename = re.search(filename_regex, txt_header).groupdict()['filename']
    
    dt1 = re.search(datetime1_regex, txt_header).groupdict()
    dt2 = re.search(datetime2_regex, txt_header).groupdict()

    info['recording_opened'] = datetime.datetime.strptime(dt1['date'] + ' ' +dt1['time'],
                                           datetimeformat)
    info['recording_closed'] = datetime.datetime.strptime(dt2['date'] + ' ' +dt2['time'],
                                           datetimeformat)
    
    #~ for k, v in info.items():
        #~ print(' ', k, ':', v)
    
    return info


ncs_dtype = np.dtype([('timestamp', 'uint64'),
                ('channel', 'uint32'),
                ('sample_rate', 'uint32'),
                ('nb_valid', 'uint32'),
                ('samples', 'int16', (BLOCK_SIZE,))
                ])


    
