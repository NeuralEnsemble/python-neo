# -*- coding: utf-8 -*-
"""
Class for reading data from NeuroExplorer (.nex)

Note:
  * NeuroExplorer have introduced a new .nex5 file format
    with 64 timestamps. This is NOT implemented here.
    If someone have some file in that new format we could also
    integrate it in neo
  * NeuroExplorer now provide there own python class for
    reading/writting nex and nex5. This could be usefull
    for testing this class.

Porting NeuroExplorerIO to NeuroExplorerRawIO have some
limitation because in neuro explorer signals can differents sampling
rate and shape. So NeuroExplorerRawIO can read only one channel
at once.

Documentation for dev :
http://www.neuroexplorer.com/downloadspage/


Author: Samuel Garcia, luc estebanez, mark hollenbeck

"""
from __future__ import  print_function, division, absolute_import
#from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, 
        _event_channel_dtype)

import numpy as np
from collections import OrderedDict
import datetime



class NeuroExplorerRawIO(BaseRawIO):
    extensions = ['nex']
    rawmode = 'one-file'
    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename
    
    def _source_name(self):
        return self.filename
    
    def _parse_header(self):
        with open(self.filename, 'rb') as fid:
            self.global_header = read_as_dict(fid, GlobalHeader, offset=0)
            offset = 544
            self._entity_headers = []
            for i in range(self.global_header['nvar']):
                self._entity_headers.append(read_as_dict(fid, EntityHeader, offset=offset + i * 208))
        
        self._memmap = np.memmap(self.filename, dtype='u1', mode='r')
        
        self._sig_lengths = []
        self._sig_t_starts = []
        sig_channels = []
        unit_channels = []
        event_channels = []
        for i in range(self.global_header['nvar']):
            entity_header = self._entity_headers[i]
            name = entity_header['name']
            _id = i
            if entity_header['type'] == 0:#Unit
                unit_channels.append((name, _id,'', 0,0, 0, 0))
            
            elif entity_header['type'] == 1:#Event
                event_channels.append((name, _id, 'event'))
            
            elif entity_header['type'] == 2:# interval = Epoch
                event_channels.append((name, _id, 'epoch'))
            
            elif entity_header['type'] == 3:# spiketrain and wavefoms
                wf_units = 'mV'
                wf_gain = entity_header['ADtoMV']
                wf_offset = entity_header['MVOffset']
                wf_left_sweep = 0
                wf_sampling_rate = entity_header['WFrequency']
                unit_channels.append((name, _id, wf_units, wf_gain, wf_offset, 
                                    wf_left_sweep, wf_sampling_rate))
            
            elif entity_header['type'] == 4:
                # popvectors
                pass

            if entity_header['type'] == 5:#Signals
                units = 'mV'
                sampling_rate = entity_header['WFrequency']
                dtype = 'int16'
                gain = entity_header['ADtoMV']
                offset = entity_header['MVOffset']
                group_id = 0
                sig_channels.append((name, _id, sampling_rate,dtype,  units,
                                                                    gain,offset, group_id))
                self._sig_lengths.append(entity_header['NPointsWave'])
                #sig t_start is the first timestamp if datablock
                offset = entity_header['offset']
                timestamps0 = self._memmap[offset:offset+4].view('int32')
                t_start = timestamps0[0]/self.global_header['freq']
                self._sig_t_starts.append(t_start)
                
            elif entity_header['type'] == 6:#Markers
                event_channels.append((name, _id, 'event'))
        
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        
        #each signal channel have a dierent groups that force reading
        #them one by one
        sig_channels['group_id'] = np.arange(sig_channels.size)
        
        #fill into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels
        
        #Annotations
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations['blocks'][0]
        seg_annotations = bl_annotations['segments'][0]
        for d in (bl_annotations, seg_annotations):
            d['neuroexplorer_version'] = self.global_header['version']
            d['comment'] = self.global_header['comment']
    
    def _segment_t_start(self, block_index, seg_index):
        t_start = self.global_header['tbeg'] / self.global_header['freq']
        return t_start
        
    def _segment_t_stop(self, block_index, seg_index):
        t_stop=self.global_header['tend'] / self.global_header['freq']
        return t_stop

    
    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        assert len(channel_indexes)==1 , 'only one channel by one channel'
        return self._sig_lengths[channel_indexes[0]]
    
    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        assert len(channel_indexes)==1 , 'only one channel by one channel'
        return self._sig_t_starts[channel_indexes[0]]

    def _get_analogsignal_chunk(self, block_index, seg_index,  i_start, i_stop, channel_indexes):
        assert len(channel_indexes)==1 , 'only one channel by one channel'
        channel_index = channel_indexes[0]
        entity_index = int(self.header['signal_channels'][channel_index]['id'])
        entity_header = self._entity_headers[entity_index]
        n = entity_header['n']
        nb_sample = entity_header['NPointsWave']
        #offset = entity_header['offset']
        #timestamps = self._memmap[offset:offset+n*4].view('int32')
        #offset2 = entity_header['offset'] + n*4
        #fragment_starts = self._memmap[offset2:offset2+n*4].view('int32')
        offset3 = entity_header['offset'] + n*4 + n*4
        raw_signal = self._memmap[offset3:offset3+nb_sample*2].view('int16')
        raw_signal = raw_signal[slice(i_start, i_stop), None]#2D for compliance
        return raw_signal

    
    def _spike_count(self,  block_index, seg_index, unit_index):
        entity_index = int(self.header['unit_channels'][unit_index]['id'])
        entity_header = self._entity_headers[entity_index]
        nb_spike = entity_header['n']
        return nb_spike
    
    def _get_spike_timestamps(self,  block_index, seg_index, unit_index, t_start, t_stop):
        entity_index = int(self.header['unit_channels'][unit_index]['id'])
        entity_header = self._entity_headers[entity_index]
        n = entity_header['n']
        offset = entity_header['offset']
        timestamps = self._memmap[offset:offset+n*4].view('int32')

        if t_start is not None:
            keep = timestamps>=int(t_start*self.global_header['freq'])
            timestamps = timestamps[keep]
        if t_stop is not None:
            keep = timestamps<=int(t_stop*self.global_header['freq'])
            timestamps = timestamps[keep]

        return timestamps
    
    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self.global_header['freq']
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        entity_index = int(self.header['unit_channels'][unit_index]['id'])
        entity_header = self._entity_headers[entity_index]
        if entity_header['type'] == 0:
            return None
        assert entity_header['type'] == 3
        
        n = entity_header['n']
        width = entity_header['NPointsWave']
        offset = entity_header['offset'] + n*2
        waveforms = self._memmap[offset:offset+n*2*width].view('int16')
        waveforms = waveforms.reshape(n, 1, width)
        
        return waveforms
    
    def _event_count(self, block_index, seg_index, event_channel_index):
        entity_index = int(self.header['event_channels'][event_channel_index]['id'])
        entity_header = self._entity_headers[entity_index]
        nb_event = entity_header['n']
        return nb_event
    
    def _get_event_timestamps(self,  block_index, seg_index, event_channel_index, t_start, t_stop):
        entity_index = int(self.header['event_channels'][event_channel_index]['id'])
        entity_header = self._entity_headers[entity_index]
        
        n = entity_header['n']
        offset = entity_header['offset']
        timestamps = self._memmap[offset:offset+n*4].view('int32')
        
        if t_start is None:
            i_start = None
        else:
            i_start = np.searchsorted(timestamps, int(t_start*self.global_header['freq']))
        if t_stop is None:
            i_stop = None
        else:
            i_stop = np.searchsorted(timestamps, int(t_stop*self.global_header['freq']))
        keep = slice(i_start, i_stop)
        
        timestamps = timestamps[keep]
        
        if entity_header['type'] == 1:#Event
            durations = None
            labels = np.array([''] * timestamps.size, dtype='U')
        elif entity_header['type'] == 2:#Epoch
            offset2 = offset + n*4
            stop_timestamps = self._memmap[offset2:offset2+n*4].view('int32')
            durations = stop_timestamps[keep] - timestamps
            labels = np.array([''] * timestamps.size, dtype='U')
        elif entity_header['type'] == 6:#Marker
            durations = None
            offset2 = offset + n*4 + 64
            s = entity_header['MarkerLength']
            labels = self._memmap[offset2:offset2+s*n].view('S'+str(s))
            labels = labels[keep].astype('U')
        
        return timestamps, durations, labels
    
    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype)
        event_times /= self.global_header['freq']
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype):
        durations = raw_duration.astype(dtype)
        durations /= self.global_header['freq']
        return durations


def read_as_dict(fid, dtype, offset=None):
    """
    Given a file descriptor
    and a numpy.dtype of the binary struct return a dict.
    Make conversion for strings.
    """
    if offset is not None:
        fid.seek(offset)
    dt =np.dtype(dtype)
    h = np.fromstring(fid.read(dt.itemsize), dt)[0]
    info = OrderedDict()
    for k in dt.names:
        v = h[k]
        
        if dt[k].kind == 'S':
            v = v.replace(b'\x00', b'')
            v = v.decode('utf8')
        
        info[k] = v
    return info

GlobalHeader = [
    ('signature', 'S4'),
    ('version', 'int32'),
    ('comment', 'S256'),
    ('freq', 'float64'),
    ('tbeg', 'int32'),
    ('tend', 'int32'),
    ('nvar', 'int32'),
]

EntityHeader = [
    ('type', 'int32'),
    ('varVersion', 'int32'),
    ('name', 'S64'),
    ('offset', 'int32'),
    ('n', 'int32'),
    ('WireNumber', 'int32'),
    ('UnitNumber', 'int32'),
    ('Gain', 'int32'),
    ('Filter', 'int32'),
    ('XPos', 'float64'),
    ('YPos', 'float64'),
    ('WFrequency', 'float64'),
    ('ADtoMV', 'float64'),
    ('NPointsWave', 'int32'),
    ('NMarkers', 'int32'),
    ('MarkerLength', 'int32'),
    ('MVOffset', 'float64'),
    ('dummy', 'S60'),
]

MarkerHeader = [
    ('type', 'int32'),
    ('varVersion', 'int32'),
    ('name', 'S64'),
    ('offset', 'int32'),
    ('n', 'int32'),
    ('WireNumber', 'int32'),
    ('UnitNumber', 'int32'),
    ('Gain', 'int32'),
    ('Filter', 'int32'),
]
