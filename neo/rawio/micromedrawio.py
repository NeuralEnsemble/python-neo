# -*- coding: utf-8 -*-
"""
Class for reading/writing data from micromed (.trc).
Inspired by the Matlab code for EEGLAB from Rami K. Niazy.

Completed with matlab Guillaume BECQ code.

Supported : Read

Author: sgarcia
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, 
        _event_channel_dtype)

import numpy as np

import datetime
import os
import struct
from io import BufferedReader


class StructFile(BufferedReader):
    def read_f(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))


class MicromedRawIO(BaseRawIO):
    """
    Class for reading  data from micromed (.trc).
    """
    extensions = ['trc', 'TRC']
    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename 
    
    def _parse_header(self):
        f = StructFile(open(self.filename, 'rb'))

        # Name
        f.seek(64, 0)
        surname = f.read(22).strip(b' ')
        firstname = f.read(20).strip(b' ')

        #Date
        f.seek(128, 0)
        day, month, year, hour, minute, sec = f.read_f('bbbbbb')
        rec_datetime = datetime.datetime(year + 1900, month, day, hour, minute,
                                         sec)

        f.seek(138, 0)
        Data_Start_Offset, Num_Chan, Multiplexer, Rate_Min, Bytes = f.read_f(
            'IHHHH')

        #header version
        f.seek(175, 0)
        header_version, = f.read_f('b')
        assert header_version == 4

        # area
        f.seek(176, 0)
        zone_names = ['ORDER', 'LABCOD', 'NOTE', 'FLAGS', 'TRONCA', 
                    'IMPED_B', 'IMPED_E', 'MONTAGE',
                    'COMPRESS', 'AVERAGE', 'HISTORY', 'DVIDEO', 'EVENT A',
                    'EVENT B', 'TRIGGER']
        zones = {}
        for zname in zone_names:
            zname2, pos, length = f.read_f('8sII')
            zones[zname] = zname2, pos, length
            assert zname==zname2.decode('ascii').strip(' ')
        
        #raw signals memmap
        self._raw_signals = np.memmap(self.filename, dtype='u' + str(Bytes), mode='r',
                        offset=Data_Start_Offset ).reshape(-1, Num_Chan)

        # Reading Code Info
        zname2, pos, length = zones['ORDER']
        f.seek(pos, 0)
        code = np.fromfile(f, dtype='u2', count=Num_Chan)

        #~ units = {-1: pq.nano * pq.V, 0: pq.uV, 1: pq.mV, 2: 1, 100: pq.percent,
                 #~ 101: pq.dimensionless, 102: pq.dimensionless}
        units_code = {-1: 'nV', 0: 'uV', 1: 'mV', 2: 1, 100: 'percent',
                 101: 'dimensionless', 102: 'dimensionless'}
        all_sampling_rate = []
        sig_channels = []
        for c in range(Num_Chan):
            zname2, pos, length = zones['LABCOD']
            f.seek(pos + code[c] * 128 + 2, 0)

            chan_name = f.read(6).strip(b"\x00").decode('ascii')
            ground = f.read(6).strip(b"\x00").decode('ascii')#TODO put in annotations
            logical_min, logical_max, logical_ground, physical_min, physical_max = f.read_f('iiiii')
            #~ print(logical_min, logical_max, logical_ground, physical_min, physical_max)
            k, = f.read_f('h')
            units = units_code.get(k, 'uV')
            
            factor = float(physical_max - physical_min) / float(
                    logical_max - logical_min + 1)
            gain = factor
            offset = -logical_ground*factor

            f.seek(8, 1)
            sampling_rate, = f.read_f('H')
            sampling_rate *= Rate_Min
            all_sampling_rate.append(sampling_rate)

            chan_id = c
            sig_channels.append((chan_name, chan_id, units, gain,offset))
            
        #~ print(all_sampling_rate)
        assert np.unique(all_sampling_rate).size==1
        self._sampling_rate = float(np.unique(all_sampling_rate)[0])
        #~ print('self._sample_rate', self._sample_rate)
        
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        #~ print(sig_channels)
        #~ exit()

        #Event channels
        event_channels = []
        event_channels.append(('Trigger', '', 'event'))
        event_channels.append(('Note', '', 'event'))
        event_channels.append(('Event A', '', 'epoch'))
        event_channels.append(('Event B', '', 'epoch'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # Read trigger and notes
        self._raw_events = []
        ev_dtypes = [('TRIGGER', [('start', 'u4'), ('label', 'u2')]),
                            ('NOTE', [('start', 'u4'), ('label', 'S40')]),
                            ('EVENT A', [('label', 'u4'), ('start', 'u4'), ('stop', 'u4')]),
                            ('EVENT B', [('label', 'u4'), ('start', 'u4'), ('stop', 'u4')]),
                            ]
        for zname, ev_dtype in ev_dtypes:
            zname2, pos, length = zones[zname]
            dtype = np.dtype(ev_dtype)
            rawevent = np.memmap(self.filename, dtype=dtype, mode='r',
                        offset=pos, shape=length//dtype.itemsize )
            
            keep = (rawevent['start'] >= rawevent['start'][0]) & (
                    rawevent['start'] < self._raw_signals.shape[0]) & (
                    rawevent['start'] != 0)
            rawevent = rawevent[keep]
            self._raw_events.append(rawevent)

        #No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        
        #fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels
        
        # insert some annotation at some place
        self._generate_empty_annotations()
        bl_annotations = self.raw_annotations['blocks'][0]
        seg_annotations = bl_annotations['segments'][0]
        
        for d in (bl_annotations, seg_annotations):
            d['rec_datetime'] = rec_datetime
            d['firstname'] = firstname
            d['surname'] = surname
            d['header_version'] = header_version
    
    def _source_name(self):
        return self.filename
    
    def _block_count(self):
        return 1
    
    def _segment_count(self, block_index):
        return 1
    
    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0]/self._sampling_rate
        return t_stop

    def _analogsignal_shape(self, block_index, seg_index):
        return self._raw_signals.shape
    
    def _analogsignal_sampling_rate(self):
        return self._sampling_rate

    def _get_analogsignal_chunk(self, block_index, seg_index,  i_start, i_stop, channel_indexes):
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals
    
    def _spike_count(self,  block_index, seg_index, unit_index):
        return 0
    
    def _event_count(self, block_index, seg_index, event_channel_index):
        n = self._raw_events[event_channel_index].size
        return n
    
    def _event_timestamps(self,  block_index, seg_index, event_channel_index, t_start, t_stop):
        
        raw_event = self._raw_events[event_channel_index]
        
        if t_start is not None:
            keep = raw_event['start']>=int(t_start*self._sampling_rate)
            raw_event = raw_event[keep]
        
        if t_stop is not None:
            keep = raw_event['start']<=int(t_stop*self._sampling_rate)
            raw_event = raw_event[keep]
        
        timestamp = raw_event['start']
        if event_channel_index<2:
            durations = None
        else:
            durations = raw_event['stop'] - raw_event['start']
        labels = raw_event['label'].astype('U')
        
        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype)/self._sampling_rate
        return event_times
    
    def _rescale_epoch_duration(self, raw_duration, dtype):
        durations = raw_duration.astype(dtype)//self._sampling_rate
        return durations

