# -*- coding: utf-8 -*-
"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV and NSE file formats.


Author: Julia Sprenger, Carlos Canova, Samuel Garcia

"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, 
        _event_channel_dtype)

import numpy as np
from xml.etree import ElementTree


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
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        
        #fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels
        
        # Annotations
        self._generate_minimal_annotations()
        #~ bl_annotations = self.raw_annotations['blocks'][0]
        #~ seg_annotations = bl_annotations['segments'][0]

    def _block_count(self):
        raise(NotImplementedError)
    
    def _segment_count(self, block_index):
        raise(NotImplementedError)
    
    def _segment_t_start(self, block_index, seg_index):
        raise(NotImplementedError)

    def _segment_t_stop(self, block_index, seg_index):
        raise(NotImplementedError)
    
    ###
    # signal and channel zone
    def _analogsignal_shape(self, block_index, seg_index):
        raise(NotImplementedError)
        
    def _analogsignal_sampling_rate(self):
        raise(NotImplementedError)

    def _get_analogsignal_chunk(self, block_index, seg_index,  i_start, i_stop, channel_indexes):
        raise(NotImplementedError)
    
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

