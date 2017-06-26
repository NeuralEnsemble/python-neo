# -*- coding: utf-8 -*-
"""
baserawio
======

Classes
-------

BaseRawIO
abstract class which should be overridden.

This manage low level acces to raw data in an efficient way.

This handle **only** one simplified but very frequent case of dataset:
    * Only one channel set  for AnalogSignal (aka ChannelIndex) stable along Segment
    * Only one channel set  for SpikeTrain (aka Unit) stable along Segment
    * AnalogSignal have all the same sampling_rate, t_start, duration inside a segment

    
"""

from __future__ import absolute_import, division, print_function

import logging
import numpy as np

from neo import logging_handler



possible_modes = ['one-file', 'multi-file', 'one-dir', 'multi-dir', 'url', 'other']

error_header = 'Header is not read yet, do parse_header() first'


channel_dtype = [
    ('name','U'),
    ('id','U'),
    ('units','U'),
    ('gain','float64'),
    ('offset','float64'),
]


class BaseRawIO(object):
    """
    Generic class to handle.

    """
    
    name = 'BaseIO'
    description = ''
    extentions = []

    mode = None # one key in possible modes
    

    def __init__(self, **kargs):
        # create a logger for the IO class
        fullname = self.__class__.__module__ + '.' + self.__class__.__name__
        self.logger = logging.getLogger(fullname)
        # create a logger for 'neo' and add a handler to it if it doesn't
        # have one already.
        # (it will also not add one if the root logger has a handler)
        corename = self.__class__.__module__.split('.')[0]
        corelogger = logging.getLogger(corename)
        rootlogger = logging.getLogger()
        if not corelogger.handlers and not rootlogger.handlers:
            corelogger.addHandler(logging_handler)
        
        self.header = None
    
    def parse_header(self):
        #if we need to cache the header somewhere
        # make some hack here
        self._parse_header()
    
    
    def _parse_header(self):
        """
        This must parse the file header to get all stuff for fast later one.
        
        This must contain
        self.header['signal_channels']
        
        """
        #
        #
        raise(NotImplementedError)
        #self.header = ...
    
    def source_name(self):
        #this is used for __repr__
        raise(NotImplementedError)
    
    def __repr__(self):
        txt = '{}: {}\n'.format(self.__class__.__name__, self.source_name())
        if self.header is not None:
            nb_block = self.block_count()
            txt += 'nb_block: {}\n'.format(nb_block)
            nb_seg = [self.segment_count(i) for i in range(nb_block)]
            txt += 'nb_segment:  {}\n'.format(nb_seg)
            ch = self.header['signal_channels']
            if len(ch)>8:
                chantxt = "[{} ... {}]".format(' '.join(e for e in ch['name'][:4]),\
                                                                            ' '.join(e for e in ch['name'][-4:]))
            else:
                chantxt = "[{}]".format(' '.join(e for e in ch['name']))
            txt += 'channel: {}\n'.format(chantxt)
            
        return txt
    
    def channel_name_to_index(self, channel_names):
        """
        Transform channel_names to channel_indexes.
        """
        ch = self.header['signal_channels']
        channel_indexes,  = np.nonzero(np.in1d(ch['name'], channel_names))
        assert len(channel_indexes) == len(channel_names), 'not match'
        return channel_indexes
    
    def channel_name_to_id(self, channel_ids):
        """
        Transform channel_ids to channel_indexes.
        """
        ch = self.header['signal_channels']
        channel_indexes,  = np.nonzero(channel_index(np.in1d(ch['id'], channel_ids)))
        assert len(channel_indexes) == len(channel_ids), 'not match'
        return channel_indexes
    
    def _get_channel_indexes(self, channel_indexes, channel_names, channel_ids):
        """
        select channel_indexes from channel_indexes/channel_names/channel_ids
        depending which is not None
        """
        if channel_indexes is None and channel_names is not None:
            channel_indexes = self.channel_name_to_index(channel_names)

        if channel_indexes is None and channel_ids is not None:
            channel_indexes = self.channel_name_to_id(channel_ids)
        
        return channel_indexes
    
    def block_count(self):
        raise(NotImplementedError)
    
    def segment_count(self, block_index):
        raise(NotImplementedError)
    
    def get_analogsignal_chunk(self, block_index=0, seg_index=0, i_start=None, i_stop=None, 
                        channel_indexes=None, channel_names=None, channel_ids=None):
        
        channel_indexes = self._get_channel_indexes(channel_indexes, channel_names, channel_ids)
        
        raw_chunk = self._get_analogsignal_chunk(block_index, seg_index,  i_start, i_stop, channel_indexes)
        
        return raw_chunk
    
    def _get_analogsignal_chunk(self, block_index, seg_index,  i_start, i_stop, channel_indexes):
        raise(NotImplementedError)
    
    def analogsignal_meta(self):
        #sampling_rate in Hz and t_start in s
        raise(NotImplementedError)
    
    def rescale_raw_to_float(self, raw_signal,  dtype='float32',
                channel_indexes=None, channel_names=None, channel_ids=None):
        
        channel_indexes = self._get_channel_indexes(channel_indexes, channel_names, channel_ids)
        if channel_indexes is None:
            channel_indexes = sl(None)
        
        channels = self.header['signal_channels'][channel_indexes]
        
        float_signal = raw_signal.astype(dtype)
        
        if np.any(channels['gain'] !=1.):
            float_signal *= channels['gain']
        
        if np.any(channels['offset'] !=0.):
            float_signal += channels['offset']
        
        return float_signal
    
    
    
    
    
    
    
    