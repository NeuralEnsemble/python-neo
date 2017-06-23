# -*- coding: utf-8 -*-
"""
baserawio
======

Classes
-------

BaseRawIO        - abstract class which should be overridden, managing how a
                file will load its data

If you want a model for developing a new IO start from ExampleRawIO.
"""

from __future__ import absolute_import, division, print_function

import logging
from neo import logging_handler


possible_modes = ['one-file', 'multi-file', 'one-dir', 'multi-dir', 'url', 'other']

error_header = 'Header is not read yet, do parse_header() first'

class BaseRawIO(object):
    """
    Generic class to handle all the file read methods.

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
        #This must parse the file header to get all stuff
        #for fast reading after
        raise(NotImplementedError)
        #self.header = ...
    
    def channel_name_to_index(self, channel_names):
        raise(NotImplementedError)
        #~ return channel_indexes

    def channel_name_to_id(self, channel_ids):
        raise(NotImplementedError)
        #~ return channel_indexes
    
    def get_nb_block(self):
        raise(NotImplementedError)
    
    def get_nb_segment(self, block_index):
        raise(NotImplementedError)

    def get_nb_analogsignal(self, block_index, seg_index):
        raise(NotImplementedError)
    
    def get_nb_analogsignal(self, block_index, seg_index):
        raise(NotImplementedError)
    
    def get_analogsignal_chunk(self, block_index=0, seg_index=0, anasig_index=0,
                        i_start=None, i_stop=None, 
                        channel_indexes=None, channel_names=None, channel_ids=None):
        
        if channel_indexes is None and channel_names is not None:
            channel_indexes = self.channel_name_to_index(channel_names)

        if channel_indexes is None and channel_ids is not None:
            channel_indexes = self.channel_name_to_id(channel_ids)
            
        return self._get_analogsignal_chunk(block_index, seg_index, anasig_index, i_start, i_stop, channel_indexes)
    
    def _get_analogsignal_chunk(self, block_index, seg_index, anasig_index, i_start, i_stop, channel_indexes):
        raise(NotImplementedError)
    
    
    
    
    
    
    
    
    
    
    