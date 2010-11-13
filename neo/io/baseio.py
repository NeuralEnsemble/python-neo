# -*- coding: utf-8 -*-

"""
baseio
==================



Classes
-------

BaseIO        - abstract class which should be overriden, managing how a file will load/write
                  its data
                  
If you want a model for develloping a new IO just start from exampleIO.
"""

import sys, os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../..'))

#from neo.core import *
from ..core import *


class BaseIO(object):
    """
    Generic class to handle all the file read/write methods for the key objects of the
    core class. This template is file reading/writing oriented but it can also handle data 
    from/to a database like TDT sytem tanks or SQLite files.
    This is an abstract class that will be implemented for each format
    The key methods of the class are:
        - read()- Read the whole object structure, generaly point to th highest Object level
        - read_block(** params)     - Read Block object from file with some params
        - read_segment(** params)     - Read Segment object from file with some params
        - read_spiketrainlist(**params)  - Read SpikeTrainList object from file with some params
        - write() - Write  the whole object structure, generaly point to th highest Object level
        - write_block(** params)    - Write Block object to file with some params
        - write_segment(** params)    - Write Segment object to file with some params
        - write_spiketrainlist(**params) - Write SpikeTrainList object to file with some params
        
    But the classe can also implement this methods :
        - read_spike(params)           - Read Spike object from file with some params
        - read_analogsignal( **params)    - Read AnalogSignal object from file with some params
        - read_spiketrain(**params)      - Read SpikeTrain object from file with some params
        - read_epoch(**params)     - Read Epoch object from file with some params
        - read_event(**params)           - Read Event object from file with some params
        - write_spikes(**params)          - Write Spike object to file with some params
        - write_analog(** params)   - Write AnalogSignal object to file with some params
        - write_spiketrain(**params)     - Write SpikeTrain object to file with some params
        - write_epoch(** params)    - Write Epoch object to file with some params
        - write_event(**params)          - Write Event object to file with some params
        
        
    Each object is able to declare what can be accessed or written
    The object types can be one of the class defined in neo.core :
       - Block (with all segments, AnalogSignals, SpikeTrains, ...)
       - Segment (with all  AnalogSignals, SpikeTrains, Events, Epoch, ...)
       - SpikeTrainList ( with all SpikeTrains )
       - Neuron ( with all SpikeTrains )
       - SpikeTrain
       - AnalogSignal
    
    
    ** start a new IO **
    If you want to implement your own file format, you just have to create an object that will 
    inherit from this BaseFile class and implement the previous functions.
    See ExampleIO in exampleio.py
    """
    
    is_readable        = False
    is_writable        = False
    
    supported_objects            = []
    readable_objects    = []
    writeable_objects    = []
    
    has_header         = False
    is_streameable     = False
    read_params        = {}
    write_params       = {}
    name               = None
    
    mode = 'file'

    def __init__(self , filename = None , **kargs ) :
        self.filename = filename
    
    ######## General read/write methods #######################
    
    def read(self, **kargs ):
        """
        bulk read the file at the highest level possible
        """
        pass

    def write(self, **kargs):
        """
        bulk write the file at the highest level possible
        """
        pass

    ######## All individual read methods #######################
    
    def read_spike(self, **kargs):
        """
        Read Spike object from a file
        """
        assert(Spike in self.readable_objects), "This type is not supported by this file format"
    
    def read_analogsignal(self, **kargs):
        """
        Read AnalogSignal object from a file
        """
        assert(AnalogSignal in self.readable_objects), "This type is not supported by this file format"
    
    def read_spiketrain(self, **kargs):
        """
        Read SpikeTrain object from a file
        """
        assert(SpikeTrain in self.readable_objects), "This type is not supported by this file format"
    
    def read_event(self, **kargs):
        """
        Read Events object from a file
        """
        assert(Event in self.readable_objects), "This type is not supported by this file format"
    
    def read_block(self, **kargs):
        """
        Read Events object from a file
        
        Examples:

        """
        assert(Block in self.readable_objects), "This type is not supported by this file format"

    def read_segment(self, **kargs):
        """
        Read Sequence object from a file
        
        Examples:

        """
        assert(Segment in self.readable_objects), "This type is not supported by this file format"

    def read_epoch(self, **kargs):
        """
        Read Epochs object from a file
        
        Examples:

        """
        assert(Epoch in self.readable_objects), "This type is not supported by this file format"
    
    def read_spiketrainlist(self, **kargs):
        """
        Read SpikeTrainList object from a file
        """
        assert(SpikeTrainList in self.readable_objects), "This type is not supported by this file format"

    ######## All individual write methods #######################

    def write_spike(self, **kargs):
        """
        Write Spike object from a file
        """
        assert(Spike in self.writeable_objects), "This type is not supported by this file format"
    
    def write_analogsignal(self, **kargs):
        """
        Write AnalogSignal objects from a file
        """
        assert(AnalogSignal in self.writeable_objects), "This type is not supported by this file format"


    def write_spiketrain(self, **kargs):
        """
        Write SpikeTrain objects from a file
        """
        assert(SpikeTrain in self.writeable_objects), "This type is not supported by this file format"
    
    def write_event(self, **kargs):
        """
        Write Event objects from a file
        """
        assert(Event in self.writeable_objects), "This type is not supported by this file format"
    
    def write_block(self, **kargs):
        """
        Write Block objects from a file
        """
        assert(Block in self.writeable_objects), "This type is not supported by this file format"
        
    def write_segment(self, **kargs):
        """
        Write Segment object from a file
        """
        assert(Segment in self.writeable_objects), "This type is not supported by this file format"
    
    def write_epoch(self, **kargs):
        """
        Write Epoch object from a file
        """
        assert(Epoch in self.writeable_objects), "This type is not supported by this file format"
     
    def write_spiketrainlist(self, **kargs):
        """
        Write SpikeTrainList objects from a file
        """
        assert(SpikeTrainList in self.writeable_objects), "This type is not supported by this file format"