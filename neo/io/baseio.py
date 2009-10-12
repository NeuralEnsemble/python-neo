# -*- coding: utf-8 -*-

"""
NeuroTools.io
==================

A collection of functions to handle all the inputs/outputs of the NeuroTools.signals
file, used by the loaders.

Classes
-------

BaseFile        - abstract class which should be overriden, managing how a file will load/write
                  its data
"""

from core import *

class BaseFile(object):
    """
    Generic class to handle all the file read/write methods for the key objects of the
    core class. This template is file reading/writing oriented but it can also handle data 
    from/to a database like TDT sytem tanks or SQLite files.
    This is an abstract class that will be implemented for each format
    The key methods of the class are:
        read(object)                  - Read the whole object structure
        write(object)                 - Write the whole object structure
        read_spikes(params)           - Read Spike object from file with some params
        read_analogs(type, params)    - Read AnalogSignal object from file with some params
        read_spiketrains(params)      - Read SpikeTrain object from file with some params
        read_spiketrainlists(params)  - Read SpikeTrainList object from file with some params
        read_epochs(type, params)     - Read Epoch object from file with some params
        read_events(params)           - Read Event object from file with some params
        read_blocks(type, params)     - Read Block object from file with some params
        write_spikes(params)          - Write Spike object to file with some params
        write_analogs(type, params)   - Write AnalogSignal object to file with some params
        write_spiketrains(params)     - Write SpikeTrain object to file with some params
        write_spiketrainlists(params) - Write SpikeTrainList object to file with some params
        write_epochs(type, params)    - Write Epoch object to file with some params
        write_events(params)          - Write Event object to file with some params
        write_blocks(type, params)    - Write Block object to file with some params
        
    Each object is able to declare what can be accessed or written
    The object types can be one of the class defined in neo.core :
        Block (with all segments, AnalogSignals, SpikeTrains, ...)
        Segment (with all  AnalogSignals, SpikeTrains, Events, Epoch, ...)
        SpikeTrain
        SpikeTrainList
        AnalogSignal
        AnalogSignalList
        Neuron
        
    ** Guidelines **
        Each IO implementation of BaseFile can also add attributs (fields) freely to all object.
        Each IO implementation of BaseFile should come with tipics files exemple.
        Each IO implementation of BaseFile should come with its documentation.
    
    If you want to implement your own file format, you just have to create an object that will 
    inherit from this BaseFile class and implement the previous functions.
    """
    
    is_readable        = False
    is_writable        = False	
    is_object_readable = False
    is_object_writable = False
    has_header         = False	
    is_streameable     = False
    read_params        = {}
    write_params       = {}   
    level              = None
    nfiles             = 0        
    name               = None
    objects            = []
    supported_types    = []
    
    def __init__(self , filename = None , **kargs ) :
        self.filename = filename
        pass
        
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
    
    def read_spikes(self, **kargs):
        """
        Read Spikes objects from a file
        
        Examples:
        """
        assert(Spike in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def read_analogs(self, **kargs):
        """
        Read AnalogSignal objects from a file
        
        Examples:

        """
        assert(AnalogSignal in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def read_spiketrains(self, **kargs):
        """
        Read SpikeTrains objects from a file
        
        Examples:

        """
        assert(SpikeTrain in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def read_events(self, **kargs):
        """
        Read Events objects from a file
        
        Examples:

        """
        assert(Event in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def read_blocks(self, **kargs):
        """
        Read Events objects from a file
        
        Examples:

        """
        assert(Block in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)

    def read_segment(self, **kargs):
        """
        Read Sequence object from a file
        
        Examples:

        """
        assert(Segment in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)

    def read_epochs(self, **kargs):
        """
        Read Epochs objects from a file
        
        Examples:

        """
        assert(Epoch in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
     
    def read_spiketrainlists(self, **kargs):
        """
        Read SpikeTrainList objects from a file
        
        Examples:

        """
        assert(SpikeTrainList in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)

    def read_header(self):
        """
        Read metadata/header from a file
        
        Examples:

        """
        return _abstract_method(self)

######## All individual write methods #######################

    def write_spikes(self, **kargs):
        """
        Write Spikes objects from a file
        
        Examples:
        """
        assert(Spike in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def write_analogs(self, **kargs):
        """
        Write AnalogSignal objects from a file
        
        Examples:

        """
        assert(AnalogSignal in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def write_spiketrains(self, **kargs):
        """
        Write SpikeTrains objects from a file
        
        Examples:

        """
        assert(SpikeTrain in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def write_events(self, **kargs):
        """
        Write Events objects from a file
        
        Examples:

        """
        assert(Event in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def write_blocks(self, **kargs):
        """
        Write Events objects from a file
        
        Examples:

        """
        assert(Block in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
        
    def write_segment(self, **kargs):
        """
        Write Sequence object from a file
        
        Examples:

        """
        assert(Segment in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
    
    def write_epochs(self, **kargs):
        """
        Write Epochs objects from a file
        
        Examples:

        """
        assert(Epoch in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)
     
    def write_spiketrainlists(self, **kargs):
        """
        Write SpikeTrainList objects from a file
        
        Examples:

        """
        assert(SpikeTrainList in self.supported_types), "This type is not supported by this file format"
        return _abstract_method(self)

    def write_header(self):
        """
        Write metadata/header from a file
        
        Examples:

        """
        return _abstract_method(self)
