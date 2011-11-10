# encoding: utf-8
"""
baseio
======

Classes
-------

BaseIO        - abstract class which should be overridden, managing how a file will load/write
                  its data
                  
If you want a model for developing a new IO start from exampleIO.
"""

from ..core import *
from .tools import create_many_to_one_relationship

read_error = "This type is not supported by this file format for reading"
write_error = "This type is not supported by this file format for writing"


class BaseIO(object):
    """
    Generic class to handle all the file read/write methods for the key objects of the
    core class. This template is file-reading/writing oriented but it can also handle data 
    read from/written to a database such as TDT sytem tanks or SQLite files.
    This is an abstract class that will be subclassed for each format
    The key methods of the class are:
        - ``read()`` - Read the whole object structure, return the object at the highest level in the hierarchy
        - ``read_block(lazy=True, cascade=True, **params)``     - Read Block object from file with some parameters
        - ``read_segment(lazy=True, cascade=True, **params)``     - Read Segment object from file with some parameters
        - ``read_spiketrainlist(lazy=True, cascade=True, **params)`` - Read SpikeTrainList object from file with some parameters
        - ``write()`` - Write the whole object structure
        - ``write_block(**params)``    - Write Block object to file with some parameters
        - ``write_segment(**params)``    - Write Segment object to file with some parameters
        - ``write_spiketrainlist(**params)`` - Write SpikeTrainList object to file with some parameters
        
    The class can also implement these methods:
        - ``read_XXX(lazy=True, cascade=True, **params)``
        - ``write_XXX(**params)``
        where XXX could be one one the object supported by the IO
    
    Each class is able to declare what can be accessed or written directly discribed by **readable_objects** and **readable_objects**.
    The object types can be one of the classes defined in neo.core (Block, Segment, AnalogSignal, ...)
    
    Each class do not necessary support all the whole neo hierarchy but part of it.
    This is discribe with **supported_objects**.
    All IOs must support at least Block with a read_block()
    
    
    ** start a new IO **
    If you want to implement your own file format, you should create a class that will 
    inherit from this BaseFile class and implement the previous methods.
    See ExampleIO in exampleio.py
    """
    
    is_readable = False
    is_writable = False
    
    supported_objects = []
    readable_objects  = []
    writeable_objects = []
    
    has_header = False
    is_streameable = False
    read_params = {}
    write_params = {}
    
    name = 'BaseIO'
    description = 'This IO does not read or write anything'
    extentions = [ ]
    
    mode = 'file' # or 'fake' or 'dir' or 'database'

    def __init__(self, filename=None, **kargs):
        self.filename = filename
    
    ######## General read/write methods #######################
    def read(self, lazy = False, cascade = True,  **kargs):
        if Block in self.readable_objects:
            return self.read_block(lazy = lazy, cascade = cascade, **kargs)
        elif Segment in self.readable_objects:
            bl = Block(name = 'One segment only')
            if not cascade:
                return bl
            seg = self.read_segment(lazy = lazy, cascade = cascade,  **kargs)
            bl.segments.append(seg)
            create_many_to_one_relationship(bl)
            return bl
        else:
            raise NotImplementedError
    
    def write(self, bl, **kargs):
        if Block in self.writeable_objects:
            self.write_block(bl, **kargs)
        elif Segment in self.writeable_objects:
            assert len(bl.segments) == 1, '%s is based on segment so if you try to write a block it must contain only one Segment'% self.__class__.__name__
            self.write_segment(bl.segments[0], **kargs)
        else:
            raise NotImplementedError

    ######## All individual read methods #######################
    def read_block(self, **kargs):
        assert(Block in self.readable_objects), read_error

    def read_segment(self, **kargs):
        assert(Segment in self.readable_objects), read_error

    def read_unit(self, **kargs):
        assert(Unit in self.readable_objects), read_error

    def read_spiketrain(self, **kargs):
        assert(SpikeTrain in self.readable_objects), read_error

    def read_spike(self, **kargs):
        assert(Spike in self.readable_objects), read_error
    
    def read_analogsignal(self, **kargs):
        assert(AnalogSignal in self.readable_objects), read_error

    def read_irregularlysampledsignal(self, **kargs):
        assert(IrregularlySampledSignal in self.readable_objects), read_error

    def read_analogsignalarray(self, **kargs):
        assert(AnalogSignalArray in self.readable_objects), read_error

    def read_recordingchannelgroup(self, **kargs):
        assert(RecordingChannelGroup in self.readable_objects), read_error

    def read_recordingchannel(self, **kargs):
        assert(RecordingChannel in self.readable_objects), read_error
    
    def read_event(self, **kargs):
        assert(Event in self.readable_objects), read_error
    
    def read_eventarray(self, **kargs):
        assert(EventArray in self.readable_objects), read_error
    
    def read_epoch(self, **kargs):
        assert(Epoch in self.readable_objects), read_error

    def read_epocharray(self, **kargs):
        assert(EpochArray in self.readable_objects), read_error
    
    ######## All individual write methods #######################
    def write_block(self, bl, **kargs):
        assert(Block in self.writeable_objects), write_error

    def write_segment(self, seg, **kargs):
        assert(Segment in self.writeable_objects), write_error

    def write_unit(self, ut, **kargs):
        assert(Unit in self.writeable_objects), write_error

    def write_spiketrain(self,sptr,  **kargs):
        assert(SpikeTrain in self.writeable_objects), write_error

    def write_spike(self, sp, **kargs):
        assert(Spike in self.writeable_objects), write_error
    
    def write_analogsignal(self, anasig,  **kargs):
        assert(AnalogSignal in self.writeable_objects), write_error

    def write_irregularlysampledsignal(self,irsig,  **kargs):
        assert(IrregularlySampledSignal in self.writeable_objects), write_error

    def write_analogsignalarray(self, anasigar, **kargs):
        assert(AnalogSignalArray in self.writeable_objects), write_error

    def write_recordingchannelgroup(self, rcg, **kargs):
        assert(RecordingChannelGroup in self.writeable_objects), write_error

    def write_recordingchannel(self, rc, **kargs):
        assert(RecordingChannel in self.writeable_objects), write_error
    
    def write_event(self,ev,  **kargs):
        assert(Event in self.writeable_objects), write_error
    
    def write_eventarray(self, ea,  **kargs):
        assert(EventArray in self.writeable_objects), write_error
    
    def write_epoch(self, ep, **kargs):
        assert(Epoch in self.writeable_objects), write_error

    def write_epocharray(self, epa,  **kargs):
        assert(EpochArray in self.writeable_objects), write_error
