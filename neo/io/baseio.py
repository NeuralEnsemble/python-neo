"""
baseio
======

Classes
-------

BaseIO        - abstract class which should be overridden, managing how a
                file will load/write its data

If you want a model for developing a new IO start from exampleIO.
"""

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import logging

from neo import logging_handler
from neo.core import (AnalogSignal, Block,
                      Epoch, Event, Group,
                      IrregularlySampledSignal,
                      ChannelView,
                      Segment, SpikeTrain, ImageSequence,
                      RectangularRegionOfInterest, CircularRegionOfInterest,
                      PolygonRegionOfInterest)

read_error = "This type is not supported by this file format for reading"
write_error = "This type is not supported by this file format for writing"


class BaseIO:
    """
    Generic class to handle all the file read/write methods for the key objects
    of the core class. This template is file-reading/writing oriented but it
    can also handle data read from/written to a database such as TDT sytem
    tanks or SQLite files.

    This is an abstract class that will be subclassed for each format
    The key methods of the class are:
        - ``read()`` - Read the whole object structure, return a list of Block
                objects
        - ``read_block(lazy=True, **params)`` - Read Block object
                from file with some parameters
        - ``read_segment(lazy=True, **params)`` - Read Segment
                object from file with some parameters
        - ``read_spiketrainlist(lazy=True, **params)`` - Read
                SpikeTrainList object from file with some parameters
        - ``write()`` - Write the whole object structure
        - ``write_block(**params)``    - Write Block object to file with some
                parameters
        - ``write_segment(**params)``    - Write Segment object to file with
                some parameters
        - ``write_spiketrainlist(**params)`` - Write SpikeTrainList object to
                file with some parameters

    The class can also implement these methods:
        - ``read_XXX(lazy=True, **params)``
        - ``write_XXX(**params)``
        where XXX could be one of the objects supported by the IO

    Each class is able to declare what can be accessed or written directly
    discribed by **readable_objects** and **readable_objects**.
    The object types can be one of the classes defined in neo.core
    (Block, Segment, AnalogSignal, ...)

    Each class does not necessary support all the whole neo hierarchy but part
    of it.
    This is described with **supported_objects**.
    All IOs must support at least Block with a read_block()


    ** start a new IO **
    If you want to implement your own file format, you should create a class
    that will inherit from this BaseFile class and implement the previous
    methods.
    See ExampleIO in exampleio.py
    """

    is_readable = False
    is_writable = False

    supported_objects = []
    readable_objects = []
    writeable_objects = []

    support_lazy = False

    read_params = {}
    write_params = {}

    name = 'BaseIO'
    description = ''
    extensions = []

    mode = 'file'  # or 'fake' or 'dir' or 'database'

    def __init__(self, filename=None, **kargs):
        self.filename = str(filename)
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

    ######## General read/write methods #######################
    def read(self, lazy=False, **kargs):
        """
        Return all data from the file as a list of Blocks
        """
        if lazy and not self.support_lazy:
            raise ValueError("This IO module does not support lazy loading")
        if Block in self.readable_objects:
            if (hasattr(self, 'read_all_blocks') and
                    callable(getattr(self, 'read_all_blocks'))):
                return self.read_all_blocks(lazy=lazy, **kargs)
            return [self.read_block(lazy=lazy, **kargs)]
        elif Segment in self.readable_objects:
            bl = Block(name='One segment only')
            seg = self.read_segment(lazy=lazy, **kargs)
            bl.segments.append(seg)
            bl.create_many_to_one_relationship()
            return [bl]
        else:
            raise NotImplementedError

    def write(self, bl, **kargs):
        if Block in self.writeable_objects:
            if isinstance(bl, Sequence):
                assert hasattr(self, 'write_all_blocks'), \
                    '%s does not offer to store a sequence of blocks' % \
                    self.__class__.__name__
                self.write_all_blocks(bl, **kargs)
            else:
                self.write_block(bl, **kargs)
        elif Segment in self.writeable_objects:
            assert len(bl.segments) == 1, \
                '%s is based on segment so if you try to write a block it ' + \
                'must contain only one Segment' % self.__class__.__name__
            self.write_segment(bl.segments[0], **kargs)
        else:
            raise NotImplementedError

    ######## All individual read methods #######################
    def read_block(self, **kargs):
        assert (Block in self.readable_objects), read_error

    def read_segment(self, **kargs):
        assert (Segment in self.readable_objects), read_error

    def read_spiketrain(self, **kargs):
        assert (SpikeTrain in self.readable_objects), read_error

    def read_analogsignal(self, **kargs):
        assert (AnalogSignal in self.readable_objects), read_error

    def read_imagesequence(self, **kargs):
        assert (ImageSequence in self.readable_objects), read_error

    def read_rectangularregionofinterest(self, **kargs):
        assert (RectangularRegionOfInterest in self.readable_objects), read_error

    def read_circularregionofinterest(self, **kargs):
        assert (CircularRegionOfInterest in self.readable_objects), read_error

    def read_polygonregionofinterest(self, **kargs):
        assert (PolygonRegionOfInterest in self.readable_objects), read_error

    def read_irregularlysampledsignal(self, **kargs):
        assert (IrregularlySampledSignal in self.readable_objects), read_error

    def read_channelview(self, **kargs):
        assert (ChannelView in self.readable_objects), read_error

    def read_event(self, **kargs):
        assert (Event in self.readable_objects), read_error

    def read_epoch(self, **kargs):
        assert (Epoch in self.readable_objects), read_error

    def read_group(self, **kargs):
        assert (Group in self.readable_objects), read_error

    ######## All individual write methods #######################
    def write_block(self, bl, **kargs):
        assert (Block in self.writeable_objects), write_error

    def write_segment(self, seg, **kargs):
        assert (Segment in self.writeable_objects), write_error

    def write_spiketrain(self, sptr, **kargs):
        assert (SpikeTrain in self.writeable_objects), write_error

    def write_analogsignal(self, anasig, **kargs):
        assert (AnalogSignal in self.writeable_objects), write_error

    def write_imagesequence(self, imseq, **kargs):
        assert (ImageSequence in self.writeable_objects), write_error

    def write_rectangularregionofinterest(self, rectroi, **kargs):
        assert (RectangularRegionOfInterest in self.writeable_objects), read_error

    def write_circularregionofinterest(self, circroi, **kargs):
        assert (CircularRegionOfInterest in self.writeable_objects), read_error

    def write_polygonregionofinterest(self, polyroi, **kargs):
        assert (PolygonRegionOfInterest in self.writeable_objects), read_error

    def write_irregularlysampledsignal(self, irsig, **kargs):
        assert (IrregularlySampledSignal in self.writeable_objects), write_error

    def write_channelview(self, chv, **kargs):
        assert (ChannelView in self.writeable_objects), write_error

    def write_event(self, ev, **kargs):
        assert (Event in self.writeable_objects), write_error

    def write_epoch(self, ep, **kargs):
        assert (Epoch in self.writeable_objects), write_error

    def write_group(self, group, **kargs):
        assert (Group in self.writeable_objects), write_error