"""
baseio
======

Classes
-------

BaseIO        - abstract class which should be overridden, managing how a
                file will load/write its data

If you want a model for developing a new IO start from exampleIO.
"""

from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
import logging

from neo import logging_handler
from neo.core import (
    AnalogSignal,
    Block,
    Epoch,
    Event,
    Group,
    IrregularlySampledSignal,
    ChannelView,
    Segment,
    SpikeTrain,
    ImageSequence,
    RectangularRegionOfInterest,
    CircularRegionOfInterest,
    PolygonRegionOfInterest,
    NeoReadWriteError,
)

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

    name = "BaseIO"
    description = ""
    extensions = []

    mode = "file"  # or 'fake' or 'dir' or 'database'

    def __init__(self, filename: str | Path = None, **kargs):
        self.filename = str(filename)
        # create a logger for the IO class
        fullname = self.__class__.__module__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(fullname)
        # create a logger for 'neo' and add a handler to it if it doesn't
        # have one already.
        # (it will also not add one if the root logger has a handler)
        corename = self.__class__.__module__.split(".")[0]
        corelogger = logging.getLogger(corename)
        rootlogger = logging.getLogger()
        if not corelogger.handlers and not rootlogger.handlers:
            corelogger.addHandler(logging_handler)

    ######## General read/write methods #######################
    def read(self, lazy: bool = False, **kargs):
        """
        Return all data from the file as a list of Blocks

        Parameters
        ----------
        lazy: bool, default: False
            Whether to lazily load the data (True) or to load into memory (False)
        kargs: dict
            IO specific additional arguments

        Returns
        ------
        block_list: list[neo.core.Block]
            Returns all the data from the file as Blocks
        """
        if lazy and not self.support_lazy:
            raise NeoReadWriteError("This IO module does not support lazy loading")
        if Block in self.readable_objects:
            if hasattr(self, "read_all_blocks") and callable(getattr(self, "read_all_blocks")):
                return self.read_all_blocks(lazy=lazy, **kargs)
            return [self.read_block(lazy=lazy, **kargs)]
        elif Segment in self.readable_objects:
            bl = Block(name="One segment only")
            seg = self.read_segment(lazy=lazy, **kargs)
            bl.segments.append(seg)
            bl.check_relationships()
            return [bl]
        else:
            raise NotImplementedError

    def write(self, bl, **kargs):
        """
        Writes a given block if IO supports writing

        Parameters
        ----------
        bl: neo.core.Block
            The neo Block to be written
        kargs: dict
            IO specific additional arguments

        """
        if Block in self.writeable_objects:
            if isinstance(bl, Sequence):
                if not hasattr(self, "write_all_blocks"):
                    raise NeoReadWriteError(f"{self.__class__.__name__} does not offer to store a sequence of blocks")
                self.write_all_blocks(bl, **kargs)
            else:
                self.write_block(bl, **kargs)
        elif Segment in self.writeable_objects:
            if len(bl.segments) != 1:
                raise NeoReadWriteError(
                    f"{self.__class__.__name__} is based on segment so if you try to write a block it "
                    + "must contain only one Segment"
                )
            self.write_segment(bl.segments[0], **kargs)
        else:
            raise NotImplementedError

    ######## All individual read methods #######################
    def read_block(self, **kargs):
        if Block not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_segment(self, **kargs):
        if Segment not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_spiketrain(self, **kargs):
        if SpikeTrain not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_analogsignal(self, **kargs):
        if AnalogSignal not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_imagesequence(self, **kargs):
        if ImageSequence not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_rectangularregionofinterest(self, **kargs):
        if RectangularRegionOfInterest not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_circularregionofinterest(self, **kargs):
        if CircularRegionOfInterest not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_polygonregionofinterest(self, **kargs):
        if PolygonRegionOfInterest not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_irregularlysampledsignal(self, **kargs):
        if IrregularlySampledSignal not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_channelview(self, **kargs):
        if ChannelView not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_event(self, **kargs):
        if Event not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_epoch(self, **kargs):
        if Epoch not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    def read_group(self, **kargs):
        if Group not in self.readable_objects:
            raise NeoReadWriteError(read_error)

    ######## All individual write methods #######################
    def write_block(self, bl, **kargs):
        if Block not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_segment(self, seg, **kargs):
        if Segment not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_spiketrain(self, sptr, **kargs):
        if SpikeTrain not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_analogsignal(self, anasig, **kargs):
        if AnalogSignal not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_imagesequence(self, imseq, **kargs):
        if ImageSequence not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_rectangularregionofinterest(self, rectroi, **kargs):
        if RectangularRegionOfInterest not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_circularregionofinterest(self, circroi, **kargs):
        if CircularRegionOfInterest not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_polygonregionofinterest(self, polyroi, **kargs):
        if PolygonRegionOfInterest not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_irregularlysampledsignal(self, irsig, **kargs):
        if IrregularlySampledSignal not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_channelview(self, chv, **kargs):
        if ChannelView not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_event(self, ev, **kargs):
        if Event not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_epoch(self, ep, **kargs):
        if Epoch not in self.writeable_objects:
            raise NeoReadWriteError(write_error)

    def write_group(self, group, **kargs):
        if Group not in self.writeable_objects:
            raise NeoReadWriteError(write_error)
