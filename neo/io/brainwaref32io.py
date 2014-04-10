# -*- coding: utf-8 -*-
'''
Class for reading from Brainware F32 files

F32 files are simplified binary files for holding spike data.  Unlike SRC
files, F32 files carry little metadata.  This also means, however, that the
file format does not change, unlike SRC files whose format changes periodically
(although ideally SRC files are backwards-compatible).

Each F32 file only holds a single Block.

The only metadata stored in the file is the length of a single repetition
of the stimulus and the values of the stimulus parameters (but not the names
of the parameters).

Brainware was developed by Dr. Jan Schnupp and is availabe from
Tucker Davis Technologies, Inc.
http://www.tdt.com/downloads.htm

Neither Dr. Jan Schnupp nor Tucker Davis Technologies, Inc. had any part in the
development of this code

The code is implemented with the permission of Dr. Jan Schnupp

Author: Todd Jennings
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

# import needed core python modules
from os import path

# numpy and quantities are already required by neo
import numpy as np
import quantities as pq

# needed core neo modules
from neo.core import Block, RecordingChannelGroup, Segment, SpikeTrain, Unit

# need to subclass BaseIO
from neo.io.baseio import BaseIO


class BrainwareF32IO(BaseIO):
    '''
    Class for reading Brainware Spike ReCord files with the extension '.f32'

    The read_block method returns the first Block of the file.  It will
    automatically close the file after reading.
    The read method is the same as read_block.

    The read_all_blocks method automatically reads all Blocks.  It will
    automatically close the file after reading.

    The read_next_block method will return one Block each time it is called.
    It will automatically close the file and reset to the first Block
    after reading the last block.
    Call the close method to close the file and reset this method
    back to the first Block.

    The isopen property tells whether the file is currently open and
    reading or closed.

    Note 1:
        There is always only one RecordingChannelGroup.  BrainWare stores the
        equivalent of RecordingChannelGroups in separate files.

    Usage:
        >>> from neo.io.brainwaref32io import BrainwareF32IO
        >>> f32file = BrainwareF32IO(filename='multi_500ms_mulitrep_ch1.f32')
        >>> blk1 = f32file.read()
        >>> blk2 = f32file.read_block()
        >>> print blk1.segments
        >>> print blk1.segments[0].spiketrains
        >>> print blk1.units
        >>> print blk1.units[0].name
        >>> print blk2
        >>> print blk2[0].segments
    '''

    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects = [Block, RecordingChannelGroup,
                         Segment, SpikeTrain, Unit]

    readable_objects = [Block]
    writeable_objects = []

    has_header = False
    is_streameable = False

    # This is for GUI stuff: a definition for parameters when reading.
    # This dict should be keyed by object (`Block`). Each entry is a list
    # of tuple. The first entry in each tuple is the parameter name. The
    # second entry is a dict with keys 'value' (for default value),
    # and 'label' (for a descriptive name).
    # Note that if the highest-level object requires parameters,
    # common_io_test will be skipped.
    read_params = {Block: []}

    # does not support write so no GUI stuff
    write_params = None
    name = 'Brainware F32 File'
    extensions = ['f32']

    mode = 'file'

    def __init__(self, filename=None):
        '''
        Arguments:
            filename: the filename
        '''
        BaseIO.__init__(self)
        self._path = filename
        self._filename = path.basename(filename)

        self._fsrc = None
        self.__lazy = False

        self._blk = None
        self.__unit = None

        self.__t_stop = None
        self.__params = None
        self.__seg = None
        self.__spiketimes = None

    def read(self, lazy=False, cascade=True, **kargs):
        '''
        Reads simple spike data file "fname" generated with BrainWare
        '''
        return self.read_block(lazy=lazy, cascade=cascade)

    def read_block(self, lazy=False, cascade=True, **kargs):
        '''
        Reads a block from the simple spike data file "fname" generated
        with BrainWare
        '''

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')
        self._fsrc = None
        self.__lazy = lazy

        self._blk = Block(file_origin=self._filename)
        block = self._blk

        # if we aren't doing cascade, don't load anything
        if not cascade:
            return block

        # create the objects to store other objects
        rcg = RecordingChannelGroup(file_origin=self._filename)
        self.__unit = Unit(file_origin=self._filename)

        # load objects into their containers
        block.recordingchannelgroups.append(rcg)
        rcg.units.append(self.__unit)

        # initialize values
        self.__t_stop = None
        self.__params = None
        self.__seg = None
        self.__spiketimes = None

        # open the file
        with open(self._path, 'rb') as self._fsrc:
            res = True
            # while the file is not done keep reading segments
            while res:
                res = self.__read_id()

        block.create_many_to_one_relationship()

        # cleanup attributes
        self._fsrc = None
        self.__lazy = False

        self._blk = None

        self.__t_stop = None
        self.__params = None
        self.__seg = None
        self.__spiketimes = None

        return block

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #   IMPORTANT!!!
    #   These are private methods implementing the internal reading mechanism.
    #   Due to the way BrainWare DAM files are structured, they CANNOT be used
    #   on their own.  Calling these manually will almost certainly alter your
    #   position in the file in an unrecoverable manner, whether they throw
    #   an exception or not.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def __read_id(self):
        '''
        Read the next ID number and do the appropriate task with it.

        Returns nothing.
        '''
        try:
            # float32 -- ID of the first data sequence
            objid = np.fromfile(self._fsrc, dtype=np.float32, count=1)[0]
        except IndexError:
            # if we have a previous segment, save it
            self.__save_segment()

            # if there are no more Segments, return
            return False

        if objid == -2:
            self.__read_condition()
        elif objid == -1:
            self.__read_segment()
        else:
            self.__spiketimes.append(objid)
        return True

    def __read_condition(self):
        '''
        Read the parameter values for a single stimulus condition.

        Returns nothing.
        '''
        # float32 -- SpikeTrain length in ms
        self.__t_stop = np.fromfile(self._fsrc, dtype=np.float32, count=1)[0]

        # float32 -- number of stimulus parameters
        numelements = int(np.fromfile(self._fsrc, dtype=np.float32,
                                      count=1)[0])

        # [float32] * numelements -- stimulus parameter values
        paramvals = np.fromfile(self._fsrc, dtype=np.float32,
                                count=numelements).tolist()

        # organize the parameers into a dictionary with arbitrary names
        paramnames = ['Param%s' % i for i in range(len(paramvals))]
        self.__params = dict(zip(paramnames, paramvals))

    def __read_segment(self):
        '''
        Setup the next Segment.

        Returns nothing.
        '''
        # if we have a previous segment, save it
        self.__save_segment()

        # create the segment
        self.__seg = Segment(file_origin=self._filename,
                             **self.__params)

        # create an empy array to save the spike times
        # this needs to be converted to a SpikeTrain before it can be used
        self.__spiketimes = []

    def __save_segment(self):
        '''
        Write the segment to the Block if it exists
        '''
        # if this is the beginning of the first condition, then we don't want
        # to save, so exit
        # but set __seg from None to False so we know next time to create a
        # segment even if there are no spike in the condition
        if self.__seg is None:
            self.__seg = False
            return

        if not self.__seg:
            # create dummy values if there are no SpikeTrains in this condition
            self.__seg = Segment(file_origin=self._filename,
                                 **self.__params)
            self.__spiketimes = []

        if self.__lazy:
            train = SpikeTrain(pq.Quantity([], dtype=np.float32,
                                           units=pq.ms),
                               t_start=0*pq.ms, t_stop=self.__t_stop * pq.ms,
                               file_origin=self._filename)
            train.lazy_shape = len(self.__spiketimes)
        else:
            times = pq.Quantity(self.__spiketimes, dtype=np.float32,
                                units=pq.ms)
            train = SpikeTrain(times,
                               t_start=0*pq.ms, t_stop=self.__t_stop * pq.ms,
                               file_origin=self._filename)

        self.__seg.spiketrains = [train]
        self.__unit.spiketrains.append(train)
        self._blk.segments.append(self.__seg)

        # set an empty segment
        # from now on, we need to set __seg to False rather than None so
        # that if there is a condition with no SpikeTrains we know
        # to create an empty Segment
        self.__seg = False
