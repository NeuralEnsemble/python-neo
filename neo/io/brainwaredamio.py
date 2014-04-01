# -*- coding: utf-8 -*-
'''
Class for reading from Brainware DAM files

DAM files are binary files for holding raw data.  They are broken up into
sequence of Segments, each containing a single raw trace and parameters.

The DAM file does NOT contain a sampling rate, nor can it be reliably
calculated from any of the parameters.  You can calculate it from
the "sweep length" attribute if it is present, but it isn't always present.
It is more reliable to get it from the corresponding SRC file or F32 file if
you have one.

The DAM file also does not divide up data into Blocks, so only a single
Block is returned..

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
import os
import os.path

# numpy and quantities are already required by neo
import numpy as np
import quantities as pq

# needed core neo modules
from neo.core import (AnalogSignal, Block, RecordingChannel,
                      RecordingChannelGroup, Segment)

# need to subclass BaseI
from neo.io.baseio import BaseIO


class BrainwareDamIO(BaseIO):
    """
    Class for reading Brainware raw data files with the extension '.dam'.

    The read_block method returns the first Block of the file.  It will
    automatically close the file after reading.
    The read method is the same as read_block.

    Note:

    The file format does not contain a sampling rate.  The sampling rate
    is set to 1 Hz, but this is arbitrary. If you have a corresponding .src
    or .f32 file, you can get the sampling rate from that. It may also be
    possible to infer it from the attributes, such as "sweep length", if
    present.

    Usage:
        >>> from neo.io.brainwaredamio import BrainwareDamIO
        >>> damfile = BrainwareDamIO(filename='multi_500ms_mulitrep_ch1.dam')
        >>> blk1 = damfile.read()
        >>> blk2 = damfile.read_block()
        >>> print blk1.segments
        >>> print blk1.segments[0].analogsignals
        >>> print blk1.units
        >>> print blk1.units[0].name
        >>> print blk2
        >>> print blk2[0].segments
    """

    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects = [Block, RecordingChannelGroup, RecordingChannel,
                         Segment, AnalogSignal]

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

    # do not support write so no GUI stuff
    write_params = None
    name = 'Brainware DAM File'
    extensions = ['dam']

    mode = 'file'

    def __init__(self, filename=None):
        '''
        Arguments:
            filename: the filename
        '''
        BaseIO.__init__(self)
        self._path = filename
        self._filename = os.path.basename(filename)
        self._fsrc = None

    def read(self, lazy=False, cascade=True, **kargs):
        '''
        Reads raw data file "fname" generated with BrainWare
        '''
        return self.read_block(lazy=lazy, cascade=cascade)

    def read_block(self, lazy=False, cascade=True, **kargs):
        '''
        Reads a block from the raw data file "fname" generated
        with BrainWare
        '''

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')
        self._fsrc = None

        block = Block(file_origin=self._filename)

        # if we aren't doing cascade, don't load anything
        if not cascade:
            return block

        # create the objects to store other objects
        rcg = RecordingChannelGroup(file_origin=self._filename)
        rchan = RecordingChannel(file_origin=self._filename,
                                 index=1, name='Chan1')

        # load objects into their containers
        rcg.recordingchannels.append(rchan)
        block.recordingchannelgroups.append(rcg)
        rcg.channel_indexes = np.array([1])
        rcg.channel_names = np.array(['Chan1'], dtype='S')

        # open the file
        with open(self._path, 'rb') as fobject:
            # while the file is not done keep reading segments
            while True:
                seg = self._read_segment(fobject, lazy)
                # if there are no more Segments, stop
                if not seg:
                    break

                # store the segment and signals
                block.segments.append(seg)
                rchan.analogsignals.append(seg.analogsignals[0])

        # remove the file object
        self._fsrc = None

        block.create_many_to_one_relationship()
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

    def _read_segment(self, fobject, lazy):
        '''
        Read a single segment with a single analogsignal

        Returns the segment or None if there are no more segments
        '''

        try:
            # float64 -- start time of the AnalogSignal
            t_start = np.fromfile(fobject, dtype=np.float64, count=1)[0]
        except IndexError:
            # if there are no more Segments, return
            return False

        # int16 -- index of the stimulus parameters
        seg_index = np.fromfile(fobject, dtype=np.int16, count=1)[0].tolist()

        # int16 -- number of stimulus parameters
        numelements = np.fromfile(fobject, dtype=np.int16, count=1)[0]

        # read the name strings for the stimulus parameters
        paramnames = []
        for _ in range(numelements):
            # unit8 -- the number of characters in the string
            numchars = np.fromfile(fobject, dtype=np.uint8, count=1)[0]

            # char * numchars -- a single name string
            name = np.fromfile(fobject, dtype=np.uint8, count=numchars)

            # exclude invalid characters
            name = str(name[name >= 32].view('c').tostring())

            # add the name to the list of names
            paramnames.append(name)

        # float32 * numelements -- the values for the stimulus parameters
        paramvalues = np.fromfile(fobject, dtype=np.float32, count=numelements)

        # combine parameter names and the parameters as a dict
        params = dict(zip(paramnames, paramvalues))

        # int32 -- the number elements in the AnalogSignal
        numpts = np.fromfile(fobject, dtype=np.int32, count=1)[0]

        # int16 * numpts -- the AnalogSignal itself
        signal = np.fromfile(fobject, dtype=np.int16, count=numpts)

        # handle lazy loading
        if lazy:
            sig = AnalogSignal([], t_start=t_start*pq.d,
                               file_origin=self._filename,
                               sampling_period=1.*pq.s,
                               units=pq.mV,
                               dtype=np.float)
            sig.lazy_shape = len(signal)
        else:
            sig = AnalogSignal(signal.astype(np.float)*pq.mV,
                               t_start=t_start*pq.d,
                               file_origin=self._filename,
                               sampling_period=1.*pq.s,
                               copy=False)
        # Note: setting the sampling_period to 1 s is arbitrary

        # load the AnalogSignal and parameters into a new Segment
        seg = Segment(file_origin=self._filename,
                      index=seg_index,
                      **params)
        seg.analogsignals = [sig]

        return seg
