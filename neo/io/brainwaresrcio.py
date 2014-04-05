# -*- coding: utf-8 -*-
"""
Class for reading from Brainware SRC files

SRC files are binary files for holding spike data.  They are broken up into
nested data sequences of different types, with each type of sequence identified
by a unique ID number.  This allows new versions of sequences to be included
without breaking backwards compatibility, since new versions can just be given
a new ID number.

The ID numbers and the format of the data they contain were taken from the
Matlab-based reader function supplied with BrainWare.  The python code,
however, was implemented from scratch in Python using Python idioms.

There are some situations where BrainWare data can overflow the SRC file,
resulting in a corrupt file.  Neither BrainWare nor the Matlab-based
reader can read such files.  This software, however, will try to recover
the data, and in most cases can do so successfully.

Each SRC file can hold the equivalent of multiple Neo Blocks.

Brainware was developed by Dr. Jan Schnupp and is availabe from
Tucker Davis Technologies, Inc.
http://www.tdt.com/downloads.htm

Neither Dr. Jan Schnupp nor Tucker Davis Technologies, Inc. had any part in the
development of this code

The code is implemented with the permission of Dr. Jan Schnupp

Author: Todd Jennings
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

# import needed core python modules
from datetime import datetime, timedelta
from itertools import chain
import logging
import os.path
import sys

# numpy and quantities are already required by neo
import numpy as np
import quantities as pq

# needed core neo modules
from neo.core import (Block, EventArray, RecordingChannel,
                      RecordingChannelGroup, Segment, SpikeTrain, Unit)

# need to subclass BaseIO
from neo.io.baseio import BaseIO

LOGHANDLER = logging.StreamHandler()

PY_VER = sys.version_info[0]


class BrainwareSrcIO(BaseIO):
    """
    Class for reading Brainware Spike ReCord files with the extension '.src'

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

    The _isopen property tells whether the file is currently open and
    reading or closed.

    Note 1:
        The first Unit in each RecordingChannelGroup is always
        UnassignedSpikes, which has a SpikeTrain for each Segment containing
        all the spikes not assigned to any Unit in that Segment.

    Note 2:
        The first Segment in each Block is always Comments, which stores all
        comments as an EventArray object.

    Note 3:
        The parameters from the BrainWare table for each condition are stored
        in the Segment annotations.  If there are multiple repetitions of
        a condition, each repetition is stored as a separate Segment.

    Note 4:
        There is always only one RecordingChannelGroup.  BrainWare stores the
        equivalent of RecordingChannelGroups in separate files.

    Usage:
        >>> from neo.io.brainwaresrcio import BrainwareSrcIO
        >>> srcfile = BrainwareSrcIO(filename='multi_500ms_mulitrep_ch1.src')
        >>> blk1 = srcfile.read()
        >>> blk2 = srcfile.read_block()
        >>> blks = srcfile.read_all_blocks()
        >>> print blk1.segments
        >>> print blk1.segments[0].spiketrains
        >>> print blk1.units
        >>> print blk1.units[0].name
        >>> print blk2
        >>> print blk2[0].segments
        >>> print blks
        >>> print blks[0].segments
    """

    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects = [Block, RecordingChannel, RecordingChannelGroup,
                         Segment, SpikeTrain, EventArray, Unit]

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
    name = 'Brainware SRC File'
    extensions = ['src']

    mode = 'file'

    def __init__(self, filename=None):
        """
        Arguments:
            filename: the filename
        """
        BaseIO.__init__(self)

        # log the __init__
        self.logger.info('__init__')

        # this stores the filename of the current object, exactly as it is
        # provided when the instance is initialized.
        self._filename = filename

        # this store the filename without the path
        self._file_origin = filename

        # This stores the file object for the current file
        self._fsrc = None

        # This stores the current Block
        self._blk = None

        # This stores the current RecordingChannelGroup for easy access
        # It is equivalent to self._blk.recordingchannelgroups[0]
        self._rcg = None

        # This stores the current Segment for easy access
        # It is equivalent to self._blk.segments[-1]
        self._seg0 = None

        # this stores a dictionary of the Block's Units by name,
        # making it easier and faster to retrieve Units by name later
        # UnassignedSpikes and Units accessed by index are not stored here
        self._unitdict = {}

        # this stores the current Unit
        self._unit0 = None

        # if the file has a list with negative length, the rest of the file's
        # list lengths are unreliable, so we need to store this value for the
        # whole file
        self._damaged = False

        # this stores whether the current file is lazy loaded
        self._lazy = False

        # this stores whether the current file is cascading
        # this is false by default so if we use read_block on its own it works
        self._cascade = False

        # this stores an empty SpikeTrain which is used in various places.
        self._default_spiketrain = None

    @property
    def _isopen(self):
        """
        This property tells whether the SRC file associated with the IO object
        is open.
        """
        return self._fsrc is not None

    def _opensrc(self):
        """
        Open the file if it isn't already open.
        """
        # if the file isn't already open, open it and clear the Blocks
        if not self._fsrc or self._fsrc.closed:
            self._fsrc = open(self._filename, 'rb')

            # figure out the filename of the current file
            self._file_origin = os.path.basename(self._filename)

    def close(self):
        """
        Close the currently-open file and reset the current reading point.
        """
        self.logger.info('close')
        if self._isopen and not self._fsrc.closed:
            self._fsrc.close()

        # we also need to reset all per-file attributes
        self._damaged = False
        self._fsrc = None
        self._seg0 = None
        self._cascade = False
        self._file_origin = None
        self._lazy = False
        self._default_spiketrain = None

    def read(self, lazy=False, cascade=True, **kargs):
        """
        Reads the first Block from the Spike ReCording file "filename"
        generated with BrainWare.

        If you wish to read more than one Block, please use read_all_blocks.
        """
        return self.read_block(lazy=lazy, cascade=cascade, **kargs)

    def read_block(self, lazy=False, cascade=True, **kargs):
        """
        Reads the first Block from the Spike ReCording file "filename"
        generated with BrainWare.

        If you wish to read more than one Block, please use read_all_blocks.
        """

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')

        blockobj = self.read_next_block(cascade=cascade, lazy=lazy)
        self.close()
        return blockobj

    def read_next_block(self, cascade=True, lazy=False, **kargs):
        """
        Reads a single Block from the Spike ReCording file "filename"
        generated with BrainWare.

        Each call of read will return the next Block until all Blocks are
        loaded.  After the last Block, the file will be automatically closed
        and the progress reset.  Call the close method manually to reset
        back to the first Block.
        """

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')

        self._lazy = lazy
        self._opensrc()

        # create _default_spiketrain here for performance reasons
        self._default_spiketrain = self._init_default_spiketrain.copy()
        self._default_spiketrain.file_origin = self._file_origin
        if lazy:
            self._default_spiketrain.lazy_shape = (0,)

        # create the Block and the contents all Blocks of from IO share
        self._blk = Block(file_origin=self._file_origin)
        if not cascade:
            return self._blk
        self._rcg = RecordingChannelGroup(file_origin=self._file_origin)
        self._seg0 = Segment(name='Comments', file_origin=self._file_origin)
        self._unit0 = Unit(name='UnassignedSpikes',
                           file_origin=self._file_origin,
                           elliptic=[], boundaries=[],
                           timestamp=[], max_valid=[])
        self._blk.recordingchannelgroups.append(self._rcg)
        self._rcg.units.append(self._unit0)
        self._blk.segments.append(self._seg0)

        # this actually reads the contents of the Block
        result = []
        while hasattr(result, '__iter__'):
            try:
                result = self._read_by_id()
            except:
                self.close()
                raise

        # set the recorging channel group names and indices
        chans = self._rcg.recordingchannels
        chan_inds = np.arange(len(chans), dtype='int')
        chan_names = np.array(['Chan'+str(i) for i in chan_inds],
                              dtype='string_')
        self._rcg.channel_indexes = chan_inds
        self._rcg.channel_names = chan_names

        # since we read at a Block level we always do this
        self._blk.create_many_to_one_relationship()

        # put the Block in a local object so it can be gargabe collected
        blockobj = self._blk

        # reset the per-Block attributes
        self._blk = None
        self._rcg = None
        self._unitdict = {}

        # combine the comments into one big eventarray
        self._combine_segment_eventarrays(self._seg0)

        # result is None iff the end of the file is reached, so we can
        # close the file
        # this notification is not helpful if using the read method with
        # cascade==True, since the user will know it is done when the method
        # returns a value
        if result is None:
            self.logger.info('Last Block read.  Closing file.')
            self.close()

        return blockobj

    def read_all_blocks(self, cascade=True, lazy=False, **kargs):
        """
        Reads all Blocks from the Spike ReCording file "filename"
        generated with BrainWare.

        The progress in the file is reset and the file closed then opened again
        prior to reading.

        The file is automatically closed after reading completes.
        """

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')

        self._lazy = lazy
        self._cascade = True

        self.close()
        self._opensrc()

        # Read each Block.
        # After the last Block self._isopen is set to False, so this make a
        # good way to determine when to stop
        blocks = []
        while self._isopen:
            try:
                blocks.append(self.read_next_block(cascade=cascade,
                                                   lazy=lazy))
            except:
                self.close()
                raise

        return blocks

    def _convert_timestamp(self, timestamp, start_date=datetime(1899, 12, 30)):
        """
        _convert_timestamp(timestamp, start_date) - convert a timestamp in
        brainware src file units to a python datetime object.

        start_date defaults to 1899.12.30 (ISO format), which is the start date
        used by all BrainWare SRC data Blocks so far.  If manually specified
        it should be a datetime object or any other object that can be added
        to a timedelta object.
        """
        # datetime + timedelta = datetime again.
        try:
            timestamp = convert_brainwaresrc_timestamp(timestamp, start_date)
        except OverflowError as err:
            timestamp = start_date
            self.logger.exception('_convert_timestamp overflow')

        return timestamp

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #  All methods from here on are private.  They are not intended to be used
    #  on their own, although methods that could theoretically be called on
    #  their own are marked as such.  All private methods could be renamed,
    #  combined, or split at any time.  All private methods prefixed by
    #  "__read" or "__skip" will alter the current place in the file.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    #@profile
    def _read_by_id(self):
        """
        Reader for generic data

        BrainWare SRC files are broken up into data sequences that are
        identified by an ID code.  This method determines the ID code and calls
        the method to read the data sequence with that ID code.  See the
        _ID_DICT attribute for a dictionary of code/method pairs.

        IMPORTANT!!!
        This is the only private method that can be called directly.
        The rest of the private methods can only safely be called by this
        method or by other private methods, since they depend on the
        current position in the file.
        """

        try:
            # uint16 -- the ID code of the next sequence
            seqid = np.asscalar(np.fromfile(self._fsrc,
                                            dtype=np.uint16, count=1))
        except ValueError:
            # return a None if at EOF.  Other methods use None to recognize
            # an EOF
            return None

        # using the seqid, get the reader function from the reader dict
        readfunc = self._ID_DICT.get(seqid)
        if readfunc is None:
            if seqid <= 0:
                # return if end-of-sequence ID code.  This has to be 0.
                # just calling "return" will return a None which is used as an
                # EOF indicator
                return 0
            else:
                # return a warning if the key is invalid
                # (this is consistent with the official behavior,
                # even the official reference files have invalid keys
                # when using the official reference reader matlab
                # scripts
                self.logger.warning('unknown ID: %s',  seqid)
                return []

        try:
            # run the function to get the data
            return readfunc(self)
        except (EOFError, UnicodeDecodeError) as err:
            # return a warning if the EOF is reached in the middle of a method
            self.logger.exception('Premature end of file')
            return None

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #   These are helper methods.  They don't read from the file, so it
    #   won't harm the reading process to call them, but they are only relevant
    #   when used in other private methods.
    #
    #   These are tuned to the particular needs of this IO class, they are
    #   unlikely to work properly if used with another file format.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _assign_sequence(self, data_obj):
        """
        _assign_sequence(data_obj) - Try to guess where an unknown sequence
        should go based on its class.  Warning are issued if this method is
        used since manual reorganization may be needed.
        """
        if isinstance(data_obj, Unit):
            self.logger.warning('Unknown Unit found, adding to Units list')
            self._rcg.units.append(data_obj)
            if data_obj.name:
                self._unitdict[data_obj.name] = data_obj
        elif isinstance(data_obj, Segment):
            self.logger.warning('Unknown Segment found, '
                                 'adding to Segments list')
            self._blk.segments.append(data_obj)
        elif isinstance(data_obj, EventArray):
            self.logger.warning('Unknown EventArray found, '
                                 'adding to comment Events list')
            self._seg0.eventarrays.append(data_obj)
        elif isinstance(data_obj, SpikeTrain):
            self.logger.warning('Unknown SpikeTrain found, '
                                 'adding to the UnassignedSpikes Unit')
            self._unit0.spiketrains.append(data_obj)
        elif hasattr(data_obj, '__iter__') and not isinstance(data_obj, str):
            for sub_obj in data_obj:
                self._assign_sequence(sub_obj)
        else:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning('Unrecognized sequence of type %s found, '
                                     'skipping', type(data_obj))

    _default_datetime = datetime(1, 1, 1)
    _default_t_start = pq.Quantity(0., units=pq.ms, dtype=np.float32)
    _init_default_spiketrain = SpikeTrain(times=pq.Quantity([], units=pq.ms,
                                                            dtype=np.float32),
                                          t_start=pq.Quantity(0, units=pq.ms,
                                                              dtype=np.float32
                                                              ),
                                          t_stop=pq.Quantity(1, units=pq.ms,
                                                             dtype=np.float32),
                                          waveforms=pq.Quantity([[[]]],
                                                                dtype=np.int8,
                                                                units=pq.mV),
                                          dtype=np.float32, copy=False,
                                          timestamp=_default_datetime,
                                          respwin=np.array([], dtype=np.int32),
                                          dama_index=-1,
                                          trig2=pq.Quantity([], units=pq.ms,
                                                            dtype=np.uint8),
                                          side='')

    def _combine_eventarrays(self, eventarrays):
        """
        _combine_eventarrays(eventarrays) - combine a list of EventArrays
        with single events into one long EventArray
        """
        if not eventarrays or self._lazy:
            eventarray = EventArray(times=pq.Quantity([], units=pq.s),
                                    labels=np.array([], dtype='S'),
                                    senders=np.array([], dtype='S'),
                                    t_start=0)
            if self._lazy:
                eventarray.lazy_shape = len(eventarrays)
            return eventarray

        times = []
        labels = []
        senders = []
        for event in eventarrays:
            times.append(event.times.magnitude)
            labels.append(event.labels)
            senders.append(event.annotations['sender'])

        times = np.array(times, dtype=np.float32)
        t_start = times.min()
        times = pq.Quantity(times-t_start, units=pq.d).rescale(pq.s)

        labels = np.array(labels)
        senders = np.array(senders)

        eventarray = EventArray(times=times, labels=labels,
                                t_start=t_start.tolist(), senders=senders)

        return eventarray

    def _combine_segment_eventarrays(self, segment):
        """
        _combine_segment_eventarrays(segment)
        Combine all EventArrays in a segment.
        """
        eventarray = self._combine_eventarrays(segment.eventarrays)
        eventarray_t_start = eventarray.annotations.pop('t_start')
        segment.rec_datetime = self._convert_timestamp(eventarray_t_start)
        segment.eventarrays = [eventarray]
        eventarray.segment = segment

    def _combine_spiketrains(self, spiketrains):
        """
        _combine_spiketrains(spiketrains) - combine a list of SpikeTrains
        with single spikes into one long SpikeTrain
        """

        if not spiketrains:
            return self._default_spiketrain.copy()

        if hasattr(spiketrains[0], 'waveforms') and len(spiketrains) == 1:
            train = spiketrains[0]
            if self._lazy and not hasattr(train, 'lazy_shape'):
                train.lazy_shape = train.shape
                train = train[:0]
            return train

        if hasattr(spiketrains[0], 't_stop'):
            # workaround for bug in some broken files
            istrain = [hasattr(utrain, 'waveforms') for utrain in spiketrains]
            if not all(istrain):
                goodtrains = [itrain for i, itrain in enumerate(spiketrains)
                              if istrain[i]]
                badtrains = [itrain for i, itrain in enumerate(spiketrains)
                             if not istrain[i]]

                spiketrains = (goodtrains +
                               [self._combine_spiketrains(badtrains)])

            spiketrains = [itrain for itrain in spiketrains if itrain.size > 0]
            if not spiketrains:
                return self._default_spiketrain.copy()

            # get the times of the spiketrains and combine them
            waveforms = [itrain.waveforms for itrain in spiketrains]
            rawtrains = np.array(np.concatenate(spiketrains, axis=1))
            times = pq.Quantity(rawtrains, units=pq.ms, copy=False)
            lens1 = np.array([wave.shape[1] for wave in waveforms])
            lens2 = np.array([wave.shape[2] for wave in waveforms])
            if lens1.max() != lens1.min() or lens2.max() != lens2.min():
                lens1 = lens1.max() - lens1
                lens2 = lens2.max() - lens2
                waveforms = [np.pad(waveform,
                                    ((0, 0), (0, len1), (0, len2)),
                                    'constant')
                             for waveform, len1, len2 in zip(waveforms,
                                                             lens1,
                                                             lens2)]

            waveforms = np.concatenate(waveforms, axis=0)

            # extract the trig2 annotation
            trig2 = np.array(np.concatenate([itrain.annotations['trig2'] for
                                             itrain in spiketrains], axis=1))
            trig2 = pq.Quantity(trig2, units=pq.ms)
        elif hasattr(spiketrains[0], 'units'):
            return self._combine_spiketrains([spiketrains])
        else:
            times, waveforms, trig2 = zip(*spiketrains)
            times = np.concatenate(times, axis=0)

            # get the times of the SpikeTrains and combine them
            times = pq.Quantity(times, units=pq.ms, copy=False)

            # get the waveforms of the SpikeTrains and combine them
            # these should be a 3D array with the first axis being the spike,
            # the second axis being the recording channel (there is only one),
            # and the third axis being the actual waveform
            waveforms = np.concatenate(waveforms, axis=0)

            # extract the trig2 annotation
            trig2 = pq.Quantity(np.concatenate(trig2, axis=1),
                                units=pq.ms, copy=False)

        if not times.size:
            return self._default_spiketrain.copy()

        # get the maximum time
        t_stop = times[-1] * 2.

        if self._lazy:
            timesshape = times.shape
            times = pq.Quantity([], units=pq.ms, copy=False)
            waveforms = pq.Quantity([[[]]], units=pq.mV)
        else:
            waveforms = pq.Quantity(waveforms, units=pq.mV, copy=False)

        train = SpikeTrain(times=times, copy=False,
                           t_start=self._default_t_start.copy(), t_stop=t_stop,
                           file_origin=self._file_origin,
                           waveforms=waveforms,
                           timestamp=self._default_datetime,
                           respwin=np.array([], dtype=np.int32),
                           dama_index=-1, trig2=trig2, side='')
        if self._lazy:
            train.lazy_shape = timesshape
        return train

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #   IMPORTANT!!!
    #   These are private methods implementing the internal reading mechanism.
    #   Due to the way BrainWare SRC files are structured, they CANNOT be used
    #   on their own.  Calling these manually will almost certainly alter your
    #   position in the file in an unrecoverable manner, whether they throw
    #   an exception or not.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def __read_str(self, numchars=1, utf=None):
        """
        Read a string of a specific length.

        This is compatible with python 2 and python 3.
        """
        rawstr = np.asscalar(np.fromfile(self._fsrc,
                                         dtype='S%s' % numchars, count=1))
        if utf or (utf is None and PY_VER == 3):
            return rawstr.decode('utf-8')
        return rawstr

    def __read_annotations(self):
        """
        Read the stimulus grid properties.

        -------------------------------------------------------------------
        Returns a dictionary containing the parameter names as keys and the
        parameter values as values.

        The returned object must be added to the Block.

        ID: 29109
        """

        # int16 -- number of stimulus parameters
        numelements = np.fromfile(self._fsrc, dtype=np.int16, count=1)[0]
        if not numelements:
            return {}

        # [data sequence] * numelements -- parameter names
        names = []
        for i in range(numelements):
            # {skip} = byte (char) -- skip one byte
            self._fsrc.seek(1, 1)

            # uint8 -- length of next string
            numchars = np.asscalar(np.fromfile(self._fsrc,
                                               dtype=np.uint8, count=1))

            # if there is no name, make one up
            if not numchars:
                name = 'param%s' % i
            else:
                # char * numchars -- parameter name string
                name = self.__read_str(numchars)

            # if the name is already in there, add a unique number to it
            # so it isn't overwritten
            if name in names:
                name = name + str(i)

            names.append(name)

        # float32 * numelements -- an array of parameter values
        values = np.fromfile(self._fsrc, dtype=np.float32,
                             count=numelements)

        # combine the names and values into a dict
        # the dict will be added to the annotations
        annotations = dict(zip(names, values))

        return annotations

    def __read_annotations_old(self):
        """
        Read the stimulus grid properties.

        Returns a dictionary containing the parameter names as keys and the
        parameter values as values.

        ------------------------------------------------
        The returned objects must be added to the Block.

        This reads an old version of the format that does not store paramater
        names, so placeholder names are created instead.

        ID: 29099
        """

        # int16 * 14 -- an array of parameter values
        values = np.fromfile(self._fsrc, dtype=np.int16, count=14)

        # create dummy names and combine them with the values in a dict
        # the dict will be added to the annotations
        params = ['param%s' % i for i in range(len(values))]
        annotations = dict(zip(params, values))

        return annotations

    def __read_comment(self):
        """
        Read a single comment.

        The comment is stored as an EventArray in Segment 0, which is
        specifically for comments.

        ----------------------
        Returns an empty list.

        The returned object is already added to the Block.

        No ID number: always called from another method
        """

        # float64 -- timestamp (number of days since dec 30th 1899)
        time = np.fromfile(self._fsrc, dtype=np.double, count=1)[0]

        # int16 -- length of next string
        numchars1 = np.asscalar(np.fromfile(self._fsrc,
                                            dtype=np.int16, count=1))

        # char * numchars -- the one who sent the comment
        sender = self.__read_str(numchars1)

        # int16 -- length of next string
        numchars2 = np.asscalar(np.fromfile(self._fsrc,
                                            dtype=np.int16, count=1))

        # char * numchars -- comment text
        text = self.__read_str(numchars2, utf=False)

        comment = EventArray(times=pq.Quantity(time, units=pq.d), labels=text,
                             sender=sender,
                             file_origin=self._file_origin)

        self._seg0.eventarrays.append(comment)

        return []

    def __read_list(self):
        """
        Read a list of arbitrary data sequences

        It only says how many data sequences should be read.  These sequences
        are then read by their ID number.

        Note that lists can be nested.

        If there are too many sequences (for instance if there are a large
        number of spikes in a Segment) then a negative number will be returned
        for the number of data sequences to read.  In this case the method
        tries to guess.  This also means that all future list data sequences
        have unreliable lengths as well.

        -------------------------------------------
        Returns a list of objects.

        Whether these objects need to be added to the Block depends on the
        object in question.

        There are several data sequences that have identical formats but are
        used in different situations.  That means this data sequences has
        multiple ID numbers.
        ID: 29082
        ID: 29083
        ID: 29091
        ID: 29093
        """

        # int16 -- number of sequences to read
        numelements = np.fromfile(self._fsrc, dtype=np.int16, count=1)[0]

        # {skip} = bytes * 4 (int16 * 2) -- skip four bytes
        self._fsrc.seek(4, 1)

        if numelements == 0:
            return []

        if not self._damaged and numelements < 0:
            self._damaged = True
            self.logger.error('Negative sequence count %s, file damaged',
                               numelements)

        if not self._damaged:
            # read the sequences into a list
            seq_list = [self._read_by_id() for _ in range(numelements)]
        else:
            # read until we get some indication we should stop
            seq_list = []

            # uint16 -- the ID of the next sequence
            seqidinit = np.fromfile(self._fsrc, dtype=np.uint16, count=1)[0]

            # {rewind} = byte * 2 (int16) -- move back 2 bytes, i.e. go back to
            # before the beginning of the seqid
            self._fsrc.seek(-2, 1)
            while 1:
                # uint16 -- the ID of the next sequence
                seqid = np.fromfile(self._fsrc, dtype=np.uint16, count=1)[0]

                # {rewind} = byte * 2 (int16) -- move back 2 bytes, i.e. go
                # back to before the beginning of the seqid
                self._fsrc.seek(-2, 1)

                # if we come across a new sequence, we are at the end of the
                # list so we should stop
                if seqidinit != seqid:
                    break

                # otherwise read the next sequence
                seq_list.append(self._read_by_id())

        return seq_list

    def __read_segment(self):
        """
        Read an individual Segment.

        A Segment contains a dictionary of parameters, the length of the
        recording, a list of Units with their Spikes, and a list of Spikes
        not assigned to any Unit.  The unassigned spikes are always stored in
        Unit 0, which is exclusively for storing these spikes.

        -------------------------------------------------
        Returns the Segment object created by the method.

        The returned object is already added to the Block.

        ID: 29106
        """

        # (data_obj) -- the stimulus parameters for this segment
        annotations = self._read_by_id()
        annotations['feature_type'] = -1
        annotations['go_by_closest_unit_center'] = False
        annotations['include_unit_bounds'] = False

        # (data_obj) -- SpikeTrain list of unassigned spikes
        # these go in the first Unit since it is for unassigned spikes
        unassigned_spikes = self._read_by_id()
        self._unit0.spiketrains.extend(unassigned_spikes)

        # read a list of units and grab the second return value, which is the
        # SpikeTrains from this Segment (if we use the Unit we will get all the
        # SpikeTrains from that Unit, resuling in duplicates if we are past
        # the first Segment
        trains = self._read_by_id()
        if not trains:
            if unassigned_spikes:
                # if there are no assigned spikes,
                # just use the unassigned spikes
                trains = zip(unassigned_spikes)
            else:
                # if there are no spiketrains at all,
                # create an empty spike train
                trains = [[self._default_spiketrain.copy()]]
        elif hasattr(trains[0], 'dtype'):
            #workaround for some broken files
            trains = [unassigned_spikes +
                      [self._combine_spiketrains([trains])]]
        else:
            # get the second element from each returned value,
            # which is the actual SpikeTrains
            trains = [unassigned_spikes] + [train[1] for train in trains]
            # re-organize by sweeps
            trains = zip(*trains)

        # int32 -- SpikeTrain length in ms
        spiketrainlen = pq.Quantity(np.fromfile(self._fsrc, dtype=np.int32,
                                    count=1)[0], units=pq.ms, copy=False)

        segments = []
        for train in trains:
            # create the Segment and add everything to it
            segment = Segment(file_origin=self._file_origin,
                              **annotations)
            segment.spiketrains = train
            self._blk.segments.append(segment)
            segments.append(segment)

            for itrain in train:
                # use the SpikeTrain length to figure out the stop time
                # t_start is always 0 so we can ignore it
                itrain.t_stop = spiketrainlen

        return segments

    def __read_segment_list(self):
        """
        Read a list of Segments with comments.

        Since comments can occur at any point, whether a recording is happening
        or not, it is impossible to reliably assign them to a specific Segment.
        For this reason they are always assigned to Segment 0, which is
        exclusively used to store comments.

        --------------------------------------------------------
        Returns a list of the Segments created with this method.

        The returned objects are already added to the Block.

        ID: 29112
        """

        # uint8 --  number of electrode channels in the Segment
        numchannels = np.fromfile(self._fsrc, dtype=np.uint8, count=1)[0]

        # [list of sequences] -- individual Segments
        segments = self.__read_list()
        while not hasattr(segments[0], 'spiketrains'):
            segments = list(chain(*segments))

        # char -- "side of brain" info
        side = self.__read_str(1)

        # int16 -- number of comments
        numelements = np.fromfile(self._fsrc, dtype=np.int16, count=1)[0]

        # comment_obj * numelements -- comments about the Segments
        # we don't know which Segment specifically, though
        for _ in range(numelements):
            self.__read_comment()

        # create an empty RecordingChannel for each of the numchannels
        for i in range(numchannels):
            chan = RecordingChannel(file_origin=self._file_origin,
                                    index=int(i), name='Chan'+str(int(i)))
            self._rcg.recordingchannels.append(chan)

        # store what side of the head we are dealing with
        for segment in segments:
            for spiketrain in segment.spiketrains:
                spiketrain.annotations['side'] = side

        return segments

    def __read_segment_list_v8(self):
        """
        Read a list of Segments with comments.

        This is version 8 of the data sequence.

        This is the same as __read_segment_list_var, but can also contain
        one or more arbitrary sequences.  The class makes an attempt to assign
        the sequences when possible, and warns the user when this happens
        (see the _assign_sequence method)

        --------------------------------------------------------
        Returns a list of the Segments created with this method.

        The returned objects are already added to the Block.

        ID: 29117
        """

        # segment_collection_var -- this is based off a segment_collection_var
        segments = self.__read_segment_list_var()

        # uint16 -- the ID of the next sequence
        seqid = np.fromfile(self._fsrc, dtype=np.uint16, count=1)[0]

        # {rewind} = byte * 2 (int16) -- move back 2 bytes, i.e. go back to
        # before the beginning of the seqid
        self._fsrc.seek(-2, 1)

        if seqid in self._ID_DICT:
            # if it is a valid seqid, read it and try to figure out where
            # to put it
            self._assign_sequence(self._read_by_id())
        else:
            # otherwise it is a Unit list
            self.__read_unit_list()

        # {skip} = byte * 2 (int16) -- skip 2 bytes
        self._fsrc.seek(2, 1)

        return segments

    def __read_segment_list_v9(self):
        """
        Read a list of Segments with comments.

        This is version 9 of the data sequence.

        This is the same as __read_segment_list_v8, but contains some
        additional annotations.  These annotations are added to the Segment.

        --------------------------------------------------------
        Returns a list of the Segments created with this method.

        The returned objects are already added to the Block.

        ID: 29120
        """

        # segment_collection_v8 -- this is based off a segment_collection_v8
        segments = self.__read_segment_list_v8()

        # uint8
        feature_type = np.fromfile(self._fsrc, dtype=np.uint8,
                                   count=1)[0]

        # uint8
        go_by_closest_unit_center = np.fromfile(self._fsrc, dtype=np.bool8,
                                                count=1)[0]

        # uint8
        include_unit_bounds = np.fromfile(self._fsrc, dtype=np.bool8,
                                          count=1)[0]

        # create a dictionary of the annotations
        annotations = {'feature_type': feature_type,
                       'go_by_closest_unit_center': go_by_closest_unit_center,
                       'include_unit_bounds': include_unit_bounds}

        # add the annotations to each Segment
        for segment in segments:
            segment.annotations.update(annotations)

        return segments

    def __read_segment_list_var(self):
        """
        Read a list of Segments with comments.

        This is the same as __read_segment_list, but contains information
        regarding the sampling period.  This information is added to the
        SpikeTrains in the Segments.

        --------------------------------------------------------
        Returns a list of the Segments created with this method.

        The returned objects are already added to the Block.

        ID: 29114
        """

        # float32 -- DA conversion clock period in microsec
        sampling_period = pq.Quantity(np.fromfile(self._fsrc,
                                                  dtype=np.float32, count=1),
                                      units=pq.us, copy=False)[0]

        # segment_collection -- this is based off a segment_collection
        segments = self.__read_segment_list()

        # add the sampling period to each SpikeTrain
        for segment in segments:
            for spiketrain in segment.spiketrains:
                spiketrain.sampling_period = sampling_period

        return segments

    def __read_spike_fixed(self, numpts=40):
        """
        Read a spike with a fixed waveform length (40 time bins)

        -------------------------------------------
        Returns the time, waveform and trig2 value.

        The returned objects must be converted to a SpikeTrain then
        added to the Block.

        ID: 29079
        """

        # float32 -- spike time stamp in ms since start of SpikeTrain
        time = np.fromfile(self._fsrc, dtype=np.float32, count=1)

        # int8 * 40 -- spike shape -- use numpts for spike_var
        waveform = np.fromfile(self._fsrc, dtype=np.int8,
                               count=numpts).reshape(1, 1, numpts)

        # uint8 -- point of return to noise
        trig2 = np.fromfile(self._fsrc, dtype=np.uint8, count=1)

        return time, waveform, trig2

    def __read_spike_fixed_old(self):
        """
        Read a spike with a fixed waveform length (40 time bins)

        This is an old version of the format.  The time is stored as ints
        representing 1/25 ms time steps.  It has no trigger information.

        -------------------------------------------
        Returns the time, waveform and trig2 value.

        The returned objects must be converted to a SpikeTrain then
        added to the Block.

        ID: 29081
        """

        # int32 -- spike time stamp in ms since start of SpikeTrain
        time = np.fromfile(self._fsrc, dtype=np.int32, count=1) / 25.
        time = time.astype(np.float32)

        # int8 * 40 -- spike shape
        # This needs to be a 3D array, one for each channel.  BrainWare
        # only ever has a single channel per file.
        waveform = np.fromfile(self._fsrc, dtype=np.int8,
                               count=40).reshape(1, 1, 40)

        # create a dummy trig2 value
        trig2 = np.array([-1], dtype=np.uint8)

        return time, waveform, trig2

    def __read_spike_var(self):
        """
        Read a spike with a variable waveform length

        -------------------------------------------
        Returns the time, waveform and trig2 value.

        The returned objects must be converted to a SpikeTrain then
        added to the Block.

        ID: 29115
        """

        # uint8 -- number of points in spike shape
        numpts = np.fromfile(self._fsrc, dtype=np.uint8, count=1)[0]

        # spike_fixed is the same as spike_var if you don't read the numpts
        # byte and set numpts = 40
        return self.__read_spike_fixed(numpts)

    def __read_spiketrain_indexed(self):
        """
        Read a SpikeTrain

        This is the same as __read_spiketrain_timestamped except it also
        contains the index of the Segment in the dam file.

        The index is stored as an annotation in the SpikeTrain.

        -------------------------------------------------
        Returns a SpikeTrain object with multiple spikes.

        The returned object must be added to the Block.

        ID: 29121
        """

        #int32 -- index of the analogsignalarray in corresponding .dam file
        dama_index = np.fromfile(self._fsrc, dtype=np.int32,
                                 count=1)[0]

        # spiketrain_timestamped -- this is based off a spiketrain_timestamped
        spiketrain = self.__read_spiketrain_timestamped()

        # add the property to the dict
        spiketrain.annotations['dama_index'] = dama_index

        return spiketrain

    def __read_spiketrain_timestamped(self):
        """
        Read a SpikeTrain

        This SpikeTrain contains a time stamp for when it was recorded

        The timestamp is stored as an annotation in the SpikeTrain.

        -------------------------------------------------
        Returns a SpikeTrain object with multiple spikes.

        The returned object must be added to the Block.

        ID: 29110
        """

        # float64 -- timeStamp (number of days since dec 30th 1899)
        timestamp = np.fromfile(self._fsrc, dtype=np.double, count=1)[0]

        # convert to datetime object
        timestamp = self._convert_timestamp(timestamp)

        # seq_list -- spike list
        # combine the spikes into a single SpikeTrain
        spiketrain = self._combine_spiketrains(self.__read_list())

        # add the timestamp
        spiketrain.annotations['timestamp'] = timestamp

        return spiketrain

    def __read_unit(self):
        """
        Read all SpikeTrains from a single Segment and Unit

        This is the same as __read_unit_unsorted except it also contains
        information on the spike sorting boundaries.

        ------------------------------------------------------------------
        Returns a single Unit and a list of SpikeTrains from that Unit and
        current Segment, in that order.  The SpikeTrains must be returned since
        it is not possible to determine from the Unit which SpikeTrains are
        from the current Segment.

        The returned objects are already added to the Block.  The SpikeTrains
        must be added to the current Segment.

        ID: 29116
        """

        # same as unsorted Unit
        unit, trains = self.__read_unit_unsorted()

        # float32 * 18 -- Unit boundaries (IEEE 32-bit floats)
        unit.annotations['boundaries'] = [np.fromfile(self._fsrc,
                                                      dtype=np.float32,
                                                      count=18)]

        # uint8 * 9 -- boolean values indicating elliptic feature boundary
        # dimensions
        unit.annotations['elliptic'] = [np.fromfile(self._fsrc,
                                                    dtype=np.uint8,
                                                    count=9)]

        return unit, trains

    def __read_unit_list(self):
        """
        A list of a list of Units

        -----------------------------------------------
        Returns a list of Units modified in the method.

        The returned objects are already added to the Block.

        No ID number: only called by other methods
        """

        # this is used to figure out which Units to return
        maxunit = 1

        # int16 -- number of time slices
        numelements = np.fromfile(self._fsrc, dtype=np.int16, count=1)[0]

        # {sequence} * numelements1 -- the number of lists of Units to read
        self._rcg.annotations['max_valid'] = []
        for i in range(numelements):

            # {skip} = byte * 2 (int16) -- skip 2 bytes
            self._fsrc.seek(2, 1)

            # double
            max_valid = np.fromfile(self._fsrc, dtype=np.double, count=1)[0]

            # int16 - the number of Units to read
            numunits = np.fromfile(self._fsrc, dtype=np.int16, count=1)[0]

            # update tha maximum Unit so far
            maxunit = max(maxunit, numunits + 1)

            # if there aren't enough Units, create them
            # remember we need to skip the UnassignedSpikes Unit
            if numunits > len(self._rcg.units) + 1:
                for ind1 in range(len(self._rcg.units), numunits + 1):
                    unit = Unit(name='unit%s' % ind1,
                                file_origin=self._file_origin,
                                elliptic=[], boundaries=[],
                                timestamp=[], max_valid=[])
                    self._rcg.units.append(unit)

            # {Block} * numelements -- Units
            for ind1 in range(numunits):
                # get the Unit with the given index
                # remember we need to skip the UnassignedSpikes Unit
                unit = self._rcg.units[ind1 + 1]

                # {skip} = byte * 2 (int16) -- skip 2 bytes
                self._fsrc.seek(2, 1)

                # int16 -- a multiplier for the elliptic and boundaries
                #          properties
                numelements3 = np.fromfile(self._fsrc, dtype=np.int16,
                                           count=1)[0]

                # uint8 * 10 * numelements3 -- boolean values indicating
                # elliptic feature boundary dimensions
                elliptic = np.fromfile(self._fsrc, dtype=np.uint8,
                                       count=10 * numelements3)

                # float32 * 20 * numelements3 -- feature boundaries
                boundaries = np.fromfile(self._fsrc, dtype=np.float32,
                                         count=20 * numelements3)

                unit.annotations['elliptic'].append(elliptic)
                unit.annotations['boundaries'].append(boundaries)
                unit.annotations['max_valid'].append(max_valid)

        return self._rcg.units[1:maxunit]

    def __read_unit_list_timestamped(self):
        """
        A list of a list of Units.

        This is the same as __read_unit_list, except that it also has a
        timestamp.  This is added ad an annotation to all Units.

        -----------------------------------------------
        Returns a list of Units modified in the method.

        The returned objects are already added to the Block.

        ID: 29119
        """

        # double -- time zero (number of days since dec 30th 1899)
        timestamp = np.fromfile(self._fsrc, dtype=np.double, count=1)[0]

        # convert to to days since UNIX epoc time:
        timestamp = self._convert_timestamp(timestamp)

        # sorter -- this is based off a sorter
        units = self.__read_unit_list()

        for unit in units:
            unit.annotations['timestamp'].append(timestamp)

        return units

    def __read_unit_old(self):
        """
        Read all SpikeTrains from a single Segment and Unit

        This is the same as __read_unit_unsorted except it also contains
        information on the spike sorting boundaries.

        This is an old version of the format that used 48-bit floating-point
        numbers for the boundaries.  These cannot easily be read and so are
        skipped.

        ------------------------------------------------------------------
        Returns a single Unit and a list of SpikeTrains from that Unit and
        current Segment, in that order.  The SpikeTrains must be returned since
        it is not possible to determine from the Unit which SpikeTrains are
        from the current Segment.

        The returned objects are already added to the Block.  The SpikeTrains
        must be added to the current Segment.

        ID: 29107
        """

        # same as Unit
        unit, trains = self.__read_unit_unsorted()

        # bytes * 108 (float48 * 18) -- Unit boundaries (48-bit floating
        # point numbers are not supported so we skip them)
        self._fsrc.seek(108, 1)

        # uint8 * 9 -- boolean values indicating elliptic feature boundary
        # dimensions
        unit.annotations['elliptic'] = np.fromfile(self._fsrc, dtype=np.uint8,
                                                   count=9).tolist()

        return unit, trains

    def __read_unit_unsorted(self):
        """
        Read all SpikeTrains from a single Segment and Unit

        This does not contain Unit boundaries.

        ------------------------------------------------------------------
        Returns a single Unit and a list of SpikeTrains from that Unit and
        current Segment, in that order.  The SpikeTrains must be returned since
        it is not possible to determine from the Unit which SpikeTrains are
        from the current Segment.

        The returned objects are already added to the Block.  The SpikeTrains
        must be added to the current Segment.

        ID: 29084
        """

        # {skip} = bytes * 2 (uint16) -- skip two bytes
        self._fsrc.seek(2, 1)

        # uint16 -- number of characters in next string
        numchars = np.asscalar(np.fromfile(self._fsrc,
                                           dtype=np.uint16, count=1))

        # char * numchars -- ID string of Unit
        name = self.__read_str(numchars)

        # int32 -- SpikeTrain length in ms
        # int32 * 4 -- response and spon period boundaries
        parts = np.fromfile(self._fsrc, dtype=np.int32, count=5)
        t_stop = pq.Quantity(parts[0].astype('float32'),
                             units=pq.ms, copy=False)
        respwin = parts[1:]

        # (data_obj) -- list of SpikeTrains
        spikeslists = self._read_by_id()

        # use the Unit if it already exists, otherwise create it
        if name in self._unitdict:
            unit = self._unitdict[name]
        else:
            unit = Unit(name=name, file_origin=self._file_origin,
                        elliptic=[], boundaries=[], timestamp=[], max_valid=[])
            self._rcg.units.append(unit)
            self._unitdict[name] = unit

        # convert the individual spikes to SpikeTrains and add them to the Unit
        trains = [self._combine_spiketrains(spikes) for spikes in spikeslists]
        unit.spiketrains.extend(trains)
        for train in trains:
            train.t_stop = t_stop.copy()
            train.annotations['respwin'] = respwin.copy()

        return unit, trains

    def __skip_information(self):
        """
        Read an information sequence.

        This is data sequence is skipped both here and in the Matlab reference
        implementation.

        ----------------------
        Returns an empty list

        Nothing is created so nothing is added to the Block.

        ID: 29113
        """

        # {skip} char * 34 -- display information
        self._fsrc.seek(34, 1)

        return []

    def __skip_information_old(self):
        """
        Read an information sequence

        This is data sequence is skipped both here and in the Matlab reference
        implementation

        This is an old version of the format

        ----------------------
        Returns an empty list.

        Nothing is created so nothing is added to the Block.

        ID: 29100
        """

        # {skip} char * 4 -- display information
        self._fsrc.seek(4, 1)

        return []

    # This dictionary maps the numeric data sequence ID codes to the data
    # sequence reading functions.
    #
    # Since functions are first-class objects in Python, the functions returned
    # from this dictionary are directly callable.
    #
    # If new data sequence ID codes are added in the future please add the code
    # here in numeric order and the method above in alphabetical order
    #
    # The naming of any private method may change at any time
    _ID_DICT = {29079: __read_spike_fixed,
                29081: __read_spike_fixed_old,
                29082: __read_list,
                29083: __read_list,
                29084: __read_unit_unsorted,
                29091: __read_list,
                29093: __read_list,
                29099: __read_annotations_old,
                29100: __skip_information_old,
                29106: __read_segment,
                29107: __read_unit_old,
                29109: __read_annotations,
                29110: __read_spiketrain_timestamped,
                29112: __read_segment_list,
                29113: __skip_information,
                29114: __read_segment_list_var,
                29115: __read_spike_var,
                29116: __read_unit,
                29117: __read_segment_list_v8,
                29119: __read_unit_list_timestamped,
                29120: __read_segment_list_v9,
                29121: __read_spiketrain_indexed
                }


def convert_brainwaresrc_timestamp(timestamp,
                                   start_date=datetime(1899, 12, 30)):
    """
    convert_brainwaresrc_timestamp(timestamp, start_date) - convert a timestamp
    in brainware src file units to a python datetime object.

    start_date defaults to 1899.12.30 (ISO format), which is the start date
    used by all BrainWare SRC data Blocks so far.  If manually specified
    it should be a datetime object or any other object that can be added
    to a timedelta object.
    """
    # datetime + timedelta = datetime again.
    return start_date + timedelta(days=timestamp)


if __name__ == '__main__':
    # run this when calling the file directly as a benchmark
    from neo.test.iotest.test_brainwaresrcio import FILES_TO_TEST
    from neo.test.iotest.common_io_test import url_for_tests
    from neo.test.iotest.tools import (create_local_temp_dir,
                                       download_test_file,
                                       get_test_file_full_path,
                                       make_all_directories)
    shortname = BrainwareSrcIO.__name__.lower().strip('io')
    local_test_dir = create_local_temp_dir(shortname)
    url = url_for_tests+shortname
    FILES_TO_TEST.remove('long_170s_1rep_1clust_ch2.src')
    make_all_directories(FILES_TO_TEST, local_test_dir)
    download_test_file(FILES_TO_TEST, local_test_dir, url)
    for path in get_test_file_full_path(ioclass=BrainwareSrcIO,
                                        filename=FILES_TO_TEST,
                                        directory=local_test_dir):
        ioobj = BrainwareSrcIO(path)
        ioobj.read_all_blocks(lazy=False)
        ioobj.read_all_blocks(lazy=True)
