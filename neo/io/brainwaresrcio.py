# -*- coding: utf-8 -*-
'''
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
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

# import needed core python modules
from datetime import datetime, timedelta
from os import path
from warnings import warn

# numpy and quantities are already required by neo
import numpy as np
import quantities as pq

# needed core neo modules
from neo.core import (Block, Event, RecordingChannel,
                      RecordingChannelGroup, Segment, SpikeTrain, Unit)

# need to subclass BaseIO
from neo.io.baseio import BaseIO

# some tools to finalize the hierachy
from neo.io.tools import create_many_to_one_relationship


class BrainwareSrcIO(BaseIO):
    '''
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

    The isopen property tells whether the file is currently open and
    reading or closed.

    Note 1:
        The first Unit in each RecordingChannelGroup is always
        UnassignedSpikes, which has a SpikeTrain for each Segment containing
        all the spikes not assigned to any Unit in that Segment.

    Note 2:
        The first Segment in each Block is always Comments, which stores all
        comments as Event objects.  The Event times are the timestamps
        of the comments as the number of days since dec 30th 1899, while the
        timestamp attribute has the same value in python datetime format

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
    '''

    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects = [Block, RecordingChannel, RecordingChannelGroup,
                         Segment, SpikeTrain, Event, Unit]

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
    read_params = {Block: [],
                   Event: [('sender',
                            {'value': '', 'type': str,
                             'label': 'The ones who sent the comments', }
                            ),
                           ('timestamp',
                            {'value': datetime(1, 1, 1), 'type': datetime,
                             'label': 'The time of the comment', }
                            )
                           ],
                   RecordingChannel: [],
                   RecordingChannelGroup: [],
                   Segment: [('feature_type',
                              {'value': -1, 'type': int}),
                             ('go_by_closest_unit_center',
                              {'value': False, 'type': bool}),
                             ('include_unit_bounds',
                              {'value': False, 'type': bool})
                             ],
                   SpikeTrain: [('dama_index',
                                 {'value': -1, 'type': int,
                                  'label': 'index of analogsignalarray in '
                                           'corresponding .dam file, if any'}),
                                ('respwin',
                                 {'value': np.asarray([], dtype=np.int32),
                                  'type': np.ndarray,
                                  'label': 'response and spon period '
                                           'boundaries'}),
                                ('trig2',
                                 {'value': pq.Quantity([], dtype=np.uint8,
                                                       units=pq.ms),
                                  'type': pq.quantity.Quantity,
                                  'label': 'point of return to noise'}),
                                ('side',
                                 {'value': '', 'type': str,
                                  'label': 'side of the brain'}),
                                ('timestamp',
                                 {'value': datetime(1, 1, 1),
                                  'type': datetime,
                                  'label': 'Start time of the SpikeTrain'})
                                ],
                   Unit: [('boundaries',
                           {'value': [], 'type': list,
                            'label': 'unit boundaries'}),
                          ('elliptic',
                           {'value': [], 'type': list,
                            'label': 'elliptic feature'}),
                          ('timestamp',
                           {'value': [], 'type': list,
                            'label': 'Start time of each unit list'}),
                          ('max_valid',
                           {'value': [], 'type': list})
                          ]
                   }

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

        # this stores the filename of the current object, exactly as it is
        # provided when the instance is initialized.
        self.__filename = filename

        # this store the filename without the path
        self.__file_origin = filename

        # This stores the file object for the current file
        self.__fsrc = None

        # This stores the current Block
        self.__blk = None

        # This stores the current RecordingChannelGroup for easy access
        # It is equivalent to self.__blk.recordingchannelgroups[0]
        self.__rcg = None

        # This stores the current Segment for easy access
        # It is equivalent to self.__blk.segments[-1]
        self.__seg = None

        # this stores a dictionary of the Block's Units by name,
        # making it easier and faster to retrieve Units by name later
        # UnassignedSpikes and Units accessed by index are not stored here
        self.__unitdict = {}

        # this stores the current Unit
        self.__unit = None

        # if the file has a list with negative length, the rest of the file's
        # list lengths are unreliable, so we need to store this value for the
        # whole file
        self.__damaged = False

        # this stores whether the current file is lazy loaded
        self.__lazy = False

        # this stores whether the current file is cascading
        # this is false by default so if we use read_block on its own it works
        self.__cascade = False

    @property
    def isopen(self):
        '''
        This property tells whether the SRC file associated with the IO object
        is open.
        '''
        return self.__fsrc is not None

    def opensrc(self):
        '''
        Open the file if it isn't already open.
        '''
        # if the file isn't already open, open it and clear the Blocks
        if not self.__fsrc or self.__fsrc.closed:
            self.__fsrc = open(self.__filename, 'rb')

            # figure out the filename of the current file
            self.__file_origin = path.basename(self.__filename)

    def close(self):
        '''
        Close the currently-open file and reset the current reading point.
        '''
        if self.isopen and not self.__fsrc.closed:
            self.__fsrc.close()

        # we also need to reset all per-file attributes
        self.__damaged = False
        self.__fsrc = None
        self.__seg = None
        self.__cascade = False
        self.__file_origin = None
        self.__lazy = False

    def read(self, lazy=False, cascade=True, **kargs):
        '''
        Reads the first Block from the Spike ReCording file "filename"
        generated with BrainWare.

        If you wish to read more than one Block, please use read_all_blocks.
        '''
        return self.read_block(lazy=lazy, cascade=cascade, **kargs)

    def read_block(self, lazy=False, cascade=True, **kargs):
        '''
        Reads the first Block from the Spike ReCording file "filename"
        generated with BrainWare.

        If you wish to read more than one Block, please use read_all_blocks.
        '''

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')

        blockobj = self.read_next_block(cascade=cascade, lazy=lazy,
                                        warnlast=False)
        self.close()
        return blockobj

    def read_next_block(self, cascade=True, lazy=False, warnlast=True,
                        **kargs):
        '''
        Reads a single Block from the Spike ReCording file "filename"
        generated with BrainWare.

        Each call of read will return the next Block until all Blocks are
        loaded.  After the last Block, the file will be automatically closed
        and the progress reset.  Call the close method manually to reset
        back to the first Block.

        If "warnlast" is set to True (default), print a warning after
        reading the last Block.
        '''

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')

        self.__lazy = lazy
        self.opensrc()

        # create the Block and the contents all Blocks of from IO share
        self.__blk = Block(file_origin=self.__file_origin)
        if not cascade:
            return self.__blk
        self.__rcg = RecordingChannelGroup(file_origin=self.__file_origin)
        self.__seg = Segment(name='Comments', file_origin=self.__file_origin)
        self.__unit = Unit(name='UnassignedSpikes',
                           file_origin=self.__file_origin,
                           elliptic=[], boundaries=[],
                           timestamp=[], max_valid=[])
        self.__blk.recordingchannelgroups.append(self.__rcg)
        self.__rcg.units.append(self.__unit)
        self.__blk.segments.append(self.__seg)

        # this actually reads the contents of the Block
        result = []
        while hasattr(result, '__iter__'):
            try:
                result = self._read_by_id()
            except:
                self.close()
                raise

        # set the recorging channel group names and indices
        chans = self.__rcg.recordingchannels
        chan_inds = np.arange(len(chans), dtype='int')
        chan_names = np.array(['Chan'+str(i) for i in chan_inds],
                              dtype='string_')
        self.__rcg.channel_indexes = chan_inds
        self.__rcg.channel_names = chan_names

        # since we read at a Block level we always do this
        create_many_to_one_relationship(self.__blk)

        # put the Block in a local object so it can be gargabe collected
        blockobj = self.__blk

        # reset the per-Block attributes
        self.__blk = None
        self.__rcg = None
        self.__unitdict = {}

        # result is None iff the end of the file is reached, so we can
        # close the file
        # this notification is not helpful if using the read method with
        # cascade==True, since the user will know it is done when the method
        # returns a value
        if result is None:
            if warnlast:
                print('Last Block read.  Closing file')
            self.close()

        return blockobj

    def read_all_blocks(self, cascade=True, lazy=False, **kargs):
        '''
        Reads all Blocks from the Spike ReCording file "filename"
        generated with BrainWare.

        The progress in the file is reset and the file closed then opened again
        prior to reading.

        The file is automatically closed after reading completes.
        '''

        # there are no keyargs implemented to so far.  If someone tries to pass
        # them they are expecting them to do something or making a mistake,
        # neither of which should pass silently
        if kargs:
            raise NotImplementedError('This method does not have any '
                                      'argument implemented yet')

        self.__lazy = lazy
        self.__cascade = True

        self.close()
        self.opensrc()

        # Read each Block.
        # After the last Block self.isopen is set to False, so this make a
        # good way to determine when to stop
        blocks = []
        while self.isopen:
            try:
                blocks.append(self.read_next_block(cascade=cascade, lazy=lazy,
                                                   warnlast=False))
            except:
                self.close()
                raise

        return blocks

    @staticmethod
    def convert_timestamp(timestamp, start_date=datetime(1899, 12, 30)):
        '''
        convert_timestamp(timestamp, start_date) - convert a timestamp in
        brainware src file units to a python datetime object.

        start_date defaults to 1899.12.30 (ISO format), which is the start date
        used by all BrainWare SRC data Blocks so far.  If manually specified
        it should be a datetime object or any other object that can be added
        to a timedelta object.
        '''
        return convert_brainwaresrc_timestamp(timestamp, start_date=start_date)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #  All methods from here on are private.  They are not intended to be used
    #  on their own, although methods that could theoretically be called on
    #  their own are marked as such.  All private methods could be renamed,
    #  combined, or split at any time.  All private methods prefixed by
    #  "__read" or "__skip" will alter the current place in the file.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def _read_by_id(self):
        '''
        Reader for generic data

        BrainWare SRC files are broken up into data sequences that are
        identified by an ID code.  This method determines the ID code and calls
        the method to read the data sequence with that ID code.  See the
        __ID_DICT attribute for a dictionary of code/method pairs.

        IMPORTANT!!!
        This is the only private method that can be called directly.
        The rest of the private methods can only safely be called by this
        method or by other private methods, since they depend on the
        current position in the file.
        '''

        try:
            # uint16 -- the ID code of the next sequence
            seqid = np.fromfile(self.__fsrc, dtype=np.uint16, count=1)[0]
        except IndexError:
            # return a None if at EOF.  Other methods use None to recognize
            # an EOF
            return None

        # using the seqid, get the reader function from the reader dict
        readfunc = self.__ID_DICT.get(seqid)
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
                warn('unknown ID: %s' % seqid)
                return []

        try:
            # run the function to get the data
            return readfunc(self)
        except (EOFError, UnicodeDecodeError) as err:
            # return a warning if the EOF is reached in the middle of a method
            warn(str(err))
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
        '''
        _assign_sequence(data_obj) - Try to guess where an unknown sequence
        should go based on its class.  Warning are issued if this method is
        used since manual reorganization may be needed.
        '''
        if isinstance(data_obj, Unit):
            warn('Unknown Unit found, adding to Units list')
            self.__rcg.units.append(data_obj)
            if data_obj.name:
                self.__unitdict[data_obj.name] = data_obj
        elif isinstance(data_obj, Segment):
            warn('Unknown Segment found, adding to Segments list')
            self.__blk.segments.append(data_obj)
        elif isinstance(data_obj, Event):
            warn('Unknown Event found, adding to comment Events list')
            self.__blk.segments[0].events.append(data_obj)
        elif isinstance(data_obj, SpikeTrain):
            warn('Unknown SpikeTrain found, ' +
                 'adding to the UnassignedSpikes Unit')
            self.__rcg.units[0].spiketrains.append(data_obj)
        elif hasattr(data_obj, '__iter__') and not isinstance(data_obj, str):
            for sub_obj in data_obj:
                self._assign_sequence(sub_obj)
        else:
            warn('Unrecognized sequence of type %s found, skipping',
                 type(data_obj))

    _default_datetime = datetime(1, 1, 1)
    _default_spiketrain = SpikeTrain(times=pq.Quantity([], units=pq.ms,
                                                       dtype=np.float32),
                                     t_start=pq.Quantity(0, units=pq.ms,
                                                         dtype=np.float32),
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

    def _combine_spiketrains(self, spiketrains):
        '''
        _combine_spiketrains(spiketrains) - combine a list of SpikeTrains
        with single spikes into one long SpikeTrain
        '''

        if not spiketrains:
            train = self._default_spiketrain.copy()
            train.file_origin = self.__file_origin
            if self.__lazy:
                train.lazy_shape = (0,)
            return train

        if hasattr(spiketrains[0], 'waveforms') and len(spiketrains) == 1:
            train = spiketrains[0]
            if self.__lazy and not hasattr(train, 'lazy_shape'):
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
                train = self._default_spiketrain.copy()
                train.file_origin = self.__file_origin
                if self.__lazy:
                    train.lazy_shape = (0,)
                return train

            # get the times of the spiketrains and combine them
            waveforms = [itrain.waveforms for itrain in spiketrains]
            rawtrains = np.array(np.hstack(spiketrains))
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

            waveforms = np.vstack(waveforms)

            # extract the trig2 annotation
            trig2 = np.array(np.hstack([itrain.annotations['trig2'] for itrain
                             in spiketrains]))
            trig2 = pq.Quantity(trig2, units=pq.ms)
        elif hasattr(spiketrains[0], 'units'):
            return self._combine_spiketrains([spiketrains])
        else:
            times, waveforms, trig2 = zip(*spiketrains)
            times = np.hstack(times)

            # get the times of the SpikeTrains and combine them
            times = pq.Quantity(times, units=pq.ms, copy=False)

            # get the waveforms of the SpikeTrains and combine them
            # these should be a 3D array with the first axis being the spike,
            # the second axis being the recording channel (there is only one),
            # and the third axis being the actual waveform
            waveforms = np.vstack(waveforms)[np.newaxis].swapaxes(0, 1)

            # extract the trig2 annotation
            trig2 = pq.Quantity(np.hstack(trig2), units=pq.ms, copy=False)

        if not times.size:
            train = self._default_spiketrain.copy()
            train.file_origin = self.__file_origin
            if self.__lazy:
                train.lazy_shape = (0,)
            return train

        # get the maximum time
        t_stop = times.max() * 2
        t_start = pq.Quantity(0, units=pq.ms, dtype=times.dtype)

        if self.__lazy:
            timesshape = times.shape
            times = pq.Quantity([], units=pq.ms, copy=False)
            waveforms = pq.Quantity([[[]]], units=pq.mV)
        else:
            waveforms = pq.Quantity(np.asarray(waveforms), units=pq.mV)

        train = SpikeTrain(times=times, copy=False,
                           t_start=t_start, t_stop=t_stop,
                           file_origin=self.__file_origin,
                           waveforms=waveforms,
                           timestamp=self._default_datetime,
                           respwin=np.array([], dtype=np.int32),
                           dama_index=-1, trig2=trig2, side='')
        if self.__lazy:
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

    def __read_annotations(self):
        '''
        Read the stimulus grid properties.

        -------------------------------------------------------------------
        Returns a dictionary containing the parameter names as keys and the
        parameter values as values.

        The returned object must be added to the Block.

        ID: 29109
        '''

        # int16 -- number of stimulus parameters
        numelements = np.fromfile(self.__fsrc, dtype=np.int16, count=1)[0]
        if not numelements:
            return {}

        # [data sequence] * numelements -- parameter names
        names = []
        for i in range(numelements):
            # {skip} = byte (char) -- skip one byte
            self.__fsrc.seek(1, 1)

            # uint8 -- length of next string
            numchars = np.fromfile(self.__fsrc, dtype=np.uint8, count=1)[0]

            # if there is no name, make one up
            if not numchars:
                name = 'param%s' % i
            else:
                # char * numchars -- parameter name string
                name = str(np.fromfile(self.__fsrc, dtype='S%s' % numchars,
                                       count=1)[0].decode('UTF-8'))

            # if the name is already in there, add a unique number to it
            # so it isn't overwritten
            if name in names:
                name = name + str(i)

            names.append(name)

        # float32 * numelements -- an array of parameter values
        values = np.fromfile(self.__fsrc, dtype=np.float32,
                             count=numelements)

        # combine the names and values into a dict
        # the dict will be added to the annotations
        annotations = dict(zip(names, values))

        return annotations

    def __read_annotations_old(self):
        '''
        Read the stimulus grid properties.

        Returns a dictionary containing the parameter names as keys and the
        parameter values as values.

        ------------------------------------------------
        The returned objects must be added to the Block.

        This reads an old version of the format that does not store paramater
        names, so placeholder names are created instead.

        ID: 29099
        '''

        # int16 * 14 -- an array of parameter values
        values = np.fromfile(self.__fsrc, dtype=np.int16, count=14)

        # create dummy names and combine them with the values in a dict
        # the dict will be added to the annotations
        params = ['param%s' % i for i in range(len(values))]
        annotations = dict(zip(params, values))

        return annotations

    def __read_comment(self):
        '''
        Read a single comment.

        The comment is stored as an Event in Segment 0, which is
        specifically for comments.

        ----------------------
        Returns an empty list.

        The returned object is already added to the Block.

        No ID number: always called from another method
        '''

        # float64 -- timestamp (number of days since dec 30th 1899)
        time = np.fromfile(self.__fsrc, dtype=np.double, count=1)[0]

        # convert the timestamp to a python datetime object
        # then convert that to a quantity containing a unix timestamp
        timestamp = self.convert_timestamp(time)
        time = pq.Quantity(time, units=pq.d)

        # int16 -- length of next string
        numchars1 = np.fromfile(self.__fsrc, dtype=np.int16, count=1)[0]

        # char * numchars -- the one who sent the comment
        sender = str(np.fromfile(self.__fsrc, dtype='S%s' % numchars1,
                                 count=1)[0].decode('UTF-8'))

        # int16 -- length of next string
        numchars2 = np.fromfile(self.__fsrc, dtype=np.int16, count=1)[0]

        # char * numchars -- comment text
        text = str(np.fromfile(self.__fsrc, dtype='S%s' % numchars2,
                               count=1)[0].decode('UTF-8'))

        comment = Event(time=time, label=text,
                        sender=sender, name='Comment',
                        description='container for a comment',
                        file_origin=self.__file_origin,
                        timestamp=timestamp)

        self.__blk.segments[0].events.append(comment)

        return []

    def __read_list(self):
        '''
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
        '''

        # int16 -- number of sequences to read
        numelements = np.fromfile(self.__fsrc, dtype=np.int16, count=1)[0]

        # {skip} = bytes * 4 (int16 * 2) -- skip four bytes
        self.__fsrc.seek(4, 1)

        if numelements == 0:
            return []

        if not self.__damaged and numelements < 0:
            self.__damaged = True
            warn('Negative sequence count, file may be damaged')

        if not self.__damaged:
            # read the sequences into a list
            seq_list = [self._read_by_id() for _ in range(numelements)]
        else:
            # read until we get some indication we should stop
            seq_list = []

            # uint16 -- the ID of the next sequence
            seqidinit = np.fromfile(self.__fsrc, dtype=np.uint16, count=1)[0]

            # {rewind} = byte * 2 (int16) -- move back 2 bytes, i.e. go back to
            # before the beginning of the seqid
            self.__fsrc.seek(-2, 1)
            while 1:
                # uint16 -- the ID of the next sequence
                seqid = np.fromfile(self.__fsrc, dtype=np.uint16, count=1)[0]

                # {rewind} = byte * 2 (int16) -- move back 2 bytes, i.e. go
                # back to before the beginning of the seqid
                self.__fsrc.seek(-2, 1)

                # if we come across a new sequence, we are at the end of the
                # list so we should stop
                if seqidinit != seqid:
                    break

                # otherwise read the next sequence
                seq_list.append(self._read_by_id())

        return seq_list

    def __read_segment(self):
        '''
        Read an individual Segment.

        A Segment contains a dictionary of parameters, the length of the
        recording, a list of Units with their Spikes, and a list of Spikes
        not assigned to any Unit.  The unassigned spikes are always stored in
        Unit 0, which is exclusively for storing these spikes.

        -------------------------------------------------
        Returns the Segment object created by the method.

        The returned object is already added to the Block.

        ID: 29106
        '''

        # (data_obj) -- the stimulus parameters for this segment
        annotations = self._read_by_id()
        annotations['feature_type'] = -1
        annotations['go_by_closest_unit_center'] = False
        annotations['include_unit_bounds'] = False

        # (data_obj) -- SpikeTrain list of unassigned spikes
        # these go in the first Unit since it is for unassigned spikes
        unassigned_spikes = self._read_by_id()
        self.__rcg.units[0].spiketrains.extend(unassigned_spikes)

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
                train = self._default_spiketrain.copy()
                train.file_origin = self.__file_origin
                if self.__lazy:
                    train.lazy_shape = (0,)
                trains = [[train]]
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
        spiketrainlen = pq.Quantity(np.fromfile(self.__fsrc, dtype=np.int32,
                                    count=1)[0], units=pq.ms, copy=False)

        segments = []
        for train in trains:
            # create the Segment and add everything to it
            segment = Segment(file_origin=self.__file_origin,
                              **annotations)
            segment.spiketrains = train
            self.__blk.segments.append(segment)
            segments.append(segment)

            for itrain in train:
                # use the SpikeTrain length to figure out the stop time
                # t_start is always 0 so we can ignore it
                itrain.t_stop = spiketrainlen

        return segments

    def __read_segment_list(self):
        '''
        Read a list of Segments with comments.

        Since comments can occur at any point, whether a recording is happening
        or not, it is impossible to reliably assign them to a specific Segment.
        For this reason they are always assigned to Segment 0, which is
        exclusively used to store comments.

        --------------------------------------------------------
        Returns a list of the Segments created with this method.

        The returned objects are already added to the Block.

        ID: 29112
        '''

        # uint8 --  number of electrode channels in the Segment
        numchannels = np.fromfile(self.__fsrc, dtype=np.uint8, count=1)[0]

        # [list of sequences] -- individual Segments
        segments = self.__read_list()
        while not hasattr(segments[0], 'spiketrains'):
            segments = sum(segments, [])

        # char -- "side of brain" info
        side = str(np.fromfile(self.__fsrc, dtype='S1',
                               count=1)[0].decode('UTF-8'))

        # int16 -- number of comments
        numelements = np.fromfile(self.__fsrc, dtype=np.int16, count=1)[0]

        # comment_obj * numelements -- comments about the Segments
        # we don't know which Segment specifically, though
        for _ in range(numelements):
            self.__read_comment()

        # create an empty RecordingChannel for each of the numchannels

        for i in range(numchannels):
            chan = RecordingChannel(file_origin=self.__file_origin,
                                    index=int(i), name='Chan'+str(int(i)))
            self.__rcg.recordingchannels.append(chan)

        # store what side of the head we are dealing with
        for segment in segments:
            for spiketrain in segment.spiketrains:
                spiketrain.annotations['side'] = side

        return segments

    def __read_segment_list_v8(self):
        '''
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
        '''

        # segment_collection_var -- this is based off a segment_collection_var
        segments = self.__read_segment_list_var()

        # uint16 -- the ID of the next sequence
        seqid = np.fromfile(self.__fsrc, dtype=np.uint16, count=1)[0]

        # {rewind} = byte * 2 (int16) -- move back 2 bytes, i.e. go back to
        # before the beginning of the seqid
        self.__fsrc.seek(-2, 1)

        if seqid in self.__ID_DICT:
            # if it is a valid seqid, read it and try to figure out where
            # to put it
            self._assign_sequence(self._read_by_id())
        else:
            # otherwise it is a Unit list
            self.__read_unit_list()

        # {skip} = byte * 2 (int16) -- skip 2 bytes
        self.__fsrc.seek(2, 1)

        return segments

    def __read_segment_list_v9(self):
        '''
        Read a list of Segments with comments.

        This is version 9 of the data sequence.

        This is the same as __read_segment_list_v8, but contains some
        additional annotations.  These annotations are added to the Segment.

        --------------------------------------------------------
        Returns a list of the Segments created with this method.

        The returned objects are already added to the Block.

        ID: 29120
        '''

        # segment_collection_v8 -- this is based off a segment_collection_v8
        segments = self.__read_segment_list_v8()

        # uint8
        feature_type = np.fromfile(self.__fsrc, dtype=np.uint8,
                                   count=1)[0]

        # uint8
        go_by_closest_unit_center = np.fromfile(self.__fsrc, dtype=np.bool8,
                                                count=1)[0]

        # uint8
        include_unit_bounds = np.fromfile(self.__fsrc, dtype=np.bool8,
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
        '''
        Read a list of Segments with comments.

        This is the same as __read_segment_list, but contains information
        regarding the sampling period.  This information is added to the
        SpikeTrains in the Segments.

        --------------------------------------------------------
        Returns a list of the Segments created with this method.

        The returned objects are already added to the Block.

        ID: 29114
        '''

        # float32 -- DA conversion clock period in microsec
        sampling_period = pq.Quantity(np.fromfile(self.__fsrc,
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
        '''
        Read a spike with a fixed waveform length (40 time bins)

        -------------------------------------------
        Returns the time, waveform and trig2 value.

        The returned objects must be converted to a SpikeTrain then
        added to the Block.

        ID: 29079
        '''

        # float32 -- spike time stamp in ms since start of SpikeTrain
        time = np.fromfile(self.__fsrc, dtype=np.float32, count=1)

        # int8 * 40 -- spike shape -- use numpts for spike_var
        waveform = np.fromfile(self.__fsrc, dtype=np.int8, count=numpts)

        # uint8 -- point of return to noise
        trig2 = np.fromfile(self.__fsrc, dtype=np.uint8, count=1)

        return time, waveform, trig2

    def __read_spike_fixed_old(self):
        '''
        Read a spike with a fixed waveform length (40 time bins)

        This is an old version of the format.  The time is stored as ints
        representing 1/25 ms time steps.  It has no trigger information.

        -------------------------------------------
        Returns the time, waveform and trig2 value.

        The returned objects must be converted to a SpikeTrain then
        added to the Block.

        ID: 29081
        '''

        # int32 -- spike time stamp in ms since start of SpikeTrain
        time = np.fromfile(self.__fsrc, dtype=np.int32, count=1) / 25.

        # int8 * 40 -- spike shape
        # This needs to be a 2D array, one for each channel.  BrainWare
        # only ever has a single channel per file.
        waveform = np.fromfile(self.__fsrc, dtype=np.int8, count=40)

        # create a dummy trig2 value
        trig2 = np.array([-1], dtype=np.uint8)

        return time, waveform, trig2

    def __read_spike_var(self):
        '''
        Read a spike with a variable waveform length

        -------------------------------------------
        Returns the time, waveform and trig2 value.

        The returned objects must be converted to a SpikeTrain then
        added to the Block.

        ID: 29115
        '''

        # uint8 -- number of points in spike shape
        numpts = np.fromfile(self.__fsrc, dtype=np.uint8, count=1)[0]

        # spike_fixed is the same as spike_var if you don't read the numpts
        # byte and set numpts = 40
        return self.__read_spike_fixed(numpts)

    def __read_spiketrain_indexed(self):
        '''
        Read a SpikeTrain

        This is the same as __read_spiketrain_timestamped except it also
        contains the index of the Segment in the dam file.

        The index is stored as an annotation in the SpikeTrain.

        -------------------------------------------------
        Returns a SpikeTrain object with multiple spikes.

        The returned object must be added to the Block.

        ID: 29121
        '''

        #int32 -- index of the analogsignalarray in corresponding .dam file
        dama_index = np.fromfile(self.__fsrc, dtype=np.int32,
                                 count=1)[0]

        # spiketrain_timestamped -- this is based off a spiketrain_timestamped
        spiketrain = self.__read_spiketrain_timestamped()

        # add the property to the dict
        spiketrain.annotations['dama_index'] = dama_index

        return spiketrain

    def __read_spiketrain_timestamped(self):
        '''
        Read a SpikeTrain

        This SpikeTrain contains a time stamp for when it was recorded

        The timestamp is stored as an annotation in the SpikeTrain.

        -------------------------------------------------
        Returns a SpikeTrain object with multiple spikes.

        The returned object must be added to the Block.

        ID: 29110
        '''

        # float64 -- timeStamp (number of days since dec 30th 1899)
        timestamp = np.fromfile(self.__fsrc, dtype=np.double, count=1)[0]

        # convert to datetime object
        timestamp = self.convert_timestamp(timestamp)

        # seq_list -- spike list
        # combine the spikes into a single SpikeTrain
        spiketrain = self._combine_spiketrains(self.__read_list())

        # add the timestamp
        spiketrain.annotations['timestamp'] = timestamp

        return spiketrain

    def __read_unit(self):
        '''
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
        '''

        # same as unsorted Unit
        unit, trains = self.__read_unit_unsorted()

        # float32 * 18 -- Unit boundaries (IEEE 32-bit floats)
        unit.annotations['boundaries'] = [np.fromfile(self.__fsrc,
                                                      dtype=np.float32,
                                                      count=18)]

        # uint8 * 9 -- boolean values indicating elliptic feature boundary
        # dimensions
        unit.annotations['elliptic'] = [np.fromfile(self.__fsrc,
                                                    dtype=np.uint8,
                                                    count=9)]

        return unit, trains

    def __read_unit_list(self):
        '''
        A list of a list of Units

        -----------------------------------------------
        Returns a list of Units modified in the method.

        The returned objects are already added to the Block.

        No ID number: only called by other methods
        '''

        # this is used to figure out which Units to return
        maxunit = 1

        # int16 -- number of time slices
        numelements = np.fromfile(self.__fsrc, dtype=np.int16, count=1)[0]

        # {sequence} * numelements1 -- the number of lists of Units to read
        self.__rcg.annotations['max_valid'] = []
        for _ in range(numelements):

            # {skip} = byte * 2 (int16) -- skip 2 bytes
            self.__fsrc.seek(2, 1)

            # double
            max_valid = np.fromfile(self.__fsrc, dtype=np.double, count=1)[0]

            # int16 - the number of Units to read
            numunits = np.fromfile(self.__fsrc, dtype=np.int16, count=1)[0]

            # update tha maximum Unit so far
            maxunit = max(maxunit, numunits + 1)

            # if there aren't enough Units, create them
            # remember we need to skip the UnassignedSpikes Unit
            if numunits > len(self.__rcg.units) + 1:
                for ind1 in range(len(self.__rcg.units), numunits + 1):
                    unit = Unit(name='unit%s' % ind1,
                                file_origin=self.__file_origin,
                                elliptic=[], boundaries=[],
                                timestamp=[], max_valid=[])
                    self.__rcg.units.append(unit)

            # {Block} * numelements -- Units
            for ind1 in range(numunits):
                # get the Unit with the given index
                # remember we need to skip the UnassignedSpikes Unit
                unit = self.__rcg.units[ind1 + 1]

                # {skip} = byte * 2 (int16) -- skip 2 bytes
                self.__fsrc.seek(2, 1)

                # int16 -- a multiplier for the elliptic and boundaries
                #          properties
                numelements3 = np.fromfile(self.__fsrc, dtype=np.int16,
                                           count=1)[0]

                # uint8 * 10 * numelements3 -- boolean values indicating
                # elliptic feature boundary dimensions
                elliptic = np.fromfile(self.__fsrc, dtype=np.uint8,
                                       count=10 * numelements3)

                # float32 * 20 * numelements3 -- feature boundaries
                boundaries = np.fromfile(self.__fsrc, dtype=np.float32,
                                         count=20 * numelements3)

                unit.annotations['elliptic'].append(elliptic)
                unit.annotations['boundaries'].append(boundaries)
                unit.annotations['max_valid'].append(max_valid)

        return self.__rcg.units[1:maxunit]

    def __read_unit_list_timestamped(self):
        '''
        A list of a list of Units.

        This is the same as __read_unit_list, except that it also has a
        timestamp.  This is added ad an annotation to all Units.

        -----------------------------------------------
        Returns a list of Units modified in the method.

        The returned objects are already added to the Block.

        ID: 29119
        '''

        # double -- time zero (number of days since dec 30th 1899)
        timestamp = np.fromfile(self.__fsrc, dtype=np.double, count=1)[0]

        # convert to to days since UNIX epoc time:
        timestamp = self.convert_timestamp(timestamp)

        # sorter -- this is based off a sorter
        units = self.__read_unit_list()

        for unit in units:
            unit.annotations['timestamp'].append(timestamp)

        return units

    def __read_unit_old(self):
        '''
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
        '''

        # same as Unit
        unit, trains = self.__read_unit_unsorted()

        # bytes * 108 (float48 * 18) -- Unit boundaries (48-bit floating
        # point numbers are not supported so we skip them)
        self.__fsrc.seek(108, 1)

        # uint8 * 9 -- boolean values indicating elliptic feature boundary
        # dimensions
        unit.annotations['elliptic'] = np.fromfile(self.__fsrc, dtype=np.uint8,
                                                   count=9).tolist()

        return unit, trains

    def __read_unit_unsorted(self):
        '''
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
        '''

        # {skip} = bytes * 2 (uint16) -- skip two bytes
        self.__fsrc.seek(2, 1)

        # uint16 -- number of characters in next string
        numchars = np.fromfile(self.__fsrc, dtype=np.uint16, count=1)[0]

        # char * numchars -- ID string of Unit
        name = str(np.fromfile(self.__fsrc, dtype='S%s' % numchars,
                               count=1)[0].decode('UTF-8'))

        # int32 -- SpikeTrain length in ms
        t_stop = pq.Quantity(np.fromfile(self.__fsrc, dtype=np.int32,
                                         count=1)[0].astype('float32'),
                             units=pq.ms, copy=False)

        # int32 * 4 -- response and spon period boundaries
        respwin = np.fromfile(self.__fsrc, dtype=np.int32, count=4)

        # (data_obj) -- list of SpikeTrains
        spikeslists = self._read_by_id()

        # use the Unit if it already exists, otherwise create it
        if name in self.__unitdict:
            unit = self.__unitdict[name]
        else:
            unit = Unit(name=name, file_origin=self.__file_origin,
                        elliptic=[], boundaries=[], timestamp=[], max_valid=[])
            self.__rcg.units.append(unit)
            self.__unitdict[name] = unit

        # convert the individual spikes to SpikeTrains and add them to the Unit
        trains = [self._combine_spiketrains(spikes) for spikes in spikeslists]
        unit.spiketrains.extend(trains)
        for train in trains:
            train.t_stop = t_stop
            train.annotations['respwin'] = respwin

        return unit, trains

    def __skip_information(self):
        '''
        Read an information sequence.

        This is data sequence is skipped both here and in the Matlab reference
        implementation.

        ----------------------
        Returns an empty list

        Nothing is created so nothing is added to the Block.

        ID: 29113
        '''

        # {skip} char * 34 -- display information
        self.__fsrc.seek(34, 1)

        return []

    def __skip_information_old(self):
        '''
        Read an information sequence

        This is data sequence is skipped both here and in the Matlab reference
        implementation

        This is an old version of the format

        ----------------------
        Returns an empty list.

        Nothing is created so nothing is added to the Block.

        ID: 29100
        '''

        # {skip} char * 4 -- display information
        self.__fsrc.seek(4, 1)

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
    __ID_DICT = {29079: __read_spike_fixed,
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
    '''
    convert_brainwaresrc_timestamp(timestamp, start_date) - convert a
    timestamp in brainware units to a python datetime object.

    start_date defaults to 1899.12.30 (ISO format), which is the start date
    used by all BrainWare SRC data blocks so far.  If manually specified
    it should be a datetime object or any other object that can be added
    to a timedelta object.
    '''

    # datetime + timedelta = datetime again.
    try:
        timestamp = start_date + timedelta(days=timestamp)
    except OverflowError as err:
        timestamp = start_date
        warn(str(err))

    return timestamp
