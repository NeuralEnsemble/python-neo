# -*- coding: utf-8 -*-
'''
This module defines :class:`Block`, the main container gathering all the data,
whether discrete or continous, for a given recording session. base class
used by all :module:`neo.core` classes.

:class:`Block` derives from :class:`BaseNeo`, from :module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from neo.core.baseneo import BaseNeo


class Block(BaseNeo):
    '''
    Main container for data.

    Main container gathering all the data, whether discrete or continous, for a
    given recording session.

    A block is not necessarily temporally homogeneous, in contrast to Segment.

    *Usage*::

        >>> from neo.core import (Block, Segment, RecordingChannelGroup,
        ...                       AnalogSignalArray)
        >>> from quantities import nA, kHz
        >>> import numpy as np
        >>>
        >>> # create a Block with 3 Segment and 2 RecordingChannelGroup objects
        ,,, blk = Block()
        >>> for ind in range(3):
        ...     seg = Segment(name='segment %d' % ind, index=ind)
        ...     blk.segments.append(seg)
        ...
        >>> for ind in range(2):
        ...     rcg = RecordingChannelGroup(name='Array probe %d' % ind,
        ...                                 channel_indexes=np.arange(64))
        ...     blk.recordingchannelgroups.append(rcg)
        ...
        >>> # Populate the Block with AnalogSignalArray objects
        ... for seg in blk.segments:
        ...     for rcg in blk.recordingchannelgroups:
        ...         a = AnalogSignalArray(np.random.randn(10000, 64)*nA,
        ...                               sampling_rate=10*kHz)
        ...         rcg.analogsignalarrays.append(a)
        ...         seg.analogsignalarrays.append(a)

    *Required attributes/properties*:
        None

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :file_datetime: (datetime) The creation date and time of the original
            data file.
        :rec_datetime: (datetime) The date and time of the original recording.
        :index: (int) You can use this to define an ordering of your Block.
            It is not used by Neo in any way.

    *Properties available on this object*:
        :list_units: descends through hierarchy and returns a list of
            :class:`Unit` objects existing in the block. This shortcut exists
            because a common analysis case is analyzing all neurons that
            you recorded in a session.
        :list_recordingchannels: descends through hierarchy and returns
            a list of :class:`RecordingChannel` objects existing in the block.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Container of*:
        :class:`Segment`
        :class:`RecordingChannelGroup`

    '''

    def __init__(self, name=None, description=None, file_origin=None,
                 file_datetime=None, rec_datetime=None, index=None,
                 **annotations):
        '''
        Initalize a new :class:`Block` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index

        self.segments = []
        self.recordingchannelgroups = []

    @property
    def list_units(self):
        '''
        Return a list of all :class:`Unit` objects in the :class:`Block`.
        '''
        units = []
        for rcg in self.recordingchannelgroups:
            for unit in rcg.units:
                if unit not in units:
                    units.append(unit)
        return units

    @property
    def list_recordingchannels(self):
        '''
        Return a list of all :class:`RecordingChannel` objects in the
        :class:`Block`.
        '''
        all_rc = []
        for rcg in self.recordingchannelgroups:
            for rc in rcg.recordingchannels:
                if rc not in all_rc:
                    all_rc.append(rc)
        return all_rc

    def merge(self, other):
        '''
        Merge the contents of another block into this one.

        For each :class:`Segment` in the other block, if its name matches that
        of a :class:`Segment` in this block, the two segments will be merged,
        otherwise it will be added as a new segment. The equivalent procedure
        is then applied to each :class:`RecordingChannelGroup`.
        '''
        for container in ("segments", "recordingchannelgroups"):
            lookup = dict((obj.name, obj) for obj in getattr(self, container))
            for obj in getattr(other, container):
                if obj.name in lookup:
                    lookup[obj.name].merge(obj)
                else:
                    lookup[obj.name] = obj
                    getattr(self, container).append(obj)
        # TODO: merge annotations

    _repr_pretty_attrs_keys_ = [
        "name", "description", "annotations",
        "file_origin", "file_datetime", "rec_datetime", "index"]

    def _repr_pretty_(self, pp, cycle):
        '''
        Handle pretty-printing the :class:`Block`.
        '''
        pp.text("{0} with {1} segments and {1} groups".format(
            self.__class__.__name__,
            len(self.segments),
            len(self.recordingchannelgroups),
        ))
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

        if self.segments:
            pp.breakable()
            pp.text("# Segments")
            pp.breakable()
            for (i, seg) in enumerate(self.segments):
                if i > 0:
                    pp.breakable()
                pp.text("{0}: ".format(i))
                with pp.indent(3):
                    pp.pretty(seg)
