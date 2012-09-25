from neo.core.baseneo import BaseNeo


class Block(BaseNeo):
    """
    Main container gathering all the data, whether discrete or continous, for a
    given recording session.

    A block is not necessarily temporally homogeneous, in contrast to Segment.

    *Usage*:

    TODO

    *Required attributes/properties*:
        None

    *Recommended attributes/properties*:
        :name: A label for the dataset
        :description: text description
        :file_origin: filesystem path or URL of the original data file.
        :file_datetime: the creation date and time of the original data file.
        :rec_datetime: the date and time of the original recording
        :index: integer. You can use this to define an ordering of your Block.
            It is not used by Neo in any way.

    *Container of*:
        :py:class:`Segment`
        :py:class:`RecordingChannelGroup`

    *Properties*
        list_units : descends through hierarchy and returns a list of
            :py:class:`Unit` existing in the block. This shortcut exists
            because a common analysis case is analyzing all neurons that
            you recorded in a session.

        list_recordingchannels: descends through hierarchy and returns
            a list of :py:class:`RecordingChannel` existing in the block.


    """
    def __init__(self, name=None, description=None, file_origin=None,
                 file_datetime=None, rec_datetime=None, index=None,
                 **annotations):
        """Initalize a new Block."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index

        self.segments = []
        self.recordingchannelgroups = []

    @property
    def list_units(self):
        """
        Return a list of all :py:class:`Unit` in a block.
        """
        units = []
        for rcg in self.recordingchannelgroups:
            for unit in rcg.units:
                if unit not in units:
                    units.append(unit)
        return units

    @property
    def list_recordingchannels(self):
        """
        Return a list of all :py:class:`RecordingChannel` in a block.
        """
        all_rc = []
        for rcg in self.recordingchannelgroups:
            for rc in rcg.recordingchannels:
                if rc not in all_rc:
                    all_rc.append(rc)
        return all_rc

    def merge(self, other):
        """
        Merge the contents of another block into this one.

        For each :class:`Segment` in the other block, if its name matches that
        of a :class:`Segment` in this block, the two segments will be merged,
        otherwise it will be added as a new segment. The equivalent procedure
        is then applied to each :class:`RecordingChannelGroup`.
        """
        for container in ("segments", "recordingchannelgroups"):
            lookup = dict((obj.name, obj) for obj in getattr(self, container))
            for obj in getattr(other, container):
                if obj.name in lookup:
                    lookup[obj.name].merge(obj)
                else:
                    getattr(self, container).append(obj)

    _repr_pretty_attrs_keys_ = [
        "name", "description", "annotations",
        "file_origin", "file_datetime", "rec_datetime", "index"]

    def _repr_pretty_(self, pp, cycle):
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
