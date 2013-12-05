from neo.core.baseneo import BaseNeo, merge_annotations

import numpy as np
import quantities as pq


class EventArray(BaseNeo):
    """
    Array of events. Introduced for performance reasons.
    An :class:`EventArray` is prefered to a list of :class:`Event` objects.

    *Usage*:
        TODO

    *Required attributes/properties*:
        :times: (quantity array 1D)
        :labels: (numpy.array 1D dtype='S')

    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:

    """
    def __init__(self, times=None, labels=None, name=None, description=None,
                 file_origin=None, **annotations):
        """Initialize a new EventArray."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        if times is None:
            times = np.array([]) * pq.s
        if labels is None:
            labels = np.array([], dtype='S')

        self.times = times
        self.labels = labels

        self.segment = None

    def __repr__(self):
        return "<EventArray: %s>" % ", ".join('%s@%s' % item for item in
                                              zip(self.labels, self.times))

    def merge(self, other):
        othertimes = other.times.rescale(self.times.units)
        times = np.hstack([self.times, othertimes]) * self.times.units
        labels = np.hstack([self.labels, other.labels])
        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge(%s, %s)" % (attr_self, attr_other)

        merged_annotations = merge_annotations(self.annotations,
                                               other.annotations)
        kwargs.update(merged_annotations)
        return EventArray(times=times, labels=labels, **kwargs)
