from neo.core.baseneo import BaseNeo, merge_annotations

import numpy as np
import quantities as pq

import sys
PY_VER = sys.version_info[0]


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
        # need to convert labels to unicode for python 3 or repr is messed up
        if PY_VER == 3:
            labels = self.labels.astype('U')
        else:
            labels = self.labels
        objs = ['%s@%s' % (label, time) for label, time in zip(labels,
                                                               self.times)]
        return '<EventArray: %s>' % ', '.join(objs)

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
