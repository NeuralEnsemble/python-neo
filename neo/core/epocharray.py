from neo.core.baseneo import BaseNeo, merge_annotations

import numpy as np
import quantities as pq

import sys
PY_VER = sys.version_info[0]


class EpochArray(BaseNeo):
    """
    Array of epochs. Introduced for performance reason.
    An :class:`EpochArray` is prefered to a list of :class:`Epoch` objects.

    *Usage*:
        TODO

    *Required attributes/properties*:
        :times: (quantity array 1D)
        :durations: (quantity array 1D)
        :labels: (numpy.array 1D dtype='S') )

    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:
    """
    def __init__(self, times=None, durations=None, labels=None,
                 name=None, description=None, file_origin=None, **annotations):
        """Initialize a new EpochArray."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

        if times is None:
            times = np.array([]) * pq.s
        if durations is None:
            durations = np.array([]) * pq.s
        if labels is None:
            labels = np.array([], dtype='S')

        self.times = times
        self.durations = durations
        self.labels = labels

        self.segment = None

    def __repr__(self):
        # need to convert labels to unicode for python 3 or repr is messed up
        if PY_VER == 3:
            labels = self.labels.astype('U')
        else:
            labels = self.labels

        objs = ['%s@%s for %s' % (label, time, dur) for
                label, time, dur in zip(labels, self.times, self.durations)]
        return '<EventArray: %s>' % ', '.join(objs)

    def merge(self, other):
        othertimes = other.times.rescale(self.times.units)
        otherdurations = other.durations.rescale(self.durations.units)
        times = np.hstack([self.times, othertimes]) * self.times.units
        durations = np.hstack([self.durations,
                               otherdurations]) * self.durations.units
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
        return EpochArray(times=times, durations=durations, labels=labels,
                          **kwargs)
