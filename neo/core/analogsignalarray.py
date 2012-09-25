from __future__ import division
import numpy as np
import quantities as pq
from .analogsignal import BaseAnalogSignal, AnalogSignal, _get_sampling_rate
from .baseneo import BaseNeo


class AnalogSignalArray(BaseAnalogSignal):
    """
    A representation of several continuous, analog signals that
    have the same duration, sampling rate and start time.
    Basically, it is a 2D array like AnalogSignal: dim 0 is time, dim 1 is
    channel index

    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    :class:`numpy.ndarray`.

    *Usage*:
        TODO

    *Required attributes/properties*:
        :t_start:         time when signal begins
        :sampling_rate: *or* :sampling_period: Quantity, number of samples per
                                               unit time or interval between
                                               two samples. If both are
                                               specified, they are checked for
                                               consistency.

    *Properties*:
        :sampling_period: interval between two samples (1/sampling_rate)
        :duration:        signal duration (size * sampling_period)
        :t_stop:          time when signal ends (t_start + duration)

    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:

    """
    def __new__(cls, signal, units=None, dtype=None, copy=True,
                t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin=None, description=None,
                channel_indexes=None, **annotations):
        """
        Create a new :class:`AnalogSignalArray` instance from a list or numpy
        array of numerical values, or from a Quantity array.
        """
        if (isinstance(signal, pq.Quantity)
                and units is not None
                and units != signal.units):
            signal = signal.rescale(units)
        if not units and hasattr(signal, "units"):
            units = signal.units
        obj = pq.Quantity.__new__(cls, signal, units=units, dtype=dtype,
                                  copy=copy)
        obj.t_start = t_start
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)

        obj.channel_indexes = channel_indexes
        obj.segment = None
        obj.recordingchannelgroup = None

        return obj

    def __init__(self, signal, units=None, dtype=None, copy=True,
                 t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                 name=None, file_origin=None, description=None,
                 channel_indexes=None, **annotations):
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __getitem__(self, i):
        obj = super(BaseAnalogSignal, self).__getitem__(i)
        if isinstance(i, int):
            return obj
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(k, int):
                if isinstance(j, slice):  # extract an AnalogSignal
                    obj = AnalogSignal(obj, sampling_rate=self.sampling_rate)
                    if j.start:
                        obj.t_start = (self.t_start +
                                       j.start * self.sampling_period)
                # return a Quantity (for some reason quantities does not
                # return a Quantity in this case)
                elif isinstance(j, int):
                    obj = pq.Quantity(obj, units=self.units)
                return obj
            elif isinstance(j, int):  # extract a quantity array
                # should be a better way to do this
                obj = pq.Quantity(np.array(obj), units=obj.units)
                return obj
            else:
                return obj
        elif isinstance(i, slice):
            if i.start:
                obj.t_start = self.t_start + i.start * self.sampling_period
            return obj
        else:
            raise IndexError("index should be an integer, tuple or slice")

    def _copy_data_complement(self, other):
        BaseAnalogSignal._copy_data_complement(self, other)
        for attr in ("channel_indexes"):
            setattr(self, attr, getattr(other, attr, None))

    def time_slice(self, t_start, t_stop):
        """
        Creates a new AnalogSignal corresponding to the time slice of the
        original AnalogSignal between times t_start, t_stop. Note, that for
        numerical stability reasons if t_start, t_stop do not fall exactly on
        the time bins defined by the sampling_period they will be rounded to
        the nearest sampling bins.
        """

        t_start = t_start.rescale(self.sampling_period.units)
        t_stop = t_stop.rescale(self.sampling_period.units)
        i = (t_start - self.t_start) / self.sampling_period
        j = (t_stop - self.t_start) / self.sampling_period
        i = int(np.rint(i.magnitude))
        j = int(np.rint(j.magnitude))

        if (i < 0) or (j > len(self)):
            raise ValueError('t_start, t_stop have to be withing the analog \
                              signal duration')

        # we're going to send the list of indicies so that we get *copy* of the
        # sliced data
        obj = super(BaseAnalogSignal, self).__getitem__(np.arange(i, j, 1))
        obj.t_start = self.t_start + i * self.sampling_period

        return obj

    def merge(self, other):
        assert self.sampling_rate == other.sampling_rate
        assert self.t_start == other.t_start
        other.units = self.units
        stack = np.hstack(map(np.array, (self, other)))
        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge(%s, %s)" % (attr_self, attr_other)
        if self.channel_indexes is None:
            kwargs['channel_indexes'] = other.channel_indexes
        elif other.channel_indexes is None:
            kwargs['channel_indexes'] = self.channel_indexes
        else:
            kwargs['channel_indexes'] = np.append(self.channel_indexes,
                                                  other.channel_indexes)
        # TODO: merge self.annotations and other.annotations
        kwargs.update(self.annotations)
        return AnalogSignalArray(stack, units=self.units, dtype=self.dtype,
                                 copy=False, t_start=self.t_start,
                                 sampling_rate=self.sampling_rate,
                                 **kwargs)
