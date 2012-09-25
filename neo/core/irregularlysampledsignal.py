from neo.core.baseneo import BaseNeo

import numpy as np
import quantities as pq


class IrregularlySampledSignal(BaseNeo, pq.Quantity):
    """
    A representation of a continuous, analog signal acquired at time ``t_start``
    with a varying sampling interval.

    *Usage*:

      >>> from quantities import ms, nA, uV
      >>> import numpy as np
      >>> a = IrregularlySampledSignal([0.0, 1.23, 6.78], [1,2,3], units='mV', time_units='ms')
      >>> b = IrregularlySampledSignal([0.01, 0.03, 0.12]*s, [4,5,6]*nA)

    *Required attributes/properties*:
        :times:  NumPy array, Quantity array or list
        :signal: Numpy array, Quantity array or list of the same size as times
        :units:  required if the signal is a list or NumPy array, not if it is a :py:class:`Quantity`
        :time_units:  required if `times` is a list or NumPy array, not if it is a :py:class:`Quantity`

    *Optional arguments*:
        :dtype:  Data type of the signal (times are always floats)
    
    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:
    """
    
    def __new__(cls, times, signal, units=None, time_units=None, dtype=None,
                copy=True, name=None, description=None, file_origin=None,
                **annotations):
        if len(times) != len(signal):
            raise ValueError("times array and signal array must have same length")
        if units is None:
            if hasattr(signal, "units"):
                units = signal.units
            else:
                raise ValueError("Units must be specified")
        elif isinstance(signal, pq.Quantity):
            if units != signal.units: # could improve this test, what if units is a string?
                signal = signal.rescale(units)
        if time_units is None:
            if hasattr(times, "units"):
                time_units = times.units
            else:
                raise ValueError("Time units must be specified")
        elif isinstance(times, pq.Quantity):
            if time_units != times.units: # could improve this test, what if units is a string?
                times = times.rescale(time_units)
        # should check time units have correct dimensions
        obj = pq.Quantity.__new__(cls, signal, units=units, dtype=dtype, copy=copy)
        obj.times = pq.Quantity(times, units=time_units, dtype=float, copy=copy)
        obj.segment = None
        obj.recordingchannel = None
        return obj
    
    def __init__(self, times, signal, units=None, time_units=None, dtype=None,
                 copy=True, name=None, description=None, file_origin=None,
                 **annotations):
        """Initalize a new IrregularlySampledSignal."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def __array_finalize__(self, obj):
        super(IrregularlySampledSignal, self).__array_finalize__(obj)
        self.times = getattr(obj, 'times', None)

        # The additional arguments
        self.annotations = getattr(obj, 'annotations', None)

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)

    def __repr__(self):
        return '<%s(%s at times %s)>' % (self.__class__.__name__,
             super(IrregularlySampledSignal, self).__repr__(), self.times)
    
    def __getslice__(self, i, j):
        # doesn't get called in Python 3 - __getitem__ is called instead
        obj = super(IrregularlySampledSignal, self).__getslice__(i, j)
        obj.times = self.times.__getslice__(i, j)
        return obj

    def __getitem__(self, i):
        obj = super(IrregularlySampledSignal, self).__getitem__(i)
        if isinstance(obj, IrregularlySampledSignal):
            obj.times = self.times.__getitem__(i)
        return obj
    
    @property
    def duration(self):
        return self.times[-1] - self.times[0]

    @property
    def t_start(self):
        return self.times[0]

    @property
    def t_stop(self):
        return self.times[-1]
    
    def __eq__(self, other):
        return super(IrregularlySampledSignal, self).__eq__(other) and self.times == other.times

    def __ne__(self, other):
        return not self.__eq__(other)
    
    @property
    def sampling_intervals(self):
        return self.times[1:] - self.times[:-1]
    
    def mean(self, interpolation=None):
        """
        Calculates the mean, optionally using interpolation between sampling times.
        
        If interpolation is None, we assume that values change stepwise at sampling times.
        """
        if interpolation is None:
            return (self[:-1]*self.sampling_intervals).sum()/self.duration
        else:
            raise NotImplementedError

    def resample(self, at=None, interpolation=None):
        """
        Resample the signal, returning either an AnalogSignal object or another
        IrregularlySampledSignal object.
        
        Arguments:
            :at:  either a Quantity array containing the times at which samples
                  should be created (times must be within the signal duration,
                  there is no extrapolation), a sampling rate with dimensions
                  (1/Time) or a sampling interval with dimensions (Time).
            :interpolation: one of: None, 'linear'
        """
        # further interpolation methods could be added
        raise NotImplementedError
        
