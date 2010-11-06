import numpy as np
import quantities as pq
from baseneo import BaseNeo


class AnalogSignal(BaseNeo, pq.Quantity):
    """
    BaseNeo should be the parent of every Neo class
    Quantities inherits from numpy.ndarray
    
    Usage:
    a = AnalogSignal([1,2,3], sampling_period=42)
    """
    def __new__(subtype, signal, dtype=None, copy=True, t_start=0., sampling_rate=None, sampling_period=None):
        # maybe some parameters are useless for the AnalogSignal use case (dtype, copy ?)
        if not sampling_period is None:
            if not sampling_rate is None:
                if sampling_period != 1./sampling_rate:
                    raise ValueError('The sampling_rate has to be 1./sampling_period')
            else:
                sampling_rate = 1./sampling_period

        if isinstance(signal, AnalogSignal):
            return signal

        if isinstance(signal, np.ndarray):
            new = signal.view(subtype)
            if copy: return new.copy()
            else: return new

        if isinstance(signal, str):
            signal = _convert_from_string(signal)

        # now convert signal to an array
        arr = pq.Quantity(signal, dtype=dtype, copy=copy)

        # added from quantities before the __new__ because it needs it
        subtype._dimensionality = arr._dimensionality
        #subtype.sampling_period = sampling_period no redundancy, thanks
        subtype.sampling_rate = float(sampling_rate)

        #ret = np.ndarray.__new__(subtype, shape, arr.dtype, buffer=arr, order=order)
        ret = pq.Quantity.__new__(subtype, arr)
        ret.signal = ret.view(np.ndarray) #, dtype=arr.dtype)
        ret.sampling_rate = float(sampling_rate)
        #ret.sampling_period = sampling_period

        return ret

    @property
    def sampling_period(self):
        return 1./self.sampling_rate


    @property
    def t_stop(self):
        return self.t_start * len(signal[0]) / self.sampling_rate

    def __array_finalize__(self, obj):
        if obj is None: return
        self.signal = getattr(obj, 'signal', None)
        self.sampling_period = getattr(obj, 'sampling_period', None)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)


