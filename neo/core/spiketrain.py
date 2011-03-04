from neo.core.baseneo import BaseNeo
import quantities as pq
import numpy


def check_has_dimensions_time(*values):
    errmsgs = []
    for value in values:
        if value._reference.dimensionality != pq.s.dimensionality:
            errmsgs.append("value %s has dimensions %s, not [time]" % (
                value,value.dimensionality.simplified))
    if errmsgs:
        raise ValueError("\n".join(errmsgs))


class SpikeTrain(BaseNeo, pq.Quantity):
    """
    An ensemble of action potentials (spikes) emitted by the same unit in a
    period of time.
    
    Always contains the spike times, may also contain the waveforms of the
    individual action potentials.
    
    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    ``numpy.ndarray``.
    """
    
    def __new__(cls, times, t_start, t_stop, units='', dtype=None, copy=True,
                waveforms=None, name='', sort=True):
        """
        Create a new :class:`SpikeTrain` instance from a list or numpy array
        of numerical values, or from a Quantity array with dimensions of time.
        
        ``times``      - a list, 1D numpy array or quantity array
        ``waveforms``  - an :class:`AnalogSignalArray` object (??)
        ``t_start``    - the start time of the spike train; must be
                         before or at the same time as the first spike
        ``t_stop``     - the end time of the spike train; must be after or at
                         the same time as the last spike
        ``units``      - if ``times`` is not a quantity array, the units of time
                         must be specified.
        """
        # check units
        if isinstance(times, pq.Quantity) and units:
            times = times.rescale(units)
        if not units and hasattr(times, "units"):
            units = times.units
        t_start = pq.Quantity(t_start, units)
        t_stop = pq.Quantity(t_stop, units)
        # sort the times and waveforms
        if sort:
            sorted = True
            times = numpy.array(times)
            sort_indices = numpy.argsort(times)
            times = times[sort_indices]
            if waveforms:
                waveforms = waveforms[sort_indices]
        else:
            sorted = False
        obj = pq.Quantity.__new__(cls, times, units=units, dtype=dtype, copy=copy)
        check_has_dimensions_time(obj, t_start, t_stop) # should perhaps check earlier
        obj.t_start = t_start
        obj.t_stop = t_stop
        obj._sorted = sorted
        obj.name = name
        obj._annotations = {}
        # check waveforms
        obj.waveforms = waveforms
        return obj

    def __array_finalize__(self, obj):
        super(SpikeTrain, self).__array_finalize__(obj)
        self.t_start = getattr(obj, 't_start', obj.min())
        self.t_stop = getattr(obj, 't_stop', obj.max())
        self.waveforms = getattr(obj, 'waveforms', None)

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def sorted(self):
        return self._sorted
