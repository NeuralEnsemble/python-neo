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
    
    def __new__(cls, times, units=None, dtype=numpy.float, copy=True, **kwargs):
        """Constructs Quantity array from data"""
        # If data is Quantity, rescale to desired units
        if isinstance(times, pq.Quantity) and units: 
            times = times.rescale(units)
        
        # Error check that units were provided somehow
        if units is None: 
            try:
                units = times.units
            except AttributeError:
                raise ValueError('you must specify units')
        
        # Construct Quantity and return
        obj = pq.Quantity.__new__(cls, times, units=units, dtype=dtype, 
            copy=copy)
        return obj

    
    def __init__(self, times, units=None,  dtype=numpy.float, copy=True,
        sampling_rate=1.0*pq.Hz, t_start=0.0, t_stop=None,  sort=True,
        waveforms=None, left_sweep=None, name='', **kwargs):
        """Create a new SpikeTrain instance from data.
        
        Required arguments:
            times : a list, 1d numpy array, or quantity array. 
            
            The Quantity array is constructed with the data in `times`, as
            well as the arguments `units`, `dtype`, and `copy`. 
            See: SpikeTrain.__new__
        
        Recommended arguments:
            t_start : time at which SpikeTrain began. This will be converted
                to the same units as the data.
            t_stop : time at which SpikeTrain ends. If not provided, the
                maximum of the data is chosen. This will be converted to the
                same units as the data.
            waveforms : the waveforms of each spike
            sampling_rate : the sampling rate of the waveforms
            left_sweep : hard to explain
            name : name of the spiketrain
            sort : if True, the spike train will be sorted

        Any other keyword arguments are stored in the `self.annotations` dict.
        """
        # Default value of t_stop
        if t_stop is None:
            try: t_stop = self.max()
            except ValueError: t_stop = 0.0

        # Store recommended attributes
        self.t_start = pq.Quantity(t_start, units=self.units)
        self.t_stop = pq.Quantity(t_stop, units=self.units)        
        self.name = name
        self.waveforms = waveforms
        self.left_sweep = left_sweep
        self.sampling_rate = sampling_rate

        # Error checking (do earlier?)
        check_has_dimensions_time(self, self.t_start, self.t_stop)

        # sort the times and waveforms
        # this should be moved to a SpikeTrain.sort() method
        if sort:
            sort_indices = self.argsort()
            if waveforms is not None and waveforms.any():
                self.waveforms = waveforms[sort_indices]
            self.sort()

        # create annotations dict
        self._annotations = kwargs

    def __repr__(self):
        return '<SpikeTrain(%s, [%s, %s], )>' % (
             super(SpikeTrain, self).__repr__(), self.t_start, self.t_stop, )

    @property
    def times(self):
        return self

    @property
    def duration(self):
        return self.t_stop - self.t_start

    @property
    def right_sweep(self):
        try:
            return self.left_sweep + self.waveforms.shape[2]/self.sampling_rate
        except:
            return None
    
    @property
    def sorted(self):
        return self._sorted
