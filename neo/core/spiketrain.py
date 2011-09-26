from neo.core.baseneo import BaseNeo
import quantities as pq
import numpy
import numpy as np

"""This module implements SpikeTrain.

SpikeTrain inherits from Quantity, which inherits from numpy.array.
Inheritance from numpy.array is explained here:
http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

In brief:
* Initialization of a new object from constructor happens in __new__.
This is where user-specified attributes are set.

* __array_finalize__ is called for all new objects, including those
created by slicing. This is where attributes are copied over from
the old object.
"""


def check_has_dimensions_time(*values):
    errmsgs = []
    for value in values:
        if value._reference.dimensionality != pq.s.dimensionality:
            errmsgs.append("value %s has dimensions %s, not [time]" % (
                value,value.dimensionality.simplified))
    if errmsgs:
        raise ValueError("\n".join(errmsgs))


def _check_time_in_range(value, t_start, t_stop):
    if hasattr(value, "min"):
        if value.min() < t_start:
            raise ValueError("The first spike (%s) is before t_start (%s)" % (value, t_start))
        if value.max() > t_stop:
            raise ValueError("The last spike (%s) is after t_stop (%s)" % (value, t_stop))
    else:
        if value < t_start:
            raise ValueError("The spike time (%s) is before t_start (%s)" % (value, t_start))
        if value > t_stop:
            raise ValueError("The spike time (%s) is after t_stop (%s)" % (value, t_stop))


class SpikeTrain(BaseNeo, pq.Quantity):
    """SpikeTrain is a :class:`Quantity` array of spike times.
    
    It is an ensemble of action potentials (spikes) emitted by the same unit
    in a period of time.

    *Required arguments*:
        :times: a list, 1d numpy array, or quantity array. 
        :t_stop: time at which SpikeTrain ends. This will be converted to the
            same units as the data.
            
        The Quantity array is constructed with the data in :attr:`times`, as
        well as the construction arguments :attr:`units`, :attr:`dtype`, and 
        :attr:`copy`. 
    
    *Recommended arguments*:
        :t_start: time at which SpikeTrain began. This will be converted
            to the same units as the data. Default is zero.
        :waveforms: the waveforms of each spike
        :sampling_rate: the sampling rate of the waveforms
        :left_sweep: Quantity, in units of time. Time from the beginning
            of the waveform to the trigger time of the spike.
        :sort: if True, the spike train will be sorted
        :name: string
        :description: string
        :file_origin: string
        
    Any other keyword arguments are stored in the :attr:`self.annotations` dict.
    
    *Slicing*:
        :class:`SpikeTrain` objects can be sliced. When this occurs, a new
        :class:`SpikeTrain` (actually a view) is returned, with the same 
        metadata, except for the following updates:

            :attr:`waveforms` is also sliced accordingly.
            :attr:`t_start` is the time of the starting index
            :attr:`t_stop` is the time of the starting index, plus the length
            of the new signal times the sampling period.

    *Example*::
    
        >>> st = SpikeTrain([3,4,5] * pq.s)
        >>> st2 = st[1:2]
        >>> st.t_start
        0. s
        >>> st2
        [4, 5] s
    """
    
    def __new__(cls, times, units=None,  dtype=numpy.float, copy=True,
        sampling_rate=1.0*pq.Hz, t_start=0.0*pq.s, t_stop=None,
        waveforms=None, left_sweep=None, **kwargs):
        """Constructs new SpikeTrain from data.
        
        This is called whenever a new SpikeTrain is created from the
        constructor, but not when slicing.
        
        First the Quantity array is constructed from the data. Then,        
        the attributes are set from the user's arguments. Finally, error
        checking and (optionally) sorting occurs.
        """
        # If data is Quantity, rescale to desired units
        if isinstance(times, pq.Quantity) and units: 
            times = times.rescale(units)
        
        # Error check that units were provided somehow
        if units is None: 
            try:
                units = times.units
            except AttributeError:
                raise ValueError('you must specify units')
        
        # Check that t_stop was provided
        if t_stop is None:
            raise ValueError("Argument t_stop must be provided.")
        
        # Construct Quantity from data
        obj = pq.Quantity.__new__(cls, times, units=units, dtype=dtype, 
                                  copy=copy)
        
        # Store attributes
        obj.t_start = pq.Quantity(t_start, units=units)
        obj.t_stop = pq.Quantity(t_stop, units=units)        
        obj.waveforms = waveforms
        obj.left_sweep = left_sweep
        obj.sampling_rate = sampling_rate

        # Error checking (do earlier?)
        check_has_dimensions_time(obj, obj.t_start, obj.t_stop)
        if obj.size > 0:
            _check_time_in_range(obj.min(), obj.t_start, obj.t_stop)
            _check_time_in_range(obj.max(), obj.t_start, obj.t_stop)
        return obj

    
    def __init__(self, times, units=None,  dtype=numpy.float, copy=True,
        sampling_rate=1.0*pq.Hz, t_start=0.0*pq.s, t_stop=None,
        waveforms=None, left_sweep=None, **kwargs):
        """Initializes newly constructed SpikeTrain."""
        # This method is only called when constructing a new SpikeTrain,
        # not when slicing or viewing. We use the same call signature
        # as __new__ for documentation purposes. Anything not in the call
        # signature is stored in annotations.
        
        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        BaseNeo.__init__(self, **kwargs)        


    def __array_finalize__(self, obj):
        """This is called every time a new SpikeTrain is created.
        
        It is the appropriate place to set default values for attributes
        for SpikeTrain constructed by slicing or viewing.
        
        User-specified values are only relevant for construction from
        constructor, and these are set in __new__. Then they are just
        copied over here.
        
        Note that the `waveforms` attibute is not sliced here. Nor is
        `t_start` or `t_stop` modified.
        """
        # This calls Quantity.__array_finalize__ which deals with dimensionality
        super(SpikeTrain, self).__array_finalize__(obj)
        
        # Supposedly, during initialization from constructor, obj is supposed
        # to be None, but this never happens. It must be something to do
        # with inheritance from Quantity.
        if obj is None: return
        
        # Set all attributes of the new object `self` from the attributes
        # of `obj`. For instance, when slicing, we want to copy over the
        # attributes of the original object.
        self.t_start = getattr(obj, 't_start', None)
        self.t_stop = getattr(obj, 't_stop', None)
        self.waveforms = getattr(obj, 'waveforms', None)
        self.left_sweep = getattr(obj, 'left_sweep', None)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        
        # The additional arguments
        self.annotations = getattr(obj, 'annotations', None)
        
        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
    


    def __repr__(self):
        return '<SpikeTrain(%s, [%s, %s], )>' % (
             super(SpikeTrain, self).__repr__(), self.t_start, self.t_stop, )

    def sort(self):
        """Sorts the spiketrain and its waveforms, if any."""
        # sort the waveforms by the times
        sort_indices = np.argsort(self)
        if self.waveforms is not None and self.waveforms.any():
            self.waveforms = self.waveforms[sort_indices]
        
        # now sort the times
        # We have sorted twice, but `self = self[sort_indices]` introduces
        # a dependency on the slicing functionality of SpikeTrain.
        super(SpikeTrain, self).sort()

    def __getslice__(self, i, j):
        # doesn't get called in Python 3 - __getitem__ is called instead
        
        # first slice the Quantity array
        obj = super(SpikeTrain, self).__getslice__(i, j)
        # somehow this knows to call SpikeTrain.__array_finalize__, though
        # I'm not sure how. (If you know, please add an explanatory comment
        # here.) That copies over all of the metadata.

        # update waveforms
        if obj.waveforms is not None:
            obj.waveforms = obj.waveforms[i:j]
        return obj

    def __setitem__(self, i, value):
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
            # or should we be strict: raise ValueError("Setting a value requires a quantity")?
        # check for values outside t_start, t_stop
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setitem__(i, value)

    def __setslice__(self, i, j, value):
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setslice__(i, j, value)

    @property
    def times(self):
        return self

    @property
    def duration(self):
        return self.t_stop - self.t_start
    
    @property
    def sampling_period(self):
        return 1.0 / self.sampling_rate

    @property
    def right_sweep(self):
        try:
            return self.left_sweep + self.waveforms.shape[2]/self.sampling_rate
        except:
            return None
    
