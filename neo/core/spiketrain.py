from neo.core.baseneo import BaseNeo
import quantities as pq
import numpy

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


class SpikeTrain(BaseNeo, pq.Quantity):
    """SpikeTrain is a :class:`Quantity` array of spike times.
    
    It is an ensemble of action potentials (spikes) emitted by the same unit
    in a period of time.

    *Required arguments*:
        :times: a list, 1d numpy array, or quantity array. 
        
        The Quantity array is constructed with the data in :attr:`times`, as
        well as the construction arguments :attr:`units`, :attr:`dtype`, and :attr:`copy`. 
    
    *Recommended arguments*:
        :t_start: time at which SpikeTrain began. This will be converted
            to the same units as the data.
        :t_stop: time at which SpikeTrain ends. If not provided, the
            maximum of the data is chosen. This will be converted to the
            same units as the data.
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
        :class:`SpikeTrain` objects can be sliced. When this occurs, a new :class:`SpikeTrain` (actually
        a view) is returned, with the same metadata, except that :attr:`waveforms` is also sliced accordingly.

    *Example*::
    
        >>> st = SpikeTrain([3,4,5] * pq.s)
        >>> st2 = st[1:2]
        >>> st.t_start
        0. s
        >>> st2
        [4, 5] s
    """
    
    def __new__(cls, times, units=None,  dtype=numpy.float, copy=True,
        sampling_rate=1.0*pq.Hz, t_start=0.0*pq.s, t_stop=None, sort=True,
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
        
        # Construct Quantity from data
        obj = pq.Quantity.__new__(cls, times, units=units, dtype=dtype, 
            copy=copy)

        # Default value of t_stop
        if t_stop is None:
            try: t_stop = obj.max()
            except ValueError: t_stop = 0.0 * pq.s

        # Store recommended attributes
        obj.t_start = pq.Quantity(t_start, units=units)
        obj.t_stop = pq.Quantity(t_stop, units=units)        
        obj.waveforms = waveforms
        obj.left_sweep = left_sweep
        obj.sampling_rate = sampling_rate

        # Error checking (do earlier?)
        check_has_dimensions_time(obj, obj.t_start, obj.t_stop)

        # sort the times and waveforms
        # this should be moved to a SpikeTrain.sort() method
        if sort:
            sort_indices = obj.argsort()
            if waveforms is not None and waveforms.any():
                obj.waveforms = waveforms[sort_indices]
            obj.sort()

        return obj

    
    def __init__(self, times, units=None,  dtype=numpy.float, copy=True,
        sampling_rate=1.0*pq.Hz, t_start=0.0*pq.s, t_stop=None,  sort=True,
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
