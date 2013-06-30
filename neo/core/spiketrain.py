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

from neo.core.baseneo import BaseNeo
import quantities as pq
import numpy as np


def check_has_dimensions_time(*values):
    errmsgs = []
    for value in values:
        dim = value.dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            errmsgs.append("value %s has dimensions %s, not [time]" %
                           (value, dim.simplified))
    if errmsgs:
        raise ValueError("\n".join(errmsgs))


def _check_time_in_range(value, t_start, t_stop, view=False):
    if not value.size:
        return

    # using views here drastically increases the speed, but is only
    # safe if we are certain that the dtype and units are the same
    if view:
        value = value.view(np.ndarray)
        t_start = t_start.view(np.ndarray)
        t_stop = t_stop.view(np.ndarray)

    if value.min() < t_start:
        raise ValueError("The first spike (%s) is before t_start (%s)" %
                         (value, t_start))
    if value.max() > t_stop:
        raise ValueError("The last spike (%s) is after t_stop (%s)" %
                         (value, t_stop))


def _new_spiketrain(cls, signal, t_stop, units=None, dtype=np.float,
                    copy=True, sampling_rate=None, t_start=0.0 * pq.s,
                    waveforms=None, left_sweep=None, name=None,
                    file_origin=None, description=None, annotations=None):
    """A function to map BaseAnalogSignal.__new__ to function that
    does not do the unit checking. This is needed for pickle to work.
    """
    return SpikeTrain(signal, t_stop, units, dtype, copy, sampling_rate,
                        t_start, waveforms, left_sweep, name, file_origin,
                        description, **annotations)


class SpikeTrain(BaseNeo, pq.Quantity):
    """SpikeTrain is a :class:`Quantity` array of spike times.

    It is an ensemble of action potentials (spikes) emitted by the same unit
    in a period of time.

    *Required arguments*:
        :times: a list, 1d numpy array, or quantity array, containing the
            times of each spike.
        :t_stop: time at which SpikeTrain ends. This will be converted to the
            same units as the data. This argument is required because it
            specifies the period of time over which spikes could have occurred.
            Note that `t_start` is highly recommended for the same reason.

        Your spike times must have units. Preferably, `times` is a Quantity
        array with units of time. Otherwise, you must specify the keyword
        argument `units`.

        If `times` contains values outside of the range [t_start, t_stop],
        an Exception is raised.

    *Recommended arguments*:
        :t_start: time at which SpikeTrain began. This will be converted
            to the same units as the data. Default is zero seconds.
        :waveforms: the waveforms of each spike
        :sampling_rate: the sampling rate of the waveforms
        :left_sweep: Quantity, in units of time. Time from the beginning
            of the waveform to the trigger time of the spike.
        :sort: if True, the spike train will be sorted
        :name: string
        :description: string
        :file_origin: string

    Any other keyword arguments are stored in the :attr:`self.annotations` dict

    *Other arguments relating to implementation*
        :attr:`dtype` : data type (float32, float64, etc)
        :attr:`copy` : boolean, whether to copy the data or use a view.

        These arguments, as well as `units`, are simply passed to the
        Quantity constructor.

        Note that `copy` must be True when you request a change
        of units or dtype.

    *Slicing*:
        :class:`SpikeTrain` objects can be sliced. When this occurs, a new
        :class:`SpikeTrain` (actually a view) is returned, with the same
        metadata, except that :attr:`waveforms` is also sliced in the same way.
        Note that t_start and t_stop are not changed automatically, though
        you can still manually change them.

    *Example*::
        >>> st = SpikeTrain([3,4,5] * pq.s, t_stop=10.0)
        >>> st2 = st[1:3]
        >>> st.t_start
        array(0.0) * s
        >>> st2
        <SpikeTrain(array([ 4.,  5.]) * s, [0.0 s, 10.0 s])>
    """

    def __new__(cls, times, t_stop, units=None, dtype=None, copy=True,
                sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s, waveforms=None,
                left_sweep=None, name=None, file_origin=None, description=None,
                **annotations):
        """Constructs new SpikeTrain from data.

        This is called whenever a new SpikeTrain is created from the
        constructor, but not when slicing.

        First the Quantity array is constructed from the data. Then,
        the attributes are set from the user's arguments. Finally, error
        checking and (optionally) sorting occurs.
        """
        # Make sure units are consistent
        # also get the dimensionality now since it is much faster to feed
        # that to Quantity rather than a unit
        if units is None:
            # No keyword units, so get from `times`
            try:
                units = times.units
                dim = units.dimensionality
            except AttributeError:
                raise ValueError('you must specify units')
        else:
            if hasattr(units, 'dimensionality'):
                dim = units.dimensionality
            else:
                dim = pq.quantity.validate_dimensionality(units)

            if (hasattr(times, 'dimensionality') and
                    times.dimensionality.items() != dim.items()):
                if not copy:
                    raise ValueError("cannot rescale and return view")
                else:
                    # this is needed because of a bug in python-quantities
                    # see issue # 65 in python-quantities github
                    # remove this if it is fixed
                    times = times.rescale(dim)

        if dtype is None:
            dtype = getattr(times, 'dtype', np.float)
        elif hasattr(times, 'dtype') and times.dtype != dtype:
            if not copy:
                raise ValueError("cannot change dtype and return view")

            # if t_start.dtype or t_stop.dtype != times.dtype != dtype,
            # _check_time_in_range can have problems, so we set the t_start
            # and t_stop dtypes to be the same as times before converting them
            # to dtype below
            # see ticket #38
            if hasattr(t_start, 'dtype') and t_start.dtype != times.dtype:
                t_start = t_start.astype(times.dtype)
            if hasattr(t_stop, 'dtype') and t_stop.dtype != times.dtype:
                t_stop = t_stop.astype(times.dtype)

        # check to make sure the units are time
        # this approach is orders of magnitude faster than comparing the
        # reference dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            ValueError("Unit %s has dimensions %s, not [time]" %
                       (units, dim.simplified))

        # Construct Quantity from data
        obj = pq.Quantity.__new__(cls, times, units=dim, dtype=dtype,
                                  copy=copy)

        # if the dtype and units match, just copy the values here instead
        # of doing the much more epxensive creation of a new Quantity
        # using items() is orders of magnitude faster
        if (hasattr(t_start, 'dtype') and t_start.dtype == obj.dtype and
                hasattr(t_start, 'dimensionality') and
                t_start.dimensionality.items() == dim.items()):
            obj.t_start = t_start.copy()
        else:
            obj.t_start = pq.Quantity(t_start, units=dim, dtype=dtype)

        if (hasattr(t_stop, 'dtype') and t_stop.dtype == obj.dtype and
                hasattr(t_stop, 'dimensionality') and
                t_stop.dimensionality.items() == dim.items()):
            obj.t_stop = t_stop.copy()
        else:
            obj.t_stop = pq.Quantity(t_stop, units=dim, dtype=dtype)

        # Store attributes
        obj.waveforms = waveforms
        obj.left_sweep = left_sweep
        obj.sampling_rate = sampling_rate

        # parents
        obj.segment = None
        obj.unit = None

        # Error checking (do earlier?)
        _check_time_in_range(obj, obj.t_start, obj.t_stop, view=True)

        return obj

    def __init__(self, times, t_stop, units=None,  dtype=np.float,
                 copy=True, sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s,
                 waveforms=None, left_sweep=None, name=None, file_origin=None,
                 description=None, **annotations):
        """Initializes newly constructed SpikeTrain."""
        # This method is only called when constructing a new SpikeTrain,
        # not when slicing or viewing. We use the same call signature
        # as __new__ for documentation purposes. Anything not in the call
        # signature is stored in annotations.

        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def rescale(self, units):
        """
        Return a copy of the SpikeTrain converted to the specified units
        """
        if self.dimensionality == pq.quantity.validate_dimensionality(units):
            return self.copy()
        spikes = self.view(pq.Quantity)
        return SpikeTrain(spikes, self.t_stop, units=units,
                          sampling_rate=self.sampling_rate,
                          t_start=self.t_start, waveforms=self.waveforms,
                          left_sweep=self.left_sweep, name=self.name,
                          file_origin=self.file_origin,
                          description=self.description, **self.annotations)

    def __reduce__(self):
        """
        Map the __new__ function onto _new_BaseAnalogSignal, so that pickle
        works
        """
        import numpy
        return _new_spiketrain, (self.__class__, numpy.array(self),
                                 self.t_stop, self.units, self.dtype, True,
                                 self.sampling_rate, self.t_start,
                                 self.waveforms, self.left_sweep,
                                 self.name, self.file_origin, self.description,
                                 self.annotations)

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
        # This calls Quantity.__array_finalize__ which deals with
        # dimensionality
        super(SpikeTrain, self).__array_finalize__(obj)

        # Supposedly, during initialization from constructor, obj is supposed
        # to be None, but this never happens. It must be something to do
        # with inheritance from Quantity.
        if obj is None:
            return

        # Set all attributes of the new object `self` from the attributes
        # of `obj`. For instance, when slicing, we want to copy over the
        # attributes of the original object.
        self.t_start = getattr(obj, 't_start', None)
        self.t_stop = getattr(obj, 't_stop', None)
        self.waveforms = getattr(obj, 'waveforms', None)
        self.left_sweep = getattr(obj, 'left_sweep', None)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        self.segment = getattr(obj, 'segment', None)
        self.unit = getattr(obj, 'unit', None)

        # The additional arguments
        self.annotations = getattr(obj, 'annotations', None)

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)

    def __repr__(self):
        return '<SpikeTrain(%s, [%s, %s])>' % (
            super(SpikeTrain, self).__repr__(), self.t_start, self.t_stop)

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

    def __getitem__(self, i):
        obj = super(SpikeTrain, self).__getitem__(i)
        if hasattr(obj, 'waveforms') and obj.waveforms is not None:
            obj.waveforms = obj.waveforms.__getitem__(i)
        return obj

    def __setitem__(self, i, value):
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
            # or should we be strict: raise ValueError("Setting a value
            # requires a quantity")?
        # check for values outside t_start, t_stop
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setitem__(i, value)

    def __setslice__(self, i, j, value):
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setslice__(i, j, value)

    def time_slice(self, t_start, t_stop):
        """Creates a new spiketrain corresponding to the time slice of the
        original spiketrain between (and including) times t_start, t_stop.
        Either parameter can also be None to use infinite endpoints for the
        time interval.
        """
        _t_start = t_start
        _t_stop = t_stop
        if t_start is None:
            _t_start = -np.inf
        if t_stop is None:
            _t_stop = np.inf
        indices = (self >= _t_start) & (self <= _t_stop)
        new_st = self[indices]

        new_st.t_start = max(_t_start, self.t_start)
        new_st.t_stop = min(_t_stop, self.t_stop)
        if self.waveforms is not None:
            new_st.waveforms = self.waveforms[indices]

        return new_st

    @property
    def times(self):
        return self

    @property
    def duration(self):
        if self.t_stop is None or self.t_start is None:
            return None
        return self.t_stop - self.t_start

    @property
    def spike_duration(self):
        if self.waveforms is None or self.sampling_rate is None:
            return None
        return self.waveform.shape[2] / self.sampling_rate

    @property
    def sampling_period(self):
        if self.sampling_rate is None:
            return None
        return 1.0 / self.sampling_rate

    @sampling_period.setter
    def sampling_period(self, period):
        if period is None:
            self.sampling_rate = None
        else:
            self.sampling_rate = 1.0 / period

    @property
    def right_sweep(self):
        dur = self.spike_duration
        if self.left_sweep is None or dur is None:
            return None
        return self.left_sweep + dur
