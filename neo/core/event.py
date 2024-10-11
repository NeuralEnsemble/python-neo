"""
This module defines :class:`Event`, an array of events.

:class:`Event` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
"""

from copy import deepcopy

import numpy as np
import quantities as pq

from neo.core.baseneo import merge_annotations
from neo.core.dataobject import DataObject, ArrayDict
from neo.core.epoch import Epoch


def _new_event(
    cls,
    times=None,
    labels=None,
    units=None,
    name=None,
    file_origin=None,
    description=None,
    array_annotations=None,
    annotations=None,
    segment=None,
):
    """
    A function to map Event.__new__ to function that
    does not do the unit checking. This is needed for pickle to work.
    """
    e = Event(
        times=times,
        labels=labels,
        units=units,
        name=name,
        file_origin=file_origin,
        description=description,
        array_annotations=array_annotations,
        **annotations,
    )
    e.segment = segment
    return e


class Event(DataObject):
    """
    Array of events which are the start times of events along with the labels
    of the events

    Parameters
    ----------
    times: quantity array 1d | list
        The times of the events
    labels: numpy.ndarray 1d dtype='U' | list
        Names or labels for the events
    units: quantity units | None, default: None
        If times are list the units of the times
        If times is a quantity array this is ignored
    name: str | None, default: None
        An optional label for the dataset
    description: str | None, default: None
        An optional text descriptoin of the dataset
    file_orgin: str | None, default: None
        The filesystem path or url of the original data file
    array_annotations: dict | None, default: None
        Dict mapping strings to numpy arrays containing annotations for all data points
    **annotations: dict
        Additional user specified metadata stored in the annotations attribue

    Examples
    --------

    >>> from neo.core import Event
    >>> from quantities import s
    >>> import numpy as np
    >>>
    >>> evt = Event(np.arange(0, 30, 10)*s,
    ...             labels=np.array(['trig0', 'trig1', 'trig2'],
    ...                             dtype='U'))
    >>>
    >>> evt.times
    array([  0.,  10.,  20.]) * s
    >>> evt.labels
    array(['trig0', 'trig1', 'trig2'],
              dtype='<U5')

    """

    _parent_objects = ("Segment",)
    _parent_attrs = ("segment",)
    _quantity_attr = "times"
    _necessary_attrs = (("times", pq.Quantity, 1), ("labels", np.ndarray, 1, np.dtype("U")))

    def __new__(
        cls,
        times=None,
        labels=None,
        units=None,
        name=None,
        description=None,
        file_origin=None,
        array_annotations=None,
        **annotations,
    ):
        if times is None:
            times = np.array([]) * pq.s
        elif isinstance(times, (list, tuple)):
            times = np.array(times)
        if len(times.shape) > 1:
            raise ValueError("Times array has more than 1 dimension")
        if labels is None:
            labels = np.array([], dtype="U")
        else:
            labels = np.array(labels)
            if labels.size != times.size and labels.size:
                raise ValueError("Labels array has different length to times")
        if units is None:
            # No keyword units, so get from `times`
            try:
                units = times.units
                dim = units.dimensionality
            except AttributeError:
                raise ValueError("you must specify units")
        else:
            if hasattr(units, "dimensionality"):
                dim = units.dimensionality
            else:
                dim = pq.quantity.validate_dimensionality(units)
        # check to make sure the units are time
        # this approach is much faster than comparing the
        # reference dimensionality
        if len(dim) != 1 or list(dim.values())[0] != 1 or not isinstance(list(dim.keys())[0], pq.UnitTime):
            ValueError(f"Unit {units} has dimensions {dim.simplified}, not [time].")

        obj = pq.Quantity(times, units=dim).view(cls)
        obj._labels = labels
        obj.segment = None
        return obj

    def __init__(
        self,
        times=None,
        labels=None,
        units=None,
        name=None,
        description=None,
        file_origin=None,
        array_annotations=None,
        **annotations,
    ):
        """
        Initialize a new :class:`Event` instance.
        """
        DataObject.__init__(
            self,
            name=name,
            file_origin=file_origin,
            description=description,
            array_annotations=array_annotations,
            **annotations,
        )

    def __reduce__(self):
        """
        Map the __new__ function onto _new_event, so that pickle
        works
        """
        return _new_event, (
            self.__class__,
            np.array(self),
            self.labels,
            self.units,
            self.name,
            self.file_origin,
            self.description,
            self.array_annotations,
            self.annotations,
            self.segment,
        )

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._labels = getattr(obj, "labels", None)
        self.annotations = getattr(obj, "annotations", None)
        self.name = getattr(obj, "name", None)
        self.file_origin = getattr(obj, "file_origin", None)
        self.description = getattr(obj, "description", None)
        self.segment = getattr(obj, "segment", None)
        # Add empty array annotations, because they cannot always be copied,
        # but do not overwrite existing ones from slicing etc.
        # This ensures the attribute exists
        if not hasattr(self, "array_annotations"):
            self.array_annotations = ArrayDict(self._get_arr_ann_length())

    def __repr__(self):
        """
        Returns a string representing the :class:`Event`.
        """

        objs = [f"{label}@{str(time)}" for label, time in zip(self.labels, self.times)]
        return "<Event: %s>" % ", ".join(objs)

    def _repr_pretty_(self, pp, cycle):
        labels = ""
        if self._labels is not None:
            labels = " with labels"
        pp.text(
            f"{self.__class__.__name__} containing {self.size} events{labels}; "
            f"time units {self.units.dimensionality.string}; datatype {self.dtype} "
        )
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

    def rescale(self, units, dtype=None):
        """
        Return a copy of the :class:`Event` converted to the specified units

        Parameters
        ----------
        units: quantity units
            The units to convert the Event to
        dtype: None
            Exists for backward compatiblity within quantities see Notes for more info

        Returns
        -------
        obj: neo.core.Event
            A copy of the event with the specified units

        Notes
        -----
        The `dtype` argument exists only for backward compatibility within quantities, see
        https://github.com/python-quantities/python-quantities/pull/204

        """
        # Use simpler functionality, if nothing will be changed
        dim = pq.quantity.validate_dimensionality(units)
        if self.dimensionality == dim:
            return self.copy()

        # Rescale the object into a new object
        obj = self.duplicate_with_new_data(times=self.view(pq.Quantity).rescale(dim), labels=self.labels, units=units)

        # Expected behavior is deepcopy, so deepcopying array_annotations
        obj.array_annotations = deepcopy(self.array_annotations)
        obj.segment = self.segment
        return obj

    @property
    def times(self):
        return pq.Quantity(self)

    def merge(self, other):
        """
        Merge another :class:`Event` into this one.

        Parameter
        ---------
        other: neo.core.Event
            The `Event` to merge into this one

        Notes
        -----
        * The :class:`Event` objects are concatenated horizontally
        (column-wise), :func:`np.hstack`).

        * If the attributes of the two :class:`Event` are not
        compatible, and Exception is raised.
        """
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
                kwargs[name] = f"merge({attr_self}, {attr_other})"

        print("Event: merge annotations")
        merged_annotations = merge_annotations(self.annotations, other.annotations)

        kwargs.update(merged_annotations)

        kwargs["array_annotations"] = self._merge_array_annotations(other)

        evt = Event(times=times, labels=labels, **kwargs)

        return evt

    def _copy_data_complement(self, other):
        """
        Copy the metadata from another :class:`Event`.
        Note: Array annotations can not be copied here because length of data can change
        """
        # Note: Array annotations, including labels, cannot be copied
        # because they are linked to their respective timestamps and length of data can be changed
        # here which would cause inconsistencies
        for attr in ("name", "file_origin", "description", "annotations"):
            setattr(self, attr, deepcopy(getattr(other, attr, None)))

    def __getitem__(self, i):
        obj = super().__getitem__(i)
        if self._labels is not None and self._labels.size > 0:
            obj.labels = self._labels[i]
        else:
            obj.labels = self._labels
        try:
            obj.array_annotate(**deepcopy(self.array_annotations_at_index(i)))
            obj._copy_data_complement(self)
        except AttributeError:  # If Quantity was returned, not Event
            obj.times = obj
        return obj

    def set_labels(self, labels):
        if self.labels is not None and self.labels.size > 0 and len(labels) != self.size:
            raise ValueError(f"Labels array has different length to times " f"({len(labels)} != {self.size})")
        self._labels = np.array(labels)

    def get_labels(self):
        return self._labels

    labels = property(get_labels, set_labels)

    def duplicate_with_new_data(self, times, labels, units=None):
        """
        Create a new :class:`Event` with the same metadata
        but different data
        Note: Array annotations can not be copied here because length of data can change
        """
        if units is None:
            units = self.units
        else:
            units = pq.quantity.validate_dimensionality(units)

        new = self.__class__(times=times, units=units)
        new._copy_data_complement(self)
        new.labels = labels
        # Note: Array annotations cannot be copied here, because length of data can be changed
        return new

    def time_slice(self, t_start, t_stop):
        """
        Creates a new `Event` corresponding to the time slice of the original `Event` between (and including) times
        `t_start` and `t_stop`.

        Parameters
        ----------
        t_start: float | None
            The starting time of the time slice
            If None will use -np.inf for the starting time
        t_stop: float | None
            The stopping time of the time slice
            If None will use np.inf for the stopping time

        Returns
        -------
        new_evt: neo.core.Event
            The new `Event` limited by the time points
        """
        _t_start = t_start
        _t_stop = t_stop
        if t_start is None:
            _t_start = -np.inf
        if t_stop is None:
            _t_stop = np.inf

        indices = (self >= _t_start) & (self <= _t_stop)

        # Time slicing should create a deep copy of the object
        new_evt = deepcopy(self[indices])

        return new_evt

    def time_shift(self, t_shift):
        """
        Shifts an :class:`Event` by an amount of time.

        Parameters
        ----------
        t_shift: Quantity (time)
            Amount of time by which to shift the :class:`Event`.

        Returns
        -------
        epoch: Event
            New instance of an :class:`Event` object starting at t_shift later than the
            original :class:`Event` (the original :class:`Event` is not modified).
        """
        new_evt = self.duplicate_with_new_data(times=self.times + t_shift, labels=self.labels)

        # Here we can safely copy the array annotations since we know that
        # the length of the Event does not change.
        new_evt.array_annotate(**self.array_annotations)

        return new_evt

    def to_epoch(self, pairwise=False, durations=None):
        """
        Returns a new Epoch object based on the times and labels in the Event object.

        This method has three modes of action.

        1. By default, an array of `n` event times will be transformed into
           `n-1` epochs, where the end of one epoch is the beginning of the next.
           This assumes that the events are ordered in time; it is the
           responsibility of the caller to check this is the case.
        2. If `pairwise` is True, then the event times will be taken as pairs
           representing the start and end time of an epoch. The number of
           events must be even, otherwise a ValueError is raised.
        3. If `durations` is given, it should be a scalar Quantity or a
           Quantity array of the same size as the Event.
           Each event time is then taken as the start of an epoch of duration
           given by `durations`.

        `pairwise=True` and `durations` are mutually exclusive. A ValueError
        will be raised if both are given.

        If `durations` is given, epoch labels are set to the corresponding
        labels of the events that indicate the epoch start
        If `durations` is not given, then the event labels A and B bounding
        the epoch are used to set the labels of the epochs in the form 'A-B'.
        """

        if pairwise:
            # Mode 2
            if durations is not None:
                raise ValueError("Inconsistent arguments. " "Cannot give both `pairwise` and `durations`")
            if self.size % 2 != 0:
                raise ValueError("Pairwise conversion of events to epochs" " requires an even number of events")
            times = self.times[::2]
            durations = self.times[1::2] - times
            labels = np.array([f"{a}-{b}" for a, b in zip(self.labels[::2], self.labels[1::2])])
        elif durations is None:
            # Mode 1
            times = self.times[:-1]
            durations = np.diff(self.times)
            labels = np.array([f"{a}-{b}" for a, b in zip(self.labels[:-1], self.labels[1:])])
        else:
            # Mode 3
            times = self.times
            labels = self.labels
        return Epoch(times=times, durations=durations, labels=labels)
