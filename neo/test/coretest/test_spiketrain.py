"""
Tests of the neo.core.spiketrain.SpikeTrain class and related functions
"""

import sys

import unittest
import warnings
from copy import deepcopy

import numpy as np
from numpy.testing import assert_array_equal
import quantities as pq

from neo.core.dataobject import ArrayDict

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.spiketrain import (check_has_dimensions_time, SpikeTrain, _check_time_in_range,
                                 _new_spiketrain)
from neo.core import Segment
from neo.core.baseneo import MergeError
from neo.test.tools import (assert_arrays_equal, assert_arrays_almost_equal,
                            assert_neo_object_is_compliant,
                            assert_same_attributes, assert_same_annotations,
                            assert_same_array_annotations)


class Testcheck_has_dimensions_time(unittest.TestCase):
    def test__check_has_dimensions_time(self):
        a = np.arange(3) * pq.ms
        b = np.arange(3) * pq.mV
        c = np.arange(3) * pq.mA
        d = np.arange(3) * pq.minute
        check_has_dimensions_time(a)
        self.assertRaises(ValueError, check_has_dimensions_time, b)
        self.assertRaises(ValueError, check_has_dimensions_time, c)
        check_has_dimensions_time(d)
        self.assertRaises(ValueError, check_has_dimensions_time, a, b, c, d)

    # Regression test for #763
    # This test ensures the function works for compound units
    def test__check_has_dimensions_time_compound_unit(self):
        a = np.arange(3) * pq.CompoundUnit("1/10*s")
        check_has_dimensions_time(a)


class Testcheck_time_in_range(unittest.TestCase):
    def test__check_time_in_range_empty_array(self):
        value = np.array([])
        t_start = 0 * pq.s
        t_stop = 10 * pq.s
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=False)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=True)

    def test__check_time_in_range_empty_array_invalid_t_stop(self):
        value = np.array([])
        t_start = 6 * pq.s
        t_stop = 4 * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop)

    def test__check_time_in_range_exact(self):
        value = np.array([0., 5., 10.]) * pq.s
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=False)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=True)

    def test__check_time_in_range_scale(self):
        value = np.array([0., 5000., 10000.]) * pq.ms
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=False)

    def test__check_time_in_range_inside(self):
        value = np.array([0.1, 5., 9.9]) * pq.s
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=False)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=True)

    def test__check_time_in_range_below(self):
        value = np.array([-0.1, 5., 10.]) * pq.s
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=False)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=True)

    def test__check_time_in_range_below_scale(self):
        value = np.array([-1., 5000., 10000.]) * pq.ms
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=False)

    def test__check_time_in_range_above(self):
        value = np.array([0., 5., 10.1]) * pq.s
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=False)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=True)

    def test__check_time_in_range_above_scale(self):
        value = np.array([0., 5000., 10001.]) * pq.ms
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=False)

    def test__check_time_in_range_above_below(self):
        value = np.array([-0.1, 5., 10.1]) * pq.s
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=False)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=True)

    def test__check_time_in_range_above_below_scale(self):
        value = np.array([-1., 5000., 10001.]) * pq.ms
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value, t_start=t_start, t_stop=t_stop,
                          view=False)


class TestConstructor(unittest.TestCase):
    def result_spike_check(self, train, st_out, t_start_out, t_stop_out, dtype, units):
        assert_arrays_equal(train, st_out)
        assert_arrays_equal(train, train.times)
        assert_neo_object_is_compliant(train)

        self.assertEqual(train.t_start, t_start_out)
        self.assertEqual(train.t_stop, t_stop_out)

        self.assertEqual(train.units, units)
        self.assertEqual(train.units, train.times.units)
        self.assertEqual(train.t_start.units, units)
        self.assertEqual(train.t_stop.units, units)

        self.assertEqual(train.dtype, dtype)
        self.assertEqual(train.dtype, train.times.dtype)
        self.assertEqual(train.t_stop.dtype, dtype)
        self.assertEqual(train.t_start.dtype, dtype)

    def test__create_minimal(self):
        t_start = 0.0
        t_stop = 10.0
        train1 = SpikeTrain([] * pq.s, t_stop)
        train2 = _new_spiketrain(SpikeTrain, [] * pq.s, t_stop)

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = [] * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_empty(self):
        t_start = 0.0
        t_stop = 10.0
        train1 = SpikeTrain([], t_start=t_start, t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, [], t_start=t_start, t_stop=t_stop, units='s')

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = [] * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_empty_no_t_start(self):
        t_start = 0.0
        t_stop = 10.0
        train1 = SpikeTrain([], t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, [], t_stop=t_stop, units='s')

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = [] * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_list(self):
        times = range(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms")
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="ms")

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_list_set_dtype(self):
        times = range(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="ms",
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = pq.Quantity(times, units=units, dtype=dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_list_no_start_stop_units(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms")
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="ms")

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_list_no_start_stop_units_set_dtype(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="ms",
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = pq.Quantity(times, units=units, dtype=dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_array(self):
        times = np.arange(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = int
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_array_with_dtype(self):
        times = np.arange(10, dtype='f4')
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = times.dtype
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_array_set_dtype(self):
        times = np.arange(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="s",
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype) * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_array_no_start_stop_units(self):
        times = np.arange(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = int
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_array_no_start_stop_units_with_dtype(self):
        times = np.arange(10, dtype='f4')
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="s")

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_array_no_start_stop_units_set_dtype(self):
        times = np.arange(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units="s",
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype) * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop)

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_with_dtype(self):
        times = np.arange(10, dtype='f4') * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop)

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_no_start_stop_units(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop)

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_no_start_stop_units_with_dtype(self):
        times = np.arange(10, dtype='f4') * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop)

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_no_start_stop_units_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_units(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units='s')

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_units_with_dtype(self):
        times = np.arange(10, dtype='f4') * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units='s')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.rescale(units).astype(dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_units_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s', dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units='s',
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.rescale(units).astype(dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_array_units_no_start_stop_units(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units='s')

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_quantity_units_no_start_stop_units_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s', dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times, t_start=t_start, t_stop=t_stop, units='s',
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out, dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out, dtype, units)

    def test__create_from_list_without_units_should_raise_ValueError(self):
        times = range(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        self.assertRaises(ValueError, SpikeTrain, times, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, times, t_start=t_start,
                          t_stop=t_stop)

    def test__create_from_array_without_units_should_raise_ValueError(self):
        times = np.arange(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        self.assertRaises(ValueError, SpikeTrain, times, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, times, t_start=t_start,
                          t_stop=t_stop)

    def test__create_from_array_with_incompatible_units_ValueError(self):
        times = np.arange(10) * pq.km
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        self.assertRaises(ValueError, SpikeTrain, times, t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, times, t_start=t_start,
                          t_stop=t_stop)

    def test__create_with_times_outside_tstart_tstop_ValueError(self):
        t_start = 23
        t_stop = 77
        train1 = SpikeTrain(np.arange(t_start, t_stop), units='ms', t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, np.arange(t_start, t_stop), units='ms',
                                 t_start=t_start, t_stop=t_stop)
        assert_neo_object_is_compliant(train1)
        assert_neo_object_is_compliant(train2)
        self.assertRaises(ValueError, SpikeTrain, np.arange(t_start - 5, t_stop), units='ms',
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, np.arange(t_start - 5, t_stop),
                          units='ms', t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, SpikeTrain, np.arange(t_start, t_stop + 5), units='ms',
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, np.arange(t_start, t_stop + 5),
                          units='ms', t_start=t_start, t_stop=t_stop)

    def test__create_with_len_times_different_size_than_waveform_shape1_ValueError(self):
        self.assertRaises(ValueError, SpikeTrain, times=np.arange(10), units='s', t_stop=4,
                          waveforms=np.ones((10, 6, 50)))

    def test__create_with_invalid_times_dimension(self):
        data2d = np.array([1, 2, 3, 4]).reshape((4, -1))
        self.assertRaises(ValueError, SpikeTrain, times=data2d * pq.s, t_stop=10 * pq.s)

    def test_defaults(self):
        # default recommended attributes
        train1 = SpikeTrain([3, 4, 5], units='sec', t_stop=10.0)
        train2 = _new_spiketrain(SpikeTrain, [3, 4, 5], units='sec', t_stop=10.0)
        assert_neo_object_is_compliant(train1)
        assert_neo_object_is_compliant(train2)
        self.assertEqual(train1.dtype, np.float_)
        self.assertEqual(train2.dtype, np.float_)
        self.assertEqual(train1.sampling_rate, 1.0 * pq.Hz)
        self.assertEqual(train2.sampling_rate, 1.0 * pq.Hz)
        self.assertEqual(train1.waveforms, None)
        self.assertEqual(train2.waveforms, None)
        self.assertEqual(train1.left_sweep, None)
        self.assertEqual(train2.left_sweep, None)
        self.assertEqual(train1.array_annotations, {})
        self.assertEqual(train2.array_annotations, {})
        self.assertIsInstance(train1.array_annotations, ArrayDict)
        self.assertIsInstance(train2.array_annotations, ArrayDict)

    def test_default_tstart(self):
        # t start defaults to zero
        train11 = SpikeTrain([3, 4, 5] * pq.s, t_stop=8000 * pq.ms)
        train21 = _new_spiketrain(SpikeTrain, [3, 4, 5] * pq.s, t_stop=8000 * pq.ms)
        assert_neo_object_is_compliant(train11)
        assert_neo_object_is_compliant(train21)
        self.assertEqual(train11.t_start, 0. * pq.s)
        self.assertEqual(train21.t_start, 0. * pq.s)

        # unless otherwise specified
        train12 = SpikeTrain([3, 4, 5] * pq.s, t_start=2.0, t_stop=8)
        train22 = _new_spiketrain(SpikeTrain, [3, 4, 5] * pq.s, t_start=2.0, t_stop=8)
        assert_neo_object_is_compliant(train12)
        assert_neo_object_is_compliant(train22)
        self.assertEqual(train12.t_start, 2. * pq.s)
        self.assertEqual(train22.t_start, 2. * pq.s)

    def test_tstop_units_conversion(self):
        train11 = SpikeTrain([3, 5, 4] * pq.s, t_stop=10)
        train21 = _new_spiketrain(SpikeTrain, [3, 5, 4] * pq.s, t_stop=10)
        assert_neo_object_is_compliant(train11)
        assert_neo_object_is_compliant(train21)
        self.assertEqual(train11.t_stop, 10. * pq.s)
        self.assertEqual(train21.t_stop, 10. * pq.s)

        train12 = SpikeTrain([3, 5, 4] * pq.s, t_stop=10000. * pq.ms)
        train22 = _new_spiketrain(SpikeTrain, [3, 5, 4] * pq.s, t_stop=10000. * pq.ms)
        assert_neo_object_is_compliant(train12)
        assert_neo_object_is_compliant(train22)
        self.assertEqual(train12.t_stop, 10. * pq.s)
        self.assertEqual(train22.t_stop, 10. * pq.s)

        train13 = SpikeTrain([3, 5, 4], units='sec', t_stop=10000. * pq.ms)
        train23 = _new_spiketrain(SpikeTrain, [3, 5, 4], units='sec', t_stop=10000. * pq.ms)
        assert_neo_object_is_compliant(train13)
        assert_neo_object_is_compliant(train23)
        self.assertEqual(train13.t_stop, 10. * pq.s)
        self.assertEqual(train23.t_stop, 10. * pq.s)


class TestSorting(unittest.TestCase):
    def test_sort(self):
        waveforms = np.array([[[0., 1.]], [[2., 3.]], [[4., 5.]]]) * pq.mV
        train = SpikeTrain([3, 4, 5] * pq.s, waveforms=waveforms, name='n', t_stop=10.0,
                           array_annotations={'a': np.arange(3)})
        assert_neo_object_is_compliant(train)
        train.sort()
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train, [3, 4, 5] * pq.s)
        assert_arrays_equal(train.waveforms, waveforms)
        self.assertEqual(train.name, 'n')
        self.assertEqual(train.t_stop, 10.0 * pq.s)
        assert_arrays_equal(train.array_annotations['a'], np.arange(3))

        train = SpikeTrain([3, 5, 4] * pq.s, waveforms=waveforms, name='n', t_stop=10.0,
                           array_annotations={'a': np.arange(3)})
        assert_neo_object_is_compliant(train)
        train.sort()
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train, [3, 4, 5] * pq.s)
        assert_arrays_equal(train.waveforms, waveforms[[0, 2, 1]])
        self.assertEqual(train.name, 'n')
        self.assertEqual(train.t_start, 0.0 * pq.s)
        self.assertEqual(train.t_stop, 10.0 * pq.s)
        assert_arrays_equal(train.array_annotations['a'], np.array([0, 2, 1]))
        self.assertIsInstance(train.array_annotations, ArrayDict)


class TestSlice(unittest.TestCase):
    def setUp(self):
        self.waveforms1 = np.array(
            [[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]]]) * pq.mV
        self.data1 = np.array([3, 4, 5])
        self.data1quant = self.data1 * pq.s
        self.arr_ann = {'index': np.arange(1, 4), 'label': ['abc', 'def', 'ghi']}
        self.train1 = SpikeTrain(self.data1quant, waveforms=self.waveforms1, name='n', arb='arbb',
                                 t_stop=10.0, array_annotations=self.arr_ann)

    def test_compliant(self):
        assert_neo_object_is_compliant(self.train1)

    def test_slice(self):
        # slice spike train, keep sliced spike times
        result = self.train1[1:2]
        assert_arrays_equal(self.train1[1:2], result)
        targwaveforms = np.array([[[2., 3.], [2.1, 3.1]]]) * pq.mV

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(self.train1.t_start, result.t_start)
        self.assertEqual(self.train1.t_stop, result.t_stop)

        # except we update the waveforms
        assert_arrays_equal(self.train1.waveforms[1:2], result.waveforms)
        assert_arrays_equal(targwaveforms, result.waveforms)

        # Also array annotations should be updated
        assert_arrays_equal(result.array_annotations['index'], np.array([2]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['def']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_slice_to_end(self):
        # slice spike train, keep sliced spike times
        result = self.train1[1:]
        assert_arrays_equal(self.train1[1:], result)
        targwaveforms = np.array([[[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]]]) * pq.mV

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(self.train1.t_start, result.t_start)
        self.assertEqual(self.train1.t_stop, result.t_stop)

        # except we update the waveforms
        assert_arrays_equal(self.train1.waveforms[1:], result.waveforms)
        assert_arrays_equal(targwaveforms, result.waveforms)

        # Also array annotations should be updated
        assert_arrays_equal(result.array_annotations['index'], np.array([2, 3]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['def', 'ghi']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_slice_from_beginning(self):
        # slice spike train, keep sliced spike times
        result = self.train1[:2]
        assert_arrays_equal(self.train1[:2], result)
        targwaveforms = np.array([[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]]]) * pq.mV

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(self.train1.t_start, result.t_start)
        self.assertEqual(self.train1.t_stop, result.t_stop)

        # except we update the waveforms
        assert_arrays_equal(self.train1.waveforms[:2], result.waveforms)
        assert_arrays_equal(targwaveforms, result.waveforms)

        # Also array annotations should be updated
        assert_arrays_equal(result.array_annotations['index'], np.array([1, 2]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['abc', 'def']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_slice_negative_idxs(self):
        # slice spike train, keep sliced spike times
        result = self.train1[:-1]
        assert_arrays_equal(self.train1[:-1], result)
        targwaveforms = np.array([[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]]]) * pq.mV

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(self.train1.t_start, result.t_start)
        self.assertEqual(self.train1.t_stop, result.t_stop)

        # except we update the waveforms
        assert_arrays_equal(self.train1.waveforms[:-1], result.waveforms)
        assert_arrays_equal(targwaveforms, result.waveforms)

        # Also array annotations should be updated
        assert_arrays_equal(result.array_annotations['index'], np.array([1, 2]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['abc', 'def']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)


class TestTimeSlice(unittest.TestCase):
    def setUp(self):
        self.waveforms1 = np.array(
            [[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]],
             [[6., 7.], [6.1, 7.1]], [[8., 9.], [8.1, 9.1]], [[10., 11.], [10.1, 11.1]]]) * pq.mV
        self.data1 = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.data1quant = self.data1 * pq.ms
        self.arr_ann = {'index': np.arange(1, 7), 'label': ['a', 'b', 'c', 'd', 'e', 'f']}
        self.train1 = SpikeTrain(self.data1quant, t_stop=10.0 * pq.ms, waveforms=self.waveforms1,
                                 array_annotations=self.arr_ann)
        self.seg = Segment()
        self.train1.segment = self.seg

    def test_compliant(self):
        assert_neo_object_is_compliant(self.train1)

    def test_time_slice_typical(self):
        # time_slice spike train, keep sliced spike times
        # this is the typical time slice falling somewhere
        # in the middle of spikes
        t_start = 0.12 * pq.ms
        t_stop = 3.5 * pq.ms
        result = self.train1.time_slice(t_start, t_stop)
        targ = SpikeTrain([0.5, 1.2, 3.3] * pq.ms, t_stop=3.3)
        assert_arrays_equal(result, targ)
        targwaveforms = np.array(
            [[[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]], [[6., 7.], [6.1, 7.1]]]) * pq.mV
        assert_arrays_equal(targwaveforms, result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(t_start, result.t_start)
        self.assertEqual(t_stop, result.t_stop)

        # Array annotations should be updated according to time slice
        assert_arrays_equal(result.array_annotations['index'], np.array([2, 3, 4]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['b', 'c', 'd']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_differnt_units(self):
        # time_slice spike train, keep sliced spike times
        t_start = 0.00012 * pq.s
        t_stop = 0.0035 * pq.s
        result = self.train1.time_slice(t_start, t_stop)
        targ = SpikeTrain([0.5, 1.2, 3.3] * pq.ms, t_stop=3.3)
        assert_arrays_equal(result, targ)
        targwaveforms = np.array(
            [[[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]], [[6., 7.], [6.1, 7.1]]]) * pq.mV
        assert_arrays_equal(targwaveforms, result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(t_start, result.t_start)
        self.assertEqual(t_stop, result.t_stop)

        # Array annotations should be updated according to time slice
        assert_arrays_equal(result.array_annotations['index'], np.array([2, 3, 4]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['b', 'c', 'd']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__time_slice_deepcopy_annotations(self):
        params1 = {'test0': 'y1', 'test1': ['deeptest'], 'test2': True}
        self.train1.annotate(**params1)
        # time_slice spike train, keep sliced spike times
        t_start = 0.00012 * pq.s
        t_stop = 0.0035 * pq.s
        result = self.train1.time_slice(t_start, t_stop)

        # Change annotations of original
        params2 = {'test0': 'y2', 'test2': False}
        self.train1.annotate(**params2)
        self.train1.annotations['test1'][0] = 'shallowtest'

        self.assertNotEqual(self.train1.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.train1.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.train1.annotations['test2'], result.annotations['test2'])

        # Change annotations of result
        params3 = {'test0': 'y3'}
        result.annotate(**params3)
        result.annotations['test1'][0] = 'shallowtest2'

        self.assertNotEqual(self.train1.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.train1.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.train1.annotations['test2'], result.annotations['test2'])

    def test__time_slice_deepcopy_array_annotations(self):
        length = len(self.train1)
        params1 = {'test0': ['y{}'.format(i) for i in range(length)],
                   'test1': ['deeptest' for i in range(length)],
                   'test2': [(-1)**i > 0 for i in range(length)]}
        self.train1.array_annotate(**params1)
        # time_slice spike train, keep sliced spike times
        t_start = 0.00012 * pq.s
        t_stop = 0.0035 * pq.s
        result = self.train1.time_slice(t_start, t_stop)

        # Change annotations of original
        params2 = {'test0': ['x{}'.format(i) for i in range(length)],
                   'test2': [(-1) ** (i + 1) > 0 for i in range(length)]}
        self.train1.array_annotate(**params2)
        self.train1.array_annotations['test1'][2] = 'shallowtest'

        self.assertFalse(all(self.train1.array_annotations['test0'][1:4]
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.train1.array_annotations['test1'][1:4]
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.train1.array_annotations['test2'][1:4]
                             == result.array_annotations['test2']))

        # Change annotations of result
        params3 = {'test0': ['z{}'.format(i) for i in range(1, 4)]}
        result.array_annotate(**params3)
        result.array_annotations['test1'][1] = 'shallow2'

        self.assertFalse(all(self.train1.array_annotations['test0'][1:4]
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.train1.array_annotations['test1'][1:4]
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.train1.array_annotations['test2'][1:4]
                             == result.array_annotations['test2']))

    def test__time_slice_deepcopy_data(self):
        result = self.train1.time_slice(None, None)

        # Change values of original array
        self.train1[2] = 7.3*self.train1.units

        self.assertFalse(all(self.train1 == result))

        # Change values of sliced array
        result[3] = 9.5*result.units

        self.assertFalse(all(self.train1 == result))

    def test_time_slice_matching_ends(self):
        # time_slice spike train, keep sliced spike times
        t_start = 0.1 * pq.ms
        t_stop = 7.0 * pq.ms
        result = self.train1.time_slice(t_start, t_stop)
        assert_arrays_equal(self.train1, result)
        assert_arrays_equal(self.waveforms1, result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(t_start, result.t_start)
        self.assertEqual(t_stop, result.t_stop)

        # Array annotations should be updated according to time slice
        assert_arrays_equal(result.array_annotations['index'], np.array(self.arr_ann['index']))
        assert_arrays_equal(result.array_annotations['label'], np.array(self.arr_ann['label']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_out_of_boundaries(self):
        self.train1.t_start = 0.1 * pq.ms
        assert_neo_object_is_compliant(self.train1)

        # time_slice spike train, keep sliced spike times
        t_start = 0.01 * pq.ms
        t_stop = 70.0 * pq.ms
        result = self.train1.time_slice(t_start, t_stop)
        assert_arrays_equal(self.train1, result)
        assert_arrays_equal(self.waveforms1, result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(self.train1.t_start, result.t_start)
        self.assertEqual(self.train1.t_stop, result.t_stop)

        # Array annotations should be updated according to time slice
        assert_arrays_equal(result.array_annotations['index'], np.array(self.arr_ann['index']))
        assert_arrays_equal(result.array_annotations['label'], np.array(self.arr_ann['label']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_completely_out_of_boundaries(self):
        # issue 831
        t_start = 20.0 * pq.ms
        t_stop = 70.0 * pq.ms
        self.assertRaises(ValueError, self.train1.time_slice, t_start, t_stop)

    def test_time_slice_empty(self):
        waveforms = np.array([[[]]]) * pq.mV
        train = SpikeTrain([] * pq.ms, t_stop=10.0, waveforms=waveforms)
        assert_neo_object_is_compliant(train)

        # time_slice spike train, keep sliced spike times
        t_start = 0.01 * pq.ms
        t_stop = 70.0 * pq.ms
        result = train.time_slice(t_start, t_stop)
        assert_arrays_equal(train, result)
        assert_arrays_equal(waveforms[:-1], result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(train.name, result.name)
        self.assertEqual(train.description, result.description)
        self.assertEqual(train.annotations, result.annotations)
        self.assertEqual(train.file_origin, result.file_origin)
        self.assertEqual(train.dtype, result.dtype)
        self.assertEqual(t_start, result.t_start)
        self.assertEqual(train.t_stop, result.t_stop)

    def test_time_slice_none_stop(self):
        # time_slice spike train, keep sliced spike times
        t_start = 1 * pq.ms
        result = self.train1.time_slice(t_start, None)
        assert_arrays_equal([1.2, 3.3, 6.4, 7] * pq.ms, result)
        targwaveforms = np.array(
            [[[4., 5.], [4.1, 5.1]], [[6., 7.], [6.1, 7.1]], [[8., 9.], [8.1, 9.1]],
             [[10., 11.], [10.1, 11.1]]]) * pq.mV
        assert_arrays_equal(targwaveforms, result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(t_start, result.t_start)
        self.assertEqual(self.train1.t_stop, result.t_stop)

        # Array annotations should be updated according to time slice
        assert_arrays_equal(result.array_annotations['index'], np.array([3, 4, 5, 6]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['c', 'd', 'e', 'f']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_start(self):
        # time_slice spike train, keep sliced spike times
        t_stop = 1 * pq.ms
        result = self.train1.time_slice(None, t_stop)
        assert_arrays_equal([0.1, 0.5] * pq.ms, result)
        targwaveforms = np.array([[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]]]) * pq.mV
        assert_arrays_equal(targwaveforms, result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(self.train1.t_start, result.t_start)
        self.assertEqual(t_stop, result.t_stop)

        # Array annotations should be updated according to time slice
        assert_arrays_equal(result.array_annotations['index'], np.array([1, 2]))
        assert_arrays_equal(result.array_annotations['label'], np.array(['a', 'b']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_both(self):
        self.train1.t_start = 0.1 * pq.ms
        assert_neo_object_is_compliant(self.train1)

        # time_slice spike train, keep sliced spike times
        result = self.train1.time_slice(None, None)
        assert_arrays_equal(self.train1, result)
        assert_arrays_equal(self.waveforms1, result.waveforms)

        # but keep everything else pristine
        assert_neo_object_is_compliant(result)
        self.assertEqual(self.train1.name, result.name)
        self.assertEqual(self.train1.description, result.description)
        self.assertEqual(self.train1.annotations, result.annotations)
        self.assertEqual(self.train1.file_origin, result.file_origin)
        self.assertEqual(self.train1.dtype, result.dtype)
        self.assertEqual(self.train1.t_start, result.t_start)
        self.assertEqual(self.train1.t_stop, result.t_stop)

        # Array annotations should be updated according to time slice
        assert_arrays_equal(result.array_annotations['index'], np.array(self.arr_ann['index']))
        assert_arrays_equal(result.array_annotations['label'], np.array(self.arr_ann['label']))
        self.assertIsInstance(result.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result.array_annotations['label'], np.ndarray)
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__time_slice_should_set_parents_to_None(self):
        # When timeslicing, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.train1.time_slice(1 * pq.ms, 3 * pq.ms)
        self.assertEqual(result.segment, None)
        self.assertEqual(result.unit, None)

    def test__deepcopy_should_set_parents_objects_to_None(self):
        # Deepcopy should destroy references to parents
        result = deepcopy(self.train1)
        self.assertEqual(result.segment, None)
        self.assertEqual(result.unit, None)


class TestTimeShift(unittest.TestCase):
    def setUp(self):
        self.waveforms1 = np.array(
            [[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]],
             [[6., 7.], [6.1, 7.1]], [[8., 9.], [8.1, 9.1]],
             [[10., 11.], [10.1, 11.1]]]) * pq.mV
        self.data1 = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.data1quant = self.data1 * pq.ms
        self.arr_ann = {'index': np.arange(1, 7), 'label': ['a', 'b', 'c', 'd', 'e', 'f']}
        self.train1 = SpikeTrain(self.data1quant, t_stop=10.0 * pq.ms,
                                 waveforms=self.waveforms1,
                                 array_annotations=self.arr_ann)
        self.seg = Segment()
        self.train1.segment = self.seg

    def test_compliant(self):
        assert_neo_object_is_compliant(self.train1)

    def test__time_shift_same_attributes(self):
        result = self.train1.time_shift(1 * pq.ms)
        assert_same_attributes(result, self.train1, exclude=['times', 't_start', 't_stop'])

    def test__time_shift_same_annotations(self):
        result = self.train1.time_shift(1 * pq.ms)
        assert_same_annotations(result, self.train1)

    def test__time_shift_same_array_annotations(self):
        result = self.train1.time_shift(1 * pq.ms)
        assert_same_array_annotations(result, self.train1)

    def test__time_shift_should_set_parents_to_None(self):
        # When time-shifting, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.train1.time_shift(1 * pq.ms)
        self.assertEqual(result.segment, None)
        self.assertEqual(result.unit, None)

    def test__time_shift_by_zero(self):
        shifted = self.train1.time_shift(0 * pq.ms)
        assert_arrays_equal(shifted.times, self.train1.times)

    def test__time_shift_same_units(self):
        shifted = self.train1.time_shift(10 * pq.ms)
        assert_arrays_equal(shifted.times, self.train1.times + 10 * pq.ms)

    def test__time_shift_different_units(self):
        shifted = self.train1.time_shift(1 * pq.s)
        assert_arrays_equal(shifted.times, self.train1.times + 1000 * pq.ms)


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.waveforms1 = np.array(
            [[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]],
             [[6., 7.], [6.1, 7.1]], [[8., 9.], [8.1, 9.1]], [[10., 11.], [10.1, 11.1]]]) * pq.mV
        self.data1 = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.data1quant = self.data1 * pq.ms
        self.arr_ann1 = {'index': np.arange(1, 7), 'label': ['a', 'b', 'c', 'd', 'e', 'f']}
        self.train1 = SpikeTrain(self.data1quant, t_stop=10.0 * pq.ms, waveforms=self.waveforms1,
                                 array_annotations=self.arr_ann1)

        self.waveforms2 = np.array(
            [[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]],
             [[6., 7.], [6.1, 7.1]], [[8., 9.], [8.1, 9.1]], [[10., 11.], [10.1, 11.1]]]) * pq.mV
        self.data2 = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.data2quant = self.data1 * pq.ms
        self.arr_ann2 = {'index': np.arange(101, 107), 'label2': ['g', 'h', 'i', 'j', 'k', 'l']}
        self.train2 = SpikeTrain(self.data1quant, t_stop=10.0 * pq.ms, waveforms=self.waveforms1,
                                 array_annotations=self.arr_ann2)

        self.segment = Segment()
        self.segment.spiketrains.extend([self.train1, self.train2])
        self.train1.segment = self.segment
        self.train2.segment = self.segment

    def test_compliant(self):
        assert_neo_object_is_compliant(self.train1)
        assert_neo_object_is_compliant(self.train2)

    def test_merge_typical(self):
        self.train1.waveforms = None
        self.train2.waveforms = None

        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(self.train2)

            self.assertTrue(len(w) == 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertSequenceEqual(str(w[0].message), "The following array annotations were "
                                                        "omitted, because they were only present"
                                                        " in one of the merged objects: "
                                                        "['label'] from the one that was merged "
                                                        "into and ['label2'] from the ones that "
                                                        "were merged into it.")

        assert_neo_object_is_compliant(result)

        # Make sure array annotations are merged correctly
        self.assertTrue('label' not in result.array_annotations)
        self.assertTrue('label2' not in result.array_annotations)
        assert_arrays_equal(result.array_annotations['index'],
                            np.array([1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6, 106]))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_merge_multiple(self):
        self.train1.waveforms = None

        train3 = self.train1.duplicate_with_new_data(self.train1.times.magnitude * pq.microsecond)
        train3.segment = self.train1.segment
        train3.array_annotate(index=np.arange(301, 307))

        train4 = self.train1.duplicate_with_new_data(self.train1.times / 2)
        train4.segment = self.train1.segment
        train4.array_annotate(index=np.arange(401, 407))

        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(train3, train4)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))

        assert_neo_object_is_compliant(result)

        self.assertEqual(len(result.shape), 1)
        self.assertEqual(result.shape[0], sum(len(st)
                                              for st in (self.train1, train3, train4)))

        self.assertEqual(self.train1.sampling_rate, result.sampling_rate)

        time_unit = result.units

        expected = np.concatenate((self.train1.rescale(time_unit).times,
                                   train3.rescale(time_unit).times,
                                   train4.rescale(time_unit).times))
        expected *= time_unit
        sorting = np.argsort(expected)
        expected = expected[sorting]
        np.testing.assert_array_equal(result.times, expected)

        # Make sure array annotations are merged correctly
        self.assertTrue('label' not in result.array_annotations)
        assert_arrays_equal(result.array_annotations['index'],
                            np.concatenate([st.array_annotations['index']
                                            for st in (self.train1, train3, train4)])[sorting])
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_merge_with_waveforms(self):
        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(self.train2)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))
        assert_neo_object_is_compliant(result)

    def test_merge_multiple_with_waveforms(self):
        train3 = self.train1.duplicate_with_new_data(self.train1.times.magnitude * pq.microsecond)
        train3.segment = self.train1.segment
        train3.array_annotate(index=np.arange(301, 307))
        train3.waveforms = self.train1.waveforms / 10

        train4 = self.train1.duplicate_with_new_data(self.train1.times / 2)
        train4.segment = self.train1.segment
        train4.array_annotate(index=np.arange(401, 407))
        train4.waveforms = self.train1.waveforms / 2

        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(train3, train4)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))

        assert_neo_object_is_compliant(result)
        self.assertEqual(len(result.shape), 1)
        self.assertEqual(result.shape[0], sum(len(st) for st in (self.train1, train3, train4)))

        time_unit = result.units

        expected = np.concatenate((self.train1.rescale(time_unit).times,
                                   train3.rescale(time_unit).times,
                                   train4.rescale(time_unit).times))
        sorting = np.argsort(expected)

        assert_arrays_equal(result.waveforms,
                            np.vstack([st.waveforms.rescale(self.train1.waveforms.units)
                                       for st in (self.train1, train3, train4)])[sorting]
                            * self.train1.waveforms.units)

    def test_correct_shape(self):
        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(self.train2)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))
        self.assertEqual(len(result.shape), 1)
        self.assertEqual(result.shape[0], self.train1.shape[0] + self.train2.shape[0])

    def test_correct_times(self):
        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(self.train2)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))
        expected = sorted(np.concatenate((self.train1.times, self.train2.times)))
        np.testing.assert_array_equal(result, expected)

        # Make sure array annotations are merged correctly
        self.assertTrue('label' not in result.array_annotations)
        self.assertTrue('label2' not in result.array_annotations)
        assert_arrays_equal(result.array_annotations['index'],
                            np.array([1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6, 106]))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_rescaling_units(self):
        train3 = self.train1.duplicate_with_new_data(self.train1.times.magnitude * pq.microsecond)
        train3.segment = self.train1.segment
        train3.array_annotate(**self.arr_ann1)
        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = train3.merge(self.train2)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))
        time_unit = result.units
        expected = sorted(np.concatenate(
            (train3.rescale(time_unit).times, self.train2.rescale(time_unit).times)))
        expected = expected * time_unit
        np.testing.assert_array_equal(result.rescale(time_unit), expected)

        # Make sure array annotations are merged correctly
        self.assertTrue('label' not in result.array_annotations)
        self.assertTrue('label2' not in result.array_annotations)
        assert_arrays_equal(result.array_annotations['index'],
                            np.array([1, 2, 3, 4, 5, 6, 101, 102, 103, 104, 105, 106]))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_name_file_origin_description(self):
        self.train1.waveforms = None
        self.train2.waveforms = None
        self.train1.name = 'name1'
        self.train1.description = 'desc1'
        self.train1.file_origin = 'file1'
        self.train2.name = 'name2'
        self.train2.description = 'desc2'
        self.train2.file_origin = 'file2'

        train3 = self.train1.duplicate_with_new_data(self.train1.times.magnitude * pq.microsecond)
        train3.segment = self.train1.segment
        train3.name = 'name3'
        train3.description = 'desc3'
        train3.file_origin = 'file3'

        train4 = self.train1.duplicate_with_new_data(self.train1.times / 2)
        train4.segment = self.train1.segment
        train4.name = 'name3'
        train4.description = 'desc3'
        train4.file_origin = 'file3'

        # merge two spiketrains with different attributes
        with warnings.catch_warnings(record=True) as w:
            merge1 = self.train1.merge(self.train2)
            self.assertTrue(len(w) > 0)

        self.assertEqual(merge1.name, 'merge(name1; name2)')
        self.assertEqual(merge1.description, 'merge(desc1; desc2)')
        self.assertEqual(merge1.file_origin, 'merge(file1; file2)')

        # merge a merged spiketrain with a regular one
        with warnings.catch_warnings(record=True) as w:
            merge2 = merge1.merge(train3)
            self.assertTrue(len(w) > 0)

        self.assertEqual(merge2.name, 'merge(name1; name2; name3)')
        self.assertEqual(merge2.description, 'merge(desc1; desc2; desc3)')
        self.assertEqual(merge2.file_origin, 'merge(file1; file2; file3)')

        # merge two merged spiketrains
        with warnings.catch_warnings(record=True) as w:
            merge3 = merge1.merge(merge2)
            self.assertTrue(len(w) > 0)

        self.assertEqual(merge3.name, 'merge(name1; name2; name3)')
        self.assertEqual(merge3.description, 'merge(desc1; desc2; desc3)')
        self.assertEqual(merge3.file_origin, 'merge(file1; file2; file3)')

        # merge two spiketrains with identical attributes
        with warnings.catch_warnings(record=True) as w:
            merge4 = train3.merge(train4)
            self.assertTrue(len(w) == 0)

        self.assertEqual(merge4.name, 'name3')
        self.assertEqual(merge4.description, 'desc3')
        self.assertEqual(merge4.file_origin, 'file3')

        # merge a reqular spiketrain with a merged spiketrain
        with warnings.catch_warnings(record=True) as w:
            merge5 = train3.merge(merge1)
            self.assertTrue(len(w) > 0)

        self.assertEqual(merge5.name, 'merge(name3; name1; name2)')
        self.assertEqual(merge5.description, 'merge(desc3; desc1; desc2)')
        self.assertEqual(merge5.file_origin, 'merge(file3; file1; file2)')

    def test_sampling_rate(self):
        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(self.train2)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))
        self.assertEqual(result.sampling_rate, self.train1.sampling_rate)

    def test_neo_relations(self):
        # Array annotations merge warning was already tested, can be ignored now
        with warnings.catch_warnings(record=True) as w:
            result = self.train1.merge(self.train2)
            self.assertEqual(len(w), 1)
            self.assertTrue("array annotations" in str(w[0].message))
        self.assertEqual(self.train1.segment, result.segment)
        # check if segment is linked bidirectionally
        self.assertTrue(any([result is r for r in result.segment.spiketrains]))

    def test_missing_waveforms_error(self):
        self.train1.waveforms = None
        with self.assertRaises(MergeError):
            self.train1.merge(self.train2)
        with self.assertRaises(MergeError):
            self.train2.merge(self.train1)

    def test_incompatible_t_start(self):
        train3 = self.train1.duplicate_with_new_data(self.train1, t_start=-1 * pq.s)
        train3.segment = self.train1.segment
        with self.assertRaises(MergeError):
            train3.merge(self.train2)
        with self.assertRaises(MergeError):
            self.train2.merge(train3)

    def test_merge_multiple_raise_merge_errors(self):
        # different t_start
        train3 = self.train1.duplicate_with_new_data(self.train1, t_start=-1 * pq.s)
        train3.segment = self.train1.segment
        with self.assertRaises(MergeError):
            train3.merge(self.train2, self.train1)
        with self.assertRaises(MergeError):
            self.train2.merge(train3, self.train1)

        # different t_stop
        train3 = self.train1.duplicate_with_new_data(self.train1, t_stop=133 * pq.s)
        train3.segment = self.train1.segment
        with self.assertRaises(MergeError):
            train3.merge(self.train2, self.train1)
        with self.assertRaises(MergeError):
            self.train2.merge(train3, self.train1)

        # different segment
        train3 = self.train1.duplicate_with_new_data(self.train1)
        seg = Segment()
        train3.segment = seg
        with self.assertRaises(MergeError):
            train3.merge(self.train2, self.train1)
        with self.assertRaises(MergeError):
            self.train2.merge(train3, self.train1)

        # missing waveforms
        train3 = self.train1.duplicate_with_new_data(self.train1)
        train3.waveforms = None
        with self.assertRaises(MergeError):
            train3.merge(self.train2, self.train1)
        with self.assertRaises(MergeError):
            self.train2.merge(train3, self.train1)

        # different sampling rate
        train3 = self.train1.duplicate_with_new_data(self.train1)
        train3.sampling_rate = 1 * pq.s
        with self.assertRaises(MergeError):
            train3.merge(self.train2, self.train1)
        with self.assertRaises(MergeError):
            self.train2.merge(train3, self.train1)

        # different left sweep
        train3 = self.train1.duplicate_with_new_data(self.train1)
        train3.left_sweep = 1 * pq.s
        with self.assertRaises(MergeError):
            train3.merge(self.train2, self.train1)
        with self.assertRaises(MergeError):
            self.train2.merge(train3, self.train1)


class TestDuplicateWithNewData(unittest.TestCase):
    def setUp(self):
        self.waveforms = np.array(
            [[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]],
             [[6., 7.], [6.1, 7.1]], [[8., 9.], [8.1, 9.1]], [[10., 11.], [10.1, 11.1]]]) * pq.mV
        self.data = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.dataquant = self.data * pq.ms
        self.arr_ann = {'index': np.arange(6)}
        self.train = SpikeTrain(self.dataquant, t_stop=10.0 * pq.ms, waveforms=self.waveforms,
                                array_annotations=self.arr_ann)

    def test_duplicate_with_new_data(self):
        signal1 = self.train
        new_t_start = -10 * pq.s
        new_t_stop = 10 * pq.s
        new_data = np.sort(np.random.uniform(new_t_start.magnitude, new_t_stop.magnitude,
                                             len(self.train))) * pq.ms

        signal1b = signal1.duplicate_with_new_data(new_data, t_start=new_t_start,
                                                   t_stop=new_t_stop)
        assert_arrays_almost_equal(np.asarray(signal1b), np.asarray(new_data), 1e-12)
        self.assertEqual(signal1b.t_start, new_t_start)
        self.assertEqual(signal1b.t_stop, new_t_stop)
        self.assertEqual(signal1b.sampling_rate, signal1.sampling_rate)
        # After duplicating, array annotations should always be empty,
        # because different length of data would cause inconsistencies
        self.assertEqual(signal1b.array_annotations, {})
        self.assertIsInstance(signal1b.array_annotations, ArrayDict)

    def test_deep_copy_attributes(self):
        signal1 = self.train
        new_t_start = -10 * pq.s
        new_t_stop = 10 * pq.s
        new_data = np.sort(np.random.uniform(new_t_start.magnitude, new_t_stop.magnitude,
                                             len(self.train))) * pq.ms

        signal1b = signal1.duplicate_with_new_data(new_data, t_start=new_t_start,
                                                   t_stop=new_t_stop)
        signal1.annotate(new_annotation='for signal 1')
        self.assertTrue('new_annotation' not in signal1b.annotations)


class TestAttributesAnnotations(unittest.TestCase):
    def test_set_universally_recommended_attributes(self):
        train = SpikeTrain([3, 4, 5], units='sec', name='Name', description='Desc',
                           file_origin='crack.txt', t_stop=99.9)
        assert_neo_object_is_compliant(train)
        self.assertEqual(train.name, 'Name')
        self.assertEqual(train.description, 'Desc')
        self.assertEqual(train.file_origin, 'crack.txt')

    def test_autoset_universally_recommended_attributes(self):
        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        self.assertEqual(train.name, None)
        self.assertEqual(train.description, None)
        self.assertEqual(train.file_origin, None)

    def test_annotations(self):
        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=11.1)
        assert_neo_object_is_compliant(train)
        self.assertEqual(train.annotations, {})

        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=11.1, ratname='Phillippe')
        assert_neo_object_is_compliant(train)
        self.assertEqual(train.annotations, {'ratname': 'Phillippe'})

    def test_array_annotations(self):
        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=11.1)
        assert_neo_object_is_compliant(train)
        self.assertEqual(train.array_annotations, {})
        self.assertIsInstance(train.array_annotations, ArrayDict)

        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=11.1,
                           array_annotations={'ratnames': ['L', 'N', 'E']})
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train.array_annotations['ratnames'], np.array(['L', 'N', 'E']))
        self.assertIsInstance(train.array_annotations, ArrayDict)

        train.array_annotate(index=[1, 2, 3])
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train.array_annotations['index'], np.arange(1, 4))
        self.assertIsInstance(train.array_annotations, ArrayDict)


class TestChanging(unittest.TestCase):
    def test_change_with_copy_default(self):
        # Default is copy = True
        # Changing spike train does not change data
        # Data source is quantity
        data = [3, 4, 5] * pq.s
        train = SpikeTrain(data, t_stop=100.0)
        train[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99 * pq.s)
        self.assertEqual(data[0], 3 * pq.s)

    def test_change_with_copy_false(self):
        # Changing spike train also changes data, because it is a view
        # Data source is quantity
        data = [3, 4, 5] * pq.s
        train = SpikeTrain(data, copy=False, t_stop=100.0)
        train[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99 * pq.s)
        self.assertEqual(data[0], 99 * pq.s)

    def test_change_with_copy_false_and_fake_rescale(self):
        # Changing spike train also changes data, because it is a view
        # Data source is quantity
        data = [3000, 4000, 5000] * pq.ms
        # even though we specify units, it still returns a view
        train = SpikeTrain(data, units='ms', copy=False, t_stop=100000)
        train[0] = 99000 * pq.ms
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99000 * pq.ms)
        self.assertEqual(data[0], 99000 * pq.ms)

    def test_change_with_copy_false_and_rescale_true(self):
        # When rescaling, a view cannot be returned
        # Changing spike train also changes data, because it is a view
        data = [3, 4, 5] * pq.s
        self.assertRaises(ValueError, SpikeTrain, data, units='ms', copy=False, t_stop=10000)

    def test_init_with_rescale(self):
        data = [3, 4, 5] * pq.s
        train = SpikeTrain(data, units='ms', t_stop=6000)
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 3000 * pq.ms)
        self.assertEqual(train._dimensionality, pq.ms._dimensionality)
        self.assertEqual(train.t_stop, 6000 * pq.ms)

    def test_change_with_copy_true(self):
        # Changing spike train does not change data
        # Data source is quantity
        data = [3, 4, 5] * pq.s
        train = SpikeTrain(data, copy=True, t_stop=100)
        train[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99 * pq.s)
        self.assertEqual(data[0], 3 * pq.s)

    def test_change_with_copy_default_and_data_not_quantity(self):
        # Default is copy = True
        # Changing spike train does not change data
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = [3, 4, 5]
        train = SpikeTrain(data, units='sec', t_stop=100)
        train[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99 * pq.s)
        self.assertEqual(data[0], 3 * pq.s)

    def test_change_with_copy_false_and_data_not_quantity(self):
        # Changing spike train also changes data, because it is a view
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = np.array([3, 4, 5])
        train = SpikeTrain(data, units='sec', copy=False, dtype=int, t_stop=101)
        train[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99 * pq.s)
        self.assertEqual(data[0], 99)

    def test_change_with_copy_false_and_dtype_change(self):
        # You cannot change dtype and request a view
        data = np.array([3, 4, 5])
        self.assertRaises(ValueError, SpikeTrain, data, units='sec', copy=False, t_stop=101,
                          dtype=np.float64)

    def test_change_with_copy_true_and_data_not_quantity(self):
        # Changing spike train does not change data
        # Data source is array
        # Array and quantity are tested separately because copy default
        # is different for these two.
        data = [3, 4, 5]
        train = SpikeTrain(data, units='sec', copy=True, t_stop=123.4)
        train[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99 * pq.s)
        self.assertEqual(data[0], 3)

    def test_changing_slice_changes_original_spiketrain(self):
        # If we slice a spiketrain and then change the slice, the
        # original spiketrain should change.
        # Whether the original data source changes is dependent on the
        # copy parameter.
        # This is compatible with both np and quantity default behavior.
        data = [3, 4, 5] * pq.s
        train = SpikeTrain(data, copy=True, t_stop=99.9)
        result = train[1:3]
        result[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[1], 99 * pq.s)
        self.assertEqual(result[0], 99 * pq.s)
        self.assertEqual(data[1], 4 * pq.s)

    def test_changing_slice_changes_original_spiketrain_with_copy_false(self):
        # If we slice a spiketrain and then change the slice, the
        # original spiketrain should change.
        # Whether the original data source changes is dependent on the
        # copy parameter.
        # This is compatible with both np and quantity default behavior.
        data = [3, 4, 5] * pq.s
        train = SpikeTrain(data, copy=False, t_stop=100.0)
        result = train[1:3]
        result[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        assert_neo_object_is_compliant(result)
        self.assertEqual(train[1], 99 * pq.s)
        self.assertEqual(result[0], 99 * pq.s)
        self.assertEqual(data[1], 99 * pq.s)

    def test__changing_spiketime_should_check_time_in_range(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        self.assertRaises(ValueError, train.__setitem__, 0, 10.1 * pq.ms)
        self.assertRaises(ValueError, train.__setitem__, 1, 5.0 * pq.s)
        self.assertRaises(ValueError, train.__setitem__, 2, 5.0 * pq.s)
        self.assertRaises(ValueError, train.__setitem__, 0, 0)

    def test__changing_multiple_spiketimes(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        train[:] = [7, 8, 9] * pq.ms
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train, np.array([7, 8, 9]))

    def test__changing_multiple_spiketimes_should_check_time_in_range(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)

    def test__adding_time_scalar(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        # t_start and t_stop are also changed
        self.assertEqual((train + 10 * pq.ms).t_start, 10.5 * pq.ms)
        self.assertEqual((train + 11 * pq.ms).t_stop, 21.0 * pq.ms)

        assert_arrays_equal(train + 1 * pq.ms, data + 1 * pq.ms)
        self.assertIsInstance(train + 10 * pq.ms, SpikeTrain)

    def test__adding_time_array(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        delta = [-2, 2, 4] * pq.ms
        assert_arrays_equal(train + delta, np.array([1, 6, 9]) * pq.ms)
        self.assertIsInstance(train + delta, SpikeTrain)
        # if new times are within t_start and t_stop, they
        # are not changed
        self.assertEqual((train + delta).t_start, train.t_start)
        self.assertEqual((train + delta).t_stop, train.t_stop)
        # if new times are outside t_start and/or t_stop, these are
        # expanded to fit
        delta = [-4, 2, 6] * pq.ms
        self.assertEqual((train + delta).t_start, -1 * pq.ms)
        self.assertEqual((train + delta).t_stop, 11 * pq.ms)

    def test__adding_two_spike_trains(self):
        data = [3, 4, 5] * pq.ms
        train1 = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        train2 = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        self.assertRaises(TypeError, train1.__add__, train2)

    def test__subtracting_time_scalar(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        # t_start and t_stop are also changed
        self.assertEqual((train - 1 * pq.ms).t_start, -0.5 * pq.ms)
        self.assertEqual((train - 3.0 * pq.ms).t_stop, 7.0 * pq.ms)
        assert_arrays_equal(train - 1 * pq.ms, data - 1 * pq.ms)
        self.assertIsInstance(train - 5 * pq.ms, SpikeTrain)

    def test__subtracting_time_array(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        delta = [2, 1, -2] * pq.ms
        self.assertIsInstance(train - delta, SpikeTrain)
        # if new times are within t_start and t_stop, they
        # are not changed
        self.assertEqual((train - delta).t_start, train.t_start)
        self.assertEqual((train - delta).t_stop, train.t_stop)
        # if new times are outside t_start and/or t_stop, these are
        # expanded to fit
        delta = [4, 1, -6] * pq.ms
        self.assertEqual((train - delta).t_start, -1 * pq.ms)
        self.assertEqual((train - delta).t_stop, 11 * pq.ms)

    def test__subtracting_two_spike_trains(self):
        train1 = SpikeTrain([3, 4, 5] * pq.ms, copy=False, t_start=0.5, t_stop=10.0)
        train2 = SpikeTrain([4, 5, 6] * pq.ms, copy=False, t_start=0.5, t_stop=10.0)
        train3 = SpikeTrain([3, 4, 5, 6] * pq.ms, copy=False, t_start=0.5, t_stop=10.0)
        self.assertRaises(TypeError, train1.__sub__, train3)
        self.assertRaises(TypeError, train3.__sub__, train1)
        self.assertIsInstance(train1 - train2, pq.Quantity)
        self.assertNotIsInstance(train1 - train2, SpikeTrain)

    def test__rescale(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, t_start=0.5, t_stop=10.0)
        train.segment = Segment()
        self.assertEqual(train.t_start.magnitude, 0.5)
        self.assertEqual(train.t_stop.magnitude, 10.0)
        result = train.rescale(pq.s)
        assert_neo_object_is_compliant(train)
        assert_neo_object_is_compliant(result)
        assert_arrays_equal(train, result)
        self.assertEqual(result.units, 1 * pq.s)
        self.assertIs(result.segment, train.segment)
        self.assertEqual(result.t_start.magnitude, 0.0005)
        self.assertEqual(result.t_stop.magnitude, 0.01)

    def test__rescale_same_units(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, t_start=0.5, t_stop=10.0)
        result = train.rescale(pq.ms)
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train, result)
        self.assertEqual(result.units, 1 * pq.ms)

    def test__rescale_incompatible_units_ValueError(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        self.assertRaises(ValueError, train.rescale, pq.m)


class TestPropertiesMethods(unittest.TestCase):
    def setUp(self):
        self.data1 = [3, 4, 5]
        self.data1quant = self.data1 * pq.ms
        self.waveforms1 = np.array(
            [[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]], [[4., 5.], [4.1, 5.1]]]) * pq.mV
        self.t_start1 = 0.5
        self.t_stop1 = 10.0
        self.t_start1quant = self.t_start1 * pq.ms
        self.t_stop1quant = self.t_stop1 * pq.ms
        self.sampling_rate1 = .1 * pq.Hz
        self.left_sweep1 = 2. * pq.s
        self.name1 = 'train 1'
        self.description1 = 'a test object'
        self.ann1 = {'targ0': [1, 2], 'targ1': 1.1}
        self.train1 = SpikeTrain(self.data1quant, t_start=self.t_start1, t_stop=self.t_stop1,
                                 waveforms=self.waveforms1, left_sweep=self.left_sweep1,
                                 sampling_rate=self.sampling_rate1, name=self.name1,
                                 description=self.description1, **self.ann1)

    def test__compliant(self):
        assert_neo_object_is_compliant(self.train1)

    def test__repr(self):
        result = repr(self.train1)
        if np.__version__.split(".")[:2] > ['1', '13']:
            # see https://github.com/numpy/numpy/blob/master/doc/release/1.14.0-notes.rst#many
            # -changes-to-array-printing-disableable-with-the-new-legacy-printing-mode  # nopep8
            targ = '<SpikeTrain(array([3., 4., 5.]) * ms, [0.5 ms, 10.0 ms])>'
        else:
            targ = '<SpikeTrain(array([ 3.,  4.,  5.]) * ms, [0.5 ms, 10.0 ms])>'
        self.assertEqual(result, targ)

    def test__duration(self):
        result1 = self.train1.duration

        self.train1.t_start = None
        assert_neo_object_is_compliant(self.train1)
        result2 = self.train1.duration

        self.train1.t_start = self.t_start1quant
        self.train1.t_stop = None
        assert_neo_object_is_compliant(self.train1)
        result3 = self.train1.duration

        self.assertEqual(result1, 9.5 * pq.ms)
        self.assertEqual(result1.units, 1. * pq.ms)
        self.assertEqual(result2, None)
        self.assertEqual(result3, None)

    def test__spike_duration(self):
        result1 = self.train1.spike_duration

        self.train1.sampling_rate = None
        assert_neo_object_is_compliant(self.train1)
        result2 = self.train1.spike_duration

        self.train1.sampling_rate = self.sampling_rate1
        self.train1.waveforms = None
        assert_neo_object_is_compliant(self.train1)
        result3 = self.train1.spike_duration

        self.assertEqual(result1, 20. / pq.Hz)
        self.assertEqual(result1.units, 1. / pq.Hz)
        self.assertEqual(result2, None)
        self.assertEqual(result3, None)

    def test__sampling_period(self):
        result1 = self.train1.sampling_period

        self.train1.sampling_rate = None
        assert_neo_object_is_compliant(self.train1)
        result2 = self.train1.sampling_period

        self.train1.sampling_rate = self.sampling_rate1
        self.train1.sampling_period = 10. * pq.ms
        assert_neo_object_is_compliant(self.train1)
        result3a = self.train1.sampling_period
        result3b = self.train1.sampling_rate

        self.train1.sampling_period = None
        result4a = self.train1.sampling_period
        result4b = self.train1.sampling_rate

        self.assertEqual(result1, 10. / pq.Hz)
        self.assertEqual(result1.units, 1. / pq.Hz)
        self.assertEqual(result2, None)
        self.assertEqual(result3a, 10. * pq.ms)
        self.assertEqual(result3a.units, 1. * pq.ms)
        self.assertEqual(result3b, .1 / pq.ms)
        self.assertEqual(result3b.units, 1. / pq.ms)
        self.assertEqual(result4a, None)
        self.assertEqual(result4b, None)

    def test__right_sweep(self):
        result1 = self.train1.right_sweep

        self.train1.left_sweep = None
        assert_neo_object_is_compliant(self.train1)
        result2 = self.train1.right_sweep

        self.train1.left_sweep = self.left_sweep1
        self.train1.sampling_rate = None
        assert_neo_object_is_compliant(self.train1)
        result3 = self.train1.right_sweep

        self.train1.sampling_rate = self.sampling_rate1
        self.train1.waveforms = None
        assert_neo_object_is_compliant(self.train1)
        result4 = self.train1.right_sweep

        self.assertEqual(result1, 22. * pq.s)
        self.assertEqual(result1.units, 1. * pq.s)
        self.assertEqual(result2, None)
        self.assertEqual(result3, None)
        self.assertEqual(result4, None)

    def test__times(self):
        result1 = self.train1.times
        self.assertIsInstance(result1, pq.Quantity)
        self.assertTrue((result1 == self.train1).all)
        self.assertEqual(len(result1), len(self.train1))
        self.assertEqual(result1.units, self.train1.units)
        self.assertEqual(result1.dtype, self.train1.dtype)

    def test__children(self):
        segment = Segment(name='seg1')
        segment.spiketrains = [self.train1]
        segment.create_many_to_one_relationship()

        self.assertEqual(self.train1._parent_objects, ('Segment',))

        self.assertEqual(self.train1._parent_containers, ('segment',))

        self.assertEqual(self.train1._parent_objects, ('Segment',))
        self.assertEqual(self.train1._parent_containers, ('segment',))

        self.assertEqual(len(self.train1.parents), 1)
        self.assertEqual(self.train1.parents[0].name, 'seg1')

        assert_neo_object_is_compliant(self.train1)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        res = pretty(self.train1)
        targ = ("SpikeTrain\n" + "name: '%s'\ndescription: '%s'\nannotations: %s"
                                 "" % (self.name1, self.description1, pretty(self.ann1)))
        self.assertEqual(res, targ)


class TestMiscellaneous(unittest.TestCase):
    def test__different_dtype_for_t_start_and_array(self):
        data = np.array([0, 9.9999999], dtype=np.float64) * pq.s
        data16 = data.astype(np.float16)
        data32 = data.astype(np.float32)
        data64 = data.astype(np.float64)
        t_start = data[0]
        t_stop = data[1]
        t_start16 = data[0].astype(dtype=np.float16)
        t_stop16 = data[1].astype(dtype=np.float16)
        t_start32 = data[0].astype(dtype=np.float32)
        t_stop32 = data[1].astype(dtype=np.float32)
        t_start64 = data[0].astype(dtype=np.float64)
        t_stop64 = data[1].astype(dtype=np.float64)
        t_start_custom = 0.0
        t_stop_custom = 10.0
        t_start_custom16 = np.array(t_start_custom, dtype=np.float16)
        t_stop_custom16 = np.array(t_stop_custom, dtype=np.float16)
        t_start_custom32 = np.array(t_start_custom, dtype=np.float32)
        t_stop_custom32 = np.array(t_stop_custom, dtype=np.float32)
        t_start_custom64 = np.array(t_start_custom, dtype=np.float64)
        t_stop_custom64 = np.array(t_stop_custom, dtype=np.float64)

        # This is OK.
        train = SpikeTrain(data64, copy=True, t_start=t_start, t_stop=t_stop)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop, dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop, dtype=np.float32)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop, dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop, dtype=np.float32)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16, dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16, dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start16, t_stop=t_stop16, dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32, dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32, dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start32, t_stop=t_stop32, dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64, dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64, dtype=np.float32)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start_custom, t_stop=t_stop_custom)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start_custom16, t_stop=t_stop_custom16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom16, t_stop=t_stop_custom16,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom16, t_stop=t_stop_custom16,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom16, t_stop=t_stop_custom16,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start_custom32, t_stop=t_stop_custom32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom32, t_stop=t_stop_custom32,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom32, t_stop=t_stop_custom32,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom32, t_stop=t_stop_custom32,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start_custom64, t_stop=t_stop_custom64)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom64, t_stop=t_stop_custom64,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom64, t_stop=t_stop_custom64,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start_custom64, t_stop=t_stop_custom64,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        # This use to bug - see ticket #38
        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop, dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop, dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start64, t_stop=t_stop64, dtype=np.float64)
        assert_neo_object_is_compliant(train)

    def test_as_array(self):
        data = np.arange(10.0)
        st = SpikeTrain(data, t_stop=10.0, units='ms')
        st_as_arr = st.as_array()
        self.assertIsInstance(st_as_arr, np.ndarray)
        assert_array_equal(data, st_as_arr)

    def test_as_quantity(self):
        data = np.arange(10.0)
        st = SpikeTrain(data, t_stop=10.0, units='ms')
        st_as_q = st.as_quantity()
        self.assertIsInstance(st_as_q, pq.Quantity)
        assert_array_equal(data * pq.ms, st_as_q)


if __name__ == "__main__":
    unittest.main()
