# -*- coding: utf-8 -*-
"""
Tests of the neo.core.spiketrain.SpikeTrain class and related functions
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

import unittest

import numpy as np
from numpy.testing import assert_array_equal
import quantities as pq

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.spiketrain import (check_has_dimensions_time, SpikeTrain,
                                 _check_time_in_range, _new_spiketrain)
from neo.core import Segment, Unit
from neo.core.baseneo import MergeError
from neo.test.tools import (assert_arrays_equal,
                            assert_arrays_almost_equal,
                            assert_neo_object_is_compliant)
from neo.test.generate_datasets import (get_fake_value, get_fake_values,
                                        fake_neo, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__get_fake_values(self):
        self.annotations['seed'] = 0
        waveforms = get_fake_value('waveforms', pq.Quantity, seed=3, dim=3)
        shape = waveforms.shape[0]
        times = get_fake_value('times', pq.Quantity, seed=0, dim=1,
                               shape=waveforms.shape[0])
        t_start = get_fake_value('t_start', pq.Quantity, seed=1, dim=0)
        t_stop = get_fake_value('t_stop', pq.Quantity, seed=2, dim=0)
        left_sweep = get_fake_value('left_sweep', pq.Quantity, seed=4, dim=0)
        sampling_rate = get_fake_value('sampling_rate', pq.Quantity,
                                       seed=5, dim=0)
        name = get_fake_value('name', str, seed=6, obj=SpikeTrain)
        description = get_fake_value('description', str,
                                     seed=7, obj='SpikeTrain')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(SpikeTrain, annotate=False, seed=0)
        res12 = get_fake_values('SpikeTrain', annotate=False, seed=0)
        res21 = get_fake_values(SpikeTrain, annotate=True, seed=0)
        res22 = get_fake_values('SpikeTrain', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('times'), times)
        assert_arrays_equal(res12.pop('times'), times)
        assert_arrays_equal(res21.pop('times'), times)
        assert_arrays_equal(res22.pop('times'), times)

        assert_arrays_equal(res11.pop('t_start'), t_start)
        assert_arrays_equal(res12.pop('t_start'), t_start)
        assert_arrays_equal(res21.pop('t_start'), t_start)
        assert_arrays_equal(res22.pop('t_start'), t_start)

        assert_arrays_equal(res11.pop('t_stop'), t_stop)
        assert_arrays_equal(res12.pop('t_stop'), t_stop)
        assert_arrays_equal(res21.pop('t_stop'), t_stop)
        assert_arrays_equal(res22.pop('t_stop'), t_stop)

        assert_arrays_equal(res11.pop('waveforms'), waveforms)
        assert_arrays_equal(res12.pop('waveforms'), waveforms)
        assert_arrays_equal(res21.pop('waveforms'), waveforms)
        assert_arrays_equal(res22.pop('waveforms'), waveforms)

        assert_arrays_equal(res11.pop('left_sweep'), left_sweep)
        assert_arrays_equal(res12.pop('left_sweep'), left_sweep)
        assert_arrays_equal(res21.pop('left_sweep'), left_sweep)
        assert_arrays_equal(res22.pop('left_sweep'), left_sweep)

        assert_arrays_equal(res11.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res12.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res21.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res22.pop('sampling_rate'), sampling_rate)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'SpikeTrain'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, SpikeTrain))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = SpikeTrain
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, SpikeTrain))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


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


class Testcheck_time_in_range(unittest.TestCase):
    def test__check_time_in_range_empty_array(self):
        value = np.array([])
        t_start = 0 * pq.s
        t_stop = 10 * pq.s
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=False)
        _check_time_in_range(value, t_start=t_start, t_stop=t_stop, view=True)

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
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=False)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=True)

    def test__check_time_in_range_below_scale(self):
        value = np.array([-1., 5000., 10000.]) * pq.ms
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=False)

    def test__check_time_in_range_above(self):
        value = np.array([0., 5., 10.1]) * pq.s
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=False)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=True)

    def test__check_time_in_range_above_scale(self):
        value = np.array([0., 5000., 10001.]) * pq.ms
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=False)

    def test__check_time_in_range_above_below(self):
        value = np.array([-0.1, 5., 10.1]) * pq.s
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=False)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=True)

    def test__check_time_in_range_above_below_scale(self):
        value = np.array([-1., 5000., 10001.]) * pq.ms
        t_start = 0. * pq.s
        t_stop = 10. * pq.s
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _check_time_in_range, value,
                          t_start=t_start, t_stop=t_stop, view=False)


class TestConstructor(unittest.TestCase):
    def result_spike_check(self, train, st_out, t_start_out, t_stop_out,
                           dtype, units):
        assert_arrays_equal(train, st_out)
        assert_arrays_equal(train, train.times)
        assert_neo_object_is_compliant(train)

        self.assertEqual(train.t_start, t_start_out)
        self.assertEqual(train.t_start, train.times.t_start)
        self.assertEqual(train.t_stop, t_stop_out)
        self.assertEqual(train.t_stop, train.times.t_stop)

        self.assertEqual(train.units, units)
        self.assertEqual(train.units, train.times.units)
        self.assertEqual(train.t_start.units, units)
        self.assertEqual(train.t_start.units, train.times.t_start.units)
        self.assertEqual(train.t_stop.units, units)
        self.assertEqual(train.t_stop.units, train.times.t_stop.units)

        self.assertEqual(train.dtype, dtype)
        self.assertEqual(train.dtype, train.times.dtype)
        self.assertEqual(train.t_stop.dtype, dtype)
        self.assertEqual(train.t_stop.dtype, train.times.t_stop.dtype)
        self.assertEqual(train.t_start.dtype, dtype)
        self.assertEqual(train.t_start.dtype, train.times.t_start.dtype)

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
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_empty(self):
        t_start = 0.0
        t_stop = 10.0
        train1 = SpikeTrain([], t_start=t_start, t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, [], t_start=t_start,
                                 t_stop=t_stop, units='s')

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = [] * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

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
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list(self):
        times = range(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms")
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units="ms")

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_set_dtype(self):
        times = range(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            units="ms", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 units="ms", dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = pq.Quantity(times, units=units, dtype=dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_no_start_stop_units(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="ms")
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units="ms")

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_no_start_stop_units_set_dtype(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            units="ms", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 units="ms", dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = pq.Quantity(times, units=units, dtype=dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array(self):
        times = np.arange(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units="s")

        dtype = np.int
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_with_dtype(self):
        times = np.arange(10, dtype='f4')
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units="s")

        dtype = times.dtype
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_set_dtype(self):
        times = np.arange(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            units="s", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 units="s", dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype) * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_no_start_stop_units(self):
        times = np.arange(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units="s")

        dtype = np.int
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_no_start_stop_units_with_dtype(self):
        times = np.arange(10, dtype='f4')
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units="s")
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units="s")

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_array_no_start_stop_units_set_dtype(self):
        times = np.arange(10)
        t_start = 0.0
        t_stop = 10000.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            units="s", dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 units="s", dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype) * units
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop)

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_with_dtype(self):
        times = np.arange(10, dtype='f4') * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop)

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.astype(dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_no_start_stop_units(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop)

        dtype = np.float64
        units = 1 * pq.ms
        t_start_out = t_start * units
        t_stop_out = t_stop * units
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_no_start_stop_units_with_dtype(self):
        times = np.arange(10, dtype='f4') * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop)

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_no_start_stop_units_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 dtype='f4')

        dtype = np.float32
        units = 1 * pq.ms
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_units(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units='s')

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = t_start
        t_stop_out = t_stop
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_units_with_dtype(self):
        times = np.arange(10, dtype='f4') * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            units='s')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units='s')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.rescale(units).astype(dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_units_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0 * pq.s
        t_stop = 12.0 * pq.ms
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            units='s', dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 units='s', dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = t_start.astype(dtype)
        t_stop_out = t_stop.rescale(units).astype(dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_array_units_no_start_stop_units(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop, units='s')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop, units='s')

        dtype = np.float64
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_quantity_units_no_start_stop_units_set_dtype(self):
        times = np.arange(10) * pq.ms
        t_start = 0.0
        t_stop = 12.0
        train1 = SpikeTrain(times, t_start=t_start, t_stop=t_stop,
                            units='s', dtype='f4')
        train2 = _new_spiketrain(SpikeTrain, times,
                                 t_start=t_start, t_stop=t_stop,
                                 units='s', dtype='f4')

        dtype = np.float32
        units = 1 * pq.s
        t_start_out = pq.Quantity(t_start, units=units, dtype=dtype)
        t_stop_out = pq.Quantity(t_stop, units=units, dtype=dtype)
        st_out = times.rescale(units).astype(dtype)
        self.result_spike_check(train1, st_out, t_start_out, t_stop_out,
                                dtype, units)
        self.result_spike_check(train2, st_out, t_start_out, t_stop_out,
                                dtype, units)

    def test__create_from_list_without_units_should_raise_ValueError(self):
        times = range(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        self.assertRaises(ValueError, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)

    def test__create_from_array_without_units_should_raise_ValueError(self):
        times = np.arange(10)
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        self.assertRaises(ValueError, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)

    def test__create_from_array_with_incompatible_units_ValueError(self):
        times = np.arange(10) * pq.km
        t_start = 0.0 * pq.s
        t_stop = 10000.0 * pq.ms
        self.assertRaises(ValueError, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain, times,
                          t_start=t_start, t_stop=t_stop)

    def test__create_with_times_outside_tstart_tstop_ValueError(self):
        t_start = 23
        t_stop = 77
        train1 = SpikeTrain(np.arange(t_start, t_stop), units='ms',
                            t_start=t_start, t_stop=t_stop)
        train2 = _new_spiketrain(SpikeTrain,
                                 np.arange(t_start, t_stop), units='ms',
                                 t_start=t_start, t_stop=t_stop)
        assert_neo_object_is_compliant(train1)
        assert_neo_object_is_compliant(train2)
        self.assertRaises(ValueError, SpikeTrain,
                          np.arange(t_start - 5, t_stop), units='ms',
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain,
                          np.arange(t_start - 5, t_stop), units='ms',
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, SpikeTrain,
                          np.arange(t_start, t_stop + 5), units='ms',
                          t_start=t_start, t_stop=t_stop)
        self.assertRaises(ValueError, _new_spiketrain, SpikeTrain,
                          np.arange(t_start, t_stop + 5), units='ms',
                          t_start=t_start, t_stop=t_stop)

    def test__create_with_len_times_different_size_than_waveform_shape1_ValueError(
            self):
        self.assertRaises(ValueError, SpikeTrain,
                          times=np.arange(10), units='s',
                          t_stop=4, waveforms=np.ones((10, 6, 50)))

    def test_defaults(self):
        # default recommended attributes
        train1 = SpikeTrain([3, 4, 5], units='sec', t_stop=10.0)
        train2 = _new_spiketrain(SpikeTrain, [3, 4, 5],
                                 units='sec', t_stop=10.0)
        assert_neo_object_is_compliant(train1)
        assert_neo_object_is_compliant(train2)
        self.assertEqual(train1.dtype, np.float)
        self.assertEqual(train2.dtype, np.float)
        self.assertEqual(train1.sampling_rate, 1.0 * pq.Hz)
        self.assertEqual(train2.sampling_rate, 1.0 * pq.Hz)
        self.assertEqual(train1.waveforms, None)
        self.assertEqual(train2.waveforms, None)
        self.assertEqual(train1.left_sweep, None)
        self.assertEqual(train2.left_sweep, None)

    def test_default_tstart(self):
        # t start defaults to zero
        train11 = SpikeTrain([3, 4, 5] * pq.s, t_stop=8000 * pq.ms)
        train21 = _new_spiketrain(SpikeTrain, [3, 4, 5] * pq.s,
                                  t_stop=8000 * pq.ms)
        assert_neo_object_is_compliant(train11)
        assert_neo_object_is_compliant(train21)
        self.assertEqual(train11.t_start, 0. * pq.s)
        self.assertEqual(train21.t_start, 0. * pq.s)

        # unless otherwise specified
        train12 = SpikeTrain([3, 4, 5] * pq.s, t_start=2.0, t_stop=8)
        train22 = _new_spiketrain(SpikeTrain, [3, 4, 5] * pq.s,
                                  t_start=2.0, t_stop=8)
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
        train22 = _new_spiketrain(SpikeTrain, [3, 5, 4] * pq.s,
                                  t_stop=10000. * pq.ms)
        assert_neo_object_is_compliant(train12)
        assert_neo_object_is_compliant(train22)
        self.assertEqual(train12.t_stop, 10. * pq.s)
        self.assertEqual(train22.t_stop, 10. * pq.s)

        train13 = SpikeTrain([3, 5, 4], units='sec', t_stop=10000. * pq.ms)
        train23 = _new_spiketrain(SpikeTrain, [3, 5, 4],
                                  units='sec', t_stop=10000. * pq.ms)
        assert_neo_object_is_compliant(train13)
        assert_neo_object_is_compliant(train23)
        self.assertEqual(train13.t_stop, 10. * pq.s)
        self.assertEqual(train23.t_stop, 10. * pq.s)


class TestSorting(unittest.TestCase):
    def test_sort(self):
        waveforms = np.array([[[0., 1.]], [[2., 3.]], [[4., 5.]]]) * pq.mV
        train = SpikeTrain([3, 4, 5] * pq.s, waveforms=waveforms, name='n',
                           t_stop=10.0)
        assert_neo_object_is_compliant(train)
        train.sort()
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train, [3, 4, 5] * pq.s)
        assert_arrays_equal(train.waveforms, waveforms)
        self.assertEqual(train.name, 'n')
        self.assertEqual(train.t_stop, 10.0 * pq.s)

        train = SpikeTrain([3, 5, 4] * pq.s, waveforms=waveforms, name='n',
                           t_stop=10.0)
        assert_neo_object_is_compliant(train)
        train.sort()
        assert_neo_object_is_compliant(train)
        assert_arrays_equal(train, [3, 4, 5] * pq.s)
        assert_arrays_equal(train.waveforms, waveforms[[0, 2, 1]])
        self.assertEqual(train.name, 'n')
        self.assertEqual(train.t_start, 0.0 * pq.s)
        self.assertEqual(train.t_stop, 10.0 * pq.s)


class TestSlice(unittest.TestCase):
    def setUp(self):
        self.waveforms1 = np.array([[[0., 1.],
                                     [0.1, 1.1]],
                                    [[2., 3.],
                                     [2.1, 3.1]],
                                    [[4., 5.],
                                     [4.1, 5.1]]]) * pq.mV
        self.data1 = np.array([3, 4, 5])
        self.data1quant = self.data1 * pq.s
        self.train1 = SpikeTrain(self.data1quant, waveforms=self.waveforms1,
                                 name='n', arb='arbb', t_stop=10.0)

    def test_compliant(self):
        assert_neo_object_is_compliant(self.train1)

    def test_slice(self):
        # slice spike train, keep sliced spike times
        result = self.train1[1:2]
        assert_arrays_equal(self.train1[1:2], result)
        targwaveforms = np.array([[[2., 3.],
                                   [2.1, 3.1]]]) * pq.mV

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

    def test_slice_to_end(self):
        # slice spike train, keep sliced spike times
        result = self.train1[1:]
        assert_arrays_equal(self.train1[1:], result)
        targwaveforms = np.array([[[2., 3.],
                                   [2.1, 3.1]],
                                  [[4., 5.],
                                   [4.1, 5.1]]]) * pq.mV

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

    def test_slice_from_beginning(self):
        # slice spike train, keep sliced spike times
        result = self.train1[:2]
        assert_arrays_equal(self.train1[:2], result)
        targwaveforms = np.array([[[0., 1.],
                                   [0.1, 1.1]],
                                  [[2., 3.],
                                   [2.1, 3.1]]]) * pq.mV

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

    def test_slice_negative_idxs(self):
        # slice spike train, keep sliced spike times
        result = self.train1[:-1]
        assert_arrays_equal(self.train1[:-1], result)
        targwaveforms = np.array([[[0., 1.],
                                   [0.1, 1.1]],
                                  [[2., 3.],
                                   [2.1, 3.1]]]) * pq.mV

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


class TestTimeSlice(unittest.TestCase):
    def setUp(self):
        self.waveforms1 = np.array([[[0., 1.],
                                     [0.1, 1.1]],
                                    [[2., 3.],
                                     [2.1, 3.1]],
                                    [[4., 5.],
                                     [4.1, 5.1]],
                                    [[6., 7.],
                                     [6.1, 7.1]],
                                    [[8., 9.],
                                     [8.1, 9.1]],
                                    [[10., 11.],
                                     [10.1, 11.1]]]) * pq.mV
        self.data1 = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.data1quant = self.data1 * pq.ms
        self.train1 = SpikeTrain(self.data1quant, t_stop=10.0 * pq.ms,
                                 waveforms=self.waveforms1)

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
        targwaveforms = np.array([[[2., 3.],
                                   [2.1, 3.1]],
                                  [[4., 5.],
                                   [4.1, 5.1]],
                                  [[6., 7.],
                                   [6.1, 7.1]]]) * pq.mV
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

    def test_time_slice_differnt_units(self):
        # time_slice spike train, keep sliced spike times
        t_start = 0.00012 * pq.s
        t_stop = 0.0035 * pq.s
        result = self.train1.time_slice(t_start, t_stop)
        targ = SpikeTrain([0.5, 1.2, 3.3] * pq.ms, t_stop=3.3)
        assert_arrays_equal(result, targ)
        targwaveforms = np.array([[[2., 3.],
                                   [2.1, 3.1]],
                                  [[4., 5.],
                                   [4.1, 5.1]],
                                  [[6., 7.],
                                   [6.1, 7.1]]]) * pq.mV
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

    def test_time_slice_out_of_boundries(self):
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
        targwaveforms = np.array([[[4., 5.],
                                   [4.1, 5.1]],
                                  [[6., 7.],
                                   [6.1, 7.1]],
                                  [[8., 9.],
                                   [8.1, 9.1]],
                                  [[10., 11.],
                                   [10.1, 11.1]]]) * pq.mV
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

    def test_time_slice_none_start(self):
        # time_slice spike train, keep sliced spike times
        t_stop = 1 * pq.ms
        result = self.train1.time_slice(None, t_stop)
        assert_arrays_equal([0.1, 0.5] * pq.ms, result)
        targwaveforms = np.array([[[0., 1.],
                                   [0.1, 1.1]],
                                  [[2., 3.],
                                   [2.1, 3.1]]]) * pq.mV
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


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.waveforms1 = np.array([[[0., 1.],
                                     [0.1, 1.1]],
                                    [[2., 3.],
                                     [2.1, 3.1]],
                                    [[4., 5.],
                                     [4.1, 5.1]],
                                    [[6., 7.],
                                     [6.1, 7.1]],
                                    [[8., 9.],
                                     [8.1, 9.1]],
                                    [[10., 11.],
                                     [10.1, 11.1]]]) * pq.mV
        self.data1 = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.data1quant = self.data1 * pq.ms
        self.train1 = SpikeTrain(self.data1quant, t_stop=10.0 * pq.ms,
                                 waveforms=self.waveforms1)

        self.waveforms2 = np.array([[[0., 1.],
                                     [0.1, 1.1]],
                                    [[2., 3.],
                                     [2.1, 3.1]],
                                    [[4., 5.],
                                     [4.1, 5.1]],
                                    [[6., 7.],
                                     [6.1, 7.1]],
                                    [[8., 9.],
                                     [8.1, 9.1]],
                                    [[10., 11.],
                                     [10.1, 11.1]]]) * pq.mV
        self.data2 = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.data2quant = self.data1 * pq.ms
        self.train2 = SpikeTrain(self.data1quant, t_stop=10.0 * pq.ms,
                                 waveforms=self.waveforms1)

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

        result = self.train1.merge(self.train2)
        assert_neo_object_is_compliant(result)

    def test_merge_with_waveforms(self):
        result = self.train1.merge(self.train2)
        assert_neo_object_is_compliant(result)

    def test_correct_shape(self):
        result = self.train1.merge(self.train2)
        self.assertEqual(len(result.shape), 1)
        self.assertEqual(result.shape[0],
                         self.train1.shape[0] + self.train2.shape[0])

    def test_correct_times(self):
        result = self.train1.merge(self.train2)
        expected = sorted(np.concatenate((self.train1.times,
                                          self.train2.times)))
        np.testing.assert_array_equal(result, expected)

    def test_rescaling_units(self):
        train3 = self.train1.duplicate_with_new_data(
            self.train1.times.magnitude * pq.microsecond)
        train3.segment = self.train1.segment
        result = train3.merge(self.train2)
        time_unit = result.units
        expected = sorted(np.concatenate((train3.rescale(time_unit).times,
                                          self.train2.rescale(
                                              time_unit).times)))
        expected = expected * time_unit
        np.testing.assert_array_equal(result.rescale(time_unit), expected)

    def test_sampling_rate(self):
        result = self.train1.merge(self.train2)
        self.assertEqual(result.sampling_rate, self.train1.sampling_rate)

    def test_neo_relations(self):
        result = self.train1.merge(self.train2)
        self.assertEqual(self.train1.segment, result.segment)
        self.assertTrue(result in result.segment.spiketrains)

    def test_missing_waveforms_error(self):
        self.train1.waveforms = None
        with self.assertRaises(MergeError):
            self.train1.merge(self.train2)
        with self.assertRaises(MergeError):
            self.train2.merge(self.train1)

    def test_incompatible_t_start(self):
        train3 = self.train1.duplicate_with_new_data(self.train1,
                                                     t_start=-1 * pq.s)
        train3.segment = self.train1.segment
        with self.assertRaises(MergeError):
            train3.merge(self.train2)
        with self.assertRaises(MergeError):
            self.train2.merge(train3)


class TestDuplicateWithNewData(unittest.TestCase):
    def setUp(self):
        self.waveforms = np.array([[[0., 1.],
                                    [0.1, 1.1]],
                                   [[2., 3.],
                                    [2.1, 3.1]],
                                   [[4., 5.],
                                    [4.1, 5.1]],
                                   [[6., 7.],
                                    [6.1, 7.1]],
                                   [[8., 9.],
                                    [8.1, 9.1]],
                                   [[10., 11.],
                                    [10.1, 11.1]]]) * pq.mV
        self.data = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.dataquant = self.data * pq.ms
        self.train = SpikeTrain(self.dataquant, t_stop=10.0 * pq.ms,
                                waveforms=self.waveforms)

    def test_duplicate_with_new_data(self):
        signal1 = self.train
        new_t_start = -10 * pq.s
        new_t_stop = 10 * pq.s
        new_data = np.sort(np.random.uniform(new_t_start.magnitude,
                                             new_t_stop.magnitude,
                                             len(self.train))) * pq.ms

        signal1b = signal1.duplicate_with_new_data(new_data,
                                                   t_start=new_t_start,
                                                   t_stop=new_t_stop)
        assert_arrays_almost_equal(np.asarray(signal1b),
                                   np.asarray(new_data), 1e-12)
        self.assertEqual(signal1b.t_start, new_t_start)
        self.assertEqual(signal1b.t_stop, new_t_stop)
        self.assertEqual(signal1b.sampling_rate, signal1.sampling_rate)

    def test_deep_copy_attributes(self):
        signal1 = self.train
        new_t_start = -10*pq.s
        new_t_stop = 10*pq.s
        new_data = np.sort(np.random.uniform(new_t_start.magnitude,
                                             new_t_stop.magnitude,
                                             len(self.train))) * pq.ms

        signal1b = signal1.duplicate_with_new_data(new_data,
                                                   t_start=new_t_start,
                                                   t_stop=new_t_stop)
        signal1.annotate(new_annotation='for signal 1')
        self.assertTrue('new_annotation' not in signal1b.annotations)

class TestAttributesAnnotations(unittest.TestCase):
    def test_set_universally_recommended_attributes(self):
        train = SpikeTrain([3, 4, 5], units='sec', name='Name',
                           description='Desc', file_origin='crack.txt',
                           t_stop=99.9)
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
        self.assertRaises(ValueError, SpikeTrain, data, units='ms',
                          copy=False, t_stop=10000)

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
        train = SpikeTrain(data, units='sec', copy=False, dtype=np.int,
                           t_stop=101)
        train[0] = 99 * pq.s
        assert_neo_object_is_compliant(train)
        self.assertEqual(train[0], 99 * pq.s)
        self.assertEqual(data[0], 99)

    def test_change_with_copy_false_and_dtype_change(self):
        # You cannot change dtype and request a view
        data = np.array([3, 4, 5])
        self.assertRaises(ValueError, SpikeTrain, data, units='sec',
                          copy=False, t_stop=101, dtype=np.float64)

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
        if sys.version_info[0] == 2:
            self.assertRaises(ValueError, train.__setslice__,
                              0, 3, [3, 4, 11] * pq.ms)
            self.assertRaises(ValueError, train.__setslice__,
                              0, 3, [0, 4, 5] * pq.ms)

    def test__adding_time(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        self.assertRaises(ValueError, train.__add__, 10 * pq.ms)
        assert_arrays_equal(train + 1 * pq.ms, data + 1 * pq.ms)

    def test__subtracting_time(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, copy=False, t_start=0.5, t_stop=10.0)
        assert_neo_object_is_compliant(train)
        self.assertRaises(ValueError, train.__sub__, 10 * pq.ms)
        assert_arrays_equal(train - 1 * pq.ms, data - 1 * pq.ms)

    def test__rescale(self):
        data = [3, 4, 5] * pq.ms
        train = SpikeTrain(data, t_start=0.5, t_stop=10.0)
        train.segment = Segment()
        train.unit = Unit()
        result = train.rescale(pq.s)
        assert_neo_object_is_compliant(train)
        assert_neo_object_is_compliant(result)
        assert_arrays_equal(train, result)
        self.assertEqual(result.units, 1 * pq.s)
        self.assertIs(result.segment, train.segment)
        self.assertIs(result.unit, train.unit)


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
        self.waveforms1 = np.array([[[0., 1.],
                                     [0.1, 1.1]],
                                    [[2., 3.],
                                     [2.1, 3.1]],
                                    [[4., 5.],
                                     [4.1, 5.1]]]) * pq.mV
        self.t_start1 = 0.5
        self.t_stop1 = 10.0
        self.t_start1quant = self.t_start1 * pq.ms
        self.t_stop1quant = self.t_stop1 * pq.ms
        self.sampling_rate1 = .1 * pq.Hz
        self.left_sweep1 = 2. * pq.s
        self.name1 = 'train 1'
        self.description1 = 'a test object'
        self.ann1 = {'targ0': [1, 2], 'targ1': 1.1}
        self.train1 = SpikeTrain(self.data1quant,
                                 t_start=self.t_start1, t_stop=self.t_stop1,
                                 waveforms=self.waveforms1,
                                 left_sweep=self.left_sweep1,
                                 sampling_rate=self.sampling_rate1,
                                 name=self.name1,
                                 description=self.description1,
                                 **self.ann1)

    def test__compliant(self):
        assert_neo_object_is_compliant(self.train1)

    def test__repr(self):
        result = repr(self.train1)
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

    def test__children(self):
        segment = Segment(name='seg1')
        segment.spiketrains = [self.train1]
        segment.create_many_to_one_relationship()

        unit = Unit(name='unit1')
        unit.spiketrains = [self.train1]
        unit.create_many_to_one_relationship()

        self.assertEqual(self.train1._single_parent_objects,
                         ('Segment', 'Unit'))
        self.assertEqual(self.train1._multi_parent_objects, ())

        self.assertEqual(self.train1._single_parent_containers,
                         ('segment', 'unit'))
        self.assertEqual(self.train1._multi_parent_containers, ())

        self.assertEqual(self.train1._parent_objects,
                         ('Segment', 'Unit'))
        self.assertEqual(self.train1._parent_containers,
                         ('segment', 'unit'))

        self.assertEqual(len(self.train1.parents), 2)
        self.assertEqual(self.train1.parents[0].name, 'seg1')
        self.assertEqual(self.train1.parents[1].name, 'unit1')

        assert_neo_object_is_compliant(self.train1)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        res = pretty(self.train1)
        targ = ("SpikeTrain\n" +
                "name: '%s'\ndescription: '%s'\nannotations: %s" %
                (self.name1, self.description1, pretty(self.ann1)))
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

        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start16, t_stop=t_stop16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start16, t_stop=t_stop16,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start16, t_stop=t_stop16,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start16, t_stop=t_stop16,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start32, t_stop=t_stop32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start32, t_stop=t_stop32,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start32, t_stop=t_stop32,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start32, t_stop=t_stop32,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start64, t_stop=t_stop64,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start64, t_stop=t_stop64,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True,
                           t_start=t_start_custom, t_stop=t_stop_custom,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom16, t_stop=t_stop_custom16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom16, t_stop=t_stop_custom16,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom16, t_stop=t_stop_custom16,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom16, t_stop=t_stop_custom16,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom32, t_stop=t_stop_custom32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom32, t_stop=t_stop_custom32,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom32, t_stop=t_stop_custom32,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom32, t_stop=t_stop_custom32,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom64, t_stop=t_stop_custom64)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom64, t_stop=t_stop_custom64,
                           dtype=np.float16)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom64, t_stop=t_stop_custom64,
                           dtype=np.float32)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start_custom64, t_stop=t_stop_custom64,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        # This use to bug - see ticket #38
        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data16, copy=True, t_start=t_start, t_stop=t_stop,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True, t_start=t_start, t_stop=t_stop,
                           dtype=np.float64)
        assert_neo_object_is_compliant(train)

        train = SpikeTrain(data32, copy=True,
                           t_start=t_start64, t_stop=t_stop64)
        assert_neo_object_is_compliant(train)
        train = SpikeTrain(data32, copy=True,
                           t_start=t_start64, t_stop=t_stop64,
                           dtype=np.float64)
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
