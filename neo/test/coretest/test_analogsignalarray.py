# -*- coding: utf-8 -*-
"""
Tests of the neo.core.analogsignalarray.AnalogSignalArrayArray class
"""

import os
import pickle

import unittest

import numpy as np
import quantities as pq

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from numpy.testing import assert_array_equal
from neo.core.analogsignal import AnalogSignal
from neo.core import Segment, ChannelIndex
from neo.test.tools import (assert_arrays_almost_equal, assert_arrays_equal,
                            assert_neo_object_is_compliant,
                            assert_same_sub_schema)
from neo.test.generate_datasets import (get_fake_value, get_fake_values,
                                        fake_neo, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__get_fake_values(self):
        self.annotations['seed'] = 0
        signal = get_fake_value('signal', pq.Quantity, seed=0, dim=2)
        sampling_rate = get_fake_value('sampling_rate', pq.Quantity,
                                       seed=1, dim=0)
        t_start = get_fake_value('t_start', pq.Quantity, seed=2, dim=0)
        name = get_fake_value('name', str, seed=3, obj=AnalogSignal)
        description = get_fake_value('description', str, seed=4,
                                     obj='AnalogSignal')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(AnalogSignal, annotate=False, seed=0)
        res12 = get_fake_values('AnalogSignal', annotate=False, seed=0)
        res21 = get_fake_values(AnalogSignal, annotate=True, seed=0)
        res22 = get_fake_values('AnalogSignal', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('signal'), signal)
        assert_arrays_equal(res12.pop('signal'), signal)
        assert_arrays_equal(res21.pop('signal'), signal)
        assert_arrays_equal(res22.pop('signal'), signal)

        assert_arrays_equal(res11.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res12.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res21.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res22.pop('sampling_rate'), sampling_rate)

        assert_arrays_equal(res11.pop('t_start'), t_start)
        assert_arrays_equal(res12.pop('t_start'), t_start)
        assert_arrays_equal(res21.pop('t_start'), t_start)
        assert_arrays_equal(res22.pop('t_start'), t_start)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'AnalogSignal'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = AnalogSignal
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestAnalogSignalArrayConstructor(unittest.TestCase):
    def test__create_from_list(self):
        data = [(i, i, i) for i in range(10)]  # 3 signals each with 10 samples
        rate = 1000 * pq.Hz
        signal = AnalogSignal(data, sampling_rate=rate, units="mV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.shape, (10, 3))
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, len(data) / rate)
        self.assertEqual(signal[9, 0], 9000 * pq.uV)

    def test__create_from_numpy_array(self):
        data = np.arange(20.0).reshape((10, 2))
        rate = 1 * pq.kHz
        signal = AnalogSignal(data, sampling_rate=rate, units="uV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, data.shape[0] / rate)
        self.assertEqual(signal[9, 0], 0.018 * pq.mV)
        self.assertEqual(signal[9, 1], 19 * pq.uV)

    def test__create_from_quantities_array(self):
        data = np.arange(20.0).reshape((10, 2)) * pq.mV
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, sampling_rate=rate)
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, data.shape[0] / rate)
        self.assertEqual(signal[9, 0], 18000 * pq.uV)

    def test__create_from_quantities_with_inconsistent_units_ValueError(self):
        data = np.arange(20.0).reshape((10, 2)) * pq.mV
        self.assertRaises(ValueError, AnalogSignal, data,
                          sampling_rate=1 * pq.kHz, units="nA")

    def test__create_with_copy_true_should_return_copy(self):
        data = np.arange(20.0).reshape((10, 2)) * pq.mV
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, copy=True, sampling_rate=rate)
        assert_neo_object_is_compliant(signal)
        data[3, 0] = 0.099 * pq.V
        self.assertNotEqual(signal[3, 0], 99 * pq.mV)

    def test__create_with_copy_false_should_return_view(self):
        data = np.arange(20.0).reshape((10, 2)) * pq.mV
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, copy=False, sampling_rate=rate)
        assert_neo_object_is_compliant(signal)
        data[3, 0] = 99 * pq.mV
        self.assertEqual(signal[3, 0], 99000 * pq.uV)

        # signal must not be 1D - should raise Exception if 1D


class TestAnalogSignalArrayProperties(unittest.TestCase):
    def setUp(self):
        self.t_start = [0.0 * pq.ms, 100 * pq.ms, -200 * pq.ms]
        self.rates = [1 * pq.kHz, 420 * pq.Hz, 999 * pq.Hz]
        self.data = [np.arange(10.0).reshape((5, 2)) * pq.nA,
                     np.arange(-100.0, 100.0, 10.0).reshape((4, 5)) * pq.mV,
                     np.random.uniform(size=(100, 4)) * pq.uV]
        self.signals = [AnalogSignal(D, sampling_rate=r, t_start=t)
                        for r, D, t in zip(self.rates,
                                           self.data,
                                           self.t_start)]

    def test__compliant(self):
        for signal in self.signals:
            assert_neo_object_is_compliant(signal)

    def test__t_stop(self):
        for i, signal in enumerate(self.signals):
            targ = self.t_start[i] + self.data[i].shape[0] / self.rates[i]
            self.assertEqual(signal.t_stop, targ)

    def test__duration(self):
        for signal in self.signals:
            self.assertAlmostEqual(signal.duration,
                                   signal.t_stop - signal.t_start,
                                   delta=1e-15)

    def test__sampling_period(self):
        for signal, rate in zip(self.signals, self.rates):
            self.assertEqual(signal.sampling_period, 1 / rate)

    def test__times(self):
        for i, signal in enumerate(self.signals):
            targ = np.arange(self.data[i].shape[0])
            targ = targ / self.rates[i] + self.t_start[i]
            assert_arrays_almost_equal(signal.times, targ, 1e-12 * pq.ms)

    def test__children(self):
        signal = self.signals[0]

        segment = Segment(name='seg1')
        segment.analogsignals = [signal]
        segment.create_many_to_one_relationship()

        chx = ChannelIndex(name='chx1', index=np.arange(signal.shape[1]))
        chx.analogsignals = [signal]
        chx.create_many_to_one_relationship()

        self.assertEqual(signal._single_parent_objects,
                         ('Segment', 'ChannelIndex'))
        self.assertEqual(signal._multi_parent_objects, ())

        self.assertEqual(signal._single_parent_containers,
                         ('segment', 'channel_index'))
        self.assertEqual(signal._multi_parent_containers, ())

        self.assertEqual(signal._parent_objects,
                         ('Segment', 'ChannelIndex'))
        self.assertEqual(signal._parent_containers,
                         ('segment', 'channel_index'))

        self.assertEqual(len(signal.parents), 2)
        self.assertEqual(signal.parents[0].name, 'seg1')
        self.assertEqual(signal.parents[1].name, 'chx1')

        assert_neo_object_is_compliant(signal)

    def test__repr(self):
        for i, signal in enumerate(self.signals):
            prepr = repr(signal)
            targ = '<AnalogSignal(%s, [%s, %s], sampling rate: %s)>' % \
                   (repr(self.data[i]),
                    self.t_start[i],
                    self.t_start[i] + len(self.data[i]) / self.rates[i],
                    self.rates[i])
            self.assertEqual(prepr, targ)


class TestAnalogSignalArrayArrayMethods(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(55.0).reshape((11, 5))
        self.data1quant = self.data1 * pq.nA
        self.signal1 = AnalogSignal(self.data1quant,
                                    sampling_rate=1 * pq.kHz,
                                    name='spam', description='eggs',
                                    file_origin='testfile.txt',
                                    arg1='test')
        self.data2 = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]).T
        self.data2quant = self.data2 * pq.mV
        self.signal2 = AnalogSignal(self.data2quant,
                                    sampling_rate=1.0 * pq.Hz,
                                    name='spam', description='eggs',
                                    file_origin='testfile.txt',
                                    arg1='test')

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})

        assert_neo_object_is_compliant(self.signal2)
        self.assertEqual(self.signal2.name, 'spam')
        self.assertEqual(self.signal2.description, 'eggs')
        self.assertEqual(self.signal2.file_origin, 'testfile.txt')
        self.assertEqual(self.signal2.annotations, {'arg1': 'test'})

    def test__index_dim1_should_return_single_channel_analogsignalarray(self):
        result = self.signal1[:, 0]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.t_stop, self.signal1.t_stop)
        self.assertEqual(result.t_start, self.signal1.t_start)
        self.assertEqual(result.sampling_rate,
                         self.signal1.sampling_rate)
        assert_arrays_equal(result, self.data1[:, 0].reshape(-1, 1))

    def test__index_dim1_and_slice_dim0_should_return_single_channel_analogsignalarray(self):
        result = self.signal1[2:7, 0]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.shape, (5, 1))
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.t_start,
                         self.signal1.t_start + 2 * self.signal1.sampling_period)
        self.assertEqual(result.t_stop,
                         self.signal1.t_start + 7 * self.signal1.sampling_period)
        self.assertEqual(result.sampling_rate,
                         self.signal1.sampling_rate)
        assert_arrays_equal(result, self.data1[2:7, 0].reshape(-1, 1))

    def test__index_dim0_should_return_quantity_array(self):
        # i.e. values from all signals for a single point in time
        result = self.signal1[3, :]
        self.assertIsInstance(result, pq.Quantity)
        self.assertFalse(hasattr(result, 'name'))
        self.assertFalse(hasattr(result, 'description'))
        self.assertFalse(hasattr(result, 'file_origin'))
        self.assertFalse(hasattr(result, 'annotations'))

        self.assertEqual(result.shape, (5,))
        self.assertFalse(hasattr(result, "t_start"))
        self.assertEqual(result.units, pq.nA)
        assert_arrays_equal(result, self.data1[3, :])

    def test__index_dim0_and_slice_dim1_should_return_quantity_array(self):
        # i.e. values from a subset of signals for a single point in time
        result = self.signal1[3, 2:5]
        self.assertIsInstance(result, pq.Quantity)
        self.assertFalse(hasattr(result, 'name'))
        self.assertFalse(hasattr(result, 'description'))
        self.assertFalse(hasattr(result, 'file_origin'))
        self.assertFalse(hasattr(result, 'annotations'))

        self.assertEqual(result.shape, (3,))
        self.assertFalse(hasattr(result, "t_start"))
        self.assertEqual(result.units, pq.nA)
        assert_arrays_equal(result, self.data1[3, 2:5])

    def test__index_as_string_IndexError(self):
        self.assertRaises(IndexError, self.signal1.__getitem__, 5.)

    def test__slice_both_dimensions_should_return_analogsignalarray(self):
        result = self.signal1[0:3, 0:3]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targ = AnalogSignal([[0, 1, 2], [5, 6, 7], [10, 11, 12]],
                            dtype=float, units="nA",
                            sampling_rate=1 * pq.kHz,
                            name='spam', description='eggs',
                            file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(targ)

        self.assertEqual(result.t_stop, targ.t_stop)
        self.assertEqual(result.t_start, targ.t_start)
        self.assertEqual(result.sampling_rate, targ.sampling_rate)
        self.assertEqual(result.shape, targ.shape)
        assert_same_sub_schema(result, targ)
        assert_arrays_equal(result, self.data1[0:3, 0:3])

    def test__slice_only_first_dimension_should_return_analogsignalarray(self):
        result = self.signal1[2:7]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.t_start,
                         self.signal1.t_start + 2 * self.signal1.sampling_period)
        self.assertEqual(result.t_stop,
                         self.signal1.t_start + 7 * self.signal1.sampling_period)
        self.assertEqual(result.sampling_rate, self.signal1.sampling_rate)
        assert_arrays_equal(result, self.data1[2:7])

    def test__getitem_should_return_single_quantity(self):
        # quantities drops the units in this case
        self.assertEqual(self.signal1[9, 3], 48000 * pq.pA)
        self.assertEqual(self.signal1[9][3], self.signal1[9, 3])
        self.assertTrue(hasattr(self.signal1[9, 3], 'units'))
        self.assertRaises(IndexError, self.signal1.__getitem__, (99, 73))

    def test_comparison_operators(self):
        assert_arrays_equal(self.signal1[0:3, 0:3] >= 5 * pq.nA,
                            np.array([[False, False, False],
                                      [True, True, True],
                                      [True, True, True]]))
        assert_arrays_equal(self.signal1[0:3, 0:3] >= 5 * pq.pA,
                            np.array([[False, True, True],
                                      [True, True, True],
                                      [True, True, True]]))

    def test__comparison_with_inconsistent_units_should_raise_Exception(self):
        self.assertRaises(ValueError, self.signal1.__gt__, 5 * pq.mV)

    def test__simple_statistics(self):
        self.assertEqual(self.signal1.max(), 54000 * pq.pA)
        self.assertEqual(self.signal1.min(), 0 * pq.nA)
        self.assertEqual(self.signal1.mean(), 27 * pq.nA)
        self.assertEqual(self.signal1.std(), self.signal1.magnitude.std() * pq.nA)
        self.assertEqual(self.signal1.var(), self.signal1.magnitude.var() * pq.nA ** 2)

    def test__rescale_same(self):
        result = self.signal1.copy()
        result = result.rescale(pq.nA)

        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.units, 1 * pq.nA)
        assert_arrays_equal(result, self.data1)
        assert_same_sub_schema(result, self.signal1)

    def test__rescale_new(self):
        result = self.signal1.copy()
        result = result.rescale(pq.pA)

        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.units, 1 * pq.pA)
        assert_arrays_almost_equal(np.array(result), self.data1 * 1000., 1e-10)

    def test__time_slice(self):
        t_start = 2 * pq.s
        t_stop = 4 * pq.s

        result = self.signal2.time_slice(t_start, t_stop)
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targ = AnalogSignal(np.array([[2., 3.], [2., 3.]]).T,
                            sampling_rate=1.0 * pq.Hz, units='mV',
                            t_start=t_start,
                            name='spam', description='eggs',
                            file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(result)

        self.assertEqual(result.t_stop, t_stop)
        self.assertEqual(result.t_start, t_start)
        self.assertEqual(result.sampling_rate, targ.sampling_rate)
        assert_array_equal(result, targ)
        assert_same_sub_schema(result, targ)

    def test__time_slice__out_of_bounds_ValueError(self):
        t_start_good = 2 * pq.s
        t_stop_good = 4 * pq.s
        t_start_bad = -2 * pq.s
        t_stop_bad = 40 * pq.s

        self.assertRaises(ValueError, self.signal2.time_slice,
                          t_start_good, t_stop_bad)
        self.assertRaises(ValueError, self.signal2.time_slice,
                          t_start_bad, t_stop_good)
        self.assertRaises(ValueError, self.signal2.time_slice,
                          t_start_bad, t_stop_bad)

    def test__time_equal(self):
        t_start = 0 * pq.s
        t_stop = 6 * pq.s

        result = self.signal2.time_slice(t_start, t_stop)
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.t_stop, t_stop)
        self.assertEqual(result.t_start, t_start)
        assert_array_equal(result, self.signal2)
        assert_same_sub_schema(result, self.signal2)

    def test__time_slice__offset(self):
        self.signal2.t_start = 10.0 * pq.s
        assert_neo_object_is_compliant(self.signal2)

        t_start = 12 * pq.s
        t_stop = 14 * pq.s

        result = self.signal2.time_slice(t_start, t_stop)
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targ = AnalogSignal(np.array([[2., 3.], [2., 3.]]).T,
                            t_start=12.0 * pq.ms,
                            sampling_rate=1.0 * pq.Hz, units='mV',
                            name='spam', description='eggs',
                            file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(result)

        self.assertEqual(self.signal2.t_start, 10.0 * pq.s)
        self.assertEqual(result.t_stop, t_stop)
        self.assertEqual(result.t_start, t_start)
        self.assertEqual(result.sampling_rate, targ.sampling_rate)
        assert_arrays_equal(result, targ)
        assert_same_sub_schema(result, targ)

    def test__time_slice__different_units(self):
        self.signal2.t_start = 10.0 * pq.ms
        assert_neo_object_is_compliant(self.signal2)

        t_start = 2 * pq.s + 10.0 * pq.ms
        t_stop = 4 * pq.s + 10.0 * pq.ms

        result = self.signal2.time_slice(t_start, t_stop)
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targ = AnalogSignal(np.array([[2., 3.], [2., 3.]]).T,
                            t_start=t_start.rescale(pq.ms),
                            sampling_rate=1.0 * pq.Hz, units='mV',
                            name='spam', description='eggs',
                            file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(result)

        assert_neo_object_is_compliant(self.signal2)
        self.assertEqual(self.signal2.t_start, 10.0 * pq.ms)
        self.assertAlmostEqual(result.t_stop, t_stop, delta=1e-12 * pq.ms)
        self.assertAlmostEqual(result.t_start, t_start, delta=1e-12 * pq.ms)
        assert_arrays_almost_equal(result.times, targ.times, 1e-12 * pq.ms)
        self.assertEqual(result.sampling_rate, targ.sampling_rate)
        assert_arrays_equal(result, targ)
        assert_same_sub_schema(result, targ)

    def test__time_slice__no_explicit_time(self):
        self.signal2.t_start = 10.0 * pq.ms
        assert_neo_object_is_compliant(self.signal2)

        t1 = 2 * pq.s + 10.0 * pq.ms
        t2 = 4 * pq.s + 10.0 * pq.ms

        for t_start, t_stop in [(t1, None), (None, None), (None, t2)]:
            t_start_targ = t1 if t_start is not None else self.signal2.t_start
            t_stop_targ = t2 if t_stop is not None else self.signal2.t_stop

            result = self.signal2.time_slice(t_start, t_stop)
            self.assertIsInstance(result, AnalogSignal)
            assert_neo_object_is_compliant(result)
            self.assertEqual(result.name, 'spam')
            self.assertEqual(result.description, 'eggs')
            self.assertEqual(result.file_origin, 'testfile.txt')
            self.assertEqual(result.annotations, {'arg1': 'test'})

            targ_ind = np.where((self.signal2.times >= t_start_targ) &
                                (self.signal2.times < t_stop_targ))
            targ_array = self.signal2.magnitude[targ_ind]

            targ = AnalogSignal(targ_array,
                                t_start=t_start_targ.rescale(pq.ms),
                                sampling_rate=1.0 * pq.Hz, units='mV',
                                name='spam', description='eggs',
                                file_origin='testfile.txt', arg1='test')
            assert_neo_object_is_compliant(result)

            assert_neo_object_is_compliant(self.signal2)
            self.assertEqual(self.signal2.t_start, 10.0 * pq.ms)
            self.assertAlmostEqual(result.t_stop, t_stop_targ, delta=1e-12 * pq.ms)
            self.assertAlmostEqual(result.t_start, t_start_targ, delta=1e-12 * pq.ms)
            assert_arrays_almost_equal(result.times, targ.times, 1e-12 * pq.ms)
            self.assertEqual(result.sampling_rate, targ.sampling_rate)
            assert_array_equal(result.magnitude, targ.magnitude)
            assert_same_sub_schema(result, targ)


class TestAnalogSignalArrayEquality(unittest.TestCase):
    def test__signals_with_different_data_complement_should_be_not_equal(self):
        signal1 = AnalogSignal(np.arange(55.0).reshape((11, 5)),
                               units="mV", sampling_rate=1 * pq.kHz)
        signal2 = AnalogSignal(np.arange(55.0).reshape((11, 5)),
                               units="mV", sampling_rate=2 * pq.kHz)
        self.assertNotEqual(signal1, signal2)
        assert_neo_object_is_compliant(signal1)
        assert_neo_object_is_compliant(signal2)


class TestAnalogSignalArrayCombination(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(55.0).reshape((11, 5))
        self.data1quant = self.data1 * pq.mV
        self.signal1 = AnalogSignal(self.data1quant,
                                    sampling_rate=1 * pq.kHz,
                                    name='spam', description='eggs',
                                    file_origin='testfile.txt',
                                    arg1='test')
        self.data2 = np.arange(100.0, 155.0).reshape((11, 5))
        self.data2quant = self.data2 * pq.mV
        self.signal2 = AnalogSignal(self.data2quant,
                                    sampling_rate=1 * pq.kHz,
                                    name='spam', description='eggs',
                                    file_origin='testfile.txt',
                                    arg1='test')

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})

        assert_neo_object_is_compliant(self.signal2)
        self.assertEqual(self.signal2.name, 'spam')
        self.assertEqual(self.signal2.description, 'eggs')
        self.assertEqual(self.signal2.file_origin, 'testfile.txt')
        self.assertEqual(self.signal2.annotations, {'arg1': 'test'})

    def test__add_const_quantity_should_preserve_data_complement(self):
        result = self.signal1 + 0.065 * pq.V
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        # time zero, signal index 4
        assert_arrays_equal(result, self.data1 + 65)
        self.assertEqual(self.signal1[0, 4], 4 * pq.mV)
        self.assertEqual(result[0, 4], 69000 * pq.uV)
        self.assertEqual(self.signal1.t_start, result.t_start)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__add_two_consistent_signals_should_preserve_data_complement(self):
        result = self.signal1 + self.signal2
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targdata = np.arange(100.0, 210.0, 2.0).reshape((11, 5))
        targ = AnalogSignal(targdata, units="mV",
                            sampling_rate=1 * pq.kHz,
                            name='spam', description='eggs',
                            file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(targ)

        assert_arrays_equal(result, targdata)
        assert_same_sub_schema(result, targ)

    def test__add_signals_with_inconsistent_data_complement_ValueError(self):
        self.signal2.sampling_rate = 0.5 * pq.kHz
        assert_neo_object_is_compliant(self.signal2)

        self.assertRaises(ValueError, self.signal1.__add__, self.signal2)

    def test__subtract_const_should_preserve_data_complement(self):
        result = self.signal1 - 65 * pq.mV
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(np.array(self.signal1[1, 4]), 9)
        self.assertEqual(np.array(result[1, 4]), -56)
        assert_arrays_equal(result, self.data1 - 65)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__subtract_from_const_should_return_signal(self):
        result = 10 * pq.mV - self.signal1
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(np.array(self.signal1[1, 4]), 9)
        self.assertEqual(np.array(result[1, 4]), 1)
        assert_arrays_equal(result, 10 - self.data1)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__mult_by_const_float_should_preserve_data_complement(self):
        result = self.signal1 * 2
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(np.array(self.signal1[1, 4]), 9)
        self.assertEqual(np.array(result[1, 4]), 18)
        assert_arrays_equal(result, self.data1 * 2)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__divide_by_const_should_preserve_data_complement(self):
        result = self.signal1 / 0.5
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(np.array(self.signal1[1, 4]), 9)
        self.assertEqual(np.array(result[1, 4]), 18)
        assert_arrays_equal(result, self.data1 / 0.5)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__merge(self):
        self.signal1.description = None
        self.signal1.file_origin = None
        assert_neo_object_is_compliant(self.signal1)

        data3 = np.arange(1000.0, 1066.0).reshape((11, 6)) * pq.uV
        data3scale = data3.rescale(self.data1quant.units)

        signal2 = AnalogSignal(self.data1quant,
                               sampling_rate=1 * pq.kHz,
                               name='signal2',
                               description='test signal',
                               file_origin='testfile.txt')
        signal3 = AnalogSignal(data3,
                               units="uV", sampling_rate=1 * pq.kHz,
                               name='signal3',
                               description='test signal',
                               file_origin='testfile.txt')
        signal4 = AnalogSignal(data3,
                               units="uV", sampling_rate=1 * pq.kHz,
                               name='signal4',
                               description='test signal',
                               file_origin='testfile.txt')

        merged13 = self.signal1.merge(signal3)
        merged23 = signal2.merge(signal3)
        merged24 = signal2.merge(signal4)
        mergeddata13 = np.array(merged13)
        mergeddata23 = np.array(merged23)
        mergeddata24 = np.array(merged24)

        targdata13 = np.hstack([self.data1quant, data3scale])
        targdata23 = np.hstack([self.data1quant, data3scale])
        targdata24 = np.hstack([self.data1quant, data3scale])

        assert_neo_object_is_compliant(signal2)
        assert_neo_object_is_compliant(signal3)
        assert_neo_object_is_compliant(merged13)
        assert_neo_object_is_compliant(merged23)
        assert_neo_object_is_compliant(merged24)

        self.assertEqual(merged13[0, 4], 4 * pq.mV)
        self.assertEqual(merged23[0, 4], 4 * pq.mV)
        self.assertEqual(merged13[0, 5], 1 * pq.mV)
        self.assertEqual(merged23[0, 5], 1 * pq.mV)
        self.assertEqual(merged13[10, 10], 1.065 * pq.mV)
        self.assertEqual(merged23[10, 10], 1.065 * pq.mV)
        self.assertEqual(merged13.t_stop, self.signal1.t_stop)
        self.assertEqual(merged23.t_stop, self.signal1.t_stop)

        self.assertEqual(merged13.name, 'merge(spam, signal3)')
        self.assertEqual(merged23.name, 'merge(signal2, signal3)')
        self.assertEqual(merged13.description, 'merge(None, test signal)')
        self.assertEqual(merged23.description, 'test signal')
        self.assertEqual(merged13.file_origin, 'merge(None, testfile.txt)')
        self.assertEqual(merged23.file_origin, 'testfile.txt')

        assert_arrays_equal(mergeddata13, targdata13)
        assert_arrays_equal(mergeddata23, targdata23)
        assert_arrays_equal(mergeddata24, targdata24)


class TestAnalogSignalArrayFunctions(unittest.TestCase):
    def test__pickle(self):
        signal1 = AnalogSignal(np.arange(55.0).reshape((11, 5)),
                               units="mV", sampling_rate=1 * pq.kHz)

        fobj = open('./pickle', 'wb')
        pickle.dump(signal1, fobj)
        fobj.close()

        fobj = open('./pickle', 'rb')
        try:
            signal2 = pickle.load(fobj)
        except ValueError:
            signal2 = None

        assert_array_equal(signal1, signal2)
        assert_neo_object_is_compliant(signal1)
        assert_neo_object_is_compliant(signal2)
        fobj.close()
        os.remove('./pickle')


if __name__ == "__main__":
    unittest.main()
