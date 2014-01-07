# -*- coding: utf-8 -*-
"""
Tests of the neo.core.irregularlysampledsignal.IrregularySampledSignal class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.test.tools import (assert_arrays_almost_equal, assert_arrays_equal,
                            assert_neo_object_is_compliant,
                            assert_same_sub_schema)


class TestIrregularlySampledSignalConstruction(unittest.TestCase):
    def test_IrregularlySampledSignal_creation_times_units_signal_units(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7]*pq.ms,
                                       signal=[20., 40., 60.]*pq.mV,
                                       name='test', description='tester',
                                       file_origin='test.file',
                                       testarg1=1, **params)
        sig.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(sig)

        assert_arrays_equal(sig.times, [1.1, 1.5, 1.7]*pq.ms)
        assert_arrays_equal(np.asarray(sig), np.array([20., 40., 60.]))
        self.assertEqual(sig.units, pq.mV)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(sig.annotations['testarg1'], 1.1)
        self.assertEqual(sig.annotations['testarg2'], 'yes')
        self.assertTrue(sig.annotations['testarg3'])

    def test_IrregularlySampledSignal_creation_units_arg(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7],
                                       signal=[20., 40., 60.],
                                       units=pq.V, time_units=pq.s,
                                       name='test', description='tester',
                                       file_origin='test.file',
                                       testarg1=1, **params)
        sig.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(sig)

        assert_arrays_equal(sig.times, [1.1, 1.5, 1.7]*pq.s)
        assert_arrays_equal(np.asarray(sig), np.array([20., 40., 60.]))
        self.assertEqual(sig.units, pq.V)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(sig.annotations['testarg1'], 1.1)
        self.assertEqual(sig.annotations['testarg2'], 'yes')
        self.assertTrue(sig.annotations['testarg3'])

    def test_IrregularlySampledSignal_creation_units_rescale(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7]*pq.s,
                                       signal=[2., 4., 6.]*pq.V,
                                       units=pq.mV, time_units=pq.ms,
                                       name='test', description='tester',
                                       file_origin='test.file',
                                       testarg1=1, **params)
        sig.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(sig)

        assert_arrays_equal(sig.times, [1100, 1500, 1700]*pq.ms)
        assert_arrays_equal(np.asarray(sig), np.array([2000., 4000., 6000.]))
        self.assertEqual(sig.units, pq.mV)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(sig.annotations['testarg1'], 1.1)
        self.assertEqual(sig.annotations['testarg2'], 'yes')
        self.assertTrue(sig.annotations['testarg3'])

    def test_IrregularlySampledSignal_different_lens_ValueError(self):
        times = [1.1, 1.5, 1.7]*pq.ms
        signal = [20., 40., 60., 70.]*pq.mV
        self.assertRaises(ValueError, IrregularlySampledSignal, times, signal)

    def test_IrregularlySampledSignal_no_signal_units_ValueError(self):
        times = [1.1, 1.5, 1.7]*pq.ms
        signal = [20., 40., 60.]
        self.assertRaises(ValueError, IrregularlySampledSignal, times, signal)

    def test_IrregularlySampledSignal_no_time_units_ValueError(self):
        times = [1.1, 1.5, 1.7]
        signal = [20., 40., 60.]*pq.mV
        self.assertRaises(ValueError, IrregularlySampledSignal, times, signal)


class TestIrregularlySampledSignalProperties(unittest.TestCase):
    def setUp(self):
        self.times = [np.arange(10.0)*pq.s,
                      np.arange(-100.0, 100.0, 10.0)*pq.ms,
                      np.arange(100)*pq.ns]
        self.data = [np.arange(10.0)*pq.nA,
                     np.arange(-100.0, 100.0, 10.0)*pq.mV,
                     np.random.uniform(size=100)*pq.uV]
        self.signals = [IrregularlySampledSignal(t, signal=D,
                                                 testattr='test')
                        for D, t in zip(self.data, self.times)]

    def test__compliant(self):
        for signal in self.signals:
            assert_neo_object_is_compliant(signal)

    def test__t_start_getter(self):
        for signal, times in zip(self.signals, self.times):
            self.assertAlmostEqual(signal.t_start,
                                   times[0],
                                   delta=1e-15)

    def test__t_stop_getter(self):
        for signal, times in zip(self.signals, self.times):
            self.assertAlmostEqual(signal.t_stop,
                                   times[-1],
                                   delta=1e-15)

    def test__duration_getter(self):
        for signal, times in zip(self.signals, self.times):
            self.assertAlmostEqual(signal.duration,
                                   times[-1] - times[0],
                                   delta=1e-15)

    def test__sampling_intervals_getter(self):
        for signal, times in zip(self.signals, self.times):
            assert_arrays_almost_equal(signal.sampling_intervals,
                                       np.diff(times),
                                       threshold=1e-15)

    def test_IrregularlySampledSignal_repr(self):
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7]*pq.s,
                                       signal=[2., 4., 6.]*pq.V,
                                       name='test', description='tester',
                                       file_origin='test.file',
                                       testarg1=1)
        assert_neo_object_is_compliant(sig)

        targ = ('<IrregularlySampledSignal(array([ 2.,  4.,  6.]) * V ' +
                'at times [ 1.1  1.5  1.7] s)>')
        res = repr(sig)
        self.assertEqual(targ, res)


class TestIrregularlySampledSignalArrayMethods(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(10.0)
        self.data1quant = self.data1 * pq.mV
        self.time1 = np.logspace(1, 5, 10)
        self.time1quant = self.time1*pq.ms
        self.signal1 = IrregularlySampledSignal(self.time1quant,
                                                signal=self.data1quant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})

    def test__slice_should_return_IrregularlySampledSignal(self):
        result = self.signal1[3:8]
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.size, 5)
        self.assertEqual(result.t_start, self.time1quant[3])
        self.assertEqual(result.t_stop, self.time1quant[7])
        assert_arrays_equal(self.time1quant[3:8], result.times)
        assert_arrays_equal(self.data1[3:8], result)

        # Test other attributes were copied over (in this case, defaults)
        self.assertEqual(result.file_origin, self.signal1.file_origin)
        self.assertEqual(result.name, self.signal1.name)
        self.assertEqual(result.description, self.signal1.description)
        self.assertEqual(result.annotations, self.signal1.annotations)

    def test__getitem_should_return_single_quantity(self):
        self.assertEqual(self.signal1[0], 0*pq.mV)
        self.assertEqual(self.signal1[9], 9*pq.mV)
        self.assertRaises(IndexError, self.signal1.__getitem__, 10)

    def test__getitem_out_of_bounds_IndexError(self):
        self.assertRaises(IndexError, self.signal1.__getitem__, 10)

    def test_comparison_operators(self):
        assert_arrays_equal(self.signal1 >= 5*pq.mV,
                            np.array([False, False, False, False, False,
                                      True, True, True, True, True]))

    def test__comparison_with_inconsistent_units_should_raise_Exception(self):
        self.assertRaises(ValueError, self.signal1.__gt__, 5*pq.nA)

    def test_simple_statistics(self):
        targmean = self.signal1[:-1]*np.diff(self.time1quant)
        targmean = targmean.sum()/(self.time1quant[-1]-self.time1quant[0])
        self.assertEqual(self.signal1.max(), 9*pq.mV)
        self.assertEqual(self.signal1.min(), 0*pq.mV)
        self.assertEqual(self.signal1.mean(), targmean)

    def test_mean_interpolation_NotImplementedError(self):
        self.assertRaises(NotImplementedError, self.signal1.mean, True)

    def test_resample_NotImplementedError(self):
        self.assertRaises(NotImplementedError, self.signal1.resample, True)

    def test__rescale_same(self):
        result = self.signal1.copy()
        result = result.rescale(pq.mV)

        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.units, 1*pq.mV)
        assert_arrays_equal(result, self.data1)
        assert_arrays_equal(result.times, self.time1quant)
        assert_same_sub_schema(result, self.signal1)

    def test__rescale_new(self):
        result = self.signal1.copy()
        result = result.rescale(pq.uV)

        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.units, 1*pq.uV)
        assert_arrays_almost_equal(np.array(result), self.data1*1000., 1e-10)
        assert_arrays_equal(result.times, self.time1quant)

    def test__rescale_new_incompatible_ValueError(self):
        self.assertRaises(ValueError, self.signal1.rescale, pq.nA)


class TestIrregularlySampledSignalCombination(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(10.0)
        self.data1quant = self.data1 * pq.mV
        self.time1 = np.logspace(1, 5, 10)
        self.time1quant = self.time1*pq.ms
        self.signal1 = IrregularlySampledSignal(self.time1quant,
                                                signal=self.data1quant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})

    def test__add_const_quantity_should_preserve_data_complement(self):
        result = self.signal1 + 0.065*pq.V
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        assert_arrays_equal(result, self.data1 + 65)
        assert_arrays_equal(result.times, self.time1quant)
        self.assertEqual(self.signal1[9], 9*pq.mV)
        self.assertEqual(result[9], 74*pq.mV)

    def test__add_two_consistent_signals_should_preserve_data_complement(self):
        data2 = np.arange(10.0, 20.0)
        data2quant = data2*pq.mV
        signal2 = IrregularlySampledSignal(self.time1quant, signal=data2quant)
        assert_neo_object_is_compliant(signal2)

        result = self.signal1 + signal2
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targ = IrregularlySampledSignal(self.time1quant,
                                        signal=np.arange(10.0, 30.0, 2.0),
                                        units="mV",
                                        name='spam', description='eggs',
                                        file_origin='testfile.txt',
                                        arg1='test')
        assert_neo_object_is_compliant(targ)

        assert_arrays_equal(result, targ)
        assert_arrays_equal(self.time1quant, targ.times)
        assert_arrays_equal(result.times, targ.times)
        assert_same_sub_schema(result, targ)

    def test__add_signals_with_inconsistent_times_AssertionError(self):
        signal2 = IrregularlySampledSignal(self.time1quant*2.,
                                           signal=np.arange(10.0), units="mV")
        assert_neo_object_is_compliant(signal2)

        self.assertRaises(ValueError, self.signal1.__add__, signal2)

    def test__add_signals_with_inconsistent_dimension_ValueError(self):
        signal2 = np.arange(20).reshape(2, 10)

        self.assertRaises(ValueError, self.signal1.__add__, signal2)

    def test__subtract_const_should_preserve_data_complement(self):
        result = self.signal1 - 65*pq.mV
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9], 9*pq.mV)
        self.assertEqual(result[9], -56*pq.mV)
        assert_arrays_equal(result, self.data1 - 65)
        assert_arrays_equal(result.times, self.time1quant)

    def test__subtract_from_const_should_return_signal(self):
        result = 10*pq.mV - self.signal1
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9], 9*pq.mV)
        self.assertEqual(result[9], 1*pq.mV)
        assert_arrays_equal(result, 10 - self.data1)
        assert_arrays_equal(result.times, self.time1quant)

    def test__mult_signal_by_const_float_should_preserve_data_complement(self):
        result = self.signal1*2.
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9], 9*pq.mV)
        self.assertEqual(result[9], 18*pq.mV)
        assert_arrays_equal(result, self.data1*2)
        assert_arrays_equal(result.times, self.time1quant)

    def test__mult_signal_by_const_array_should_preserve_data_complement(self):
        result = self.signal1*np.array(2.)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9], 9*pq.mV)
        self.assertEqual(result[9], 18*pq.mV)
        assert_arrays_equal(result, self.data1*2)
        assert_arrays_equal(result.times, self.time1quant)

    def test__divide_signal_by_const_should_preserve_data_complement(self):
        result = self.signal1/0.5
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9], 9*pq.mV)
        self.assertEqual(result[9], 18*pq.mV)
        assert_arrays_equal(result, self.data1/0.5)
        assert_arrays_equal(result.times, self.time1quant)


class TestIrregularlySampledSignalEquality(unittest.TestCase):
    def test__signals_with_different_times_should_be_not_equal(self):
            signal1 = IrregularlySampledSignal(np.arange(10.0)/100*pq.s,
                                               np.arange(10.0), units="mV")
            signal2 = IrregularlySampledSignal(np.arange(10.0)/100*pq.ms,
                                               np.arange(10.0), units="mV")
            self.assertNotEqual(signal1, signal2)


if __name__ == "__main__":
    unittest.main()
