"""
Tests of the neo.core.irregularlysampledsignal.IrregularySampledSignal class
"""

import unittest

import os
import pickle
import warnings
from copy import deepcopy

import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal

from neo.core.dataobject import ArrayDict

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

try:
    import scipy
except ImportError:
    HAVE_SCIPY = False
else:
    HAVE_SCIPY = True

from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.core import Segment
from neo.core.baseneo import MergeError
from neo.test.tools import (assert_arrays_almost_equal, assert_arrays_equal,
                            assert_neo_object_is_compliant, assert_same_sub_schema,
                            assert_same_attributes, assert_same_annotations,
                            assert_same_array_annotations)


class TestIrregularlySampledSignalConstruction(unittest.TestCase):
    def test_IrregularlySampledSignal_creation_times_units_signal_units(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'anno1': [23], 'anno2': ['A']}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7] * pq.ms, signal=[20., 40., 60.] * pq.mV,
                                       name='test', description='tester', file_origin='test.file',
                                       test1=1, array_annotations=arr_ann, **params)
        sig.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(sig)

        assert_array_equal(sig.times, [1.1, 1.5, 1.7] * pq.ms)
        assert_array_equal(np.asarray(sig).flatten(), np.array([20., 40., 60.]))
        self.assertEqual(sig.units, pq.mV)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['test0'], [1, 2])
        self.assertEqual(sig.annotations['test1'], 1.1)
        self.assertEqual(sig.annotations['test2'], 'y1')
        self.assertTrue(sig.annotations['test3'])

        assert_arrays_equal(sig.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(sig.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(sig.array_annotations, ArrayDict)

    def test_IrregularlySampledSignal_creation_units_arg(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'anno1': [23], 'anno2': ['A']}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7], signal=[20., 40., 60.], units=pq.V,
                                       time_units=pq.s, name='test', description='tester',
                                       file_origin='test.file', test1=1,
                                       array_annotations=arr_ann, **params)
        sig.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(sig)

        assert_array_equal(sig.times, [1.1, 1.5, 1.7] * pq.s)
        assert_array_equal(np.asarray(sig).flatten(), np.array([20., 40., 60.]))
        self.assertEqual(sig.units, pq.V)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['test0'], [1, 2])
        self.assertEqual(sig.annotations['test1'], 1.1)
        self.assertEqual(sig.annotations['test2'], 'y1')
        self.assertTrue(sig.annotations['test3'])

        assert_arrays_equal(sig.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(sig.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(sig.array_annotations, ArrayDict)

    def test_IrregularlySampledSignal_creation_units_rescale(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'anno1': [23], 'anno2': ['A']}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7] * pq.s, signal=[2., 4., 6.] * pq.V,
                                       units=pq.mV, time_units=pq.ms, name='test',
                                       description='tester', file_origin='test.file', test1=1,
                                       array_annotations=arr_ann, **params)
        sig.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(sig)

        assert_array_equal(sig.times, [1100, 1500, 1700] * pq.ms)
        assert_array_equal(np.asarray(sig).flatten(), np.array([2000., 4000., 6000.]))
        self.assertEqual(sig.units, pq.mV)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['test0'], [1, 2])
        self.assertEqual(sig.annotations['test1'], 1.1)
        self.assertEqual(sig.annotations['test2'], 'y1')
        self.assertTrue(sig.annotations['test3'])

        assert_arrays_equal(sig.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(sig.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(sig.array_annotations, ArrayDict)

    def test_IrregularlySampledSignal_different_lens_ValueError(self):
        times = [1.1, 1.5, 1.7] * pq.ms
        signal = [20., 40., 60., 70.] * pq.mV
        self.assertRaises(ValueError, IrregularlySampledSignal, times, signal)

    def test_IrregularlySampledSignal_no_signal_units_ValueError(self):
        times = [1.1, 1.5, 1.7] * pq.ms
        signal = [20., 40., 60.]
        self.assertRaises(ValueError, IrregularlySampledSignal, times, signal)

    def test_IrregularlySampledSignal_no_time_units_ValueError(self):
        times = [1.1, 1.5, 1.7]
        signal = [20., 40., 60.] * pq.mV
        self.assertRaises(ValueError, IrregularlySampledSignal, times, signal)


class TestIrregularlySampledSignalProperties(unittest.TestCase):
    def setUp(self):
        self.times = [np.arange(10.0) * pq.s, np.arange(-100.0, 100.0, 10.0) * pq.ms,
                      np.arange(100) * pq.ns]
        self.data = [np.arange(10.0) * pq.nA, np.arange(-100.0, 100.0, 10.0) * pq.mV,
                     np.random.uniform(size=100) * pq.uV]
        self.signals = [IrregularlySampledSignal(t, signal=D, testattr='test') for D, t in
                        zip(self.data, self.times)]

    def test__compliant(self):
        for signal in self.signals:
            assert_neo_object_is_compliant(signal)

    def test__t_start_getter(self):
        for signal, times in zip(self.signals, self.times):
            self.assertAlmostEqual(signal.t_start, times[0], delta=1e-15)

    def test__t_stop_getter(self):
        for signal, times in zip(self.signals, self.times):
            self.assertAlmostEqual(signal.t_stop, times[-1], delta=1e-15)

    def test__duration_getter(self):
        for signal, times in zip(self.signals, self.times):
            self.assertAlmostEqual(signal.duration, times[-1] - times[0], delta=1e-15)

    def test__sampling_intervals_getter(self):
        for signal, times in zip(self.signals, self.times):
            assert_arrays_almost_equal(signal.sampling_intervals, np.diff(times), threshold=1e-15)

    def test_IrregularlySampledSignal_repr(self):
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7] * pq.s, signal=[2., 4., 6.] * pq.V,
                                       name='test', description='tester', file_origin='test.file',
                                       test1=1)
        assert_neo_object_is_compliant(sig)

        if np.__version__.split(".")[:2] > ['1', '13']:
            # see https://github.com/numpy/numpy/blob/master/doc/release/1.14.0-notes.rst#many
            # -changes-to-array-printing-disableable-with-the-new-legacy-printing-mode
            targ = ('<IrregularlySampledSignal(array([[2.],\n       [4.],\n       [6.]]) * V '
                    '' + 'at times [1.1 1.5 1.7] s)>')
        else:
            targ = ('<IrregularlySampledSignal(array([[ 2.],\n       [ 4.],\n       [ 6.]]) '
                    '* V ' + 'at times [ 1.1  1.5  1.7] s)>')
        res = repr(sig)
        self.assertEqual(targ, res)


class TestIrregularlySampledSignalArrayMethods(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(10.0)
        self.data1quant = self.data1 * pq.mV
        self.time1 = np.logspace(1, 5, 10)
        self.time1quant = self.time1 * pq.ms
        self.arr_ann = {'anno1': [23], 'anno2': ['A']}
        self.signal1 = IrregularlySampledSignal(self.time1quant, signal=self.data1quant,
                                                name='spam', description='eggs',
                                                file_origin='testfile.txt', arg1='test',
                                                array_annotations=self.arr_ann)
        self.signal1.segment = Segment()

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})
        assert_arrays_equal(self.signal1.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(self.signal1.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(self.signal1.array_annotations, ArrayDict)

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
        assert_array_equal(self.time1quant[3:8], result.times)
        assert_array_equal(self.data1[3:8].reshape(-1, 1), result.magnitude)

        # Test other attributes were copied over (in this case, defaults)
        self.assertEqual(result.file_origin, self.signal1.file_origin)
        self.assertEqual(result.name, self.signal1.name)
        self.assertEqual(result.description, self.signal1.description)
        self.assertEqual(result.annotations, self.signal1.annotations)
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__getitem_should_return_single_quantity(self):
        self.assertEqual(self.signal1[0], 0 * pq.mV)
        self.assertEqual(self.signal1[9], 9 * pq.mV)
        self.assertRaises(IndexError, self.signal1.__getitem__, 10)

    def test__getitem_out_of_bounds_IndexError(self):
        self.assertRaises(IndexError, self.signal1.__getitem__, 10)

    def test_comparison_operators(self):
        assert_array_equal(self.signal1 >= 5 * pq.mV, np.array(
            [[False, False, False, False, False, True, True, True, True, True]]).T)
        assert_array_equal(self.signal1 == 5 * pq.mV, np.array(
            [[False, False, False, False, False, True, False, False, False, False]]).T)
        assert_array_equal(self.signal1 == self.signal1, np.array(
            [[True, True, True, True, True, True, True, True, True, True]]).T)

    def test__comparison_as_indexing_single_trace(self):
        self.assertEqual(self.signal1[self.signal1 == 5], [5 * pq.mV])

    def test__comparison_as_indexing_multi_trace(self):
        signal = IrregularlySampledSignal(self.time1quant, np.arange(20).reshape((-1, 2)) * pq.V)
        assert_array_equal(signal[signal < 10],
                           np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]).T * pq.V)

    def test__indexing_keeps_order_across_channels(self):
        # AnalogSignals with 10 traces each having 5 samples (eg. data[0] = [0,10,20,30,40])
        data = np.array([range(10), range(10, 20), range(20, 30), range(30, 40), range(40, 50)])
        mask = np.full((5, 10), fill_value=False, dtype=bool)
        # selecting one entry per trace
        mask[[0, 1, 0, 3, 0, 2, 4, 3, 1, 4], range(10)] = True

        signal = IrregularlySampledSignal(np.arange(5) * pq.s, np.array(data) * pq.V)
        assert_array_equal(signal[mask], np.array([[0, 11, 2, 33, 4, 25, 46, 37, 18, 49]]) * pq.V)

    def test__indexing_keeps_order_across_time(self):
        # AnalogSignals with 10 traces each having 5 samples (eg. data[0] = [0,10,20,30,40])
        data = np.array([range(10), range(10, 20), range(20, 30), range(30, 40), range(40, 50)])
        mask = np.full((5, 10), fill_value=False, dtype=bool)
        # selecting two entries per trace
        temporal_ids = [0, 1, 0, 3, 1, 2, 4, 2, 1, 4] + [4, 3, 2, 1, 0, 1, 2, 3, 2, 1]
        mask[temporal_ids, list(range(10)) + list(range(10))] = True

        signal = IrregularlySampledSignal(np.arange(5) * pq.s, np.array(data) * pq.V)
        assert_array_equal(signal[mask], np.array([[0, 11, 2, 13, 4, 15, 26, 27, 18, 19],
                                                   [40, 31, 22, 33, 14, 25, 46, 37, 28,
                                                    49]]) * pq.V)

    def test__comparison_with_inconsistent_units_should_raise_Exception(self):
        self.assertRaises(ValueError, self.signal1.__gt__, 5 * pq.nA)

    def test_simple_statistics(self):
        targmean = self.signal1[:-1] * np.diff(self.time1quant).reshape(-1, 1)
        targmean = targmean.sum() / (self.time1quant[-1] - self.time1quant[0])
        self.assertEqual(self.signal1.max(), 9 * pq.mV)
        self.assertEqual(self.signal1.min(), 0 * pq.mV)
        self.assertEqual(self.signal1.mean(), targmean)

    def test_mean_interpolation_NotImplementedError(self):
        self.assertRaises(NotImplementedError, self.signal1.mean, True)

    def test__rescale_same(self):
        result = self.signal1.copy()
        result = result.rescale(pq.mV)

        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.units, 1 * pq.mV)
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1))
        assert_array_equal(result.times, self.time1quant)
        assert_same_sub_schema(result, self.signal1)

        self.assertIsInstance(result.segment, Segment)
        self.assertIs(result.segment, self.signal1.segment)

    def test__rescale_new(self):
        result = self.signal1.copy()
        result = result.rescale(pq.uV)

        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.units, 1 * pq.uV)
        assert_arrays_almost_equal(np.array(result), self.data1.reshape(-1, 1) * 1000., 1e-10)
        assert_array_equal(result.times, self.time1quant)

        self.assertIsInstance(result.segment, Segment)
        self.assertIs(result.segment, self.signal1.segment)

    def test__rescale_new_incompatible_ValueError(self):
        self.assertRaises(ValueError, self.signal1.rescale, pq.nA)

    def test_time_slice(self):
        targdataquant = [[1.0], [2.0], [3.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime[1:4] * pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant, signal=targdataquant, name='spam',
                                               description='eggs', file_origin='testfile.txt',
                                               arg1='test')

        t_start = 15
        t_stop = 250
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1 * pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__time_slice_deepcopy_annotations(self):
        params1 = {'test0': 'y1', 'test1': ['deeptest'], 'test2': True}
        self.signal1.annotate(**params1)

        result = self.signal1.time_slice(None, None)

        # Change annotations of original
        params2 = {'test0': 'y2', 'test2': False}
        self.signal1.annotate(**params2)
        self.signal1.annotations['test1'][0] = 'shallowtest'

        self.assertNotEqual(self.signal1.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.signal1.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.signal1.annotations['test2'], result.annotations['test2'])

        # Change annotations of result
        params3 = {'test0': 'y3'}
        result.annotate(**params3)
        result.annotations['test1'][0] = 'shallowtest2'

        self.assertNotEqual(self.signal1.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.signal1.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.signal1.annotations['test2'], result.annotations['test2'])

    def test__time_slice_deepcopy_array_annotations(self):
        length = self.signal1.shape[-1]
        params1 = {'test0': ['y{}'.format(i) for i in range(length)],
                   'test1': ['deeptest' for i in range(length)],
                   'test2': [(-1) ** i > 0 for i in range(length)]}
        self.signal1.array_annotate(**params1)
        result = self.signal1.time_slice(None, None)

        # Change annotations of original
        params2 = {'test0': ['x{}'.format(i) for i in range(length)],
                   'test2': [(-1) ** (i + 1) > 0 for i in range(length)]}
        self.signal1.array_annotate(**params2)
        self.signal1.array_annotations['test1'][0] = 'shallowtest'

        self.assertFalse(all(self.signal1.array_annotations['test0']
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.signal1.array_annotations['test1']
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.signal1.array_annotations['test2']
                             == result.array_annotations['test2']))

        # Change annotations of result
        params3 = {'test0': ['z{}'.format(i) for i in range(1, result.shape[-1] + 1)]}
        result.array_annotate(**params3)
        result.array_annotations['test1'][0] = 'shallow2'
        self.assertFalse(all(self.signal1.array_annotations['test0']
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.signal1.array_annotations['test1']
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.signal1.array_annotations['test2']
                             == result.array_annotations['test2']))

    def test__time_slice_deepcopy_data(self):
        result = self.signal1.time_slice(None, None)

        # Change values of original array
        self.signal1[2] = 7.3 * self.signal1.units

        self.assertFalse(all(self.signal1 == result))

        # Change values of sliced array
        result[3] = 9.5 * result.units

        self.assertFalse(all(self.signal1 == result))

    def test_time_slice_out_of_boundries(self):
        targdataquant = self.data1quant
        targtimequant = self.time1quant
        targ_signal = IrregularlySampledSignal(targtimequant, signal=targdataquant, name='spam',
                                               description='eggs', file_origin='testfile.txt',
                                               arg1='test')

        t_start = 0
        t_stop = 2500000
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1 * pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_empty(self):
        targdataquant = [] * pq.mV
        targtimequant = [] * pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant, signal=targdataquant, name='spam',
                                               description='eggs', file_origin='testfile.txt',
                                               arg1='test')

        t_start = 15
        t_stop = 250
        result = targ_signal.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1 * pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        self.assertEqual(result.array_annotations, {})
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_stop(self):
        targdataquant = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime[1:10] * pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant, signal=targdataquant, name='spam',
                                               description='eggs', file_origin='testfile.txt',
                                               arg1='test')

        t_start = 15
        t_stop = None
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1 * pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_start(self):
        targdataquant = [[0.0], [1.0], [2.0], [3.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime[0:4] * pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant, signal=targdataquant, name='spam',
                                               description='eggs', file_origin='testfile.txt',
                                               arg1='test')

        t_start = None
        t_stop = 250
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1 * pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_both(self):
        targdataquant = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],
                         [9.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime[0:10] * pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant, signal=targdataquant, name='spam',
                                               description='eggs', file_origin='testfile.txt',
                                               arg1='test')

        t_start = None
        t_stop = None
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1 * pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_differnt_units(self):
        targdataquant = [[1.0], [2.0], [3.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime[1:4] * pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant, signal=targdataquant, name='spam',
                                               description='eggs', file_origin='testfile.txt',
                                               arg1='test')

        t_start = 15
        t_stop = 250

        t_start = 0.015 * pq.s
        t_stop = .250 * pq.s

        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1 * pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__time_slice_should_set_parents_to_None(self):
        # When timeslicing, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.signal1.time_slice(1 * pq.ms, 3 * pq.ms)
        self.assertEqual(result.segment, None)

    def test__deepcopy_should_set_parents_objects_to_None(self):
        # Deepcopy should destroy references to parents
        result = deepcopy(self.signal1)
        self.assertEqual(result.segment, None)

    def test__time_shift_same_attributes(self):
        result = self.signal1.time_shift(1 * pq.ms)
        assert_same_attributes(result, self.signal1, exclude=['times', 't_start', 't_stop'])

    def test__time_shift_same_annotations(self):
        result = self.signal1.time_shift(1 * pq.ms)
        assert_same_annotations(result, self.signal1)

    def test__time_shift_same_array_annotations(self):
        result = self.signal1.time_shift(1 * pq.ms)
        assert_same_array_annotations(result, self.signal1)

    def test__time_shift_should_set_parents_to_None(self):
        # When time-shifting, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.signal1.time_shift(1 * pq.ms)
        self.assertEqual(result.segment, None)

    def test__time_shift_by_zero(self):
        shifted = self.signal1.time_shift(0 * pq.ms)
        assert_arrays_equal(shifted.times, self.signal1.times)

    def test__time_shift_same_units(self):
        shifted = self.signal1.time_shift(10 * pq.ms)
        assert_arrays_equal(shifted.times, self.signal1.times + 10 * pq.ms)

    def test__time_shift_different_units(self):
        shifted = self.signal1.time_shift(1 * pq.s)
        assert_arrays_equal(shifted.times, self.signal1.times + 1000 * pq.ms)

    def test_as_array(self):
        sig_as_arr = self.signal1.as_array()
        self.assertIsInstance(sig_as_arr, np.ndarray)
        assert_array_equal(self.data1, sig_as_arr.flat)

    def test_as_quantity(self):
        sig_as_q = self.signal1.as_quantity()
        self.assertIsInstance(sig_as_q, pq.Quantity)
        assert_array_equal(self.data1, sig_as_q.magnitude.flat)

    def test__copy_should_preserve_parent_objects(self):
        result = self.signal1.copy()
        self.assertIs(result.segment, self.signal1.segment)

    @unittest.skipUnless(HAVE_SCIPY, "requires Scipy")
    def test_resample(self):
        factors = [1, 2, 10]
        for factor in factors:
            result = self.signal1.resample(self.signal1.shape[0] * factor)
            np.testing.assert_allclose(self.signal1.magnitude, result.magnitude[::factor],
                                       rtol=1e-7, atol=0)


class TestIrregularlySampledSignalCombination(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(10.0)
        self.data1quant = self.data1 * pq.mV
        self.time1 = np.logspace(1, 5, 10)
        self.time1quant = self.time1 * pq.ms
        self.arr_ann = {'anno1': [23], 'anno2': ['A']}
        self.signal1 = IrregularlySampledSignal(self.time1quant, signal=self.data1quant,
                                                name='spam', description='eggs',
                                                file_origin='testfile.txt', arg1='test',
                                                array_annotations=self.arr_ann)

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})
        assert_arrays_equal(self.signal1.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(self.signal1.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(self.signal1.array_annotations, ArrayDict)

    def test__add_const_quantity_should_preserve_data_complement(self):
        result = self.signal1 + 0.065 * pq.V
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1) + 65)
        assert_array_equal(result.times, self.time1quant)
        self.assertEqual(self.signal1[9], 9 * pq.mV)
        self.assertEqual(result[9], 74 * pq.mV)

    def test__add_two_consistent_signals_should_preserve_data_complement(self):
        data2 = np.arange(10.0, 20.0)
        data2quant = data2 * pq.mV
        signal2 = IrregularlySampledSignal(self.time1quant, signal=data2quant)
        assert_neo_object_is_compliant(signal2)

        result = self.signal1 + signal2
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        targ = IrregularlySampledSignal(self.time1quant, signal=np.arange(10.0, 30.0, 2.0),
                                        units="mV", name='spam', description='eggs',
                                        file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(targ)

        assert_array_equal(result, targ)
        assert_array_equal(self.time1quant, targ.times)
        assert_array_equal(result.times, targ.times)
        assert_same_sub_schema(result, targ)

    def test__add_signals_with_inconsistent_times_AssertionError(self):
        signal2 = IrregularlySampledSignal(self.time1quant * 2., signal=np.arange(10.0),
                                           units="mV")
        assert_neo_object_is_compliant(signal2)

        self.assertRaises(ValueError, self.signal1.__add__, signal2)

    def test__add_signals_with_inconsistent_dimension_ValueError(self):
        signal2 = np.arange(20).reshape(2, 10)

        self.assertRaises(ValueError, self.signal1.__add__, signal2)

    def test__subtract_const_should_preserve_data_complement(self):
        result = self.signal1 - 65 * pq.mV
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(self.signal1[9], 9 * pq.mV)
        self.assertEqual(result[9], -56 * pq.mV)
        assert_array_equal(result.magnitude, (self.data1 - 65).reshape(-1, 1))
        assert_array_equal(result.times, self.time1quant)

    def test__subtract_from_const_should_return_signal(self):
        result = 10 * pq.mV - self.signal1
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(self.signal1[9], 9 * pq.mV)
        self.assertEqual(result[9], 1 * pq.mV)
        assert_array_equal(result.magnitude, (10 - self.data1).reshape(-1, 1))
        assert_array_equal(result.times, self.time1quant)

    def test__mult_signal_by_const_float_should_preserve_data_complement(self):
        result = self.signal1 * 2.
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(self.signal1[9], 9 * pq.mV)
        self.assertEqual(result[9], 18 * pq.mV)
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1) * 2)
        assert_array_equal(result.times, self.time1quant)

    def test__mult_signal_by_const_array_should_preserve_data_complement(self):
        result = self.signal1 * np.array(2.)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(self.signal1[9], 9 * pq.mV)
        self.assertEqual(result[9], 18 * pq.mV)
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1) * 2)
        assert_array_equal(result.times, self.time1quant)

    def test__divide_signal_by_const_should_preserve_data_complement(self):
        result = self.signal1 / 0.5
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([23]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['A']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(self.signal1[9], 9 * pq.mV)
        self.assertEqual(result[9], 18 * pq.mV)
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1) / 0.5)
        assert_array_equal(result.times, self.time1quant)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        res = pretty(self.signal1)
        signal = self.signal1
        targ = (("IrregularlySampledSignal with %d channels of length %d; units %s; datatype %s \n"
                 "" % (signal.shape[1], signal.shape[0], signal.units.dimensionality.unicode,
                       signal.dtype))
                + ("name: '{}'\ndescription: '{}'\n".format(signal.name, signal.description))
                + ("annotations: %s\n" % str(signal.annotations))
                + ("sample times: {}".format(signal.times[:10])))
        self.assertEqual(res, targ)

    def test__merge(self):
        data1 = np.arange(1000.0, 1066.0).reshape((11, 6)) * pq.uV
        data2 = np.arange(2.0, 2.033, 0.001).reshape((11, 3)) * pq.mV
        times1 = np.arange(11.0) * pq.ms
        times2 = np.arange(1.0, 12.0) * pq.ms
        arr_ann1 = {'anno1': np.arange(6), 'anno2': ['a', 'b', 'c', 'd', 'e', 'f']}
        arr_ann2 = {'anno1': np.arange(100, 103), 'anno3': []}

        signal1 = IrregularlySampledSignal(times1, data1, name='signal1',
                                           description='test signal', file_origin='testfile.txt',
                                           array_annotations=arr_ann1)
        signal2 = IrregularlySampledSignal(times1, data2, name='signal2',
                                           description='test signal', file_origin='testfile.txt',
                                           array_annotations=arr_ann2)
        signal3 = IrregularlySampledSignal(times2, data2, name='signal3',
                                           description='test signal', file_origin='testfile.txt')

        with warnings.catch_warnings(record=True) as w:
            merged12 = signal1.merge(signal2)

            self.assertTrue(len(w) == 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertSequenceEqual(str(w[0].message), "The following array annotations were "
                                                        "omitted, because they were only present"
                                                        " in one of the merged objects: "
                                                        "['anno2'] from the one that was merged "
                                                        "into and ['anno3'] from the one that "
                                                        "was merged into the other")

        target_data12 = np.hstack([data1, data2.rescale(pq.uV)])

        assert_neo_object_is_compliant(signal1)
        assert_neo_object_is_compliant(signal2)
        assert_neo_object_is_compliant(merged12)

        self.assertAlmostEqual(merged12[5, 0], 1030.0 * pq.uV, 9)
        self.assertAlmostEqual(merged12[5, 6], 2015.0 * pq.uV, 9)

        self.assertEqual(merged12.name, 'merge(signal1, signal2)')
        self.assertEqual(merged12.file_origin, 'testfile.txt')
        assert_arrays_equal(merged12.array_annotations['anno1'],
                            np.array([0, 1, 2, 3, 4, 5, 100, 101, 102]))
        self.assertIsInstance(merged12.array_annotations, ArrayDict)

        assert_arrays_equal(merged12.magnitude, target_data12)

        self.assertRaises(MergeError, signal1.merge, signal3)

    def test_concatenate_simple(self):
        signal1 = IrregularlySampledSignal(signal=[0, 1, 2, 3] * pq.s,
                                           times=[1, 10, 11, 14] * pq.s)
        signal2 = IrregularlySampledSignal(signal=[4, 5, 6] * pq.s, times=[15, 16, 21] * pq.s)

        result = signal1.concatenate(signal2)
        assert_array_equal(np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1, 1)), result.magnitude)
        assert_array_equal(np.array([1, 10, 11, 14, 15, 16, 21]), result.times)
        for attr in signal1._necessary_attrs:
            if attr[0] == 'times':
                continue
            self.assertEqual(getattr(signal1, attr[0], None), getattr(result, attr[0], None))

    def test_concatenate_no_overlap(self):
        signal1 = IrregularlySampledSignal(signal=[0, 1, 2, 3] * pq.s, times=range(4) * pq.s)
        signal2 = IrregularlySampledSignal(signal=[4, 5, 6] * pq.s, times=range(4, 7) * pq.s)

        for allow_overlap in [True, False]:
            result = signal1.concatenate(signal2, allow_overlap=allow_overlap)
            assert_array_equal(np.arange(7).reshape((-1, 1)), result.magnitude)
            assert_array_equal(np.arange(7), result.times)

    def test_concatenate_overlap_exception(self):
        signal1 = IrregularlySampledSignal(signal=[0, 1, 2, 3] * pq.s, times=range(4) * pq.s)
        signal2 = IrregularlySampledSignal(signal=[4, 5, 6] * pq.s, times=range(2, 5) * pq.s)

        self.assertRaises(ValueError, signal1.concatenate, signal2, allow_overlap=False)

    def test_concatenate_overlap(self):
        signal1 = IrregularlySampledSignal(signal=[0, 1, 2, 3] * pq.s, times=range(4) * pq.s)
        signal2 = IrregularlySampledSignal(signal=[4, 5, 6] * pq.s, times=range(2, 5) * pq.s)

        result = signal1.concatenate(signal2, allow_overlap=True)
        assert_array_equal(np.array([0, 1, 2, 4, 3, 5, 6]).reshape((-1, 1)), result.magnitude)
        assert_array_equal(np.array([0, 1, 2, 2, 3, 3, 4]), result.times)

    def test_concatenate_multi_trace(self):
        data1 = np.arange(4).reshape(2, 2)
        data2 = np.arange(4, 8).reshape(2, 2)
        n1 = len(data1)
        n2 = len(data2)
        signal1 = IrregularlySampledSignal(signal=data1 * pq.s, times=range(n1) * pq.s)
        signal2 = IrregularlySampledSignal(signal=data2 * pq.s, times=range(n1, n1 + n2) * pq.s)

        result = signal1.concatenate(signal2)
        data_expected = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        assert_array_equal(data_expected, result.magnitude)

    def test_concatenate_array_annotations(self):
        array_anno1 = {'first': ['a', 'b']}
        array_anno2 = {'first': ['a', 'b'],
                       'second': ['c', 'd']}
        data1 = np.arange(4).reshape(2, 2)
        data2 = np.arange(4, 8).reshape(2, 2)
        n1 = len(data1)
        n2 = len(data2)
        signal1 = IrregularlySampledSignal(signal=data1 * pq.s, times=range(n1) * pq.s,
                                           array_annotations=array_anno1)
        signal2 = IrregularlySampledSignal(signal=data2 * pq.s, times=range(n1, n1 + n2) * pq.s,
                                           array_annotations=array_anno2)

        result = signal1.concatenate(signal2)
        assert_array_equal(array_anno1.keys(), result.array_annotations.keys())

        for k in array_anno1.keys():
            assert_array_equal(np.asarray(array_anno1[k]), result.array_annotations[k])


class TestAnalogSignalFunctions(unittest.TestCase):
    def test__pickle(self):
        signal1 = IrregularlySampledSignal(np.arange(10.0) / 100 * pq.s, np.arange(10.0),
                                           units="mV")

        fobj = open('./pickle', 'wb')
        pickle.dump(signal1, fobj)
        fobj.close()

        fobj = open('./pickle', 'rb')
        try:
            signal2 = pickle.load(fobj)
        except ValueError:
            signal2 = None

        assert_array_equal(signal1, signal2)
        fobj.close()
        os.remove('./pickle')


class TestIrregularlySampledSignalEquality(unittest.TestCase):
    def test__signals_with_different_times_should_be_not_equal(self):
        signal1 = IrregularlySampledSignal(np.arange(10.0) / 100 * pq.s, np.arange(10.0),
                                           units="mV")
        signal2 = IrregularlySampledSignal(np.arange(10.0) / 100 * pq.ms, np.arange(10.0),
                                           units="mV")
        self.assertNotEqual(signal1, signal2)


if __name__ == "__main__":
    unittest.main()
