"""
Tests of the neo.core.analogsignal.AnalogSignal class and related functions
"""

import os
import pickle
import copy
import warnings
import unittest

import numpy as np
import quantities as pq

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

from numpy.testing import assert_array_equal
from neo.core.baseneo import MergeError
from neo.core.analogsignal import AnalogSignal, _get_sampling_rate
from neo.core import Segment

from neo.test.tools import (assert_arrays_almost_equal, assert_neo_object_is_compliant,
                            assert_same_sub_schema,
                            assert_same_attributes, assert_same_sub_schema, assert_arrays_equal,
                            assert_same_annotations, assert_same_array_annotations)


class TestAnalogSignalConstructor(unittest.TestCase):

    def test__create_from_list(self):
        data = range(10)
        rate = 1000 * pq.Hz
        signal = AnalogSignal(data, sampling_rate=rate, units="mV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, len(data) / rate)
        self.assertEqual(signal[9, 0], 9000 * pq.uV)

    def test__create_from_2d_list(self):
        data = [(i, i, i) for i in range(10)]  # 3 signals each with 10 samples
        rate = 1000 * pq.Hz
        signal = AnalogSignal(data, sampling_rate=rate, units="mV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.shape, (10, 3))
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, len(data) / rate)
        self.assertEqual(signal[9, 0], 9000 * pq.uV)

    def test__create_from_1d_np_array(self):
        data = np.arange(10.0)
        rate = 1 * pq.kHz
        signal = AnalogSignal(data, sampling_rate=rate, units="uV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, data.size / rate)
        self.assertEqual(signal[9, 0], 0.009 * pq.mV)

    def test__create_from_2d_numpy_array(self):
        data = np.arange(20.0).reshape((10, 2))
        rate = 1 * pq.kHz
        signal = AnalogSignal(data, sampling_rate=rate, units="uV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, data.shape[0] / rate)
        self.assertEqual(signal[9, 0], 0.018 * pq.mV)
        self.assertEqual(signal[9, 1], 19 * pq.uV)

    def test__create_from_1d_quantities_array(self):
        data = np.arange(10.0) * pq.mV
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, sampling_rate=rate)
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, data.size / rate)
        self.assertEqual(signal[9, 0], 0.009 * pq.V)

    def test__create_from_2d_quantities_array(self):
        data = np.arange(20.0).reshape((10, 2)) * pq.mV
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, sampling_rate=rate)
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0 * pq.ms)
        self.assertEqual(signal.t_stop, data.shape[0] / rate)
        self.assertEqual(signal[9, 0], 18000 * pq.uV)

    def test__create_from_array_no_units_ValueError(self):
        data = np.arange(10.0)
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=1 * pq.kHz)

    def test__create_from_1d_quantities_array_inconsistent_units_ValueError(self):
        data = np.arange(10.0) * pq.mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=1 * pq.kHz, units="nA")

    def test__create_from_2d_quantities_with_inconsistent_units_ValueError(self):
        data = np.arange(20.0).reshape((10, 2)) * pq.mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=1 * pq.kHz, units="nA")

    def test__create_without_sampling_rate_or_period_ValueError(self):
        data = np.arange(10.0) * pq.mV
        self.assertRaises(ValueError, AnalogSignal, data)

    def test__create_with_None_sampling_rate_should_raise_ValueError(self):
        data = np.arange(10.0) * pq.mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=None)

    def test__create_with_None_t_start_should_raise_ValueError(self):
        data = np.arange(10.0) * pq.mV
        rate = 5000 * pq.Hz
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=rate, t_start=None)

    def test__create_inconsistent_sampling_rate_and_period_ValueError(self):
        data = np.arange(10.0) * pq.mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=1 * pq.kHz,
                          sampling_period=5 * pq.s)

    def test__create_with_copy_true_should_return_copy(self):
        data = np.arange(10.0) * pq.mV
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, copy=True, sampling_rate=rate)
        data[3] = 99 * pq.mV
        assert_neo_object_is_compliant(signal)
        self.assertNotEqual(signal[3, 0], 99 * pq.mV)

    def test__create_with_copy_false_should_return_view(self):
        data = np.arange(10.0) * pq.mV
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, copy=False, sampling_rate=rate)
        data[3] = 99 * pq.mV
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal[3, 0], 99 * pq.mV)

    def test__create2D_with_copy_false_should_return_view(self):
        data = np.arange(10.0) * pq.mV
        data = data.reshape((5, 2))
        rate = 5000 * pq.Hz
        signal = AnalogSignal(data, copy=False, sampling_rate=rate)
        data[3, 0] = 99 * pq.mV
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal[3, 0], 99 * pq.mV)

    def test__create_with_additional_argument(self):
        signal = AnalogSignal([1, 2, 3], units="mV", sampling_rate=1 * pq.kHz,
                              file_origin='crack.txt', ratname='Nicolas')
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.annotations, {'ratname': 'Nicolas'})

        # This one is universally recommended and handled by BaseNeo
        self.assertEqual(signal.file_origin, 'crack.txt')

        # signal must be 1D - should raise Exception if not 1D


class TestAnalogSignalProperties(unittest.TestCase):
    def setUp(self):
        self.t_start = [0.0 * pq.ms, 100 * pq.ms, -200 * pq.ms] * 2
        self.rates = [1 * pq.kHz, 420 * pq.Hz, 999 * pq.Hz] * 2
        self.rates2 = [2 * pq.kHz, 290 * pq.Hz, 1111 * pq.Hz] * 2
        self.data_1d = [np.arange(10.0) * pq.nA, np.arange(-100.0, 100.0, 10.0) * pq.mV,
                        np.random.uniform(size=100) * pq.uV]
        self.data_2d = [np.arange(10.0).reshape((5, 2)) * pq.nA,
                        np.arange(-100.0, 100.0, 10.0).reshape((4, 5)) * pq.mV,
                        np.random.uniform(size=(100, 4)) * pq.uV]
        self.data = self.data_1d + self.data_2d
        self.signals = [AnalogSignal(D, sampling_rate=r, t_start=t, testattr='test') for r, D, t
                        in zip(self.rates, self.data, self.t_start)]

    def test__compliant(self):
        for signal in self.signals:
            assert_neo_object_is_compliant(signal)

    def test__t_stop(self):
        for i, signal in enumerate(self.signals):
            targ = self.t_start[i] + self.data[i].shape[0] / self.rates[i]
            self.assertEqual(signal.t_stop, targ)

    def test__duration(self):
        for signal in self.signals:
            self.assertAlmostEqual(signal.duration, signal.t_stop - signal.t_start, delta=1e-15)

    def test__sampling_rate_getter(self):
        for signal, rate in zip(self.signals, self.rates):
            self.assertEqual(signal.sampling_rate, rate)

    def test__sampling_period_getter(self):
        for signal, rate in zip(self.signals, self.rates):
            self.assertEqual(signal.sampling_period, 1 / rate)

    def test__sampling_rate_setter(self):
        for signal, rate in zip(self.signals, self.rates2):
            signal.sampling_rate = rate
            assert_neo_object_is_compliant(signal)
            self.assertEqual(signal.sampling_rate, rate)
            self.assertEqual(signal.sampling_period, 1 / rate)

    def test__sampling_period_setter(self):
        for signal, rate in zip(self.signals, self.rates2):
            signal.sampling_period = 1 / rate
            assert_neo_object_is_compliant(signal)
            self.assertEqual(signal.sampling_rate, rate)
            self.assertEqual(signal.sampling_period, 1 / rate)

    def test__sampling_rate_setter_None_ValueError(self):
        self.assertRaises(ValueError, setattr, self.signals[0], 'sampling_rate', None)

    def test__sampling_rate_setter_not_quantity_ValueError(self):
        self.assertRaises(ValueError, setattr, self.signals[0], 'sampling_rate', 5.5)

    def test__sampling_period_setter_None_ValueError(self):
        signal = self.signals[0]
        assert_neo_object_is_compliant(signal)
        self.assertRaises(ValueError, setattr, signal, 'sampling_period', None)

    def test__sampling_period_setter_not_quantity_ValueError(self):
        self.assertRaises(ValueError, setattr, self.signals[0], 'sampling_period', 5.5)

    def test__t_start_setter_None_ValueError(self):
        signal = self.signals[0]
        assert_neo_object_is_compliant(signal)
        self.assertRaises(ValueError, setattr, signal, 't_start', None)

    def test__times_getter(self):
        for i, signal in enumerate(self.signals):
            targ = np.arange(self.data[i].shape[0])
            targ = targ / self.rates[i] + self.t_start[i]
            assert_arrays_almost_equal(signal.times, targ, 1e-12 * pq.ms)

    def test__duplicate_with_new_data(self):
        signal1 = self.signals[1]
        signal2 = self.signals[2]
        data2 = self.data[2]
        signal1.array_annotate(ann=np.arange(signal1.shape[-1]))
        signal1b = signal1.duplicate_with_new_data(data2)
        assert_arrays_almost_equal(np.asarray(signal1b), np.asarray(signal2 / 1000.), 1e-12)
        self.assertEqual(signal1b.t_start, signal1.t_start)
        self.assertEqual(signal1b.sampling_rate, signal1.sampling_rate)
        # After duplicating, array annotations should always be empty,
        # because different length of data would cause inconsistencies
        self.assertEqual(signal1b.array_annotations, {})
        self.assertIsInstance(signal1b.array_annotations, ArrayDict)

    def test__children(self):
        signal = self.signals[0]

        segment = Segment(name='seg1')
        segment.analogsignals = [signal]
        segment.create_many_to_one_relationship()

        self.assertEqual(signal._parent_objects, ('Segment',))

        self.assertEqual(signal._parent_containers, ('segment',))

        self.assertEqual(signal._parent_objects, ('Segment',))
        self.assertEqual(signal._parent_containers, ('segment',))

        self.assertEqual(len(signal.parents), 1)
        self.assertEqual(signal.parents[0].name, 'seg1')

        assert_neo_object_is_compliant(signal)

    def test__repr(self):
        for i, signal in enumerate(self.signals):
            prepr = repr(signal)
            #  reshaping 1d arrays for correct representation
            d = self.data[i]
            if len(d.shape) == 1:
                d = d.reshape(-1, 1)
            targ = '<AnalogSignal(%s, [%s, %s], sampling rate: %s)>' \
                   '' % (repr(d), self.t_start[i],
                         self.t_start[i] + len(self.data[i]) / self.rates[i], self.rates[i])
            self.assertEqual(prepr, targ)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        for i, signal in enumerate(self.signals):
            prepr = pretty(signal)
            targ = (('AnalogSignal with %d channels of length %d; units %s; datatype %s \n'
                     '' % (signal.shape[1], signal.shape[0],
                           signal.units.dimensionality.unicode, signal.dtype))
                    + ('annotations: %s\n' % signal.annotations)
                    + ('sampling rate: {} {}\n'.format(
                        float(signal.sampling_rate),
                        signal.sampling_rate.dimensionality.unicode))
                    + ('time: {} {} to {} {}'.format(float(signal.t_start),
                                                     signal.t_start.dimensionality.unicode,
                                                     float(signal.t_stop),
                                                     signal.t_stop.dimensionality.unicode)))
            self.assertEqual(prepr, targ)


class TestAnalogSignalArrayMethods(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(55.0).reshape((11, 5))
        self.data1quant = self.data1 * pq.nA
        self.arr_ann1 = {'anno1': np.arange(5), 'anno2': ['a', 'b', 'c', 'd', 'e']}
        self.signal1 = AnalogSignal(self.data1quant, sampling_rate=1 * pq.kHz, name='spam',
                                    description='eggs', file_origin='testfile.txt',
                                    array_annotations=self.arr_ann1, arg1='test')
        self.signal1.segment = Segment()

        self.data2 = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]).T
        self.data2quant = self.data2 * pq.mV
        self.arr_ann2 = {'anno1': [10, 11], 'anno2': ['k', 'l']}
        self.signal2 = AnalogSignal(self.data2quant, sampling_rate=1.0 * pq.Hz, name='spam',
                                    description='eggs', file_origin='testfile.txt',
                                    array_annotations=self.arr_ann2, arg1='test')

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})
        assert_arrays_equal(self.signal1.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(self.signal1.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(self.signal1.array_annotations, ArrayDict)

        assert_neo_object_is_compliant(self.signal2)
        self.assertEqual(self.signal2.name, 'spam')
        self.assertEqual(self.signal2.description, 'eggs')
        self.assertEqual(self.signal2.file_origin, 'testfile.txt')
        self.assertEqual(self.signal2.annotations, {'arg1': 'test'})
        assert_arrays_equal(self.signal2.array_annotations['anno1'], np.array([10, 11]))
        assert_arrays_equal(self.signal2.array_annotations['anno2'], np.array(['k', 'l']))
        self.assertIsInstance(self.signal2.array_annotations, ArrayDict)

    def test__slice_should_return_AnalogSignal(self):
        # slice
        for index in (0, np.int64(0)):
            result = self.signal1[3:8, index]
            self.assertIsInstance(result, AnalogSignal)
            assert_neo_object_is_compliant(result)
            # should slicing really preserve name and description?
            self.assertEqual(result.name, 'spam')
            # perhaps these should be modified to indicate the slice?
            self.assertEqual(result.description, 'eggs')
            self.assertEqual(result.file_origin, 'testfile.txt')
            self.assertEqual(result.annotations, {'arg1': 'test'})
            # Array annotations remain the same, because number of signals was not altered
            self.assertEqual(result.array_annotations, {'anno1': np.arange(1), 'anno2': ['a']})
            self.assertIsInstance(result.array_annotations, ArrayDict)

            self.assertEqual(result.size, 5)
            self.assertEqual(result.sampling_period, self.signal1.sampling_period)
            self.assertEqual(result.sampling_rate, self.signal1.sampling_rate)
            self.assertEqual(result.t_start, self.signal1.t_start + 3 * result.sampling_period)
            self.assertEqual(result.t_stop, result.t_start + 5 * result.sampling_period)
            assert_array_equal(result.magnitude, self.data1[3:8, index:index + 1])

            # Test other attributes were copied over (in this case, defaults)
            self.assertEqual(result.file_origin, self.signal1.file_origin)
            self.assertEqual(result.name, self.signal1.name)
            self.assertEqual(result.description, self.signal1.description)
            self.assertEqual(result.annotations, self.signal1.annotations)

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
        assert_arrays_equal(result.array_annotations['anno1'], np.array([10, 11]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['k', 'l']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        targ = AnalogSignal(np.array([[2., 3.], [2., 3.]]).T, t_start=12.0 * pq.ms,
                            sampling_rate=1.0 * pq.Hz, units='mV', name='spam', description='eggs',
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
        assert_arrays_equal(result.array_annotations['anno1'], np.array([10, 11]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['k', 'l']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        targ = AnalogSignal(np.array([[2., 3.], [2., 3.]]).T, t_start=t_start.rescale(pq.ms),
                            sampling_rate=1.0 * pq.Hz, units='mV', name='spam', description='eggs',
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
            assert_arrays_equal(result.array_annotations['anno1'], np.array([10, 11]))
            assert_arrays_equal(result.array_annotations['anno2'], np.array(['k', 'l']))
            self.assertIsInstance(result.array_annotations, ArrayDict)

            targ_ind = np.where(
                (self.signal2.times >= t_start_targ) & (self.signal2.times < t_stop_targ))
            targ_array = self.signal2.magnitude[targ_ind]

            targ = AnalogSignal(targ_array, t_start=t_start_targ.rescale(pq.ms),
                                sampling_rate=1.0 * pq.Hz, units='mV', name='spam',
                                description='eggs', file_origin='testfile.txt', arg1='test')
            assert_neo_object_is_compliant(result)

            assert_neo_object_is_compliant(self.signal2)
            self.assertEqual(self.signal2.t_start, 10.0 * pq.ms)
            self.assertAlmostEqual(result.t_stop, t_stop_targ, delta=1e-12 * pq.ms)
            self.assertAlmostEqual(result.t_start, t_start_targ, delta=1e-12 * pq.ms)
            assert_arrays_almost_equal(result.times, targ.times, 1e-12 * pq.ms)
            self.assertEqual(result.sampling_rate, targ.sampling_rate)
            assert_array_equal(result.magnitude, targ.magnitude)
            assert_same_sub_schema(result, targ)

    def test__time_slice_deepcopy_data(self):
        result = self.signal1.time_slice(None, None)

        # Change values of original array
        self.signal1[2] = 7.3 * self.signal1.units

        np.testing.assert_raises(AssertionError, assert_array_equal, self.signal1, result)

        # Change values of sliced array
        result[3] = 9.5 * result.units

        np.testing.assert_raises(AssertionError, assert_array_equal, self.signal1, result)

    def test__slice_should_change_sampling_period(self):
        result1 = self.signal1[:2, 0]
        result2 = self.signal1[::2, 0]
        result3 = self.signal1[1:7:2, 0]

        self.assertIsInstance(result1, AnalogSignal)
        assert_neo_object_is_compliant(result1)
        self.assertEqual(result1.name, 'spam')
        self.assertEqual(result1.description, 'eggs')
        self.assertEqual(result1.file_origin, 'testfile.txt')
        self.assertEqual(result1.annotations, {'arg1': 'test'})
        assert_arrays_equal(result1.array_annotations['anno1'], np.arange(1))
        assert_arrays_equal(result1.array_annotations['anno2'], np.array(['a']))
        self.assertIsInstance(result1.array_annotations, ArrayDict)

        self.assertIsInstance(result2, AnalogSignal)
        assert_neo_object_is_compliant(result2)
        self.assertEqual(result2.name, 'spam')
        self.assertEqual(result2.description, 'eggs')
        self.assertEqual(result2.file_origin, 'testfile.txt')
        self.assertEqual(result2.annotations, {'arg1': 'test'})
        assert_arrays_equal(result2.array_annotations['anno1'], np.arange(1))
        assert_arrays_equal(result2.array_annotations['anno2'], np.array(['a']))
        self.assertIsInstance(result2.array_annotations, ArrayDict)

        self.assertIsInstance(result3, AnalogSignal)
        assert_neo_object_is_compliant(result3)
        self.assertEqual(result3.name, 'spam')
        self.assertEqual(result3.description, 'eggs')
        self.assertEqual(result3.file_origin, 'testfile.txt')
        self.assertEqual(result3.annotations, {'arg1': 'test'})
        assert_arrays_equal(result3.array_annotations['anno1'], np.arange(1))
        assert_arrays_equal(result3.array_annotations['anno2'], np.array(['a']))
        self.assertIsInstance(result3.array_annotations, ArrayDict)

        self.assertEqual(result1.sampling_period, self.signal1.sampling_period)
        self.assertEqual(result2.sampling_period, self.signal1.sampling_period * 2)
        self.assertEqual(result3.sampling_period, self.signal1.sampling_period * 2)

        assert_array_equal(result1.magnitude, self.data1[:2, :1])
        assert_array_equal(result2.magnitude, self.data1[::2, :1])
        assert_array_equal(result3.magnitude, self.data1[1:7:2, :1])

    def test__index_dim1_and_slice_dim0_should_return_single_channel_analogsignal(self):
        result = self.signal1[2:7, 0]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.shape, (5, 1))
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([0]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['a']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.t_start, self.signal1.t_start + 2 * self.signal1.sampling_period)
        self.assertEqual(result.t_stop, self.signal1.t_start + 7 * self.signal1.sampling_period)
        self.assertEqual(result.sampling_rate, self.signal1.sampling_rate)
        assert_arrays_equal(result, self.data1[2:7, 0].reshape(-1, 1))

    def test__index_dim0_should_return_quantity_array(self):
        # i.e. values from all signals for a single point in time
        result = self.signal1[3, :]
        self.assertIsInstance(result, pq.Quantity)
        self.assertFalse(hasattr(result, 'name'))
        self.assertFalse(hasattr(result, 'description'))
        self.assertFalse(hasattr(result, 'file_origin'))
        self.assertFalse(hasattr(result, 'annotations'))
        self.assertFalse(hasattr(result, 'array_annotations'))

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
        self.assertFalse(hasattr(result, 'array_annotations'))

        self.assertEqual(result.shape, (3,))
        self.assertFalse(hasattr(result, "t_start"))
        self.assertEqual(result.units, pq.nA)
        assert_arrays_equal(result, self.data1[3, 2:5])

    def test__invalid_index_raises_IndexError(self):
        self.assertRaises(IndexError, self.signal1.__getitem__, 5.)
        self.assertRaises(IndexError, self.signal1.__getitem__, '5.')

    def test__getitem_should_return_single_quantity(self):
        # quantities drops the units in this case
        self.assertEqual(self.signal1[9, 3], 48000 * pq.pA)
        self.assertEqual(self.signal1[9][3], self.signal1[9, 3])
        self.assertTrue(hasattr(self.signal1[9, 3], 'units'))
        self.assertRaises(IndexError, self.signal1.__getitem__, (99, 73))

    def test__getitem__with_tuple_slice_none(self):
        self.assertRaises(IndexError, self.signal1.__getitem__, (slice(None), None))

    def test__time_index(self):
        # scalar arguments
        self.assertEqual(self.signal2.time_index(2.0 * pq.s), 2)
        self.assertEqual(self.signal2.time_index(1.99 * pq.s), 2)
        self.assertEqual(self.signal2.time_index(2.01 * pq.s), 2)

        # vector arguments
        assert_array_equal(self.signal2.time_index([2.0, 0.99, 3.01] * pq.s), [2, 1, 3])
        assert_array_equal(self.signal2.time_index([2.0] * pq.s), [2])

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
        assert_arrays_equal(result.array_annotations['anno1'], np.array([10, 11]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['k', 'l']))

        targ = AnalogSignal(np.array([[2., 3.], [2., 3.]]).T, sampling_rate=1.0 * pq.Hz,
                            units='mV', t_start=t_start, name='spam', description='eggs',
                            file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(result)

        self.assertEqual(result.t_stop, t_stop)
        self.assertEqual(result.t_start, t_start)
        self.assertEqual(result.sampling_rate, targ.sampling_rate)
        assert_array_equal(result, targ)
        assert_same_sub_schema(result, targ)

    def test__slice_both_dimensions_should_return_analogsignal(self):
        result = self.signal1[0:3, 0:3]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.array([0, 1, 2]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['a', 'b', 'c']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        targ = AnalogSignal([[0, 1, 2], [5, 6, 7], [10, 11, 12]], dtype=float, units="nA",
                            sampling_rate=1 * pq.kHz, name='spam', description='eggs',
                            file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(targ)

        self.assertEqual(result.t_stop, targ.t_stop)
        self.assertEqual(result.t_start, targ.t_start)
        self.assertEqual(result.sampling_rate, targ.sampling_rate)
        self.assertEqual(result.shape, targ.shape)
        assert_same_sub_schema(result, targ)
        assert_arrays_equal(result, self.data1[0:3, 0:3])

    def test__slice_only_first_dimension_should_return_analogsignal(self):
        result = self.signal1[2:7]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.t_start, self.signal1.t_start + 2 * self.signal1.sampling_period)
        self.assertEqual(result.t_stop, self.signal1.t_start + 7 * self.signal1.sampling_period)
        self.assertEqual(result.sampling_rate, self.signal1.sampling_rate)
        assert_arrays_equal(result, self.data1[2:7])

    def test__time_slice_should_set_parents_to_None(self):
        # When timeslicing, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.signal1.time_slice(1 * pq.ms, 3 * pq.ms)
        self.assertEqual(result.segment, None)

    def test__time_slice_close_to_sample_boundaries(self):
        # see issue 530

        sig = AnalogSignal(np.arange(25000) * pq.uV,
                           t_start=0 * pq.ms,
                           sampling_rate=25 * pq.kHz)

        window_size = 3.0 * pq.ms

        expected_shape = int(np.rint((window_size * sig.sampling_rate).simplified.magnitude))

        # test with random times
        t_start = (window_size / 2).magnitude
        t_stop = (sig.t_stop.rescale(pq.ms) - window_size / 2).magnitude
        for t in np.random.uniform(t_start, t_stop, size=1000):
            tq = t * pq.ms
            sliced_sig = sig.time_slice(tq - window_size / 2, tq + window_size / 2)
            self.assertEqual(expected_shape, sliced_sig.shape[0])

        # test with times on or close to sample boundaries
        for i in np.random.randint(1000, sig.size - 1000, size=1000):
            tq = i * sig.sampling_period
            sliced_sig = sig.time_slice(tq - window_size / 2, tq + window_size / 2)
            self.assertEqual(expected_shape, sliced_sig.shape[0])

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

    def test__copy_should_let_access_to_parents_objects(self):
        result = self.signal1.copy()
        self.assertIs(result.segment, self.signal1.segment)

    def test__deepcopy_should_set_parents_objects_to_None(self):
        # Deepcopy should destroy references to parents
        result = copy.deepcopy(self.signal1)
        self.assertEqual(result.segment, None)

    def test__getitem_should_return_single_quantity(self):
        result1 = self.signal1[0, 0]
        result2 = self.signal1[1, 4]

        self.assertIsInstance(result1, pq.Quantity)
        self.assertFalse(hasattr(result1, 'name'))
        self.assertFalse(hasattr(result1, 'description'))
        self.assertFalse(hasattr(result1, 'file_origin'))
        self.assertFalse(hasattr(result1, 'annotations'))
        self.assertFalse(hasattr(result1, 'array_annotations'))

        self.assertIsInstance(result2, pq.Quantity)
        self.assertFalse(hasattr(result2, 'name'))
        self.assertFalse(hasattr(result2, 'description'))
        self.assertFalse(hasattr(result2, 'file_origin'))
        self.assertFalse(hasattr(result2, 'annotations'))
        self.assertFalse(hasattr(result2, 'array_annotations'))

        self.assertEqual(result1, 0 * pq.nA)
        self.assertEqual(result2, 9 * pq.nA)

    def test__getitem_out_of_bounds_IndexError(self):
        self.assertRaises(IndexError, self.signal1.__getitem__, (11, 0))

    def test__time_slice__out_of_bounds_ValueError(self):
        t_start_good = 2 * pq.s
        t_stop_good = 4 * pq.s
        t_start_bad = -2 * pq.s
        t_stop_bad = 40 * pq.s

        self.assertRaises(ValueError, self.signal2.time_slice, t_start_good, t_stop_bad)
        self.assertRaises(ValueError, self.signal2.time_slice, t_start_bad, t_stop_good)
        self.assertRaises(ValueError, self.signal2.time_slice, t_start_bad, t_stop_bad)

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
        assert_arrays_equal(result.array_annotations['anno1'], np.array([10, 11]))
        assert_arrays_equal(result.array_annotations['anno2'], np.array(['k', 'l']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.t_stop, t_stop)
        self.assertEqual(result.t_start, t_start)
        assert_array_equal(result, self.signal2)
        assert_same_sub_schema(result, self.signal2)

    def test_comparison_operators_in_1d(self):
        assert_array_equal(self.signal1[:, 0] >= 25 * pq.nA,
                           np.array([False, False, False, False, False, True, True, True, True,
                                     True, True]).reshape(-1, 1))
        assert_array_equal(self.signal1[:, 0] >= 25 * pq.pA, np.array(
            [False, True, True, True, True, True, True, True, True, True, True]).reshape(-1, 1))
        assert_array_equal(self.signal1[:, 0] == 25 * pq.nA,
                           np.array([False, False, False, False, False, True, False, False,
                                     False, False, False]).reshape(-1, 1))
        assert_array_equal(self.signal1[:, 0] == self.signal1[:, 0], np.array(
            [True, True, True, True, True, True, True, True, True, True, True]).reshape(-1, 1))

    def test_comparison_operators_in_2d(self):
        assert_arrays_equal(self.signal1[0:3, 0:3] >= 5 * pq.nA, np.array(
            [[False, False, False], [True, True, True], [True, True, True]]))
        assert_arrays_equal(self.signal1[0:3, 0:3] >= 5 * pq.pA, np.array(
            [[False, True, True], [True, True, True], [True, True, True]]))
        assert_arrays_equal(self.signal1[0:3, 0:3] == 5 * pq.nA, np.array(
            [[False, False, False], [True, False, False], [False, False, False]]))
        assert_arrays_equal(self.signal1[0:3, 0:3] == self.signal1[0:3, 0:3],
                            np.array([[True, True, True], [True, True, True],
                                      [True, True, True]]))

    def test__comparison_as_indexing_single_trace(self):
        self.assertEqual(self.signal1[:, 0][self.signal1[:, 0] == 5], [5 * pq.mV])

    def test__comparison_as_indexing_double_trace(self):
        signal = AnalogSignal(np.arange(20).reshape((-1, 2)) * pq.V, sampling_rate=1 * pq.Hz)
        assert_array_equal(signal[signal < 10],
                           np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]).T * pq.V)

    def test__indexing_keeps_order_across_channels(self):
        # AnalogSignals with 10 traces each having 5 samples (eg. data[0] = [0,10,20,30,40])
        data = np.array([range(10), range(10, 20), range(20, 30), range(30, 40), range(40, 50)])
        mask = np.full((5, 10), fill_value=False, dtype=bool)
        # selecting one entry per trace
        mask[[0, 1, 0, 3, 0, 2, 4, 3, 1, 4], range(10)] = True

        signal = AnalogSignal(np.array(data) * pq.V, sampling_rate=1 * pq.Hz)
        assert_array_equal(signal[mask], np.array([[0, 11, 2, 33, 4, 25, 46, 37, 18, 49]]) * pq.V)

    def test__indexing_keeps_order_across_time(self):
        # AnalogSignals with 10 traces each having 5 samples (eg. data[0] = [0,10,20,30,40])
        data = np.array([range(10), range(10, 20), range(20, 30), range(30, 40), range(40, 50)])
        mask = np.full((5, 10), fill_value=False, dtype=bool)
        # selecting two entries per trace
        temporal_ids = [0, 1, 0, 3, 1, 2, 4, 2, 1, 4] + [4, 3, 2, 1, 0, 1, 2, 3, 2, 1]
        mask[temporal_ids, list(range(10)) + list(range(10))] = True

        signal = AnalogSignal(np.array(data) * pq.V, sampling_rate=1 * pq.Hz)
        assert_array_equal(signal[mask], np.array([[0, 11, 2, 13, 4, 15, 26, 27, 18, 19],
                                                   [40, 31, 22, 33, 14, 25, 46, 37, 28,
                                                    49]]) * pq.V)

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
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.units, 1 * pq.nA)
        assert_arrays_equal(result, self.data1)
        assert_same_sub_schema(result, self.signal1)

        self.assertIsInstance(result.segment, Segment)
        self.assertIs(result.segment, self.signal1.segment)

    def test__rescale_new(self):
        result = self.signal1.copy()
        result = result.rescale(pq.pA)

        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.units, 1 * pq.pA)
        assert_arrays_almost_equal(np.array(result), self.data1 * 1000., 1e-10)

        self.assertIsInstance(result.segment, Segment)
        self.assertIs(result.segment, self.signal1.segment)

    def test__rescale_new_incompatible_ValueError(self):
        self.assertRaises(ValueError, self.signal1.rescale, pq.mV)

    def test_as_array(self):
        sig_as_arr = self.signal1.as_array()
        self.assertIsInstance(sig_as_arr, np.ndarray)
        assert_array_equal(self.data1, sig_as_arr)

    def test_as_quantity(self):
        sig_as_q = self.signal1.as_quantity()
        self.assertIsInstance(sig_as_q, pq.Quantity)
        assert_array_equal(self.data1, sig_as_q.magnitude)

    def test_splice_1channel_inplace(self):
        signal_for_splicing = AnalogSignal([0.1, 0.1, 0.1], t_start=3 * pq.ms,
                                           sampling_rate=self.signal1.sampling_rate, units=pq.uA,
                                           array_annotations={'anno1': [0], 'anno2': ['C']})
        original_1d_signal = self.signal1[:, 0]
        result = original_1d_signal.splice(signal_for_splicing, copy=False)
        targ = np.arange(55).reshape((11, 5))
        targ[3:6, 0] = 100.
        assert_array_equal(self.signal1.magnitude, targ)
        assert_array_equal(self.signal1[:, 0], result)  # in-place
        self.assertEqual(result.segment, self.signal1.segment)
        assert_array_equal(self.signal1.array_annotations['anno1'], np.arange(5))
        assert_array_equal(self.signal1.array_annotations['anno2'],
                           np.array(['a', 'b', 'c', 'd', 'e']))
        assert_array_equal(result.array_annotations['anno1'], np.arange(1))
        assert_array_equal(result.array_annotations['anno2'], np.array(['a']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_splice_1channel_with_copy(self):
        signal_for_splicing = AnalogSignal([0.1, 0.1, 0.1], t_start=3 * pq.ms,
                                           sampling_rate=self.signal1.sampling_rate, units=pq.uA,
                                           array_annotations={'anno1': [0], 'anno2': ['C']})
        result = self.signal1[:, 0].splice(signal_for_splicing, copy=True)
        assert_array_equal(result.magnitude.flatten(),
                           np.array([0.0, 5.0, 10.0, 100.0, 100.0, 100.0, 30.0, 35.0, 40.0, 45.0,
                                     50.]))
        assert_array_equal(self.signal1.magnitude, np.arange(55).reshape(11, 5))
        self.assertIs(result.segment, None)
        assert_array_equal(result.array_annotations['anno1'], np.arange(1))
        assert_array_equal(result.array_annotations['anno2'], np.array(['a']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_splice_2channels_inplace(self):
        arr_ann1 = {'index': np.arange(10, 12)}
        arr_ann2 = {'index': np.arange(2), 'test': ['a', 'b']}
        signal = AnalogSignal(np.arange(20.0).reshape((10, 2)), sampling_rate=1 * pq.kHz,
                              units="mV", array_annotations=arr_ann1)
        signal_for_splicing = AnalogSignal(np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]),
                                           t_start=3 * pq.ms, array_annotations=arr_ann2,
                                           sampling_rate=self.signal1.sampling_rate, units=pq.V)
        result = signal.splice(signal_for_splicing, copy=False)
        assert_array_equal(result.magnitude, np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0],
             [12.0, 13.0], [14.0, 15.0], [16.0, 17.0], [18.0, 19.0]]))
        assert_array_equal(signal, result)  # in-place
        # Array annotations are taken from the main signal
        assert_array_equal(result.array_annotations['index'], np.arange(10, 12))
        self.assertIsInstance(result.array_annotations, ArrayDict)
        self.assertNotIn('test', result.array_annotations)

    def test_splice_1channel_invalid_t_start(self):
        signal_for_splicing = AnalogSignal([0.1, 0.1, 0.1], t_start=12 * pq.ms,
                                           # after the end of the signal
                                           sampling_rate=self.signal1.sampling_rate, units=pq.uA)
        self.assertRaises(ValueError, self.signal1.splice, signal_for_splicing, copy=False)

    def test_splice_1channel_invalid_t_stop(self):
        signal_for_splicing = AnalogSignal([0.1, 0.1, 0.1], t_start=9 * pq.ms,
                                           # too close to the end of the signal
                                           sampling_rate=self.signal1.sampling_rate, units=pq.uA)
        self.assertRaises(ValueError, self.signal1.splice, signal_for_splicing, copy=False)

    def test_splice_1channel_invalid_sampling_rate(self):
        signal_for_splicing = AnalogSignal([0.1, 0.1, 0.1], t_start=3 * pq.ms,
                                           sampling_rate=2 * self.signal1.sampling_rate,
                                           units=pq.uA)
        self.assertRaises(ValueError, self.signal1.splice, signal_for_splicing, copy=False)

    def test_splice_1channel_invalid_units(self):
        signal_for_splicing = AnalogSignal([0.1, 0.1, 0.1], t_start=3 * pq.ms,
                                           sampling_rate=self.signal1.sampling_rate, units=pq.uV)
        self.assertRaises(ValueError, self.signal1.splice, signal_for_splicing, copy=False)

    def test_array_annotations_getitem(self):
        data = np.arange(15).reshape(5, 3) * pq.mV
        arr_ann1 = [10, 15, 20]
        arr_ann2 = ['abc', 'def', 'ghi']
        arr_anns = {'index': arr_ann1, 'label': arr_ann2}
        signal = AnalogSignal(data, sampling_rate=30000 * pq.Hz, array_annotations=arr_anns)

        # A time slice of all signals is selected, so all array annotations need to remain
        result1 = signal[0:2]
        assert_arrays_equal(result1.array_annotations['index'], np.array(arr_ann1))
        assert_arrays_equal(result1.array_annotations['label'], np.array(arr_ann2))
        self.assertIsInstance(result1.array_annotations, ArrayDict)

        # Only elements from signal with index 2 are selected,
        # so only those array_annotations should be returned
        result2 = signal[1:2, 2]
        assert_arrays_equal(result2.array_annotations['index'], np.array([20]))
        assert_arrays_equal(result2.array_annotations['label'], np.array(['ghi']))
        self.assertIsInstance(result2.array_annotations, ArrayDict)
        # Because comparison of list with single element to scalar is possible,
        # we need to make sure that array_annotations remain arrays
        self.assertIsInstance(result2.array_annotations['index'], np.ndarray)
        self.assertIsInstance(result2.array_annotations['label'], np.ndarray)

        # Signals 0 and 1 are selected completely,
        # so their respective array_annotations should be returned
        result3 = signal[:, 0:2]
        assert_arrays_equal(result3.array_annotations['index'], np.array([10, 15]))
        assert_arrays_equal(result3.array_annotations['label'], np.array(['abc', 'def']))
        self.assertIsInstance(result3.array_annotations, ArrayDict)

    @unittest.skipUnless(HAVE_SCIPY, "requires Scipy")
    def test_downsample(self):
        # generate signal long enough for decimating
        data = np.sin(np.arange(1500) / 30).reshape(500, 3) * pq.mV
        signal = AnalogSignal(data, sampling_rate=30000 * pq.Hz)

        # test decimation using different decimation factors
        factors = [1, 9, 10, 11]
        for factor in factors:
            desired = signal[::factor].magnitude
            result = signal.downsample(factor)

            self.assertEqual(np.ceil(signal.shape[0] / factor), result.shape[0])
            self.assertEqual(signal.shape[-1],
                             result.shape[-1])  # preserve number of recording traces
            self.assertAlmostEqual(signal.sampling_rate, factor * result.sampling_rate)
            # only comparing center values due to border effects
            np.testing.assert_allclose(desired[3:-3], result.magnitude[3:-3], rtol=0.05, atol=0.1)

    @unittest.skipUnless(HAVE_SCIPY, "requires Scipy")
    def test_resample_less_samples(self):
        # generate signal long enough for resampling
        data = np.sin(np.arange(1500) / 30).reshape(3, 500).T * pq.mV
        signal = AnalogSignal(data, sampling_rate=30000 * pq.Hz)

        # test resampling using different numbers of desired samples
        sample_counts = [10, 100, 400]
        for sample_count in sample_counts:
            sample_ids = np.linspace(0, signal.shape[0], sample_count, dtype=int, endpoint=False)
            desired = signal.magnitude[sample_ids]
            result = signal.resample(sample_count)

            self.assertEqual(sample_count, result.shape[0])
            self.assertEqual(signal.shape[-1],
                             result.shape[-1])  # preserve number of recording traces
            self.assertAlmostEqual(sample_count / signal.shape[0] * signal.sampling_rate,
                                   result.sampling_rate)
            # only comparing center values due to border effects
            np.testing.assert_allclose(desired[3:-3], result.magnitude[3:-3], rtol=0.05, atol=0.1)

    @unittest.skipUnless(HAVE_SCIPY, "requires Scipy")
    def test_resample_more_samples(self):
        # generate signal long enough for resampling
        data = np.sin(np.arange(1500) / 100).T * pq.mV
        signal = AnalogSignal(data, sampling_rate=30000 * pq.Hz)

        # test resampling using different numbers of desired samples
        factor = 2
        sample_count = factor * signal.shape[0]
        desired = np.interp(np.arange(sample_count) / factor, np.arange(signal.shape[0]),
                            signal.magnitude.flatten()).reshape(-1, 1)
        result = signal.resample(sample_count)

        self.assertEqual(sample_count, result.shape[0])
        self.assertEqual(signal.shape[-1], result.shape[-1])  # preserve number of recording traces
        self.assertAlmostEqual(sample_count / signal.shape[0] * signal.sampling_rate,
                               result.sampling_rate)
        # only comparing center values due to border effects
        np.testing.assert_allclose(desired[10:-10], result.magnitude[10:-10], rtol=0.0, atol=0.1)

    @unittest.skipUnless(HAVE_SCIPY, "requires Scipy")
    def test_compare_resample_and_downsample(self):
        # generate signal long enough for resampling
        data = np.sin(np.arange(1500) / 30).reshape(3, 500).T * pq.mV
        signal = AnalogSignal(data, sampling_rate=30000 * pq.Hz)

        # test resampling using different numbers of desired samples
        sample_counts = [10, 100, 250]
        for sample_count in sample_counts:
            downsampling_factor = int(signal.shape[0] / sample_count)
            desired = signal.downsample(downsampling_factor=downsampling_factor)
            result = signal.resample(sample_count)

            self.assertEqual(desired.shape[0], result.shape[0])
            self.assertEqual(desired.shape[-1],
                             result.shape[-1])  # preserve number of recording traces
            self.assertAlmostEqual(desired.sampling_rate, result.sampling_rate)
            # only comparing center values due to border effects
            np.testing.assert_allclose(desired.magnitude[3:-3], result.magnitude[3:-3], rtol=0.05,
                                       atol=0.1)

    def test_rectify(self):
        # generate signal long enough for testing the rectification
        data = np.sin(np.arange(1500) / 30).reshape(500, 3)
        target_data = np.abs(data)

        array_anno = {'anno1': [0, 1, 2], 'anno2': ['C', 'P', 'F']}

        signal = AnalogSignal(data * pq.mV,
                              sampling_rate=30000 * pq.Hz,
                              units=pq.mV,
                              array_annotations=array_anno)

        target_signal = AnalogSignal(target_data * pq.mV,
                                     sampling_rate=30000 * pq.Hz,
                                     units=pq.mV,
                                     array_annotations=array_anno)

        # Use the rectify method
        rectified_signal = signal.rectify()

        # Assert that nothing changed
        assert_arrays_equal(rectified_signal.magnitude, target_signal.magnitude)
        self.assertEqual(rectified_signal.sampling_rate,
                         target_signal.sampling_rate)
        self.assertEqual(rectified_signal.units, target_signal.units)
        self.assertEqual(rectified_signal.annotations,
                         target_signal.annotations)
        assert_arrays_equal(rectified_signal.array_annotations['anno1'],
                            target_signal.array_annotations['anno1'])
        assert_arrays_equal(rectified_signal.array_annotations['anno2'],
                            target_signal.array_annotations['anno2'])


class TestAnalogSignalEquality(unittest.TestCase):
    def test__signals_with_different_data_complement_should_be_not_equal(self):
        signal1 = AnalogSignal(np.arange(55.0).reshape((11, 5)), units="mV",
                               sampling_rate=1 * pq.kHz)
        signal2 = AnalogSignal(np.arange(55.0).reshape((11, 5)), units="mV",
                               sampling_rate=2 * pq.kHz)
        self.assertNotEqual(signal1, signal2)
        assert_neo_object_is_compliant(signal1)
        assert_neo_object_is_compliant(signal2)


class TestAnalogSignalCombination(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(55.0).reshape((11, 5))
        self.data1quant = self.data1 * pq.mV
        self.arr_ann1 = {'anno1': np.arange(5), 'anno2': ['a', 'b', 'c', 'd', 'e']}
        self.signal1 = AnalogSignal(self.data1quant, sampling_rate=1 * pq.kHz, name='spam',
                                    description='eggs', file_origin='testfile.txt',
                                    array_annotations=self.arr_ann1, arg1='test')
        self.data2 = np.arange(100.0, 155.0).reshape((11, 5))
        self.data2quant = self.data2 * pq.mV
        self.arr_ann2 = {'anno1': np.arange(10, 15), 'anno2': ['k', 'l', 'm', 'n', 'o']}
        self.signal2 = AnalogSignal(self.data2quant, sampling_rate=1 * pq.kHz, name='spam',
                                    description='eggs', file_origin='testfile.txt',
                                    array_annotations=self.arr_ann2, arg1='test')

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})
        assert_arrays_equal(self.signal1.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(self.signal1.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(self.signal1.array_annotations, ArrayDict)

        assert_neo_object_is_compliant(self.signal2)
        self.assertEqual(self.signal2.name, 'spam')
        self.assertEqual(self.signal2.description, 'eggs')
        self.assertEqual(self.signal2.file_origin, 'testfile.txt')
        self.assertEqual(self.signal2.annotations, {'arg1': 'test'})
        assert_arrays_equal(self.signal2.array_annotations['anno1'], np.arange(10, 15))
        assert_arrays_equal(self.signal2.array_annotations['anno2'],
                            np.array(['k', 'l', 'm', 'n', 'o']))
        self.assertIsInstance(self.signal2.array_annotations, ArrayDict)

    def test__add_const_quantity_should_preserve_data_complement(self):
        result = self.signal1 + 0.065 * pq.V
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        # time zero, signal index 4
        assert_arrays_equal(result, self.data1 + 65)
        self.assertEqual(self.signal1[0, 4], 4 * pq.mV)
        self.assertEqual(result[0, 4], 69000 * pq.uV)
        self.assertEqual(self.signal1.t_start, result.t_start)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__add_quantity_should_preserve_data_complement(self):
        result = self.signal1 + self.signal2
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        targdata = np.arange(100.0, 210.0, 2.0).reshape((11, 5))
        targ = AnalogSignal(targdata, units="mV", sampling_rate=1 * pq.kHz, name='spam',
                            description='eggs', file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(targ)

        assert_arrays_equal(result, targdata)
        assert_same_sub_schema(result, targ)

    def test__add_two_consistent_signals_should_preserve_data_complement(self):
        data2 = np.arange(10.0, 20.0)
        data2quant = data2 * pq.mV
        signal2 = AnalogSignal(data2quant, sampling_rate=1 * pq.kHz,
                               array_annotations={'abc': [1]})
        assert_neo_object_is_compliant(signal2)

        result = self.signal1 + self.signal2
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        targdata = np.arange(100.0, 210.0, 2.0).reshape((11, 5))
        targ = AnalogSignal(targdata, units="mV", sampling_rate=1 * pq.kHz, name='spam',
                            description='eggs', file_origin='testfile.txt', arg1='test')
        assert_neo_object_is_compliant(targ)

        assert_arrays_equal(result, targdata)
        assert_same_sub_schema(result, targ)

    def test__add_signals_with_inconsistent_data_complement_ValueError(self):
        self.signal1.t_start = 0.0 * pq.ms
        assert_neo_object_is_compliant(self.signal1)

        signal2 = AnalogSignal(np.arange(10.0), units="mV", t_start=100.0 * pq.ms,
                               sampling_rate=0.5 * pq.kHz)
        assert_neo_object_is_compliant(signal2)

        self.assertRaises(ValueError, self.signal1.__add__, signal2)

    def test__subtract_const_should_preserve_data_complement(self):
        result = self.signal1 - 65 * pq.mV
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

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
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

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
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

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
        assert_arrays_equal(result.array_annotations['anno1'], np.arange(5))
        assert_arrays_equal(result.array_annotations['anno2'],
                            np.array(['a', 'b', 'c', 'd', 'e']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

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
        arr_ann3 = {'anno1': np.arange(5, 11), 'anno3': ['h', 'i', 'j', 'k', 'l', 'm']}
        arr_ann4 = {'anno1': np.arange(100, 106), 'anno3': ['o', 'p', 'q', 'r', 's', 't']}

        signal2 = AnalogSignal(self.data1quant, sampling_rate=1 * pq.kHz, name='signal2',
                               description='test signal', file_origin='testfile.txt',
                               array_annotations=self.arr_ann1)
        signal3 = AnalogSignal(data3, units="uV", sampling_rate=1 * pq.kHz, name='signal3',
                               description='test signal', file_origin='testfile.txt',
                               array_annotations=arr_ann3)
        signal4 = AnalogSignal(data3, units="uV", sampling_rate=1 * pq.kHz, name='signal4',
                               description='test signal', file_origin='testfile.txt',
                               array_annotations=arr_ann4)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always')
            merged13 = self.signal1.merge(signal3)
            merged23 = signal2.merge(signal3)
            merged24 = signal2.merge(signal4)

            self.assertTrue(len(w) == 3)
            self.assertEqual(w[-1].category, UserWarning)
            self.assertSequenceEqual(str(w[2].message), str(w[0].message))
            self.assertSequenceEqual(str(w[2].message), str(w[1].message))
            self.assertSequenceEqual(str(w[2].message), "The following array annotations were "
                                                        "omitted, because they were only present"
                                                        " in one of the merged objects: "
                                                        "['anno2'] from the one that was merged "
                                                        "into and ['anno3'] from the one that "
                                                        "was merged into the other")

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

        assert_arrays_equal(merged13.array_annotations['anno1'], np.arange(11))
        self.assertIsInstance(merged13.array_annotations, ArrayDict)
        self.assertNotIn('anno2', merged13.array_annotations)
        self.assertNotIn('anno3', merged13.array_annotations)
        assert_arrays_equal(merged23.array_annotations['anno1'], np.arange(11))
        self.assertIsInstance(merged23.array_annotations, ArrayDict)
        self.assertNotIn('anno2', merged23.array_annotations)
        self.assertNotIn('anno3', merged23.array_annotations)
        assert_arrays_equal(merged24.array_annotations['anno1'],
                            np.array([0, 1, 2, 3, 4, 100, 101, 102, 103, 104, 105]))
        self.assertIsInstance(merged24.array_annotations, ArrayDict)
        self.assertNotIn('anno2', merged24.array_annotations)
        self.assertNotIn('anno3', merged24.array_annotations)

        assert_arrays_equal(mergeddata13, targdata13)
        assert_arrays_equal(mergeddata23, targdata23)
        assert_arrays_equal(mergeddata24, targdata24)

    def test_concatenate_simple(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop)

        result = signal1.concatenate(signal2)
        assert_array_equal(np.arange(7).reshape((-1, 1)), result.magnitude)
        for attr in signal1._necessary_attrs:
            self.assertEqual(getattr(signal1, attr[0], None), getattr(result, attr[0], None))

    def test_concatenate_no_signals(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        self.assertIs(signal1, signal1.concatenate())

    def test_concatenate_reverted_order(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop)

        result = signal2.concatenate(signal1)
        assert_array_equal(np.arange(7).reshape((-1, 1)), result.magnitude)
        for attr in signal1._necessary_attrs:
            self.assertEqual(getattr(signal1, attr[0], None), getattr(result, attr[0], None))

    def test_concatenate_no_overlap(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz, t_start=10 * pq.s)

        with self.assertRaises(MergeError):
            signal1.concatenate(signal2)

    def test_concatenate_multi_trace(self):
        data1 = np.arange(4).reshape(2, 2)
        data2 = np.arange(4, 8).reshape(2, 2)
        signal1 = AnalogSignal(data1 * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal(data2 * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop)

        result = signal1.concatenate(signal2)
        data_expected = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        assert_array_equal(data_expected, result.magnitude)
        for attr in signal1._necessary_attrs:
            self.assertEqual(getattr(signal1, attr[0], None), getattr(result, attr[0], None))

    def test_concatenate_overwrite_true(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop - signal1.sampling_period)

        result = signal1.concatenate(signal2, overwrite=True)
        assert_array_equal(np.array([0, 1, 2, 4, 5, 6]).reshape((-1, 1)), result.magnitude)

    def test_concatenate_overwrite_false(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop - signal1.sampling_period)

        result = signal1.concatenate(signal2, overwrite=False)
        assert_array_equal(np.array([0, 1, 2, 3, 5, 6]).reshape((-1, 1)), result.magnitude)

    def test_concatenate_padding_False(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=10 * pq.s)

        with self.assertRaises(MergeError):
            result = signal1.concatenate(signal2, overwrite=False, padding=False)

    def test_concatenate_padding_True(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop + 3 * signal1.sampling_period)

        result = signal1.concatenate(signal2, overwrite=False, padding=True)
        assert_array_equal(
            np.array([0, 1, 2, 3, np.NaN, np.NaN, np.NaN, 4, 5, 6]).reshape((-1, 1)),
            result.magnitude)

    def test_concatenate_padding_quantity(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop + 3 * signal1.sampling_period)

        result = signal1.concatenate(signal2, overwrite=False, padding=-1 * pq.mV)
        assert_array_equal(np.array([0, 1, 2, 3, -1e-3, -1e-3, -1e-3, 4, 5, 6]).reshape((-1, 1)),
                           result.magnitude)

    def test_concatenate_padding_invalid(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop + 3 * signal1.sampling_period)

        with self.assertRaises(MergeError):
            result = signal1.concatenate(signal2, overwrite=False, padding=1)
        with self.assertRaises(MergeError):
            result = signal1.concatenate(signal2, overwrite=False, padding=[1])
        with self.assertRaises(MergeError):
            result = signal1.concatenate(signal2, overwrite=False, padding='a')
        with self.assertRaises(MergeError):
            result = signal1.concatenate(signal2, overwrite=False, padding=np.array([1, 2, 3]))

    def test_concatenate_array_annotations(self):
        array_anno1 = {'first': ['a', 'b']}
        array_anno2 = {'first': ['a', 'b'],
                       'second': ['c', 'd']}
        data1 = np.arange(4).reshape(2, 2)
        data2 = np.arange(4, 8).reshape(2, 2)
        signal1 = AnalogSignal(data1 * pq.V, sampling_rate=1 * pq.Hz,
                               array_annotations=array_anno1)
        signal2 = AnalogSignal(data2 * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop,
                               array_annotations=array_anno2)

        result = signal1.concatenate(signal2)
        assert_array_equal(array_anno1.keys(), result.array_annotations.keys())

        for k in array_anno1.keys():
            assert_array_equal(np.asarray(array_anno1[k]), result.array_annotations[k])

    def test_concatenate_complex(self):
        signal1 = self.signal1
        assert_neo_object_is_compliant(self.signal1)

        signal2 = AnalogSignal(self.data1quant, sampling_rate=1 * pq.kHz, name='signal2',
                               description='test signal', file_origin='testfile.txt',
                               array_annotations=self.arr_ann1,
                               t_start=signal1.t_stop)

        concatenated12 = self.signal1.concatenate(signal2)

        for attr in signal1._necessary_attrs:
            self.assertEqual(getattr(signal1, attr[0], None),
                             getattr(concatenated12, attr[0], None))

        assert_array_equal(np.vstack((signal1.magnitude, signal2.magnitude)),
                           concatenated12.magnitude)

    def test_concatenate_multi_signal(self):
        signal1 = AnalogSignal([0, 1, 2, 3] * pq.V, sampling_rate=1 * pq.Hz)
        signal2 = AnalogSignal([4, 5, 6] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop + 3 * signal1.sampling_period)
        signal3 = AnalogSignal([40] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop + 3 * signal1.sampling_period)
        signal4 = AnalogSignal([30, 35] * pq.V, sampling_rate=1 * pq.Hz,
                               t_start=signal1.t_stop - signal1.sampling_period)

        concatenated = signal1.concatenate(signal2, signal3, signal4, padding=-1 * pq.V,
                                           overwrite=True)
        for attr in signal1._necessary_attrs:
            self.assertEqual(getattr(signal1, attr[0], None),
                             getattr(concatenated, attr[0], None))
        assert_arrays_equal(np.array([0, 1, 2, 30, 35, -1, -1, 40, 5, 6]).reshape((-1, 1)),
                            concatenated.magnitude)


class TestAnalogSignalFunctions(unittest.TestCase):
    def test__pickle_1d(self):
        signal1 = AnalogSignal([1, 2, 3, 4], sampling_period=1 * pq.ms, units=pq.S)
        signal1.annotations['index'] = 2
        signal1.array_annotate(**{'anno1': [23], 'anno2': ['A']})

        fobj = open('./pickle', 'wb')
        pickle.dump(signal1, fobj)
        fobj.close()

        fobj = open('./pickle', 'rb')
        try:
            signal2 = pickle.load(fobj)
        except ValueError:
            signal2 = None

        assert_array_equal(signal1, signal2)
        assert_array_equal(signal2.array_annotations['anno1'], np.array([23]))
        self.assertIsInstance(signal2.array_annotations, ArrayDict)
        # Make sure the dict can perform correct checks after unpickling
        signal2.array_annotations['anno3'] = [2]
        with self.assertRaises(ValueError):
            signal2.array_annotations['anno4'] = [2, 1]
        fobj.close()
        os.remove('./pickle')

    def test__pickle_2d(self):
        signal1 = AnalogSignal(np.arange(55.0).reshape((11, 5)), units="mV",
                               sampling_rate=1 * pq.kHz)

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


class TestAnalogSignalSampling(unittest.TestCase):
    def test___get_sampling_rate__period_none_rate_none_ValueError(self):
        sampling_rate = None
        sampling_period = None
        self.assertRaises(ValueError, _get_sampling_rate, sampling_rate, sampling_period)

    def test___get_sampling_rate__period_quant_rate_none(self):
        sampling_rate = None
        sampling_period = pq.Quantity(10., units=pq.s)
        targ_rate = 1 / sampling_period
        out_rate = _get_sampling_rate(sampling_rate, sampling_period)
        self.assertEqual(targ_rate, out_rate)

    def test___get_sampling_rate__period_none_rate_quant(self):
        sampling_rate = pq.Quantity(10., units=pq.Hz)
        sampling_period = None
        targ_rate = sampling_rate
        out_rate = _get_sampling_rate(sampling_rate, sampling_period)
        self.assertEqual(targ_rate, out_rate)

    def test___get_sampling_rate__period_rate_equivalent(self):
        sampling_rate = pq.Quantity(10., units=pq.Hz)
        sampling_period = pq.Quantity(0.1, units=pq.s)
        targ_rate = sampling_rate
        out_rate = _get_sampling_rate(sampling_rate, sampling_period)
        self.assertEqual(targ_rate, out_rate)

    def test___get_sampling_rate__period_rate_not_equivalent_ValueError(self):
        sampling_rate = pq.Quantity(10., units=pq.Hz)
        sampling_period = pq.Quantity(10, units=pq.s)
        self.assertRaises(ValueError, _get_sampling_rate, sampling_rate, sampling_period)

    def test___get_sampling_rate__period_none_rate_float_TypeError(self):
        sampling_rate = 10.
        sampling_period = None
        self.assertRaises(TypeError, _get_sampling_rate, sampling_rate, sampling_period)

    def test___get_sampling_rate__period_array_rate_none_TypeError(self):
        sampling_rate = None
        sampling_period = np.array(10.)
        self.assertRaises(TypeError, _get_sampling_rate, sampling_rate, sampling_period)


if __name__ == "__main__":
    unittest.main()
