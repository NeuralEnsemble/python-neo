# -*- coding: utf-8 -*-
"""
Tests of the neo.core.analogsignal.AnalogSignal class and related functions
"""

# needed for python 3 compatibility
from __future__ import division

import os
import pickle

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
from neo import units as un

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from numpy.testing import assert_array_equal
from neo.core.analogsignal import AnalogSignal, _get_sampling_rate
from neo.core import Segment
from neo.test.tools import (assert_arrays_almost_equal,
                            assert_neo_object_is_compliant,
                            assert_same_sub_schema)
from neo.test.generate_datasets import (get_fake_value, get_fake_values,
                                        fake_neo, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = AnalogSignal
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'AnalogSignal'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestAnalogSignalConstructor(unittest.TestCase):
    def test__create_from_list(self):
        data = range(10)
        rate = 1000*un.Hz
        signal = AnalogSignal(data, sampling_rate=rate, units="mV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0*un.ms)
        self.assertEqual(signal.t_stop, len(data)/rate)
        self.assertEqual(signal[9, 0], 9000*un.uV)

    def test__create_from_np_array(self):
        data = np.arange(10.0)
        rate = 1*un.kHz
        signal = AnalogSignal(data, sampling_rate=rate, units="uV")
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0*un.ms)
        self.assertEqual(signal.t_stop, data.size/rate)
        self.assertEqual(signal[9, 0], 0.009*un.mV)

    def test__create_from_quantities_array(self):
        data = np.arange(10.0) * un.mV
        rate = 5000*un.Hz
        signal = AnalogSignal(data, sampling_rate=rate)
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.t_start, 0*un.ms)
        self.assertEqual(signal.t_stop, data.size/rate)
        self.assertEqual(signal[9, 0], 0.009*un.V)

    def test__create_from_array_no_units_ValueError(self):
        data = np.arange(10.0)
        self.assertRaises(ValueError, AnalogSignal, data,
                          sampling_rate=1 * un.kHz)

    def test__create_from_quantities_array_inconsistent_units_ValueError(self):
        data = np.arange(10.0) * un.mV
        self.assertRaises(ValueError, AnalogSignal, data,
                          sampling_rate=1 * un.kHz, units="nA")

    def test__create_without_sampling_rate_or_period_ValueError(self):
        data = np.arange(10.0) * un.mV
        self.assertRaises(ValueError, AnalogSignal, data)

    def test__create_with_None_sampling_rate_should_raise_ValueError(self):
        data = np.arange(10.0) * un.mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=None)

    def test__create_with_None_t_start_should_raise_ValueError(self):
        data = np.arange(10.0) * un.mV
        rate = 5000 * un.Hz
        self.assertRaises(ValueError, AnalogSignal, data,
                          sampling_rate=rate, t_start=None)

    def test__create_inconsistent_sampling_rate_and_period_ValueError(self):
        data = np.arange(10.0) * un.mV
        self.assertRaises(ValueError, AnalogSignal, data,
                          sampling_rate=1 * un.kHz, sampling_period=5 * un.s)

    def test__create_with_copy_true_should_return_copy(self):
        data = np.arange(10.0) * un.mV
        rate = 5000*un.Hz
        signal = AnalogSignal(data, copy=True, sampling_rate=rate)
        data[3] = 99*un.mV
        assert_neo_object_is_compliant(signal)
        self.assertNotEqual(signal[3, 0], 99*un.mV)

    def test__create_with_copy_false_should_return_view(self):
        data = np.arange(10.0) * un.mV
        rate = 5000*un.Hz
        signal = AnalogSignal(data, copy=False, sampling_rate=rate)
        data[3] = 99*un.mV
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal[3, 0], 99*un.mV)

    def test__create2D_with_copy_false_should_return_view(self):
        data = np.arange(10.0) * un.mV
        data = data.reshape((5, 2))
        rate = 5000*un.Hz
        signal = AnalogSignal(data, copy=False, sampling_rate=rate)
        data[3, 0] = 99*un.mV
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal[3, 0], 99*un.mV)

    def test__create_with_additional_argument(self):
        signal = AnalogSignal([1, 2, 3], units="mV", sampling_rate=1*un.kHz,
                              file_origin='crack.txt', ratname='Nicolas')
        assert_neo_object_is_compliant(signal)
        self.assertEqual(signal.annotations, {'ratname': 'Nicolas'})

        # This one is universally recommended and handled by BaseNeo
        self.assertEqual(signal.file_origin, 'crack.txt')

    # signal must be 1D - should raise Exception if not 1D


class TestAnalogSignalProperties(unittest.TestCase):
    def setUp(self):
        self.t_start = [0.0*un.ms, 100*un.ms, -200*un.ms]
        self.rates = [1*un.kHz, 420*un.Hz, 999*un.Hz]
        self.rates2 = [2*un.kHz, 290*un.Hz, 1111*un.Hz]
        self.data = [np.arange(10.0)*un.nA,
                     np.arange(-100.0, 100.0, 10.0)*un.mV,
                     np.random.uniform(size=100)*un.uV]
        self.signals = [AnalogSignal(D, sampling_rate=r, t_start=t,
                                          testattr='test')
                        for r, D, t in zip(self.rates,
                                           self.data,
                                           self.t_start)]

    def test__compliant(self):
        for signal in self.signals:
            assert_neo_object_is_compliant(signal)

    def test__t_stop_getter(self):
        for i, signal in enumerate(self.signals):
            self.assertEqual(signal.t_stop,
                             self.t_start[i] + self.data[i].size/self.rates[i])

    def test__duration_getter(self):
        for signal in self.signals:
            self.assertAlmostEqual(signal.duration,
                                   signal.t_stop - signal.t_start,
                                   delta=1e-15)

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
        self.assertRaises(ValueError, setattr, self.signals[0],
                          'sampling_rate', None)

    def test__sampling_rate_setter_not_quantity_ValueError(self):
        self.assertRaises(ValueError, setattr, self.signals[0],
                          'sampling_rate', 5.5)

    def test__sampling_period_setter_None_ValueError(self):
        signal = self.signals[0]
        assert_neo_object_is_compliant(signal)
        self.assertRaises(ValueError, setattr, signal, 'sampling_period', None)

    def test__sampling_period_setter_not_quantity_ValueError(self):
        self.assertRaises(ValueError, setattr, self.signals[0],
                          'sampling_period', 5.5)

    def test__t_start_setter_None_ValueError(self):
        signal = self.signals[0]
        assert_neo_object_is_compliant(signal)
        self.assertRaises(ValueError, setattr, signal, 't_start', None)

    def test__times_getter(self):
        for i, signal in enumerate(self.signals):
            targ = np.arange(self.data[i].size)
            targ = targ/self.rates[i] + self.t_start[i]
            assert_neo_object_is_compliant(signal)
            assert_arrays_almost_equal(signal.times, targ, 1e-12*un.ms)

    def test__duplicate_with_new_array(self):
        signal1 = self.signals[1]
        signal2 = self.signals[2]
        data2 = self.data[2]
        signal1b = signal1.duplicate_with_new_array(data2)
        assert_arrays_almost_equal(np.asarray(signal1b),
                                   np.asarray(signal2/1000.), 1e-12)
        self.assertEqual(signal1b.t_start, signal1.t_start)
        self.assertEqual(signal1b.sampling_rate, signal1.sampling_rate)

    # def test__children(self):
    #     signal = self.signals[0]
    #
    #     segment = Segment(name='seg1')
    #     segment.analogsignals = [signal]
    #     segment.create_many_to_one_relationship()
    #
    #     rchan = RecordingChannel(name='rchan1')
    #     rchan.analogsignals = [signal]
    #     rchan.create_many_to_one_relationship()
    #
    #     self.assertEqual(signal._single_parent_objects,
    #                      ('Segment', 'RecordingChannel'))
    #     self.assertEqual(signal._multi_parent_objects, ())
    #
    #     self.assertEqual(signal._single_parent_containers,
    #                      ('segment', 'recordingchannel'))
    #     self.assertEqual(signal._multi_parent_containers, ())
    #
    #     self.assertEqual(signal._parent_objects,
    #                      ('Segment', 'RecordingChannel'))
    #     self.assertEqual(signal._parent_containers,
    #                      ('segment', 'recordingchannel'))
    #
    #     self.assertEqual(len(signal.parents), 2)
    #     self.assertEqual(signal.parents[0].name, 'seg1')
    #     self.assertEqual(signal.parents[1].name, 'rchan1')
    #
    #     assert_neo_object_is_compliant(signal)

    def test__repr(self):
        for i, signal in enumerate(self.signals):
            prepr = repr(signal)
            targ = '<AnalogSignal(%s, [%s, %s], sampling rate: %s)>' % \
                (repr(self.data[i].reshape(-1, 1)),
                 self.t_start[i],
                 self.t_start[i] + len(self.data[i])/self.rates[i],
                 self.rates[i])
            self.assertEqual(prepr, targ)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        for i, signal in enumerate(self.signals):
            prepr = pretty(signal)
            targ = (('AnalogSignal with %d channels of length %d; units %s; datatype %s \n' %
                     (signal.shape[1], signal.shape[0], signal.units.dimensionality.unicode, signal.dtype)) +
                    ('annotations: %s\n' % signal.annotations) +
                    ('sampling rate: %s\n' % signal.sampling_rate) +
                    ('time: %s to %s' % (signal.t_start, signal.t_stop)))
            self.assertEqual(prepr, targ)


class TestAnalogSignalArrayMethods(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(10.0)
        self.data1quant = self.data1 * un.nA
        self.signal1 = AnalogSignal(self.data1quant, sampling_rate=1*un.kHz,
                                         name='spam', description='eggs',
                                         file_origin='testfile.txt', arg1='test')
        self.signal1.segment = 1
        self.signal1.channel_index = 5

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)

    def test__slice_should_return_AnalogSignalArray(self):
        # slice
        result = self.signal1[3:8, 0]
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')         # should slicing really preserve name and description?
        self.assertEqual(result.description, 'eggs')  # perhaps these should be modified to indicate the slice?
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.size, 5)
        self.assertEqual(result.sampling_period, self.signal1.sampling_period)
        self.assertEqual(result.sampling_rate, self.signal1.sampling_rate)
        self.assertEqual(result.t_start,
                         self.signal1.t_start+3*result.sampling_period)
        self.assertEqual(result.t_stop,
                         result.t_start + 5*result.sampling_period)
        assert_array_equal(result.magnitude, self.data1[3:8].reshape(-1, 1))

        # Test other attributes were copied over (in this case, defaults)
        self.assertEqual(result.file_origin, self.signal1.file_origin)
        self.assertEqual(result.name, self.signal1.name)
        self.assertEqual(result.description, self.signal1.description)
        self.assertEqual(result.annotations, self.signal1.annotations)

    def test__slice_should_let_access_to_parents_objects(self):
        result =  self.signal1.time_slice(1*pq.ms,3*pq.ms)
        self.assertEqual(result.segment, self.signal1.segment)
        self.assertEqual(result.channel_index, self.signal1.channel_index)

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

        self.assertIsInstance(result2, AnalogSignal)
        assert_neo_object_is_compliant(result2)
        self.assertEqual(result2.name, 'spam')
        self.assertEqual(result2.description, 'eggs')
        self.assertEqual(result2.file_origin, 'testfile.txt')
        self.assertEqual(result2.annotations, {'arg1': 'test'})

        self.assertIsInstance(result3, AnalogSignal)
        assert_neo_object_is_compliant(result3)
        self.assertEqual(result3.name, 'spam')
        self.assertEqual(result3.description, 'eggs')
        self.assertEqual(result3.file_origin, 'testfile.txt')
        self.assertEqual(result3.annotations, {'arg1': 'test'})

        self.assertEqual(result1.sampling_period, self.signal1.sampling_period)
        self.assertEqual(result2.sampling_period,
                         self.signal1.sampling_period * 2)
        self.assertEqual(result3.sampling_period,
                         self.signal1.sampling_period * 2)

        assert_array_equal(result1.magnitude, self.data1[:2].reshape(-1, 1))
        assert_array_equal(result2.magnitude, self.data1[::2].reshape(-1, 1))
        assert_array_equal(result3.magnitude, self.data1[1:7:2].reshape(-1, 1))

    def test__copy_should_let_access_to_parents_objects(self):
        ##copy
        result =  self.signal1.copy()
        self.assertEqual(result.segment, self.signal1.segment)
        self.assertEqual(result.channel_index, self.signal1.channel_index)
        ## deep copy (not fixed yet)
        #result = copy.deepcopy(self.signal1)
        #self.assertEqual(result.segment, self.signal1.segment)
        #self.assertEqual(result.channel_index, self.signal1.channel_index)

    def test__getitem_should_return_single_quantity(self):
        result1 = self.signal1[0, 0]
        result2 = self.signal1[9, 0]

        self.assertIsInstance(result1, un.Quantity)
        self.assertFalse(hasattr(result1, 'name'))
        self.assertFalse(hasattr(result1, 'description'))
        self.assertFalse(hasattr(result1, 'file_origin'))
        self.assertFalse(hasattr(result1, 'annotations'))

        self.assertIsInstance(result2, un.Quantity)
        self.assertFalse(hasattr(result2, 'name'))
        self.assertFalse(hasattr(result2, 'description'))
        self.assertFalse(hasattr(result2, 'file_origin'))
        self.assertFalse(hasattr(result2, 'annotations'))

        self.assertEqual(result1, 0*un.nA)
        self.assertEqual(result2, 9*un.nA)

    def test__getitem_out_of_bounds_IndexError(self):
        self.assertRaises(IndexError, self.signal1.__getitem__, (10, 0))

    def test_comparison_operators(self):
        assert_array_equal(self.signal1 >= 5*un.nA,
                            np.array([False, False, False, False, False,
                                      True, True, True, True, True]).reshape(-1, 1))
        assert_array_equal(self.signal1 >= 5*un.pA,
                            np.array([False, True, True, True, True,
                                      True, True, True, True, True]).reshape(-1, 1))

    def test__comparison_with_inconsistent_units_should_raise_Exception(self):
        self.assertRaises(ValueError, self.signal1.__gt__, 5*un.mV)

    def test__simple_statistics(self):
        self.assertEqual(self.signal1.max(), 9*un.nA)
        self.assertEqual(self.signal1.min(), 0*un.nA)
        self.assertEqual(self.signal1.mean(), 4.5*un.nA)

    def test__rescale_same(self):
        result = self.signal1.copy()
        result = result.rescale(un.nA)

        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.units, 1*un.nA)
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1))
        assert_same_sub_schema(result, self.signal1)

    def test__rescale_new(self):
        result = self.signal1.copy()
        result = result.rescale(un.pA)

        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(result.units, 1*un.pA)
        assert_arrays_almost_equal(np.array(result), self.data1.reshape(-1, 1)*1000., 1e-10)

    def test__rescale_new_incompatible_ValueError(self):
        self.assertRaises(ValueError, self.signal1.rescale, un.mV)


class TestAnalogSignalEquality(unittest.TestCase):
    def test__signals_with_different_data_complement_should_be_not_equal(self):
        signal1 = AnalogSignal(np.arange(10.0), units="mV",
                                    sampling_rate=1*un.kHz)
        signal2 = AnalogSignal(np.arange(10.0), units="mV",
                                    sampling_rate=2*un.kHz)
        assert_neo_object_is_compliant(signal1)
        assert_neo_object_is_compliant(signal2)
        self.assertNotEqual(signal1, signal2)


class TestAnalogSignalCombination(unittest.TestCase):
    def setUp(self):
        self.data1 = np.arange(10.0)
        self.data1quant = self.data1 * un.mV
        self.signal1 = AnalogSignal(self.data1quant,
                                         sampling_rate=1*un.kHz,
                                         name='spam', description='eggs',
                                         file_origin='testfile.txt',
                                         arg1='test')

    def test__compliant(self):
        assert_neo_object_is_compliant(self.signal1)
        self.assertEqual(self.signal1.name, 'spam')
        self.assertEqual(self.signal1.description, 'eggs')
        self.assertEqual(self.signal1.file_origin, 'testfile.txt')
        self.assertEqual(self.signal1.annotations, {'arg1': 'test'})

    def test__add_const_quantity_should_preserve_data_complement(self):
        result = self.signal1 + 0.065*un.V
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        assert_array_equal(result.magnitude.flatten(), self.data1 + 65)
        self.assertEqual(self.signal1[9, 0], 9*un.mV)
        self.assertEqual(result[9, 0], 74*un.mV)
        self.assertEqual(self.signal1.t_start, result.t_start)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__add_quantity_should_preserve_data_complement(self):
        data2 = np.arange(10.0, 20.0).reshape(-1, 1)
        data2quant = data2*un.mV

        result = self.signal1 + data2quant
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targ = AnalogSignal(np.arange(10.0, 30.0, 2.0),  units="mV",
                                 sampling_rate=1*un.kHz,
                                 name='spam', description='eggs',
                                 file_origin='testfile.txt',  arg1='test')
        assert_neo_object_is_compliant(targ)

        assert_array_equal(result, targ)
        assert_same_sub_schema(result, targ)

    def test__add_two_consistent_signals_should_preserve_data_complement(self):
        data2 = np.arange(10.0, 20.0)
        data2quant = data2*un.mV
        signal2 = AnalogSignal(data2quant, sampling_rate=1*un.kHz)
        assert_neo_object_is_compliant(signal2)

        result = self.signal1 + signal2
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        targ = AnalogSignal(np.arange(10.0, 30.0, 2.0),  units="mV",
                                 sampling_rate=1*un.kHz,
                                 name='spam', description='eggs',
                                 file_origin='testfile.txt',  arg1='test')
        assert_neo_object_is_compliant(targ)

        assert_array_equal(result, targ)
        assert_same_sub_schema(result, targ)

    def test__add_signals_with_inconsistent_data_complement_ValueError(self):
        self.signal1.t_start = 0.0*un.ms
        assert_neo_object_is_compliant(self.signal1)

        signal2 = AnalogSignal(np.arange(10.0), units="mV",
                                    t_start=100.0*un.ms, sampling_rate=0.5*un.kHz)
        assert_neo_object_is_compliant(signal2)

        self.assertRaises(ValueError, self.signal1.__add__, signal2)

    def test__subtract_const_should_preserve_data_complement(self):
        result = self.signal1 - 65*un.mV
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9, 0], 9*un.mV)
        self.assertEqual(result[9, 0], -56*un.mV)
        assert_array_equal(result.magnitude.flatten(), self.data1 - 65)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__subtract_from_const_should_return_signal(self):
        result = 10*un.mV - self.signal1
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9, 0], 9*un.mV)
        self.assertEqual(result[9, 0], 1*un.mV)
        assert_array_equal(result.magnitude.flatten(), 10 - self.data1)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__mult_by_const_float_should_preserve_data_complement(self):
        result = self.signal1*2
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9, 0], 9*un.mV)
        self.assertEqual(result[9, 0], 18*un.mV)
        assert_array_equal(result.magnitude.flatten(), self.data1*2)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)

    def test__divide_by_const_should_preserve_data_complement(self):
        result = self.signal1/0.5
        self.assertIsInstance(result, AnalogSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

        self.assertEqual(self.signal1[9, 0], 9*un.mV)
        self.assertEqual(result[9, 0], 18*un.mV)
        assert_array_equal(result.magnitude.flatten(), self.data1/0.5)
        self.assertEqual(self.signal1.sampling_rate, result.sampling_rate)


class TestAnalogSignalFunctions(unittest.TestCase):
    def test__pickle(self):
        signal1 = AnalogSignal([1, 2, 3, 4], sampling_period=1*un.ms,
                                    units=un.S)
        signal1.annotations['index'] = 2

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


class TestAnalogSignalSampling(unittest.TestCase):
    def test___get_sampling_rate__period_none_rate_none_ValueError(self):
        sampling_rate = None
        sampling_period = None
        self.assertRaises(ValueError, _get_sampling_rate,
                          sampling_rate, sampling_period)

    def test___get_sampling_rate__period_quant_rate_none(self):
        sampling_rate = None
        sampling_period = un.Quantity(10., units=un.s)
        targ_rate = 1/sampling_period
        out_rate = _get_sampling_rate(sampling_rate, sampling_period)
        self.assertEqual(targ_rate, out_rate)

    def test___get_sampling_rate__period_none_rate_quant(self):
        sampling_rate = un.Quantity(10., units=un.Hz)
        sampling_period = None
        targ_rate = sampling_rate
        out_rate = _get_sampling_rate(sampling_rate, sampling_period)
        self.assertEqual(targ_rate, out_rate)

    def test___get_sampling_rate__period_rate_equivalent(self):
        sampling_rate = un.Quantity(10., units=un.Hz)
        sampling_period = un.Quantity(0.1, units=un.s)
        targ_rate = sampling_rate
        out_rate = _get_sampling_rate(sampling_rate, sampling_period)
        self.assertEqual(targ_rate, out_rate)

    def test___get_sampling_rate__period_rate_not_equivalent_ValueError(self):
        sampling_rate = un.Quantity(10., units=un.Hz)
        sampling_period = un.Quantity(10, units=un.s)
        self.assertRaises(ValueError, _get_sampling_rate,
                          sampling_rate, sampling_period)

    def test___get_sampling_rate__period_none_rate_float_TypeError(self):
        sampling_rate = 10.
        sampling_period = None
        self.assertRaises(TypeError, _get_sampling_rate,
                          sampling_rate, sampling_period)

    def test___get_sampling_rate__period_array_rate_none_TypeError(self):
        sampling_rate = None
        sampling_period = np.array(10.)
        self.assertRaises(TypeError, _get_sampling_rate,
                          sampling_rate, sampling_period)


if __name__ == "__main__":
    unittest.main()
