# -*- coding: utf-8 -*-
"""
Tests of the neo.core.irregularlysampledsignal.IrregularySampledSignal class
"""

import unittest

import os
import pickle
import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.core import Segment, ChannelIndex
from neo.core.baseneo import MergeError
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
        times = get_fake_value('times', pq.Quantity, seed=0, dim=1)
        signal = get_fake_value('signal', pq.Quantity, seed=1, dim=2)
        name = get_fake_value('name', str, seed=2,
                              obj=IrregularlySampledSignal)
        description = get_fake_value('description', str, seed=3,
                                     obj='IrregularlySampledSignal')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(IrregularlySampledSignal,
                                annotate=False, seed=0)
        res12 = get_fake_values('IrregularlySampledSignal',
                                annotate=False, seed=0)
        res21 = get_fake_values(IrregularlySampledSignal,
                                annotate=True, seed=0)
        res22 = get_fake_values('IrregularlySampledSignal',
                                annotate=True, seed=0)

        assert_array_equal(res11.pop('times'), times)
        assert_array_equal(res12.pop('times'), times)
        assert_array_equal(res21.pop('times'), times)
        assert_array_equal(res22.pop('times'), times)

        assert_array_equal(res11.pop('signal'), signal)
        assert_array_equal(res12.pop('signal'), signal)
        assert_array_equal(res21.pop('signal'), signal)
        assert_array_equal(res22.pop('signal'), signal)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = IrregularlySampledSignal
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, IrregularlySampledSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'IrregularlySampledSignal'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, IrregularlySampledSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestIrregularlySampledSignalConstruction(unittest.TestCase):
    def test_IrregularlySampledSignal_creation_times_units_signal_units(self):
        params = {'test2': 'y1', 'test3': True}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7]*pq.ms,
                                       signal=[20., 40., 60.]*pq.mV,
                                       name='test', description='tester',
                                       file_origin='test.file',
                                       test1=1, **params)
        sig.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(sig)

        assert_array_equal(sig.times, [1.1, 1.5, 1.7]*pq.ms)
        assert_array_equal(np.asarray(sig).flatten(), np.array([20., 40., 60.]))
        self.assertEqual(sig.units, pq.mV)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['test0'], [1, 2])
        self.assertEqual(sig.annotations['test1'], 1.1)
        self.assertEqual(sig.annotations['test2'], 'y1')
        self.assertTrue(sig.annotations['test3'])

    def test_IrregularlySampledSignal_creation_units_arg(self):
        params = {'test2': 'y1', 'test3': True}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7],
                                       signal=[20., 40., 60.],
                                       units=pq.V, time_units=pq.s,
                                       name='test', description='tester',
                                       file_origin='test.file',
                                       test1=1, **params)
        sig.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(sig)

        assert_array_equal(sig.times, [1.1, 1.5, 1.7]*pq.s)
        assert_array_equal(np.asarray(sig).flatten(), np.array([20., 40., 60.]))
        self.assertEqual(sig.units, pq.V)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['test0'], [1, 2])
        self.assertEqual(sig.annotations['test1'], 1.1)
        self.assertEqual(sig.annotations['test2'], 'y1')
        self.assertTrue(sig.annotations['test3'])

    def test_IrregularlySampledSignal_creation_units_rescale(self):
        params = {'test2': 'y1', 'test3': True}
        sig = IrregularlySampledSignal([1.1, 1.5, 1.7]*pq.s,
                                       signal=[2., 4., 6.]*pq.V,
                                       units=pq.mV, time_units=pq.ms,
                                       name='test', description='tester',
                                       file_origin='test.file',
                                       test1=1, **params)
        sig.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(sig)

        assert_array_equal(sig.times, [1100, 1500, 1700]*pq.ms)
        assert_array_equal(np.asarray(sig).flatten(), np.array([2000., 4000., 6000.]))
        self.assertEqual(sig.units, pq.mV)
        self.assertEqual(sig.name, 'test')
        self.assertEqual(sig.description, 'tester')
        self.assertEqual(sig.file_origin, 'test.file')
        self.assertEqual(sig.annotations['test0'], [1, 2])
        self.assertEqual(sig.annotations['test1'], 1.1)
        self.assertEqual(sig.annotations['test2'], 'y1')
        self.assertTrue(sig.annotations['test3'])

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
                                       test1=1)
        assert_neo_object_is_compliant(sig)

        if np.__version__.split(".")[:2] > ['1', '13']:
            # see https://github.com/numpy/numpy/blob/master/doc/release/1.14.0-notes.rst#many-changes-to-array-printing-disableable-with-the-new-legacy-printing-mode
            targ = ('<IrregularlySampledSignal(array([[2.],\n       [4.],\n       [6.]]) * V ' +
                    'at times [1.1 1.5 1.7] s)>')
        else:
            targ = ('<IrregularlySampledSignal(array([[ 2.],\n       [ 4.],\n       [ 6.]]) * V ' +
                    'at times [ 1.1  1.5  1.7] s)>')
        res = repr(sig)
        self.assertEqual(targ, res)

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
        self.signal1.segment = Segment()
        self.signal1.channel_index = ChannelIndex([0])

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
        assert_array_equal(self.time1quant[3:8], result.times)
        assert_array_equal(self.data1[3:8].reshape(-1, 1), result.magnitude)

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
        assert_array_equal(self.signal1 >= 5*pq.mV,
                            np.array([[False, False, False, False, False,
                                      True, True, True, True, True]]).T)

    def test__comparison_with_inconsistent_units_should_raise_Exception(self):
        self.assertRaises(ValueError, self.signal1.__gt__, 5*pq.nA)

    def test_simple_statistics(self):
        targmean = self.signal1[:-1]*np.diff(self.time1quant).reshape(-1, 1)
        targmean = targmean.sum()/(self.time1quant[-1] - self.time1quant[0])
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
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1))
        assert_array_equal(result.times, self.time1quant)
        assert_same_sub_schema(result, self.signal1)

        self.assertIsInstance(result.channel_index, ChannelIndex)
        self.assertIsInstance(result.segment, Segment)
        self.assertIs(result.channel_index, self.signal1.channel_index)
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

        self.assertEqual(result.units, 1*pq.uV)
        assert_arrays_almost_equal(np.array(result), self.data1.reshape(-1, 1)*1000., 1e-10)
        assert_array_equal(result.times, self.time1quant)

        self.assertIsInstance(result.channel_index, ChannelIndex)
        self.assertIsInstance(result.segment, Segment)
        self.assertIs(result.channel_index, self.signal1.channel_index)
        self.assertIs(result.segment, self.signal1.segment)

    def test__rescale_new_incompatible_ValueError(self):
        self.assertRaises(ValueError, self.signal1.rescale, pq.nA)

    def test_time_slice(self):
        targdataquant = [[1.0], [2.0], [3.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime [1:4] *pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant,
                                                signal=targdataquant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

        t_start = 15
        t_stop = 250
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1*pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})
    
    def test_time_slice_out_of_boundries(self):
        targdataquant = self.data1quant
        targtimequant = self.time1quant
        targ_signal = IrregularlySampledSignal(targtimequant,
                                                signal=targdataquant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

        t_start = 0
        t_stop = 2500000
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1*pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

    def test_time_slice_empty(self):
        targdataquant = [] * pq.mV
        targtimequant = [] *pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant,
                                                signal=targdataquant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

        t_start = 15
        t_stop = 250
        result = targ_signal.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1*pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

    def test_time_slice_none_stop(self):
        targdataquant = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime [1:10] *pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant,
                                                signal=targdataquant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

        t_start = 15
        t_stop = None
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1*pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

    def test_time_slice_none_start(self):
        targdataquant = [[0.0], [1.0], [2.0], [3.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime [0:4] *pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant,
                                                signal=targdataquant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

        t_start = None
        t_stop = 250
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1*pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

    def test_time_slice_none_both(self):
        targdataquant = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime [0:10] *pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant,
                                                signal=targdataquant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

        t_start = None
        t_stop = None
        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1*pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

    def test_time_slice_differnt_units(self):
        targdataquant = [[1.0], [2.0], [3.0]] * pq.mV
        targtime = np.logspace(1, 5, 10)
        targtimequant = targtime [1:4] *pq.ms
        targ_signal = IrregularlySampledSignal(targtimequant,
                                                signal=targdataquant,
                                                name='spam',
                                                description='eggs',
                                                file_origin='testfile.txt',
                                                arg1='test')

        t_start = 15
        t_stop = 250

        t_start = 0.015  * pq.s
        t_stop = .250 * pq.s

        result = self.signal1.time_slice(t_start, t_stop)

        assert_array_equal(result, targ_signal)
        assert_array_equal(result.times, targtimequant)
        self.assertEqual(result.units, 1*pq.mV)
        self.assertIsInstance(result, IrregularlySampledSignal)
        assert_neo_object_is_compliant(result)
        self.assertEqual(result.name, 'spam')
        self.assertEqual(result.description, 'eggs')
        self.assertEqual(result.file_origin, 'testfile.txt')
        self.assertEqual(result.annotations, {'arg1': 'test'})

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
        self.assertIs(result.channel_index, self.signal1.channel_index)


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

        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1) + 65)
        assert_array_equal(result.times, self.time1quant)
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

        assert_array_equal(result, targ)
        assert_array_equal(self.time1quant, targ.times)
        assert_array_equal(result.times, targ.times)
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
        assert_array_equal(result.magnitude, (self.data1 - 65).reshape(-1, 1))
        assert_array_equal(result.times, self.time1quant)

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
        assert_array_equal(result.magnitude, (10 - self.data1).reshape(-1, 1))
        assert_array_equal(result.times, self.time1quant)

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
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1)*2)
        assert_array_equal(result.times, self.time1quant)

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
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1)*2)
        assert_array_equal(result.times, self.time1quant)

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
        assert_array_equal(result.magnitude, self.data1.reshape(-1, 1)/0.5)
        assert_array_equal(result.times, self.time1quant)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        res = pretty(self.signal1)
        signal = self.signal1
        targ = (("IrregularlySampledSignal with %d channels of length %d; units %s; datatype %s \n" % (signal.shape[1], signal.shape[0], signal.units.dimensionality.unicode, signal.dtype)) +
                ("name: '%s'\ndescription: '%s'\n" % (signal.name, signal.description)) +
                ("annotations: %s\n" % str(signal.annotations)) +
                ("sample times: %s" % (signal.times[:10],)))
        self.assertEqual(res, targ)

    def test__merge(self):
        data1 = np.arange(1000.0, 1066.0).reshape((11, 6)) * pq.uV
        data2 = np.arange(2.0, 2.033, 0.001).reshape((11, 3)) * pq.mV
        times1 = np.arange(11.0) * pq.ms
        times2 = np.arange(1.0, 12.0) * pq.ms

        signal1 = IrregularlySampledSignal(times1, data1,
                                           name='signal1',
                                           description='test signal',
                                           file_origin='testfile.txt')
        signal2 = IrregularlySampledSignal(times1, data2,
                                           name='signal2',
                                           description='test signal',
                                           file_origin='testfile.txt')
        signal3 = IrregularlySampledSignal(times2, data2,
                                           name='signal3',
                                           description='test signal',
                                           file_origin='testfile.txt')

        merged12 = signal1.merge(signal2)

        target_data12 = np.hstack([data1, data2.rescale(pq.uV)])

        assert_neo_object_is_compliant(signal1)
        assert_neo_object_is_compliant(signal2)
        assert_neo_object_is_compliant(merged12)

        self.assertAlmostEqual(merged12[5, 0], 1030.0 * pq.uV, 9)
        self.assertAlmostEqual(merged12[5, 6], 2015.0 * pq.uV, 9)

        self.assertEqual(merged12.name, 'merge(signal1, signal2)')
        self.assertEqual(merged12.file_origin, 'testfile.txt')

        assert_arrays_equal(merged12.magnitude, target_data12)

        self.assertRaises(MergeError, signal1.merge, signal3)


class TestAnalogSignalFunctions(unittest.TestCase):
    def test__pickle(self):
        signal1 = IrregularlySampledSignal(np.arange(10.0)/100*pq.s,
                                           np.arange(10.0), units="mV")

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
            signal1 = IrregularlySampledSignal(np.arange(10.0)/100*pq.s,
                                               np.arange(10.0), units="mV")
            signal2 = IrregularlySampledSignal(np.arange(10.0)/100*pq.ms,
                                               np.arange(10.0), units="mV")
            self.assertNotEqual(signal1, signal2)


if __name__ == "__main__":
    unittest.main()
