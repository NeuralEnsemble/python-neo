# -*- coding: utf-8 -*-
"""
Tests of the neo.core.irregularlysampledsignal.IrregularySampledSignal class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import os
import pickle
import numpy as np
import quantities as pq

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.core import Segment, RecordingChannel
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
        signal = get_fake_value('signal', pq.Quantity, seed=1, dim=1)
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

        assert_arrays_equal(res11.pop('times'), times)
        assert_arrays_equal(res12.pop('times'), times)
        assert_arrays_equal(res21.pop('times'), times)
        assert_arrays_equal(res22.pop('times'), times)

        assert_arrays_equal(res11.pop('signal'), signal)
        assert_arrays_equal(res12.pop('signal'), signal)
        assert_arrays_equal(res21.pop('signal'), signal)
        assert_arrays_equal(res22.pop('signal'), signal)

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

        assert_arrays_equal(sig.times, [1.1, 1.5, 1.7]*pq.ms)
        assert_arrays_equal(np.asarray(sig), np.array([20., 40., 60.]))
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

        assert_arrays_equal(sig.times, [1.1, 1.5, 1.7]*pq.s)
        assert_arrays_equal(np.asarray(sig), np.array([20., 40., 60.]))
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

        assert_arrays_equal(sig.times, [1100, 1500, 1700]*pq.ms)
        assert_arrays_equal(np.asarray(sig), np.array([2000., 4000., 6000.]))
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

        targ = ('<IrregularlySampledSignal(array([ 2.,  4.,  6.]) * V ' +
                'at times [ 1.1  1.5  1.7] s)>')
        res = repr(sig)
        self.assertEqual(targ, res)

    def test__children(self):
        signal = self.signals[0]

        segment = Segment(name='seg1')
        segment.analogsignals = [signal]
        segment.create_many_to_one_relationship()

        rchan = RecordingChannel(name='rchan1')
        rchan.analogsignals = [signal]
        rchan.create_many_to_one_relationship()

        self.assertEqual(signal._single_parent_objects,
                         ('Segment', 'RecordingChannel'))
        self.assertEqual(signal._multi_parent_objects, ())

        self.assertEqual(signal._single_parent_containers,
                         ('segment', 'recordingchannel'))
        self.assertEqual(signal._multi_parent_containers, ())

        self.assertEqual(signal._parent_objects,
                         ('Segment', 'RecordingChannel'))
        self.assertEqual(signal._parent_containers,
                         ('segment', 'recordingchannel'))

        self.assertEqual(len(signal.parents), 2)
        self.assertEqual(signal.parents[0].name, 'seg1')
        self.assertEqual(signal.parents[1].name, 'rchan1')

        assert_neo_object_is_compliant(signal)


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

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        res = pretty(self.signal1)
        targ = ("IrregularlySampledSignal\n" +
                "name: '%s'\ndescription: '%s'\nannotations: %s" %
                (self.signal1.name, self.signal1.description,
                 pretty(self.signal1.annotations)))
        self.assertEqual(res, targ)



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

        assert_arrays_equal(signal1, signal2)
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
