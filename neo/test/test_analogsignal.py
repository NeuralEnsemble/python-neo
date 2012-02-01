# encoding: utf-8
"""
Tests of the AnalogSignal class
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.core.analogsignal import AnalogSignal
import numpy
import quantities as pq
import pickle
from neo.test.tools import assert_arrays_almost_equal, assert_arrays_equal
import os

V = pq.V
mV = pq.mV
uV = pq.uV
Hz = pq.Hz
kHz = pq.kHz
ms = pq.ms
nA = pq.nA
pA = pq.pA


class TestConstructor(unittest.TestCase):
    
    def test__create_from_list(self):
        data = range(10)
        rate = 1000*Hz
        a = AnalogSignal(data, sampling_rate=rate, units="mV")
        self.assertEqual(a.t_start, 0*ms)
        self.assertEqual(a.t_stop, len(data)/rate)
        self.assertEqual(a[9], 9000*uV)
        
    def test__create_from_numpy_array(self):
        data = numpy.arange(10.0)
        rate = 1*kHz
        a = AnalogSignal(data, sampling_rate=rate, units="uV")
        self.assertEqual(a.t_start, 0*ms)
        self.assertEqual(a.t_stop, data.size/rate)
        self.assertEqual(a[9], 0.009*mV)
        
    def test__create_from_quantities_array(self):
        data = numpy.arange(10.0) * mV
        rate = 5000*Hz
        a = AnalogSignal(data, sampling_rate=rate)
        self.assertEqual(a.t_start, 0*ms)
        self.assertEqual(a.t_stop, data.size/rate)
        self.assertEqual(a[9], 0.009*V)
        
    def test__create_from_quantities_array_with_inconsistent_units_should_raise_ValueError(self):
        data = numpy.arange(10.0) * mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=1*kHz, units="nA")
    
    def test__create_with_copy_true_should_return_copy(self):
        data = numpy.arange(10.0) * mV
        rate = 5000*Hz
        a = AnalogSignal(data, copy=True, sampling_rate=rate)
        data[3] = 99*mV
        self.assertNotEqual(a[3], 99*mV)
    
    def test__create_with_copy_false_should_return_view(self):
        data = numpy.arange(10.0) * mV
        rate = 5000*Hz
        a = AnalogSignal(data, copy=False, sampling_rate=rate)
        data[3] = 99*mV
        self.assertEqual(a[3], 99*mV)
    
    def test__create_with_additional_argument(self):
        a = AnalogSignal([1,2,3], units="mV", sampling_rate=1*kHz, file_origin='crack.txt', ratname='Nicolas')
        self.assertEqual(a.annotations, {'ratname':'Nicolas'})
        
        # This one is universally recommended and handled by BaseNeo
        self.assertEqual(a.file_origin, 'crack.txt')

    # signal must be 1D - should raise Exception if not 1D
    

class TestProperties(unittest.TestCase):
    
    def setUp(self):
        self.t_start = [0.0*ms, 100*ms, -200*ms]
        self.rates = [1*kHz, 420*Hz, 999*Hz]
        self.data = [numpy.arange(10.0)*nA, numpy.arange(-100.0, 100.0, 10.0)*mV,
                     numpy.random.uniform(size=100)*uV]
        self.signals = [AnalogSignal(D, sampling_rate=r, t_start=t)
                        for r,D,t in zip(self.rates, self.data, self.t_start)]
    
    def test__t_stop(self):
        for i in range(3):
            self.assertEqual(self.signals[i].t_stop,
                             self.t_start[i] + self.data[i].size/self.rates[i])
            
    def test__duration(self):
        for signal in self.signals:
            self.assertAlmostEqual(signal.duration,
                                   signal.t_stop - signal.t_start,
                                   delta=1e-15)
            
    def test__sampling_period(self):
        for signal, rate in zip(self.signals, self.rates):
            self.assertEqual(signal.sampling_period, 1/rate)
            
    def test__times(self):
        for i in range(3):
            assert_arrays_almost_equal(self.signals[i].times,
                                       numpy.arange(self.data[i].size)/self.rates[i] + self.t_start[i],
                                       1e-12*ms)


class TestArrayMethods(unittest.TestCase):
    
    def setUp(self):
        self.signal = AnalogSignal(numpy.arange(10.0), units="nA", sampling_rate=1*kHz)
    
    def test__slice_should_return_AnalogSignal(self):
        sub = self.signal[3:8]
        self.assertIsInstance(sub, AnalogSignal)
        self.assertEqual(sub.size, 5)
        self.assertEqual(sub.sampling_period, self.signal.sampling_period)
        self.assertEqual(sub.sampling_rate, self.signal.sampling_rate)
        self.assertEqual(sub.t_start,
                         self.signal.t_start+3*sub.sampling_period)
        self.assertEqual(sub.t_stop,
                         sub.t_start + 5*sub.sampling_period)
        
        # Test other attributes were copied over (in this case, defaults)
        self.assertEqual(sub.file_origin, self.signal.file_origin)
        self.assertEqual(sub.name, self.signal.name)
        self.assertEqual(sub.description, self.signal.description)        
        self.assertEqual(sub.annotations, self.signal.annotations)
        

        sub = self.signal[3:8]
        self.assertEqual(sub.file_origin, self.signal.file_origin)
        self.assertEqual(sub.name, self.signal.name)
        self.assertEqual(sub.description, self.signal.description)
        self.assertEqual(sub.annotations, self.signal.annotations)
    
    def test__slice_with_attributes(self):
        # Set attributes, slice, test that they are copied
        self.signal.file_origin = 'crack.txt'
        self.signal.name = 'sig'
        self.signal.description = 'a signal'
        self.signal.annotate(ratname='Georges')

        # slice
        sub = self.signal[3:8]
        
        # tests from other slice test
        self.assertIsInstance(sub, AnalogSignal)
        self.assertEqual(sub.size, 5)
        self.assertEqual(sub.sampling_period, self.signal.sampling_period)
        self.assertEqual(sub.sampling_rate, self.signal.sampling_rate)
        self.assertEqual(sub.t_start,
                         self.signal.t_start+3*sub.sampling_period)
        self.assertEqual(sub.t_stop,
                         sub.t_start + 5*sub.sampling_period)
        
        # Test other attributes were copied over (in this case, set by user)
        self.assertEqual(sub.file_origin, self.signal.file_origin)
        self.assertEqual(sub.name, self.signal.name)
        self.assertEqual(sub.description, self.signal.description)        
        self.assertEqual(sub.annotations, self.signal.annotations)
        self.assertEqual(sub.annotations, {'ratname': 'Georges'})

    def test__getitem_should_return_single_quantity(self):
        self.assertEqual(self.signal[0], 0*nA)
        self.assertEqual(self.signal[9], 9*nA)
        self.assertRaises(IndexError, self.signal.__getitem__, 10)

    def test_comparison_operators(self):
        assert_arrays_equal(self.signal >= 5*nA,
                            numpy.array([False, False, False, False, False, True, True, True, True, True]))
        assert_arrays_equal(self.signal >= 5*pA,
                            numpy.array([False, True, True, True, True, True, True, True, True, True]))

    def test__comparison_with_inconsistent_units_should_raise_Exception(self):
        self.assertRaises(ValueError, self.signal.__gt__, 5*mV)
        
    def test_simple_statistics(self):
        self.assertEqual(self.signal.max(), 9*nA)
        self.assertEqual(self.signal.min(), 0*nA)
        self.assertEqual(self.signal.mean(), 4.5*nA)

class TestEquality(unittest.TestCase):
    
    def test__signals_with_different_data_complement_should_be_non_equal(self):
            signal1 = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=1*kHz)
            signal2 = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=2*kHz)
            self.assertNotEqual(signal1, signal2)


class TestCombination(unittest.TestCase):
    
    def test__adding_a_constant_to_a_signal_should_preserve_data_complement(self):
        signal = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=1*kHz, name="foo")
        signal_with_offset = signal + 65*mV
        self.assertEqual(signal[9], 9*mV)
        self.assertEqual(signal_with_offset[9], 74*mV)
        for attr in "t_start", "sampling_rate":
            self.assertEqual(getattr(signal, attr),
                             getattr(signal_with_offset, attr))

    def test__adding_two_consistent_signals_should_preserve_data_complement(self):
        signal1 = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=1*kHz)
        signal2 = AnalogSignal(numpy.arange(10.0, 20.0), units="mV", sampling_rate=1*kHz)
        sum = signal1 + signal2
        assert_arrays_equal(sum, AnalogSignal(numpy.arange(10.0, 30.0, 2.0), units="mV", sampling_rate=1*kHz))

    def test__adding_signals_with_inconsistent_data_complement_should_raise_Exception(self):
        signal1 = AnalogSignal(numpy.arange(10.0), units="mV", t_start=0.0*ms, sampling_rate=1*kHz)
        signal2 = AnalogSignal(numpy.arange(10.0), units="mV", t_start=100.0*ms, sampling_rate=0.5*kHz)
        self.assertRaises(Exception, signal1.__add__, signal2)

    def test__subtracting_a_constant_from_a_signal_should_preserve_data_complement(self):
        signal = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=1*kHz, name="foo")
        signal_with_offset = signal - 65*mV
        self.assertEqual(signal[9], 9*mV)
        self.assertEqual(signal_with_offset[9], -56*mV)
        for attr in "t_start", "sampling_rate":
            self.assertEqual(getattr(signal, attr),
                             getattr(signal_with_offset, attr))
            
    def test__subtracting_a_signal_from_a_constant_should_return_a_signal(self):
        signal = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=1*kHz, name="foo")
        signal_with_offset = 10*mV - signal
        self.assertEqual(signal[9], 9*mV)
        self.assertEqual(signal_with_offset[9], 1*mV)
        for attr in "t_start", "sampling_rate":
            self.assertEqual(getattr(signal, attr),
                             getattr(signal_with_offset, attr))

    def test__multiplying_a_signal_by_a_constant_should_preserve_data_complement(self):
        signal = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=1*kHz, name="foo")
        amplified_signal = signal * 2
        self.assertEqual(signal[9], 9*mV)
        self.assertEqual(amplified_signal[9], 18*mV)
        for attr in "t_start", "sampling_rate":
            self.assertEqual(getattr(signal, attr),
                             getattr(amplified_signal, attr))
            
    def test__dividing_a_signal_by_a_constant_should_preserve_data_complement(self):
        signal = AnalogSignal(numpy.arange(10.0), units="mV", sampling_rate=1*kHz, name="foo")
        amplified_signal = signal/0.5
        self.assertEqual(signal[9], 9*mV)
        self.assertEqual(amplified_signal[9], 18*mV)
        for attr in "t_start", "sampling_rate":
            self.assertEqual(getattr(signal, attr),
                             getattr(amplified_signal, attr))

class TestFunctions(unittest.TestCase):
    
    def test__pickle(self):
        a = AnalogSignal([1,2,3,4],sampling_period=1*pq.ms,units=pq.S)
        a.annotations['index'] = 2

        f = open('./pickle','wb')
        pickle.dump(a,f)
        f.close()
       
        f = open('./pickle','rb')
        try:
            b = pickle.load(f)
        except ValueError:
            b = None

        assert_arrays_equal(a, b)
        f.close()
        os.remove('./pickle')


if __name__ == "__main__":
    unittest.main()
