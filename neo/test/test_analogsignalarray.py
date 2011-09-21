"""
Tests of the AnalogSignalArrayArray class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from neo.core.analogsignalarray import AnalogSignalArray
from neo.core.analogsignal import AnalogSignal

import numpy
import quantities as pq
from neo.test.tools import assert_arrays_almost_equal, assert_arrays_equal

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
        data = [(i,i,i) for i in range(10)] # 3 signals each with 10 samples
        rate = 1000*Hz
        a = AnalogSignalArray(data, sampling_rate=rate, units="mV")
        self.assertEqual(a.shape, (10, 3)) 
        self.assertEqual(a.t_start, 0*ms)
        self.assertEqual(a.t_stop, len(data)/rate)
        self.assertEqual(a[9, 0], 9000*uV)
        
    def test__create_from_numpy_array(self):
        data = numpy.arange(20.0).reshape((10,2))
        rate = 1*kHz
        a = AnalogSignalArray(data, sampling_rate=rate, units="uV")
        self.assertEqual(a.t_start, 0*ms)
        self.assertEqual(a.t_stop, data.shape[0]/rate)
        self.assertEqual(a[9, 0], 0.018*mV)
        self.assertEqual(a[9, 1], 19*uV)
        
    def test__create_from_quantities_array(self):
        data = numpy.arange(20.0).reshape((10,2)) * mV
        rate = 5000*Hz
        a = AnalogSignalArray(data, sampling_rate=rate)
        self.assertEqual(a.t_start, 0*ms)
        self.assertEqual(a.t_stop, data.shape[0]/rate)
        self.assertEqual(a[9, 0], 18000*uV)
        
    def test__create_from_quantities_array_with_inconsistent_units_should_raise_ValueError(self):
        data = numpy.arange(20.0).reshape((10,2)) * mV
        self.assertRaises(ValueError, AnalogSignalArray, data, sampling_rate=1*kHz, units="nA")
    
    def test__create_with_copy_true_should_return_copy(self):
        data = numpy.arange(20.0).reshape((10,2)) * mV
        rate = 5000*Hz
        a = AnalogSignalArray(data, copy=True, sampling_rate=rate)
        data[3, 0] = 0.099*V
        self.assertNotEqual(a[3, 0], 99*mV)
    
    def test__create_with_copy_false_should_return_view(self):
        data = numpy.arange(20.0).reshape((10,2)) * mV
        rate = 5000*Hz
        a = AnalogSignalArray(data, copy=False, sampling_rate=rate)
        data[3, 0] = 99*mV
        self.assertEqual(a[3, 0], 99000*uV)

    # signal must not be 1D - should raise Exception if 1D
    

class TestProperties(unittest.TestCase):
    
    def setUp(self):
        self.t_start = [0.0*ms, 100*ms, -200*ms]
        self.rates = [1*kHz, 420*Hz, 999*Hz]
        self.data = [numpy.arange(10.0).reshape((5,2))*nA, numpy.arange(-100.0, 100.0, 10.0).reshape((4,5))*mV,
                     numpy.random.uniform(size=(100,4))*uV]
        self.signals = [AnalogSignalArray(D, sampling_rate=r, t_start=t)
                        for r,D,t in zip(self.rates, self.data, self.t_start)]
  
    def test__t_stop(self):
        for i in range(3):
            self.assertEqual(self.signals[i].t_stop,
                             self.t_start[i] + self.data[i].shape[0]/self.rates[i])
            
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
                                       numpy.arange(self.data[i].shape[0])/self.rates[i] + self.t_start[i],
                                       1e-12*ms)


class TestArrayMethods(unittest.TestCase):
    
    def setUp(self):
        self.signal = AnalogSignalArray(numpy.arange(55.0).reshape((11,5)), units="nA", sampling_rate=1*kHz)
    
    def test__index_dim1_should_return_analogsignal(self):
        single_signal = self.signal[:, 0]
        self.assertIsInstance(single_signal, AnalogSignal)
        self.assertEqual(single_signal.t_stop, self.signal.t_stop)
        self.assertEqual(single_signal.t_start, self.signal.t_start)
        self.assertEqual(single_signal.sampling_rate, self.signal.sampling_rate)
    
    def test__index_dim1_and_slice_dim0_should_return_analogsignal(self):
        single_signal = self.signal[2:7,0]
        self.assertIsInstance(single_signal, AnalogSignal)
        self.assertEqual(single_signal.t_start, self.signal.t_start+2*self.signal.sampling_period)
        self.assertEqual(single_signal.t_stop, self.signal.t_start+7*self.signal.sampling_period)
        self.assertEqual(single_signal.sampling_rate, self.signal.sampling_rate)
    
    def test__index_dim0_should_return_quantity_array(self):
        # i.e. values from all signals for a single point in time
        values = self.signal[3, :]
        self.assertNotIsInstance(values, AnalogSignalArray)
        self.assertEqual(values.shape, (5,))
        assert not hasattr(values, "t_start")
        self.assertEqual(values.units, pq.nA)
   
    def test__index_dim0_and_slice_dim1_should_return_quantity_array(self):
        # i.e. values from a subset of signals for a single point in time
        values = self.signal[3, 2:5]
        self.assertNotIsInstance(values, AnalogSignalArray)
        self.assertEqual(values.shape, (3,))
        assert not hasattr(values, "t_start")
        self.assertEqual(values.units, pq.nA)
    
    def test__slice_both_dimensions_should_return_analogsignalarray(self):
        values = self.signal[0:3, 0:3]
        assert_arrays_equal(values, AnalogSignalArray([[0, 1, 2], [5, 6, 7], [10, 11, 12]], dtype=float, units="nA", sampling_rate=1*kHz))
    
    def test__slice_only_first_dimension_should_return_analogsignalarray(self):
        signal2 = self.signal[2:7]
        self.assertIsInstance(signal2, AnalogSignalArray)
        self.assertEqual(signal2.shape, (5,5))
        self.assertEqual(signal2.t_start, self.signal.t_start+2*self.signal.sampling_period)
        self.assertEqual(signal2.t_stop, self.signal.t_start+7*self.signal.sampling_period)
        self.assertEqual(signal2.sampling_rate, self.signal.sampling_rate)
    
    def test__getitem_should_return_single_quantity(self):
        self.assertEqual(self.signal[9, 3], 48000*pA) # quantities drops the units in this case
        self.assertEqual(self.signal[9][3], self.signal[9, 3])
        assert hasattr(self.signal[9, 3], 'units')
        self.assertRaises(IndexError, self.signal.__getitem__, (99,73))

    def test_comparison_operators(self):
        assert_arrays_equal(self.signal[0:3, 0:3] >= 5*nA,
                            numpy.array([[False, False, False], [True, True, True], [True, True, True]]))
        assert_arrays_equal(self.signal[0:3, 0:3] >= 5*pA,
                            numpy.array([[False, True, True], [True, True, True], [True, True, True]]))

    def test__comparison_with_inconsistent_units_should_raise_Exception(self):
        self.assertRaises(ValueError, self.signal.__gt__, 5*mV)
        
    def test_simple_statistics(self):
        self.assertEqual(self.signal.max(), 54000*pA)
        self.assertEqual(self.signal.min(), 0*nA)
        self.assertEqual(self.signal.mean(), 27*nA)


class TestEquality(unittest.TestCase):
    
    def test__signals_with_different_data_complement_should_be_non_equal(self):
            signal1 = AnalogSignalArray(numpy.arange(55.0).reshape((11,5)), units="mV", sampling_rate=1*kHz)
            signal2 = AnalogSignalArray(numpy.arange(55.0).reshape((11,5)), units="mV", sampling_rate=2*kHz)
            self.assertNotEqual(signal1, signal2)


class TestCombination(unittest.TestCase):
    
    def test__adding_a_constant_to_a_signal_should_preserve_data_complement(self):
        signal = AnalogSignalArray(numpy.arange(55.0).reshape((11,5)), units="mV", sampling_rate=1*kHz)
        signal_with_offset = signal + 0.065*V
        self.assertEqual(signal[0, 4], 4*mV)
        self.assertEqual(signal_with_offset[0, 4], 69000*uV)
        for attr in "t_start", "sampling_rate":
            self.assertEqual(getattr(signal, attr),
                             getattr(signal_with_offset, attr))

    def test__adding_two_consistent_signals_should_preserve_data_complement(self):
        signal1 = AnalogSignalArray(numpy.arange(55.0).reshape((11, 5)), units="mV", sampling_rate=1*kHz)
        signal2 = AnalogSignalArray(numpy.arange(100.0, 155.0).reshape((11, 5)), units="mV", sampling_rate=1*kHz)
        sum = signal1 + signal2
        assert_arrays_equal(sum, AnalogSignalArray(numpy.arange(100.0, 210.0, 2.0).reshape((11, 5)), units="mV", sampling_rate=1*kHz))

    def test__adding_signals_with_inconsistent_data_complement_should_raise_Exception(self):
        signal1 = AnalogSignalArray(numpy.arange(55.0).reshape((11, 5)), units="mV", sampling_rate=1*kHz)
        signal2 = AnalogSignalArray(numpy.arange(100.0, 155.0).reshape((11, 5)), units="mV", sampling_rate=0.5*kHz)
        self.assertRaises(Exception, signal1.__add__, signal2)


if __name__ == "__main__":
    unittest.main()
