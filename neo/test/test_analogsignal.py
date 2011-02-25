# encoding: utf-8
"""
Tests of the AnalogSignal class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
from __future__ import division

from neo.core.analogsignal import AnalogSignal
import numpy
import quantities as pq

mV = pq.mV
uV = pq.uV
Hz = pq.Hz
kHz = pq.kHz
ms = pq.ms
nA = pq.nA

class TestConstructor(unittest.TestCase):
    
    def test__create_from_list(self):
        data = range(10)
        rate = 1000*Hz
        a = AnalogSignal(data, sampling_rate=rate, unit="mV")
        self.assertEqual(a.t_stop, len(data)/rate)
        
    def test__create_from_numpy_array(self):
        data = numpy.arange(10.0)
        rate = 1*kHz
        a = AnalogSignal(data, sampling_rate=rate, unit="uV")
        self.assertEqual(a.t_stop, data.size/rate)
        
    def test__create_from_quantities_array(self):
        data = numpy.arange(10.0) * mV
        rate = 5000*Hz
        a = AnalogSignal(data, sampling_rate=rate)
        self.assertEqual(a.t_stop, data.size/rate)
        
    def test__create_from_quantities_array_with_inconsistent_units_should_raise_ValueError(self):
        data = numpy.arange(10.0) * mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=1*kHz, unit="nA")
        

class TestProperties(unittest.TestCase):
    
    def setUp(self):
        self.t_start = [0.0, 100*ms, -200*ms]
        self.rates = [1*kHz, 420*Hz, 999*Hz]
        self.data = [numpy.arange(10.0)*nA, numpy.arange(-100.0, 100.0, 10.0)*mV,
                     numpy.random.uniform(size=100)*uV]
        self.signals = [AnalogSignal(D, sampling_rate=r, t_start=t)
                        for r,D,t in zip(self.rates, self.data, self.t_start)]
    
    def test__t_stop(self):
        for i in range(3):
            self.assertEqual(self.signals[i].tstop,
                             self.t_start[i] + self.data[i].size/self.rates[i])
            
    def test__duration(self):
        for signal in self.signals:
            self.assertEqual(signal.duration, signal.t_stop - signal.t_start)
            
    def test__sampling_period(self):
        for signal, rate in zip(self.signals, self.rates):
            self.assertEqual(signals.sampling_period, 1/rate)
            
    def test__times(self):
        for i in range(3):
            self.assertEqual(self.signals[i].times,
                             numpy.arange(self.data[i].size)/self.rates[i] + self.t_start)


class TestArrayMethods(unittest.TestCase):
    
    def setUp(self):
        self.signal = AnalogSignal(numpy.arange(10.0), sampling_rate=1*kHz)
    
    def test__slice_should_return_AnalogSignal(self):
        sub = self.signal[3:8]
        self.assertIsInstance(sub, AnalogSignal)
        self.assertEqual(sub.size, 5)
        self.assertEqual(sub.sampling_interval, self.signal.sampling_interval)
        self.assertEqual(sub.t_start,
                         self.signal.t_start+3*sub.sampling_interval)
        self.assertEqual(sub.t_stop,
                         sub.t_start + 5*sub.sampling_interval)

    

if __name__ == "__main__":
    unittest.main()
