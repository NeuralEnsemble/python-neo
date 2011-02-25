# encoding: utf-8
"""
Tests of the AnalogSignal class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from neo.core.analogsignal import AnalogSignal
import numpy
import quantities as pq

mV = pq.mV
Hz = pq.Hz
kHz = pq.kHz
ms = pq.ms

class TestConstructor(unittest.TestCase):
    
    def test__create_from_list(self):
        data = range(10)
        rate = 1000*Hz
        a = AnalogSignal(data, sampling_rate=rate, unit="mV")
        self.assertEqual(a.t_stop, len(data)/rate)
        
    def test__create_from_numpy_array(self):
        data = numpy.arange(10.0)
        rate = 1*kHz
        a = AnalogSignal(data, sampling_rate=rate, unit="µV")
        self.assertEqual(a.t_stop, data.size/rate)
        
    def test__create_from_quantities_array(self):
        data = numpy.arange(10.0) * mV
        rate = 5000*Hz
        a = AnalogSignal(data, sampling_rate=rate)
        self.assertEqual(a.t_stop, data.size/rate)
        
    def test__create_from_quantities_array_with_inconsistent_units_should_raise_ValueError(self):
        data = numpy.arange(10.0) * mV
        self.assertRaises(ValueError, AnalogSignal, data, sampling_rate=1*kHz, unit="nA")
        
        
if __name__ == "__main__":
    unittest.main()
