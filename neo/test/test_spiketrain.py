from __future__ import absolute_import
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from neo.core.spiketrain import check_has_dimensions_time, SpikeTrain
import quantities as pq
import numpy
from neo.test.tools import assert_arrays_equal

class TestFunctions(unittest.TestCase):

    def test__check_has_dimensions_time(self):
        a = numpy.arange(3) * pq.ms
        b = numpy.arange(3) * pq.mV
        c = numpy.arange(3) * pq.mA
        d = numpy.arange(3) * pq.minute
        check_has_dimensions_time(a)
        self.assertRaises(ValueError, check_has_dimensions_time, b)
        self.assertRaises(ValueError, check_has_dimensions_time, c)
        check_has_dimensions_time(d)
        self.assertRaises(ValueError, check_has_dimensions_time, a, b, c, d)
        

class TestConstructor(unittest.TestCase):
    def test__create_empty(self):
        t_start = 0.0
        t_stop = 10.0
        st = SpikeTrain([ ], t_start, t_stop, units = 's')
    
    def test__create_from_list(self):
        times = range(10)
        t_start = 0.0
        t_stop = 10.0
        st = SpikeTrain(times, t_start, t_stop, units="ms")
        self.assertEqual(st.t_start, t_start*pq.ms)
        self.assertEqual(st.t_stop, t_stop*pq.ms)
        assert_arrays_equal(st, times*pq.ms)

    def test__create_from_array(self):
        times = numpy.arange(10)
        t_start = 0.0*pq.s
        t_stop = 10000.0*pq.ms
        st = SpikeTrain(times, t_start, t_stop, units="s")
        self.assertEqual(st.t_stop, t_stop)
        assert_arrays_equal(st, times*pq.s)

    def test__create_from_array_without_units_should_raise_ValueError(self):
        times = numpy.arange(10)
        t_start = 0.0*pq.s
        t_stop = 10.0*pq.s
        self.assertRaises(ValueError, SpikeTrain, times, t_start, t_stop)

    def test__create_from_quantity_array(self):
        times = numpy.arange(10) * pq.ms
        t_start = 0.0*pq.s
        t_stop = 12.0*pq.ms
        st = SpikeTrain(times, t_start, t_stop)
        assert_arrays_equal(st, times)


if __name__ == "__main__":
    unittest.main()