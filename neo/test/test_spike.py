# -*- coding: utf-8 -*-
"""
Tests of the neo.core.spike.Spike class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import quantities as pq

from neo.core.spike import Spike
from neo.test.tools import assert_arrays_equal, assert_neo_object_is_compliant


class TestSpike(unittest.TestCase):
    def setUp(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        self.sampling_rate1 = .1*pq.Hz
        self.left_sweep1 = 2.*pq.s
        self.spike1 = Spike(1.5*pq.ms,  waveform=[[1.1, 1.5, 1.7],
                                                  [2.2, 2.6, 2.8]]*pq.mV,
                            sampling_rate=self.sampling_rate1,
                            left_sweep=self.left_sweep1,
                            name='test', description='tester',
                            file_origin='test.file',
                            testarg1=1, **params)
        self.spike1.annotate(testarg1=1.1, testarg0=[1, 2, 3])

    def test_spike_creation(self):
        assert_neo_object_is_compliant(self.spike1)

        self.assertEqual(self.spike1.time, 1.5*pq.ms)
        assert_arrays_equal(self.spike1.waveform, [[1.1, 1.5, 1.7],
                                                   [2.2, 2.6, 2.8]]*pq.mV)
        self.assertEqual(self.spike1.sampling_rate, .1*pq.Hz)
        self.assertEqual(self.spike1.left_sweep, 2.*pq.s)
        self.assertEqual(self.spike1.description, 'tester')
        self.assertEqual(self.spike1.file_origin, 'test.file')
        self.assertEqual(self.spike1.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(self.spike1.annotations['testarg1'], 1.1)
        self.assertEqual(self.spike1.annotations['testarg2'], 'yes')
        self.assertTrue(self.spike1.annotations['testarg3'])

    def test__duration(self):
        result1 = self.spike1.duration

        self.spike1.sampling_rate = None
        assert_neo_object_is_compliant(self.spike1)
        result2 = self.spike1.duration

        self.spike1.sampling_rate = self.sampling_rate1
        self.spike1.waveform = None
        assert_neo_object_is_compliant(self.spike1)
        result3 = self.spike1.duration

        self.assertEqual(result1, 30./pq.Hz)
        self.assertEqual(result1.units, 1./pq.Hz)
        self.assertEqual(result2, None)
        self.assertEqual(result3, None)

    def test__sampling_period(self):
        result1 = self.spike1.sampling_period

        self.spike1.sampling_rate = None
        assert_neo_object_is_compliant(self.spike1)
        result2 = self.spike1.sampling_period

        self.spike1.sampling_rate = self.sampling_rate1
        self.spike1.sampling_period = 10.*pq.ms
        assert_neo_object_is_compliant(self.spike1)
        result3a = self.spike1.sampling_period
        result3b = self.spike1.sampling_rate

        self.spike1.sampling_period = None
        result4a = self.spike1.sampling_period
        result4b = self.spike1.sampling_rate

        self.assertEqual(result1, 10./pq.Hz)
        self.assertEqual(result1.units, 1./pq.Hz)
        self.assertEqual(result2, None)
        self.assertEqual(result3a, 10.*pq.ms)
        self.assertEqual(result3a.units, 1.*pq.ms)
        self.assertEqual(result3b, .1/pq.ms)
        self.assertEqual(result3b.units, 1./pq.ms)
        self.assertEqual(result4a, None)
        self.assertEqual(result4b, None)

    def test__right_sweep(self):
        result1 = self.spike1.right_sweep

        self.spike1.left_sweep = None
        assert_neo_object_is_compliant(self.spike1)
        result2 = self.spike1.right_sweep

        self.spike1.left_sweep = self.left_sweep1
        self.spike1.sampling_rate = None
        assert_neo_object_is_compliant(self.spike1)
        result3 = self.spike1.right_sweep

        self.spike1.sampling_rate = self.sampling_rate1
        self.spike1.waveform = None
        assert_neo_object_is_compliant(self.spike1)
        result4 = self.spike1.right_sweep

        self.assertEqual(result1, 32.*pq.s)
        self.assertEqual(result1.units, 1.*pq.s)
        self.assertEqual(result2, None)
        self.assertEqual(result3, None)
        self.assertEqual(result4, None)


if __name__ == "__main__":
    unittest.main()
