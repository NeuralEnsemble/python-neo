# -*- coding: utf-8 -*-
"""
Tests of the neo.core.epoch.Epoch class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import quantities as pq

from neo.core.epoch import Epoch
from neo.test.tools import assert_neo_object_is_compliant


class TestEpoch(unittest.TestCase):
    def test_epoch_creation(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        epc = Epoch(1.5*pq.ms, duration=20*pq.ns,
                    label='test epoch', name='test', description='tester',
                    file_origin='test.file',
                    testarg1=1, **params)
        epc.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(epc)

        self.assertEqual(epc.time, 1.5*pq.ms)
        self.assertEqual(epc.duration, 20*pq.ns)
        self.assertEqual(epc.label, 'test epoch')
        self.assertEqual(epc.name, 'test')
        self.assertEqual(epc.description, 'tester')
        self.assertEqual(epc.file_origin, 'test.file')
        self.assertEqual(epc.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(epc.annotations['testarg1'], 1.1)
        self.assertEqual(epc.annotations['testarg2'], 'yes')
        self.assertTrue(epc.annotations['testarg3'])

    def test_epoch_merge_NotImplementedError(self):
        epc1 = Epoch(1.5*pq.ms, duration=20*pq.ns,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file')
        epc2 = Epoch(1.5*pq.ms, duration=20*pq.ns,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file')
        self.assertRaises(NotImplementedError, epc1.merge, epc2)

if __name__ == "__main__":
    unittest.main()
