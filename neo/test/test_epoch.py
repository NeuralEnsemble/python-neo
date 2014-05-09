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
from neo.core import Segment
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

    def test__children(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        epc = Epoch(1.5*pq.ms, duration=20*pq.ns,
                    label='test epoch', name='test', description='tester',
                    file_origin='test.file',
                    testarg1=1, **params)
        epc.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(epc)

        segment = Segment(name='seg1')
        segment.epochs = [epc]
        segment.create_many_to_one_relationship()

        self.assertEqual(epc._container_child_objects, ())
        self.assertEqual(epc._data_child_objects, ())
        self.assertEqual(epc._single_parent_objects, ('Segment',))
        self.assertEqual(epc._multi_child_objects, ())
        self.assertEqual(epc._multi_parent_objects, ())
        self.assertEqual(epc._child_properties, ())

        self.assertEqual(epc._single_child_objects, ())

        self.assertEqual(epc._container_child_containers, ())
        self.assertEqual(epc._data_child_containers, ())
        self.assertEqual(epc._single_child_containers, ())
        self.assertEqual(epc._single_parent_containers, ('segment',))
        self.assertEqual(epc._multi_child_containers, ())
        self.assertEqual(epc._multi_parent_containers, ())

        self.assertEqual(epc._child_objects, ())
        self.assertEqual(epc._child_containers, ())
        self.assertEqual(epc._parent_objects, ('Segment',))
        self.assertEqual(epc._parent_containers, ('segment',))

        self.assertEqual(epc.children, ())
        self.assertEqual(len(epc.parents), 1)
        self.assertEqual(epc.parents[0].name, 'seg1')

        epc.create_many_to_one_relationship()
        epc.create_many_to_many_relationship()
        epc.create_relationship()
        assert_neo_object_is_compliant(epc)

if __name__ == "__main__":
    unittest.main()
