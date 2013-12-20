# -*- coding: utf-8 -*-
"""
Tests of the neo.core.event.Event class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import quantities as pq

from neo.core.event import Event
from neo.test.tools import assert_neo_object_is_compliant


class TestEvent(unittest.TestCase):
    def test_Event_creation(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        evt = Event(1.5*pq.ms,
                    label='test epoch', name='test', description='tester',
                    file_origin='test.file',
                    testarg1=1, **params)
        evt.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(evt)

        self.assertEqual(evt.time, 1.5*pq.ms)
        self.assertEqual(evt.label, 'test epoch')
        self.assertEqual(evt.name, 'test')
        self.assertEqual(evt.description, 'tester')
        self.assertEqual(evt.file_origin, 'test.file')
        self.assertEqual(evt.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(evt.annotations['testarg1'], 1.1)
        self.assertEqual(evt.annotations['testarg2'], 'yes')
        self.assertTrue(evt.annotations['testarg3'])

    def test_epoch_merge_NotImplementedError(self):
        evt1 = Event(1.5*pq.ms,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file', testarg1=1)
        evt2 = Event(1.5*pq.ms,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file', testarg1=1)
        self.assertRaises(NotImplementedError, evt1.merge, evt2)


if __name__ == "__main__":
    unittest.main()
