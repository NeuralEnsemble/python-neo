# -*- coding: utf-8 -*-
"""
Tests of the neo.core.eventarray.EventArray class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.eventarray import EventArray
from neo.core import Segment
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal, assert_same_sub_schema)


class TestEventArray(unittest.TestCase):
    def test_EventArray_creation(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        evta = EventArray([1.1, 1.5, 1.7]*pq.ms,
                          labels=np.array(['test event 1',
                                           'test event 2',
                                           'test event 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          testarg1=1, **params)
        evta.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(evta)

        assert_arrays_equal(evta.times, [1.1, 1.5, 1.7]*pq.ms)
        assert_arrays_equal(evta.labels, np.array(['test event 1',
                                                   'test event 2',
                                                   'test event 3'], dtype='S'))
        self.assertEqual(evta.name, 'test')
        self.assertEqual(evta.description, 'tester')
        self.assertEqual(evta.file_origin, 'test.file')
        self.assertEqual(evta.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(evta.annotations['testarg1'], 1.1)
        self.assertEqual(evta.annotations['testarg2'], 'yes')
        self.assertTrue(evta.annotations['testarg3'])

    def test_EventArray_repr(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        evta = EventArray([1.1, 1.5, 1.7]*pq.ms,
                          labels=np.array(['test event 1',
                                           'test event 2',
                                           'test event 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          testarg1=1, **params)
        evta.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(evta)

        targ = ('<EventArray: test event 1@1.1 ms, test event 2@1.5 ms, ' +
                'test event 3@1.7 ms>')

        res = repr(evta)

        self.assertEqual(targ, res)

    def test_EventArray_merge(self):
        params1 = {'testarg2': 'yes', 'testarg3': True}
        params2 = {'testarg2': 'no', 'testarg4': False}
        paramstarg = {'testarg2': 'yes;no',
                      'testarg3': True,
                      'testarg4': False}
        epca1 = EventArray([1.1, 1.5, 1.7]*pq.ms,
                           labels=np.array(['test event 1 1',
                                            'test event 1 2',
                                            'test event 1 3'], dtype='S'),
                           name='test', description='tester 1',
                           file_origin='test.file',
                           testarg1=1, **params1)
        epca2 = EventArray([2.1, 2.5, 2.7]*pq.us,
                           labels=np.array(['test event 2 1',
                                            'test event 2 2',
                                            'test event 2 3'], dtype='S'),
                           name='test', description='tester 2',
                           file_origin='test.file',
                           testarg1=1, **params2)
        epcatarg = EventArray([1.1, 1.5, 1.7, .0021, .0025, .0027]*pq.ms,
                              labels=np.array(['test event 1 1',
                                               'test event 1 2',
                                               'test event 1 3',
                                               'test event 2 1',
                                               'test event 2 2',
                                               'test event 2 3'], dtype='S'),
                              name='test',
                              description='merge(tester 1, tester 2)',
                              file_origin='test.file',
                              testarg1=1, **paramstarg)
        assert_neo_object_is_compliant(epca1)
        assert_neo_object_is_compliant(epca2)
        assert_neo_object_is_compliant(epcatarg)

        epcares = epca1.merge(epca2)
        assert_neo_object_is_compliant(epcares)
        assert_same_sub_schema(epcatarg, epcares)

    def test__children(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        evta = EventArray([1.1, 1.5, 1.7]*pq.ms,
                          labels=np.array(['test event 1',
                                           'test event 2',
                                           'test event 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          testarg1=1, **params)
        evta.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        assert_neo_object_is_compliant(evta)

        segment = Segment(name='seg1')
        segment.eventarrays = [evta]
        segment.create_many_to_one_relationship()

        self.assertEqual(evta._container_child_objects, ())
        self.assertEqual(evta._data_child_objects, ())
        self.assertEqual(evta._single_parent_objects, ('Segment',))
        self.assertEqual(evta._multi_child_objects, ())
        self.assertEqual(evta._multi_parent_objects, ())
        self.assertEqual(evta._child_properties, ())

        self.assertEqual(evta._single_child_objects, ())

        self.assertEqual(evta._container_child_containers, ())
        self.assertEqual(evta._data_child_containers, ())
        self.assertEqual(evta._single_child_containers, ())
        self.assertEqual(evta._single_parent_containers, ('segment',))
        self.assertEqual(evta._multi_child_containers, ())
        self.assertEqual(evta._multi_parent_containers, ())

        self.assertEqual(evta._child_objects, ())
        self.assertEqual(evta._child_containers, ())
        self.assertEqual(evta._parent_objects, ('Segment',))
        self.assertEqual(evta._parent_containers, ('segment',))

        self.assertEqual(evta.children, ())
        self.assertEqual(len(evta.parents), 1)
        self.assertEqual(evta.parents[0].name, 'seg1')

        evta.create_many_to_one_relationship()
        evta.create_many_to_many_relationship()
        evta.create_relationship()
        assert_neo_object_is_compliant(evta)


if __name__ == "__main__":
    unittest.main()
