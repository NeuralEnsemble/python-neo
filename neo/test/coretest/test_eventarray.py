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

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.eventarray import EventArray
from neo.core import Segment
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal, assert_same_sub_schema)
from neo.test.generate_datasets import (get_fake_value, get_fake_values,
                                        fake_neo, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__get_fake_values(self):
        self.annotations['seed'] = 0
        times = get_fake_value('times', pq.Quantity, seed=0, dim=1)
        labels = get_fake_value('labels', np.ndarray, seed=1, dim=1, dtype='S')
        name = get_fake_value('name', str, seed=2, obj=EventArray)
        description = get_fake_value('description', str,
                                     seed=3, obj='EventArray')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(EventArray, annotate=False, seed=0)
        res12 = get_fake_values('EventArray', annotate=False, seed=0)
        res21 = get_fake_values(EventArray, annotate=True, seed=0)
        res22 = get_fake_values('EventArray', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('times'), times)
        assert_arrays_equal(res12.pop('times'), times)
        assert_arrays_equal(res21.pop('times'), times)
        assert_arrays_equal(res22.pop('times'), times)

        assert_arrays_equal(res11.pop('labels'), labels)
        assert_arrays_equal(res12.pop('labels'), labels)
        assert_arrays_equal(res21.pop('labels'), labels)
        assert_arrays_equal(res22.pop('labels'), labels)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = EventArray
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, EventArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'EventArray'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, EventArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestEventArray(unittest.TestCase):
    def test_EventArray_creation(self):
        params = {'test2': 'y1', 'test3': True}
        evta = EventArray([1.1, 1.5, 1.7]*pq.ms,
                          labels=np.array(['test event 1',
                                           'test event 2',
                                           'test event 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          test1=1, **params)
        evta.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evta)

        assert_arrays_equal(evta.times, [1.1, 1.5, 1.7]*pq.ms)
        assert_arrays_equal(evta.labels, np.array(['test event 1',
                                                   'test event 2',
                                                   'test event 3'], dtype='S'))
        self.assertEqual(evta.name, 'test')
        self.assertEqual(evta.description, 'tester')
        self.assertEqual(evta.file_origin, 'test.file')
        self.assertEqual(evta.annotations['test0'], [1, 2])
        self.assertEqual(evta.annotations['test1'], 1.1)
        self.assertEqual(evta.annotations['test2'], 'y1')
        self.assertTrue(evta.annotations['test3'])

    def test_EventArray_repr(self):
        params = {'test2': 'y1', 'test3': True}
        evta = EventArray([1.1, 1.5, 1.7]*pq.ms,
                          labels=np.array(['test event 1',
                                           'test event 2',
                                           'test event 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          test1=1, **params)
        evta.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evta)

        targ = ('<EventArray: test event 1@1.1 ms, test event 2@1.5 ms, ' +
                'test event 3@1.7 ms>')

        res = repr(evta)

        self.assertEqual(targ, res)

    def test_EventArray_merge(self):
        params1 = {'test2': 'y1', 'test3': True}
        params2 = {'test2': 'no', 'test4': False}
        paramstarg = {'test2': 'yes;no',
                      'test3': True,
                      'test4': False}
        epca1 = EventArray([1.1, 1.5, 1.7]*pq.ms,
                           labels=np.array(['test event 1 1',
                                            'test event 1 2',
                                            'test event 1 3'], dtype='S'),
                           name='test', description='tester 1',
                           file_origin='test.file',
                           test1=1, **params1)
        epca2 = EventArray([2.1, 2.5, 2.7]*pq.us,
                           labels=np.array(['test event 2 1',
                                            'test event 2 2',
                                            'test event 2 3'], dtype='S'),
                           name='test', description='tester 2',
                           file_origin='test.file',
                           test1=1, **params2)
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
                              test1=1, **paramstarg)
        assert_neo_object_is_compliant(epca1)
        assert_neo_object_is_compliant(epca2)
        assert_neo_object_is_compliant(epcatarg)

        epcares = epca1.merge(epca2)
        assert_neo_object_is_compliant(epcares)
        assert_same_sub_schema(epcatarg, epcares)

    def test__children(self):
        params = {'test2': 'y1', 'test3': True}
        evta = EventArray([1.1, 1.5, 1.7]*pq.ms,
                          labels=np.array(['test event 1',
                                           'test event 2',
                                           'test event 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file',
                          test1=1, **params)
        evta.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evta)

        segment = Segment(name='seg1')
        segment.eventarrays = [evta]
        segment.create_many_to_one_relationship()

        self.assertEqual(evta._single_parent_objects, ('Segment',))
        self.assertEqual(evta._multi_parent_objects, ())

        self.assertEqual(evta._single_parent_containers, ('segment',))
        self.assertEqual(evta._multi_parent_containers, ())

        self.assertEqual(evta._parent_objects, ('Segment',))
        self.assertEqual(evta._parent_containers, ('segment',))

        self.assertEqual(len(evta.parents), 1)
        self.assertEqual(evta.parents[0].name, 'seg1')

        assert_neo_object_is_compliant(evta)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        evta = EventArray([1.1, 1.5, 1.7]*pq.ms,
                          labels=np.array(['test event 1',
                                           'test event 2',
                                           'test event 3'], dtype='S'),
                          name='test', description='tester',
                          file_origin='test.file')
        evta.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evta)

        prepr = pretty(evta)
        targ = ("EventArray\nname: '%s'\ndescription: '%s'\nannotations: %s" %
                (evta.name, evta.description, pretty(evta.annotations)))

        self.assertEqual(prepr, targ)


if __name__ == "__main__":
    unittest.main()
