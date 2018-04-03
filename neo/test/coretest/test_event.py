# -*- coding: utf-8 -*-
"""
Tests of the neo.core.event.Event class
"""

import unittest

import numpy as np
import quantities as pq
import pickle
import os
from numpy.testing import assert_array_equal

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.event import Event
from neo.core import Segment
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal,
                            assert_arrays_almost_equal,
                            assert_same_sub_schema)
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
        name = get_fake_value('name', str, seed=2, obj=Event)
        description = get_fake_value('description', str,
                                     seed=3, obj='Event')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Event, annotate=False, seed=0)
        res12 = get_fake_values('Event', annotate=False, seed=0)
        res21 = get_fake_values(Event, annotate=True, seed=0)
        res22 = get_fake_values('Event', annotate=True, seed=0)

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
        obj_type = Event
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Event))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'Event'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Event))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestEvent(unittest.TestCase):
    def test_Event_creation(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1',
                                     'test event 2',
                                     'test event 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        assert_arrays_equal(evt.times, [1.1, 1.5, 1.7] * pq.ms)
        assert_arrays_equal(evt.labels, np.array(['test event 1',
                                                  'test event 2',
                                                  'test event 3'], dtype='S'))
        self.assertEqual(evt.name, 'test')
        self.assertEqual(evt.description, 'tester')
        self.assertEqual(evt.file_origin, 'test.file')
        self.assertEqual(evt.annotations['test0'], [1, 2])
        self.assertEqual(evt.annotations['test1'], 1.1)
        self.assertEqual(evt.annotations['test2'], 'y1')
        self.assertTrue(evt.annotations['test3'])

    def tests_time_slice(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        targ = Event([2.2, 2.9, 3.0] * pq.ms)
        result = evt.time_slice(t_start=2.0, t_stop=3.0)

        assert_arrays_equal(targ, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])

    def test_time_slice_out_of_boundries(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        targ = evt
        result = evt.time_slice(t_start=0.0001, t_stop=30.0)

        assert_arrays_equal(targ, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])

    def test_time_slice_empty(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        result = evt.time_slice(t_start=0.0001, t_stop=30.0)
        assert_neo_object_is_compliant(evt)

        assert_arrays_equal(evt, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])

    def test_time_slice_none_stop(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        targ = Event([2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms)
        assert_neo_object_is_compliant(evt)

        t_start = 2.0
        t_stop = None
        result = evt.time_slice(t_start, t_stop)

        assert_arrays_equal(targ, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])

    def test_time_slice_none_start(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        targ = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0] * pq.ms)
        t_start = None
        t_stop = 3.0
        result = evt.time_slice(t_start, t_stop)

        assert_arrays_equal(targ, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])

    def test_time_slice_none_both(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        assert_neo_object_is_compliant(evt)

        evt.annotate(test1=1.1, test0=[1, 2])
        t_start = None
        t_stop = None
        result = evt.time_slice(t_start, t_stop)

        assert_arrays_equal(evt, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])

    def test_time_slice_differnt_units(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.1, 3.3] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        assert_neo_object_is_compliant(evt)
        evt.annotate(test1=1.1, test0=[1, 2])

        targ = Event([2.2, 2.9] * pq.ms,
                     name='test', description='tester',
                     file_origin='test.file',
                     test1=1, **params)
        assert_neo_object_is_compliant(targ)
        targ.annotate(test1=1.1, test0=[1, 2])

        t_start = 0.002 * pq.s
        t_stop = 0.003 * pq.s

        result = evt.time_slice(t_start, t_stop)

        assert_arrays_equal(targ, result)
        self.assertEqual(targ.name, result.name)
        self.assertEqual(targ.description, result.description)
        self.assertEqual(targ.file_origin, result.file_origin)
        self.assertEqual(targ.annotations['test0'], result.annotations['test0'])
        self.assertEqual(targ.annotations['test1'], result.annotations['test1'])
        self.assertEqual(targ.annotations['test2'], result.annotations['test2'])

    def test_Event_repr(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1',
                                     'test event 2',
                                     'test event 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        targ = ('<Event: test event 1@1.1 ms, test event 2@1.5 ms, ' +
                'test event 3@1.7 ms>')

        res = repr(evt)

        self.assertEqual(targ, res)

    def test_Event_merge(self):
        params1 = {'test2': 'y1', 'test3': True}
        params2 = {'test2': 'no', 'test4': False}
        paramstarg = {'test2': 'yes;no',
                      'test3': True,
                      'test4': False}
        evt1 = Event([1.1, 1.5, 1.7] * pq.ms,
                     labels=np.array(['test event 1 1',
                                      'test event 1 2',
                                      'test event 1 3'], dtype='S'),
                     name='test', description='tester 1',
                     file_origin='test.file',
                     test1=1, **params1)
        evt2 = Event([2.1, 2.5, 2.7] * pq.us,
                     labels=np.array(['test event 2 1',
                                      'test event 2 2',
                                      'test event 2 3'], dtype='S'),
                     name='test', description='tester 2',
                     file_origin='test.file',
                     test1=1, **params2)
        evttarg = Event([1.1, 1.5, 1.7, .0021, .0025, .0027] * pq.ms,
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
        assert_neo_object_is_compliant(evt1)
        assert_neo_object_is_compliant(evt2)
        assert_neo_object_is_compliant(evttarg)

        evtres = evt1.merge(evt2)
        assert_neo_object_is_compliant(evtres)
        assert_same_sub_schema(evttarg, evtres)

    def test__children(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1',
                                     'test event 2',
                                     'test event 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        segment = Segment(name='seg1')
        segment.events = [evt]
        segment.create_many_to_one_relationship()

        self.assertEqual(evt._single_parent_objects, ('Segment',))
        self.assertEqual(evt._multi_parent_objects, ())

        self.assertEqual(evt._single_parent_containers, ('segment',))
        self.assertEqual(evt._multi_parent_containers, ())

        self.assertEqual(evt._parent_objects, ('Segment',))
        self.assertEqual(evt._parent_containers, ('segment',))

        self.assertEqual(len(evt.parents), 1)
        self.assertEqual(evt.parents[0].name, 'seg1')

        assert_neo_object_is_compliant(evt)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1',
                                     'test event 2',
                                     'test event 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file')
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        prepr = pretty(evt)
        targ = ("Event\nname: '%s'\ndescription: '%s'\nannotations: %s" %
                (evt.name, evt.description, pretty(evt.annotations)))

        self.assertEqual(prepr, targ)

    def test__time_slice(self):
        data = [2, 3, 4, 5] * pq.ms
        evt = Event(data, foo='bar')

        evt1 = evt.time_slice(2.2 * pq.ms, 4.2 * pq.ms)
        assert_arrays_equal(evt1.times, [3, 4] * pq.ms)
        self.assertEqual(evt.annotations, evt1.annotations)

        evt2 = evt.time_slice(None, 4.2 * pq.ms)
        assert_arrays_equal(evt2.times, [2, 3, 4] * pq.ms)

        evt3 = evt.time_slice(2.2 * pq.ms, None)
        assert_arrays_equal(evt3.times, [3, 4, 5] * pq.ms)

    def test_as_array(self):
        data = [2, 3, 4, 5]
        evt = Event(data * pq.ms)
        evt_as_arr = evt.as_array()
        self.assertIsInstance(evt_as_arr, np.ndarray)
        assert_array_equal(data, evt_as_arr)

    def test_as_quantity(self):
        data = [2, 3, 4, 5]
        evt = Event(data * pq.ms)
        evt_as_q = evt.as_quantity()
        self.assertIsInstance(evt_as_q, pq.Quantity)
        assert_array_equal(data * pq.ms, evt_as_q)


class TestDuplicateWithNewData(unittest.TestCase):
    def setUp(self):
        self.data = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.dataquant = self.data * pq.ms
        self.event = Event(self.dataquant)

    def test_duplicate_with_new_data(self):
        signal1 = self.event
        new_data = np.sort(np.random.uniform(0, 100, (self.event.size))) * pq.ms
        signal1b = signal1.duplicate_with_new_data(new_data)
        assert_arrays_almost_equal(np.asarray(signal1b),
                                   np.asarray(new_data), 1e-12)


class TestEventFunctions(unittest.TestCase):
    def test__pickle(self):

        event1 = Event(np.arange(0, 30, 10) * pq.s, labels=np.array(['t0', 't1', 't2'], dtype='S'),
                       units='s', annotation1="foo", annotation2="bar")
        fobj = open('./pickle', 'wb')
        pickle.dump(event1, fobj)
        fobj.close()

        fobj = open('./pickle', 'rb')
        try:
            event2 = pickle.load(fobj)
        except ValueError:
            event2 = None

        fobj.close()
        assert_array_equal(event1.times, event2.times)
        os.remove('./pickle')
        self.assertEqual(event2.annotations, event1.annotations)


if __name__ == "__main__":
    unittest.main()
