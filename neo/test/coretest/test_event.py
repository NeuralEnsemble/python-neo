# -*- coding: utf-8 -*-
"""
Tests of the neo.core.event.Event class
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

from neo.core.event import Event
from neo.core import Segment
from neo.test.tools import assert_neo_object_is_compliant, assert_arrays_equal
from neo.test.generate_datasets import (get_fake_value, get_fake_values,
                                        fake_neo, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__get_fake_values(self):
        self.annotations['seed'] = 0
        time = get_fake_value('time', pq.Quantity, seed=0, dim=0)
        label = get_fake_value('label', str, seed=1)
        name = get_fake_value('name', str, seed=2, obj=Event)
        description = get_fake_value('description', str, seed=3, obj='Event')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'label': label,
                  'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Event, annotate=False, seed=0)
        res12 = get_fake_values('Event', annotate=False, seed=0)
        res21 = get_fake_values(Event, annotate=True, seed=0)
        res22 = get_fake_values('Event', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('time'), time)
        assert_arrays_equal(res12.pop('time'), time)
        assert_arrays_equal(res21.pop('time'), time)
        assert_arrays_equal(res22.pop('time'), time)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'Event'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Event))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = Event
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Event))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestEvent(unittest.TestCase):
    def test_Event_creation(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event(1.5*pq.ms,
                    label='test epoch', name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        self.assertEqual(evt.time, 1.5*pq.ms)
        self.assertEqual(evt.label, 'test epoch')
        self.assertEqual(evt.name, 'test')
        self.assertEqual(evt.description, 'tester')
        self.assertEqual(evt.file_origin, 'test.file')
        self.assertEqual(evt.annotations['test0'], [1, 2])
        self.assertEqual(evt.annotations['test1'], 1.1)
        self.assertEqual(evt.annotations['test2'], 'y1')
        self.assertTrue(evt.annotations['test3'])

    def test_epoch_merge_NotImplementedError(self):
        evt1 = Event(1.5*pq.ms,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file', test1=1)
        evt2 = Event(1.5*pq.ms,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file', test1=1)
        self.assertRaises(NotImplementedError, evt1.merge, evt2)

    def test__children(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event(1.5*pq.ms,
                    label='test epoch', name='test', description='tester',
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
        evt = Event(1.5*pq.ms,
                    label='test epoch', name='test', description='tester',
                    file_origin='test.file')
        evt.annotate(targ1=1.1, targ0=[1])
        assert_neo_object_is_compliant(evt)

        prepr = pretty(evt)
        targ = ("Event\nname: '%s'\ndescription: '%s'\nannotations: %s" %
                (evt.name, evt.description, pretty(evt.annotations)))

        self.assertEqual(prepr, targ)


if __name__ == "__main__":
    unittest.main()
