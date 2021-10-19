"""
Tests of the neo.core.event.Event class
"""

import unittest
import warnings
from copy import deepcopy

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

from neo.core.dataobject import ArrayDict
from neo.core.event import Event
from neo.core.epoch import Epoch
from neo.core import Segment
from neo.test.tools import (assert_neo_object_is_compliant, assert_arrays_equal,
                            assert_arrays_almost_equal, assert_same_sub_schema,
                            assert_same_attributes, assert_same_annotations,
                            assert_same_array_annotations)

warnings.simplefilter("always")


class TestEvent(unittest.TestCase):

    def setUp(self):
        self.params = {'test2': 'y1', 'test3': True}
        self.arr_ann = {'index': np.arange(10), 'test': np.arange(100, 110)}
        self.seg = Segment()
        self.evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms, name='test',
                    description='tester', file_origin='test.file', test1=1,
                    array_annotations=self.arr_ann, **self.params)
        self.evt.annotate(test1=1.1, test0=[1, 2])
        self.evt.segment = self.seg

    def test_setup_compliant(self):
        assert_neo_object_is_compliant(self.evt)

    def test_Event_creation(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'names': ['a', 'b', 'c'], 'index': np.arange(10, 13)}
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1', 'test event 2', 'test event 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        assert_arrays_equal(evt.times, [1.1, 1.5, 1.7] * pq.ms)
        assert_arrays_equal(evt.labels,
                            np.array(['test event 1', 'test event 2', 'test event 3'], dtype='U'))
        self.assertEqual(evt.name, 'test')
        self.assertEqual(evt.description, 'tester')
        self.assertEqual(evt.file_origin, 'test.file')
        self.assertEqual(evt.annotations['test0'], [1, 2])
        self.assertEqual(evt.annotations['test1'], 1.1)
        self.assertEqual(evt.annotations['test2'], 'y1')
        self.assertTrue(evt.annotations['test3'])
        assert_arrays_equal(evt.array_annotations['names'], np.array(['a', 'b', 'c']))
        assert_arrays_equal(evt.array_annotations['index'], np.arange(10, 13))
        self.assertIsInstance(evt.array_annotations, ArrayDict)

    def test_Event_invalid_times_dimension(self):
        data2d = np.array([1, 2, 3, 4]).reshape((4, -1))
        self.assertRaises(ValueError, Event, times=data2d * pq.s)

    def test_Event_creation_invalid_labels(self):
        self.assertRaises(ValueError, Event, [1.1, 1.5, 1.7] * pq.ms,
                          labels=["A", "B"])

    def test_Event_creation_from_lists(self):
        evt = Event([1.1, 1.5, 1.7],
                    ['test event 1', 'test event 2', 'test event 3'],
                    units=pq.ms)
        assert_arrays_equal(evt.times, [1.1, 1.5, 1.7] * pq.ms)
        assert_arrays_equal(evt.labels,
                            np.array(['test event 1', 'test event 2', 'test event 3']))

    def tests_time_slice(self):

        evt = self.evt

        targ = Event([2.2, 2.9, 3.0] * pq.ms)
        result = evt.time_slice(t_start=2.0, t_stop=3.0)

        assert_arrays_equal(targ, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.arange(5, 8))
        assert_arrays_equal(result.array_annotations['test'], np.arange(105, 108))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def tests_time_slice_deepcopy_annotations(self):
        params = {'test0': 'y1', 'test1': ['deeptest'], 'test2': True}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms,
                    name='test', description='tester',
                    file_origin='test.file', **params)
        result = evt.time_slice(t_start=2.0, t_stop=3.0)
        evt.annotate(test0='y2', test2=False)
        evt.annotations['test1'][0] = 'shallowtest'

        self.assertNotEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(evt.annotations['test2'], result.annotations['test2'])

    def test__time_slice_deepcopy_annotations(self):
        params1 = {'test0': 'y1', 'test1': ['deeptest'], 'test2': True}
        self.evt.annotate(**params1)
        # time_slice spike train, keep sliced spike times
        t_start = 2.1 * pq.ms
        t_stop = 3.05 * pq.ms
        result = self.evt.time_slice(t_start, t_stop)

        # Change annotations of original
        params2 = {'test0': 'y2', 'test2': False}
        self.evt.annotate(**params2)
        self.evt.annotations['test1'][0] = 'shallowtest'

        self.assertNotEqual(self.evt.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.evt.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.evt.annotations['test2'], result.annotations['test2'])

        # Change annotations of result
        params3 = {'test0': 'y3'}
        result.annotate(**params3)
        result.annotations['test1'][0] = 'shallowtest2'

        self.assertNotEqual(self.evt.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.evt.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.evt.annotations['test2'], result.annotations['test2'])

    def test__time_slice_deepcopy_array_annotations(self):
        length = self.evt.shape[-1]
        params1 = {'test0': ['y{}'.format(i) for i in range(length)],
                   'test1': ['deeptest' for i in range(length)],
                   'test2': [(-1)**i > 0 for i in range(length)]}
        self.evt.array_annotate(**params1)
        # time_slice spike train, keep sliced spike times
        t_start = 2.1 * pq.ms
        t_stop = 3.05 * pq.ms
        result = self.evt.time_slice(t_start, t_stop)

        # Change annotations of original
        params2 = {'test0': ['x{}'.format(i) for i in range(length)],
                   'test2': [(-1) ** (i + 1) > 0 for i in range(length)]}
        self.evt.array_annotate(**params2)
        self.evt.array_annotations['test1'][6] = 'shallowtest'

        self.assertFalse(all(self.evt.array_annotations['test0'][5:8]
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.evt.array_annotations['test1'][5:8]
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.evt.array_annotations['test2'][5:8]
                             == result.array_annotations['test2']))

        # Change annotations of result
        params3 = {'test0': ['z{}'.format(i) for i in range(5, 8)]}
        result.array_annotate(**params3)
        result.array_annotations['test1'][1] = 'shallow2'

        self.assertFalse(all(self.evt.array_annotations['test0'][5:8]
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.evt.array_annotations['test1'][5:8]
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.evt.array_annotations['test2'][5:8]
                             == result.array_annotations['test2']))

    def test__time_slice_deepcopy_data(self):
        result = self.evt.time_slice(None, None)

        # Change values of original array
        self.evt[2] = 7.3*self.evt.units

        self.assertFalse(all(self.evt == result))

        # Change values of sliced array
        result[3] = 9.5*result.units

        self.assertFalse(all(self.evt == result))

    def test_time_slice_out_of_boundries(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(10), 'test': np.arange(100, 110)}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms, name='test',
                    description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
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
        assert_arrays_equal(result.array_annotations['index'], np.arange(10))
        assert_arrays_equal(result.array_annotations['test'], np.arange(100, 110))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_empty(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.array([]), 'test': np.array([])}
        evt = Event([] * pq.ms, name='test', description='tester', file_origin='test.file',
                    test1=1, array_annotations=arr_ann, **params)
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
        assert_arrays_equal(result.array_annotations['index'], np.asarray([]))
        assert_arrays_equal(result.array_annotations['test'], np.asarray([]))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_stop(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(10), 'test': np.arange(100, 110)}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms, name='test',
                    description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
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
        assert_arrays_equal(result.array_annotations['index'], np.arange(5, 10))
        assert_arrays_equal(result.array_annotations['test'], np.arange(105, 110))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_start(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(10), 'test': np.arange(100, 110)}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms, name='test',
                    description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
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
        assert_arrays_equal(result.array_annotations['index'], np.arange(8))
        assert_arrays_equal(result.array_annotations['test'], np.arange(100, 108))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_both(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(10), 'test': np.arange(100, 110)}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms, name='test',
                    description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
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
        assert_arrays_equal(result.array_annotations['index'], np.arange(10))
        assert_arrays_equal(result.array_annotations['test'], np.arange(100, 110))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_differnt_units(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(9), 'test': np.arange(100, 109)}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.1, 3.3] * pq.ms, name='test',
                    description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        assert_neo_object_is_compliant(evt)
        evt.annotate(test1=1.1, test0=[1, 2])

        targ = Event([2.2, 2.9] * pq.ms, name='test', description='tester',
                     file_origin='test.file', test1=1, **params)
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
        assert_arrays_equal(result.array_annotations['index'], np.arange(5, 7))
        assert_arrays_equal(result.array_annotations['test'], np.arange(105, 107))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__time_slice_should_set_parents_to_None(self):
        # When timeslicing, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.evt.time_slice(1 * pq.ms, 3 * pq.ms)
        self.assertEqual(result.segment, None)

    def test__deepcopy_should_set_parents_objects_to_None(self):
        # Deepcopy should destroy references to parents
        result = deepcopy(self.evt)
        self.assertEqual(result.segment, None)

    def test_slice(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(10), 'test': np.arange(100, 110)}
        evt = Event([0.1, 0.5, 1.1, 1.5, 1.7, 2.2, 2.9, 3.0, 3.1, 3.3] * pq.ms, name='test',
                    description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        targ = Event([2.2, 2.9, 3.0] * pq.ms)
        result = evt[5:8]

        assert_arrays_equal(targ, result)
        self.assertEqual(evt.name, result.name)
        self.assertEqual(evt.description, result.description)
        self.assertEqual(evt.file_origin, result.file_origin)
        self.assertEqual(evt.annotations['test0'], result.annotations['test0'])
        self.assertEqual(evt.annotations['test1'], result.annotations['test1'])
        self.assertEqual(evt.annotations['test2'], result.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.arange(5, 8))
        assert_arrays_equal(result.array_annotations['test'], np.arange(105, 108))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_Event_repr(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1', 'test event 2', 'test event 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        targ = ('<Event: test event 1@1.1 ms, test event 2@1.5 ms, ' + 'test event 3@1.7 ms>')

        res = repr(evt)

        self.assertEqual(targ, res)

    def test_Event_merge(self):
        warnings.simplefilter("always")
        params1 = {'test2': 'y1', 'test3': True}
        params2 = {'test2': 'no', 'test4': False}
        paramstarg = {'test2': 'yes;no', 'test3': True, 'test4': False}
        arr_ann1 = {'index': np.arange(10, 13)}
        arr_ann2 = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        evt1 = Event([1.1, 1.5, 1.7] * pq.ms,
                     labels=np.array(['test event 1 1', 'test event 1 2', 'test event 1 3'],
                                     dtype='U'), name='test', description='tester 1',
                     file_origin='test.file', array_annotations=arr_ann1, test1=1, **params1)
        evt2 = Event([2.1, 2.5, 2.7] * pq.us,
                     labels=np.array(['test event 2 1', 'test event 2 2', 'test event 2 3'],
                                     dtype='U'), name='test', description='tester 2',
                     file_origin='test.file', array_annotations=arr_ann2, test1=1, **params2)
        evttarg = Event([1.1, 1.5, 1.7, .0021, .0025, .0027] * pq.ms,
                        labels=np.array(['test event 1 1', 'test event 1 2', 'test event 1 3',
                                         'test event 2 1', 'test event 2 2', 'test event 2 3'],
                                        dtype='U'),
                        name='test',
                        description='merge(tester 1, tester 2)', file_origin='test.file',
                        array_annotations={'index': [10, 11, 12, 0, 1, 2]}, test1=1, **paramstarg)
        assert_neo_object_is_compliant(evt1)
        assert_neo_object_is_compliant(evt2)
        assert_neo_object_is_compliant(evttarg)

        with warnings.catch_warnings(record=True) as w:
            evtres = evt1.merge(evt2)

            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertSequenceEqual(str(w[0].message), "The following array annotations were "
                                                        "omitted, because they were only present"
                                                        " in one of the merged objects: "
                                                        "[] from the one that was merged "
                                                        "into and ['test'] from the one that "
                                                        "was merged into the other")

        assert_neo_object_is_compliant(evtres)
        assert_same_sub_schema(evttarg, evtres)
        # Remove this, when array_annotations are added to assert_same_sub_schema
        assert_arrays_equal(evtres.array_annotations['index'], np.array([10, 11, 12, 0, 1, 2]))
        self.assertTrue('test' not in evtres.array_annotations)
        self.assertIsInstance(evtres.array_annotations, ArrayDict)

    def test_set_labels(self):
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=['A', 'B', 'C'])
        assert_array_equal(evt.labels, np.array(['A', 'B', 'C']))
        evt.labels = ['D', 'E', 'F']
        assert_array_equal(evt.labels, np.array(['D', 'E', 'F']))
        self.assertRaises(ValueError, setattr, evt, "labels", ['X', 'Y'])

    def test__children(self):
        params = {'test2': 'y1', 'test3': True}
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1', 'test event 2', 'test event 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1, **params)
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        segment = Segment(name='seg1')
        segment.events = [evt]
        segment.create_many_to_one_relationship()

        self.assertEqual(evt._parent_objects, ('Segment',))

        self.assertEqual(evt._parent_containers, ('segment',))

        self.assertEqual(evt._parent_objects, ('Segment',))
        self.assertEqual(evt._parent_containers, ('segment',))

        self.assertEqual(len(evt.parents), 1)
        self.assertEqual(evt.parents[0].name, 'seg1')

        assert_neo_object_is_compliant(evt)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        evt = Event([1.1, 1.5, 1.7] * pq.ms,
                    labels=np.array(['test event 1', 'test event 2', 'test event 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file')
        evt.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(evt)

        prepr = pretty(evt)
        targ = ("Event\nname: '%s'\ndescription: '%s'\nannotations: %s"
                "" % (evt.name, evt.description, pretty(evt.annotations)))

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

    def test__time_shift_same_attributes(self):
        result = self.evt.time_shift(1 * pq.ms)
        assert_same_attributes(result, self.evt, exclude=['times'])

    def test__time_shift_same_annotations(self):
        result = self.evt.time_shift(1 * pq.ms)
        assert_same_annotations(result, self.evt)

    def test__time_shift_same_array_annotations(self):
        result = self.evt.time_shift(1 * pq.ms)
        assert_same_array_annotations(result, self.evt)

    def test__time_shift_should_set_parents_to_None(self):
        # When time-shifting, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.evt.time_shift(1 * pq.ms)
        self.assertEqual(result.segment, None)

    def test__time_shift_by_zero(self):
        shifted = self.evt.time_shift(0 * pq.ms)
        assert_arrays_equal(shifted.times, self.evt.times)

    def test__time_shift_same_units(self):
        shifted = self.evt.time_shift(10 * pq.ms)
        assert_arrays_equal(shifted.times, self.evt.times + 10 * pq.ms)

    def test__time_shift_different_units(self):
        shifted = self.evt.time_shift(1 * pq.s)
        assert_arrays_equal(shifted.times, self.evt.times + 1000 * pq.ms)

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

    def test_to_epoch(self):
        seg = Segment(name="test")
        event = Event(times=np.array([5.0, 12.0, 23.0, 45.0]), units="ms",
                      labels=np.array(["A", "B", "C", "D"]))
        event.segment = seg

        # Mode 1
        epoch = event.to_epoch()
        self.assertIsInstance(epoch, Epoch)
        assert_array_equal(epoch.times.magnitude, np.array([5.0, 12.0, 23.0]))
        assert_array_equal(epoch.durations.magnitude, np.array([7.0, 11.0, 22.0]))
        assert_array_equal(epoch.labels, np.array(['A-B', 'B-C', 'C-D']))

        # Mode 2
        epoch = event.to_epoch(pairwise=True)
        assert_array_equal(epoch.times.magnitude, np.array([5.0, 23.0]))
        assert_array_equal(epoch.durations.magnitude, np.array([7.0, 22.0]))
        assert_array_equal(epoch.labels, np.array(['A-B', 'C-D']))

        # Mode 3 (scalar)
        epoch = event.to_epoch(durations=2.0 * pq.ms)
        assert_array_equal(epoch.times.magnitude, np.array([5.0, 12.0, 23.0, 45.0]))
        assert_array_equal(epoch.durations.magnitude, np.array([2.0, 2.0, 2.0, 2.0]))
        self.assertEqual(epoch.durations.size, 4)
        assert_array_equal(epoch.labels, np.array(['A', 'B', 'C', 'D']))

        # Mode 3 (array)
        epoch = event.to_epoch(durations=np.array([2.0, 3.0, 4.0, 5.0]) * pq.ms)
        assert_array_equal(epoch.times.magnitude, np.array([5.0, 12.0, 23.0, 45.0]))
        assert_array_equal(epoch.durations.magnitude, np.array([2.0, 3.0, 4.0, 5.0]))
        assert_array_equal(epoch.labels, np.array(['A', 'B', 'C', 'D']))

        # Error conditions
        self.assertRaises(ValueError, event.to_epoch, pairwise=True, durations=2.0 * pq.ms)

        odd_event = Event(times=np.array([5.0, 12.0, 23.0]), units="ms",
                          labels=np.array(["A", "B", "C"]))
        self.assertRaises(ValueError, odd_event.to_epoch, pairwise=True)

        # todo: fix Epoch, as the following does not raise a ValueError  # self.assertRaises(
        # ValueError, event.to_epoch, durations=2.0)  # missing units

    def test_rescale(self):
        times = [2, 3, 4, 5]
        labels = ["A", "B", "C", "D"]
        arr_ann = {'index': np.arange(4), 'test': ['a', 'b', 'c', 'd']}
        evt = Event(times * pq.ms, labels=labels,
                    array_annotations=arr_ann)
        result = evt.rescale(pq.us)

        self.assertIsInstance(result, Event)
        assert_neo_object_is_compliant(result)
        assert_arrays_equal(result.array_annotations['index'], np.arange(4))
        assert_arrays_equal(result.array_annotations['test'],
                            np.array(['a', 'b', 'c', 'd']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.units, 1 * pq.us)
        assert_array_equal(evt.labels, result.labels)
        assert_arrays_almost_equal(result.times, [2000, 3000, 4000, 5000] * pq.us, 1e-9)
        assert_arrays_almost_equal(result.times.magnitude,
                                   np.array([2000, 3000, 4000, 5000]),
                                   1e-9)

class TestDuplicateWithNewData(unittest.TestCase):
    def setUp(self):
        self.data = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.dataquant = self.data * pq.ms
        self.arr_ann = {'index': np.arange(6), 'test': ['a', 'b', 'c', 'd', 'e', 'f']}
        self.event = Event(times=self.dataquant, labels=np.array(['a', 'b', 'c', 'd', 'e', 'f']),
                           array_annotations=self.arr_ann)

    def test_duplicate_with_new_data(self):
        signal1 = self.event
        new_times = np.sort(np.random.uniform(0, 100, (self.event.size))) * pq.ms
        new_labels = np.array(list("zyxwvutsrqponmlkjihgfedcba"[:self.event.size]))
        signal1b = signal1.duplicate_with_new_data(new_times, new_labels)
        assert_arrays_almost_equal(np.asarray(signal1b), np.asarray(new_times), 1e-12)
        assert_arrays_equal(signal1b.labels, new_labels)
        # After duplicating, array annotations should always be empty,
        # because different length of data would cause inconsistencies
        self.assertTrue('index' not in signal1b.array_annotations)
        self.assertTrue('test' not in signal1b.array_annotations)
        self.assertIsInstance(signal1b.array_annotations, ArrayDict)


class TestEventFunctions(unittest.TestCase):
    def test__pickle(self):
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        event1 = Event(np.arange(0, 30, 10) * pq.s, labels=np.array(['t0', 't1', 't2'], dtype='U'),
                       units='s', annotation1="foo", annotation2="bar", array_annotations=arr_ann)
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
        assert_arrays_equal(event2.array_annotations['index'], np.array(arr_ann['index']))
        assert_arrays_equal(event2.array_annotations['test'], np.array(arr_ann['test']))
        self.assertIsInstance(event2.array_annotations, ArrayDict)
        # Make sure the dict can perform correct checks after unpickling
        event2.array_annotations['anno3'] = list(range(3, 6))
        with self.assertRaises(ValueError):
            event2.array_annotations['anno4'] = [2, 1]
        os.remove('./pickle')
        self.assertEqual(event2.annotations, event1.annotations)


if __name__ == "__main__":
    unittest.main()
