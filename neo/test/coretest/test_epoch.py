"""
Tests of the neo.core.epoch.Epoch class
"""

import unittest
import warnings
from copy import deepcopy

import numpy as np
import quantities as pq
import pickle
import os
from numpy.testing import assert_array_equal

from neo.core.dataobject import ArrayDict

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.epoch import Epoch
from neo.core import Segment
from neo.test.tools import (assert_neo_object_is_compliant, assert_arrays_equal,
                            assert_arrays_almost_equal, assert_same_sub_schema,
                            assert_same_attributes, assert_same_annotations,
                            assert_same_array_annotations)


class TestEpoch(unittest.TestCase):

    def setUp(self):
        self.params = {'test0': 'y1', 'test1': ['deeptest'], 'test2': True}
        self.seg = Segment()
        self.epc = Epoch(times=[10, 20, 30, 40, 50] * pq.s, durations=[10, 5, 7, 14, 9] * pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2', 'btn0', 'btn3'], dtype='S'),
                         **self.params)
        self.epc.segment = self.seg

    def test_Epoch_creation(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'names': ['a', 'b', 'c'], 'index': np.arange(10, 13)}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        assert_arrays_equal(epc.times, [1.1, 1.5, 1.7] * pq.ms)
        assert_arrays_equal(epc.durations, [20, 40, 60] * pq.ns)
        assert_arrays_equal(epc.labels,
                            np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'))
        self.assertEqual(epc.name, 'test')
        self.assertEqual(epc.description, 'tester')
        self.assertEqual(epc.file_origin, 'test.file')
        self.assertEqual(epc.annotations['test0'], [1, 2])
        self.assertEqual(epc.annotations['test1'], 1.1)
        self.assertEqual(epc.annotations['test2'], 'y1')
        self.assertTrue(epc.annotations['test3'])
        assert_arrays_equal(epc.array_annotations['names'], np.array(['a', 'b', 'c']))
        assert_arrays_equal(epc.array_annotations['index'], np.arange(10, 13))
        self.assertIsInstance(epc.array_annotations, ArrayDict)

    def test_Epoch_invalid_times_dimension(self):
        data2d = np.array([1, 2, 3, 4]).reshape((4, -1))
        durations = np.array([1, 1, 1, 1])
        self.assertRaises(ValueError, Epoch, times=data2d * pq.s, durations=durations)

    def test_Epoch_creation_invalid_durations_labels(self):
        self.assertRaises(ValueError, Epoch, [1.1, 1.5, 1.7] * pq.ms,
                          durations=[20, 40, 60, 80] * pq.ns)
        self.assertRaises(ValueError, Epoch, [1.1, 1.5, 1.7] * pq.ms,
                          durations=[20, 40, 60] * pq.ns,
                          labels=["A", "B"])

    def test_Epoch_creation_scalar_duration(self):
        # test with scalar for durations
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=20 * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'))
        assert_neo_object_is_compliant(epc)

        assert_arrays_equal(epc.times, [1.1, 1.5, 1.7] * pq.ms)
        assert_arrays_equal(epc.durations, [20, 20, 20] * pq.ns)
        self.assertEqual(epc.durations.size, 3)
        assert_arrays_equal(epc.labels,
                            np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'))

    def test_Epoch_creation_from_lists(self):
        epc = Epoch([1.1, 1.5, 1.7],
                    [20.0, 20.0, 20.0],
                    ['test event 1', 'test event 2', 'test event 3'],
                    units=pq.ms)
        assert_arrays_equal(epc.times, [1.1, 1.5, 1.7] * pq.ms)
        assert_arrays_equal(epc.durations, [20.0, 20.0, 20.0] * pq.ms)
        assert_arrays_equal(epc.labels,
                            np.array(['test event 1', 'test event 2', 'test event 3']))

    def test_Epoch_repr(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = ('<Epoch: test epoch 1@1.1 ms for 20.0 ns, '
                + 'test epoch 2@1.5 ms for 40.0 ns, '
                + 'test epoch 3@1.7 ms for 60.0 ns>')

        res = repr(epc)

        self.assertEqual(targ, res)

    def test_Epoch_merge(self):
        params1 = {'test2': 'y1', 'test3': True}
        params2 = {'test2': 'no', 'test4': False}
        paramstarg = {'test2': 'yes;no', 'test3': True, 'test4': False}
        arr_ann1 = {'index': np.arange(10, 13)}
        arr_ann2 = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc1 = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.us,
                     labels=np.array(['test epoch 1 1', 'test epoch 1 2', 'test epoch 1 3'],
                                     dtype='U'), name='test', description='tester 1',
                     file_origin='test.file', test1=1, array_annotations=arr_ann1, **params1)
        epc2 = Epoch([2.1, 2.5, 2.7] * pq.us, durations=[3, 5, 7] * pq.ms,
                     labels=np.array(['test epoch 2 1', 'test epoch 2 2', 'test epoch 2 3'],
                                     dtype='U'), name='test', description='tester 2',
                     file_origin='test.file', test1=1, array_annotations=arr_ann2, **params2)
        epctarg = Epoch([1.1, 1.5, 1.7, .0021, .0025, .0027] * pq.ms,
                        durations=[20, 40, 60, 3000, 5000, 7000] * pq.us,
                        labels=np.array(['test epoch 1 1', 'test epoch 1 2', 'test epoch 1 3',
                                         'test epoch 2 1', 'test epoch 2 2', 'test epoch 2 3'],
                                        dtype='U'),
                        name='test',
                        description='merge(tester 1, tester 2)', file_origin='test.file',
                        array_annotations={'index': [10, 11, 12, 0, 1, 2]}, test1=1, **paramstarg)
        assert_neo_object_is_compliant(epc1)
        assert_neo_object_is_compliant(epc2)
        assert_neo_object_is_compliant(epctarg)

        with warnings.catch_warnings(record=True) as w:
            epcres = epc1.merge(epc2)

            self.assertTrue(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertSequenceEqual(str(w[0].message), "The following array annotations were "
                                                        "omitted, because they were only present"
                                                        " in one of the merged objects: "
                                                        "[] from the one that was merged "
                                                        "into and ['test'] from the one that "
                                                        "was merged into the other")

        assert_neo_object_is_compliant(epcres)
        assert_same_sub_schema(epctarg, epcres)
        # Remove this, when array_annotations are added to assert_same_sub_schema
        assert_arrays_equal(epcres.array_annotations['index'], np.array([10, 11, 12, 0, 1, 2]))
        self.assertTrue('test' not in epcres.array_annotations)
        self.assertIsInstance(epcres.array_annotations, ArrayDict)

    def test_set_labels_duration(self):
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms,
                    durations=20 * pq.ns,
                    labels=['A', 'B', 'C'])
        assert_array_equal(epc.durations.magnitude, np.array([20, 20, 20]))
        epc.durations = [20.0, 21.0, 22.0] * pq.ns
        assert_array_equal(epc.durations.magnitude, np.array([20, 21, 22]))
        self.assertRaises(ValueError, setattr, epc, "durations", [25.0, 26.0] * pq.ns)

        assert_array_equal(epc.labels, np.array(['A', 'B', 'C']))
        epc.labels = ['D', 'E', 'F']
        assert_array_equal(epc.labels, np.array(['D', 'E', 'F']))
        self.assertRaises(ValueError, setattr, epc, "labels", ['X', 'Y'])

    def test__children(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        segment = Segment(name='seg1')
        segment.epochs = [epc]
        segment.create_many_to_one_relationship()

        self.assertEqual(epc._parent_objects, ('Segment',))

        self.assertEqual(epc._parent_containers, ('segment',))

        self.assertEqual(epc._parent_objects, ('Segment',))
        self.assertEqual(epc._parent_containers, ('segment',))

        self.assertEqual(len(epc.parents), 1)
        self.assertEqual(epc.parents[0].name, 'seg1')

        assert_neo_object_is_compliant(epc)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file')
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        prepr = pretty(epc)
        targ = ("Epoch\nname: '%s'\ndescription: '%s'\nannotations: %s"
                "" % (epc.name, epc.description, pretty(epc.annotations)))

        self.assertEqual(prepr, targ)

    def test__time_slice(self):
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc = Epoch(times=[10, 20, 30] * pq.s, durations=[10, 5, 7] * pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='U'), foo='bar',
                    array_annotations=arr_ann)

        epc2 = epc.time_slice(10 * pq.s, 20 * pq.s)
        assert_arrays_equal(epc2.times, [10, 20] * pq.s)
        assert_arrays_equal(epc2.durations, [10, 5] * pq.ms)
        assert_arrays_equal(epc2.labels, np.array(['btn0', 'btn1'], dtype='U'))
        self.assertEqual(epc.annotations, epc2.annotations)
        assert_arrays_equal(epc2.array_annotations['index'], np.arange(2))
        assert_arrays_equal(epc2.array_annotations['test'], np.array(['a', 'b']))
        self.assertIsInstance(epc2.array_annotations, ArrayDict)

    def test_time_slice2(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.5] * pq.ms, durations=[40] * pq.ns,
                     labels=np.array(['test epoch 2'], dtype='U'), name='test',
                     description='tester', file_origin='test.file', test1=1,
                     array_annotations={'index': [1], 'test': ['b']}, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)

        t_start = 1.2
        t_stop = 1.6
        result = epc.time_slice(t_start, t_stop)

        assert_arrays_equal(result.times, targ.times)
        assert_arrays_equal(result.durations, targ.durations)
        assert_arrays_equal(result.labels, targ.labels)
        self.assertEqual(result.name, targ.name)
        self.assertEqual(result.description, targ.description)
        self.assertEqual(result.file_origin, targ.file_origin)
        self.assertEqual(result.annotations['test0'], targ.annotations['test0'])
        self.assertEqual(result.annotations['test1'], targ.annotations['test1'])
        self.assertEqual(result.annotations['test2'], targ.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.array([1]))
        assert_arrays_equal(result.array_annotations['test'], np.array(['b']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__time_slice_deepcopy_annotations(self):
        params1 = {'test0': 'y1', 'test1': ['deeptest'], 'test2': True}
        self.epc.annotate(**params1)
        # time_slice spike train, keep sliced spike times
        t_start = 15 * pq.s
        t_stop = 45 * pq.s
        result = self.epc.time_slice(t_start, t_stop)

        # Change annotations of original
        params2 = {'test0': 'y2', 'test2': False}
        self.epc.annotate(**params2)
        self.epc.annotations['test1'][0] = 'shallowtest'

        self.assertNotEqual(self.epc.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.epc.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.epc.annotations['test2'], result.annotations['test2'])

        # Change annotations of result
        params3 = {'test0': 'y3'}
        result.annotate(**params3)
        result.annotations['test1'][0] = 'shallowtest2'

        self.assertNotEqual(self.epc.annotations['test0'], result.annotations['test0'])
        self.assertNotEqual(self.epc.annotations['test1'], result.annotations['test1'])
        self.assertNotEqual(self.epc.annotations['test2'], result.annotations['test2'])

    def test__time_slice_deepcopy_array_annotations(self):
        length = self.epc.shape[-1]
        params1 = {'test0': ['y{}'.format(i) for i in range(length)],
                   'test1': ['deeptest' for i in range(length)],
                   'test2': [(-1)**i > 0 for i in range(length)]}
        self.epc.array_annotate(**params1)
        # time_slice spike train, keep sliced spike times
        t_start = 15 * pq.s
        t_stop = 45 * pq.s
        result = self.epc.time_slice(t_start, t_stop)

        # Change annotations of original
        params2 = {'test0': ['x{}'.format(i) for i in range(length)],
                   'test2': [(-1) ** (i + 1) > 0 for i in range(length)]}
        self.epc.array_annotate(**params2)
        self.epc.array_annotations['test1'][2] = 'shallowtest'

        self.assertFalse(all(self.epc.array_annotations['test0'][1:4]
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.epc.array_annotations['test1'][1:4]
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.epc.array_annotations['test2'][1:4]
                             == result.array_annotations['test2']))

        # Change annotations of result
        params3 = {'test0': ['z{}'.format(i) for i in range(1, 4)]}
        result.array_annotate(**params3)
        result.array_annotations['test1'][1] = 'shallow2'

        self.assertFalse(all(self.epc.array_annotations['test0'][1:4]
                             == result.array_annotations['test0']))
        self.assertFalse(all(self.epc.array_annotations['test1'][1:4]
                             == result.array_annotations['test1']))
        self.assertFalse(all(self.epc.array_annotations['test2'][1:4]
                             == result.array_annotations['test2']))

    def test__time_slice_deepcopy_data(self):
        result = self.epc.time_slice(None, None)

        # Change values of original array
        self.epc[2] = 7.3*self.epc.units

        self.assertFalse(all(self.epc == result))

        # Change values of sliced array
        result[3] = 9.5*result.units

        self.assertFalse(all(self.epc == result))

    def test_time_slice_out_of_boundries(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                     labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                     name='test', description='tester', file_origin='test.file', test1=1,
                     array_annotations=arr_ann, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)

        t_start = 0.0001
        t_stop = 30
        result = epc.time_slice(t_start, t_stop)

        assert_arrays_equal(result.times, targ.times)
        assert_arrays_equal(result.durations, targ.durations)
        assert_arrays_equal(result.labels, targ.labels)
        self.assertEqual(result.name, targ.name)
        self.assertEqual(result.description, targ.description)
        self.assertEqual(result.file_origin, targ.file_origin)
        self.assertEqual(result.annotations['test0'], targ.annotations['test0'])
        self.assertEqual(result.annotations['test1'], targ.annotations['test1'])
        self.assertEqual(result.annotations['test2'], targ.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.array(arr_ann['index']))
        assert_arrays_equal(result.array_annotations['test'], np.array(arr_ann['test']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_empty(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([] * pq.ms, durations=[] * pq.ns, labels=np.array([], dtype='U'), name='test',
                    description='tester', file_origin='test.file', test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([] * pq.ms, durations=[] * pq.ns, labels=np.array([], dtype='U'), name='test',
                     description='tester', file_origin='test.file', test1=1, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)

        t_start = 1.2
        t_stop = 1.6
        result = epc.time_slice(t_start, t_stop)

        assert_arrays_equal(result.times, targ.times)
        assert_arrays_equal(result.durations, targ.durations)
        assert_arrays_equal(result.labels, targ.labels)
        self.assertEqual(result.name, targ.name)
        self.assertEqual(result.description, targ.description)
        self.assertEqual(result.file_origin, targ.file_origin)
        self.assertEqual(result.annotations['test0'], targ.annotations['test0'])
        self.assertEqual(result.annotations['test1'], targ.annotations['test1'])
        self.assertEqual(result.annotations['test2'], targ.annotations['test2'])
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_stop(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.5, 1.7] * pq.ms, durations=[40, 60] * pq.ns,
                     labels=np.array(['test epoch 2', 'test epoch 3'], dtype='U'), name='test',
                     description='tester', file_origin='test.file', test1=1,
                     array_annotations={'index': [1, 2], 'test': ['b', 'c']}, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)

        t_start = 1.2
        t_stop = None
        result = epc.time_slice(t_start, t_stop)

        assert_arrays_equal(result.times, targ.times)
        assert_arrays_equal(result.durations, targ.durations)
        assert_arrays_equal(result.labels, targ.labels)
        self.assertEqual(result.name, targ.name)
        self.assertEqual(result.description, targ.description)
        self.assertEqual(result.file_origin, targ.file_origin)
        self.assertEqual(result.annotations['test0'], targ.annotations['test0'])
        self.assertEqual(result.annotations['test1'], targ.annotations['test1'])
        self.assertEqual(result.annotations['test2'], targ.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.array([1, 2]))
        assert_arrays_equal(result.array_annotations['test'], np.array(['b', 'c']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_start(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.1, 1.5] * pq.ms, durations=[20, 40] * pq.ns,
                     labels=np.array(['test epoch 1', 'test epoch 2'], dtype='U'), name='test',
                     description='tester', file_origin='test.file', test1=1,
                     array_annotations={'index': [0, 1], 'test': ['a', 'b']}, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)

        t_start = None
        t_stop = 1.6
        result = epc.time_slice(t_start, t_stop)

        assert_arrays_equal(result.times, targ.times)
        assert_arrays_equal(result.durations, targ.durations)
        assert_arrays_equal(result.labels, targ.labels)
        self.assertEqual(result.name, targ.name)
        self.assertEqual(result.description, targ.description)
        self.assertEqual(result.file_origin, targ.file_origin)
        self.assertEqual(result.annotations['test0'], targ.annotations['test0'])
        self.assertEqual(result.annotations['test1'], targ.annotations['test1'])
        self.assertEqual(result.annotations['test2'], targ.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.array([0, 1]))
        assert_arrays_equal(result.array_annotations['test'], np.array(['a', 'b']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_none_both(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                     labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                     name='test', description='tester', file_origin='test.file', test1=1,
                     array_annotations=arr_ann, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)

        t_start = None
        t_stop = None
        result = epc.time_slice(t_start, t_stop)

        assert_arrays_equal(result.times, targ.times)
        assert_arrays_equal(result.durations, targ.durations)
        assert_arrays_equal(result.labels, targ.labels)
        self.assertEqual(result.name, targ.name)
        self.assertEqual(result.description, targ.description)
        self.assertEqual(result.file_origin, targ.file_origin)
        self.assertEqual(result.annotations['test0'], targ.annotations['test0'])
        self.assertEqual(result.annotations['test1'], targ.annotations['test1'])
        self.assertEqual(result.annotations['test2'], targ.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.array([0, 1, 2]))
        assert_arrays_equal(result.array_annotations['test'], np.array(['a', 'b', 'c']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test_time_slice_differnt_units(self):
        params = {'test2': 'y1', 'test3': True}
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epc = Epoch([1.1, 1.5, 1.7] * pq.ms, durations=[20, 40, 60] * pq.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2', 'test epoch 3'], dtype='U'),
                    name='test', description='tester', file_origin='test.file', test1=1,
                    array_annotations=arr_ann, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.5] * pq.ms, durations=[40] * pq.ns,
                     labels=np.array(['test epoch 2'], dtype='U'), name='test',
                     description='tester', file_origin='test.file', test1=1,
                     array_annotations={'index': [1], 'test': ['b']}, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)

        t_start = 0.0012 * pq.s
        t_stop = 0.0016 * pq.s
        result = epc.time_slice(t_start, t_stop)

        assert_arrays_equal(result.times, targ.times)
        assert_arrays_equal(result.durations, targ.durations)
        assert_arrays_equal(result.labels, targ.labels)
        self.assertEqual(result.name, targ.name)
        self.assertEqual(result.description, targ.description)
        self.assertEqual(result.file_origin, targ.file_origin)
        self.assertEqual(result.annotations['test0'], targ.annotations['test0'])
        self.assertEqual(result.annotations['test1'], targ.annotations['test1'])
        self.assertEqual(result.annotations['test2'], targ.annotations['test2'])
        assert_arrays_equal(result.array_annotations['index'], np.array([1]))
        assert_arrays_equal(result.array_annotations['test'], np.array(['b']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

    def test__time_slice_should_set_parents_to_None(self):
        # When timeslicing, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.epc.time_slice(1 * pq.ms, 3 * pq.ms)
        self.assertEqual(result.segment, None)

    def test__deepcopy_should_set_parents_objects_to_None(self):
        # Deepcopy should destroy references to parents
        result = deepcopy(self.epc)
        self.assertEqual(result.segment, None)

    def test__time_shift_same_attributes(self):
        result = self.epc.time_shift(1 * pq.ms)
        assert_same_attributes(result, self.epc, exclude=['times'])

    def test__time_shift_same_annotations(self):
        result = self.epc.time_shift(1 * pq.ms)
        assert_same_annotations(result, self.epc)

    def test__time_shift_same_array_annotations(self):
        result = self.epc.time_shift(1 * pq.ms)
        assert_same_array_annotations(result, self.epc)

    def test__time_shift_should_set_parents_to_None(self):
        # When time-shifting, a deep copy is made,
        # thus the reference to parent objects should be destroyed
        result = self.epc.time_shift(1 * pq.ms)
        self.assertEqual(result.segment, None)

    def test__time_shift_by_zero(self):
        shifted = self.epc.time_shift(0 * pq.ms)
        assert_arrays_equal(shifted.times, self.epc.times)

    def test__time_shift_same_units(self):
        shifted = self.epc.time_shift(10 * pq.ms)
        assert_arrays_equal(shifted.times, self.epc.times + 10 * pq.ms)

    def test__time_shift_different_units(self):
        shifted = self.epc.time_shift(1 * pq.s)
        assert_arrays_equal(shifted.times, self.epc.times + 1000 * pq.ms)

    def test_as_array(self):
        times = [2, 3, 4, 5]
        durations = [0.1, 0.2, 0.3, 0.4]
        epc = Epoch(times * pq.ms, durations=durations * pq.ms)
        epc_as_arr = epc.as_array(units='ms')
        self.assertIsInstance(epc_as_arr, np.ndarray)
        assert_array_equal(times, epc_as_arr)

    def test_as_quantity(self):
        times = [2, 3, 4, 5]
        durations = [0.1, 0.2, 0.3, 0.4]
        epc = Epoch(times * pq.ms, durations=durations * pq.ms)
        epc_as_q = epc.as_quantity()
        self.assertIsInstance(epc_as_q, pq.Quantity)
        assert_array_equal(times * pq.ms, epc_as_q)

    def test_getitem_scalar(self):
        times = [2, 3, 4, 5]
        durations = [0.1, 0.2, 0.3, 0.4]
        labels = ["A", "B", "C", "D"]
        epc = Epoch(times * pq.ms, durations=durations * pq.ms, labels=labels)
        single_epoch = epc[2]
        self.assertIsInstance(single_epoch, pq.Quantity)
        assert_array_equal(single_epoch.times, np.array([4.0]))
        assert_array_equal(single_epoch.durations, np.array([0.3]))
        assert_array_equal(single_epoch.labels, np.array(["C"]))

    def test_slice(self):
        times = [2, 3, 4, 5]
        durations = [0.1, 0.2, 0.3, 0.4]
        labels = ["A", "B", "C", "D"]
        arr_ann = {'index': np.arange(4), 'test': ['a', 'b', 'c', 'd']}
        epc = Epoch(times * pq.ms, durations=durations * pq.ms, labels=labels,
                    array_annotations=arr_ann)
        single_epoch = epc[1:3]
        self.assertIsInstance(single_epoch, Epoch)
        assert_array_equal(single_epoch.times, np.array([3.0, 4.0]))
        assert_array_equal(single_epoch.durations, np.array([0.2, 0.3]))
        assert_array_equal(single_epoch.labels, np.array(["B", "C"]))
        assert_arrays_equal(single_epoch.array_annotations['index'], np.arange(1, 3))
        assert_arrays_equal(single_epoch.array_annotations['test'], np.array(['b', 'c']))
        self.assertIsInstance(single_epoch.array_annotations, ArrayDict)

    def test_rescale(self):
        times = [2, 3, 4, 5]
        durations = [0.1, 0.2, 0.3, 0.4]
        labels = ["A", "B", "C", "D"]
        arr_ann = {'index': np.arange(4), 'test': ['a', 'b', 'c', 'd']}
        epc = Epoch(times * pq.ms, durations=durations * pq.ms, labels=labels,
                    array_annotations=arr_ann)
        result = epc.rescale(pq.us)

        self.assertIsInstance(result, Epoch)
        assert_neo_object_is_compliant(result)
        assert_arrays_equal(result.array_annotations['index'], np.arange(4))
        assert_arrays_equal(result.array_annotations['test'],
                            np.array(['a', 'b', 'c', 'd']))
        self.assertIsInstance(result.array_annotations, ArrayDict)

        self.assertEqual(result.units, 1 * pq.us)
        assert_array_equal(epc.labels, result.labels)
        assert_arrays_almost_equal(result.times, [2000, 3000, 4000, 5000] * pq.us, 1e-9)
        assert_arrays_almost_equal(result.times.magnitude,
                                   np.array([2000, 3000, 4000, 5000]),
                                   1e-9)
        assert_arrays_almost_equal(result.durations.magnitude,
                                   np.array([100, 200, 300, 400]),
                                   1e-9)


class TestDuplicateWithNewData(unittest.TestCase):
    def setUp(self):
        self.data = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.durations = np.array([0.2, 0.4, 1.1, 2.4, 0.2, 2.0])
        self.quant = pq.ms
        self.arr_ann = {'index': np.arange(6), 'test': ['a', 'b', 'c', 'd', 'e', 'f']}
        self.epoch = Epoch(self.data * self.quant, durations=self.durations * self.quant,
                           array_annotations=self.arr_ann)

    def test_duplicate_with_new_data(self):
        signal1 = self.epoch
        new_times = np.sort(np.random.uniform(0, 100, self.epoch.size)) * pq.ms
        new_durations = np.ones_like(new_times)
        new_labels = np.array(list("zyxwvutsrqponmlkjihgfedcba"[:self.epoch.size]))
        signal1b = signal1.duplicate_with_new_data(new_times, new_durations, new_labels)
        # After duplicating, array annotations should always be empty,
        # because different length of data would cause inconsistencies
        assert_arrays_equal(signal1b.labels, new_labels)
        assert_arrays_equal(signal1b.durations, new_durations)
        self.assertTrue('index' not in signal1b.array_annotations)
        self.assertTrue('test' not in signal1b.array_annotations)
        self.assertIsInstance(signal1b.array_annotations, ArrayDict)


class TestEpochFunctions(unittest.TestCase):
    def test__pickle(self):
        arr_ann = {'index': np.arange(3), 'test': ['a', 'b', 'c']}
        epoch1 = Epoch(np.arange(0, 30, 10) * pq.s, durations=[1, 2, 3] * pq.s,
                       labels=np.array(['t0', 't1', 't2'], dtype='U'), units='s',
                       annotation1="foo", annotation2="bar", array_annotations=arr_ann)
        fobj = open('./pickle', 'wb')
        pickle.dump(epoch1, fobj)
        fobj.close()

        fobj = open('./pickle', 'rb')
        try:
            epoch2 = pickle.load(fobj)
        except ValueError:
            epoch2 = None

        fobj.close()
        assert_array_equal(epoch1.times, epoch2.times)
        self.assertEqual(epoch2.annotations, epoch1.annotations)
        assert_arrays_equal(epoch2.array_annotations['index'], np.array(arr_ann['index']))
        assert_arrays_equal(epoch2.array_annotations['test'], np.array(arr_ann['test']))
        self.assertIsInstance(epoch2.array_annotations, ArrayDict)
        # Make sure the dict can perform correct checks after unpickling
        epoch2.array_annotations['anno3'] = list(range(3, 6))
        with self.assertRaises(ValueError):
            epoch2.array_annotations['anno4'] = [2, 1]
        os.remove('./pickle')


if __name__ == "__main__":
    unittest.main()
