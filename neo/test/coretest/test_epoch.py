# -*- coding: utf-8 -*-
"""
Tests of the neo.core.epoch.Epoch class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
from neo import units as un


try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.epoch import Epoch
from neo.core import Segment
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal, assert_arrays_almost_equal,
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
        times = get_fake_value('times', un.Quantity, seed=0, dim=1)
        durations = get_fake_value('durations', un.Quantity, seed=1, dim=1)
        labels = get_fake_value('labels', np.ndarray, seed=2, dim=1, dtype='S')
        name = get_fake_value('name', str, seed=3, obj=Epoch)
        description = get_fake_value('description', str,
                                     seed=4, obj='Epoch')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Epoch, annotate=False, seed=0)
        res12 = get_fake_values('Epoch', annotate=False, seed=0)
        res21 = get_fake_values(Epoch, annotate=True, seed=0)
        res22 = get_fake_values('Epoch', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('times'), times)
        assert_arrays_equal(res12.pop('times'), times)
        assert_arrays_equal(res21.pop('times'), times)
        assert_arrays_equal(res22.pop('times'), times)

        assert_arrays_equal(res11.pop('durations'), durations)
        assert_arrays_equal(res12.pop('durations'), durations)
        assert_arrays_equal(res21.pop('durations'), durations)
        assert_arrays_equal(res22.pop('durations'), durations)

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
        obj_type = Epoch
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Epoch))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'Epoch'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Epoch))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestEpoch(unittest.TestCase):
    def test_Epoch_creation(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        assert_arrays_equal(epc.times, [1.1, 1.5, 1.7]*un.ms)
        assert_arrays_equal(epc.durations, [20, 40, 60]*un.ns)
        assert_arrays_equal(epc.labels, np.array(['test epoch 1',
                                                  'test epoch 2',
                                                  'test epoch 3'], dtype='S'))
        self.assertEqual(epc.name, 'test')
        self.assertEqual(epc.description, 'tester')
        self.assertEqual(epc.file_origin, 'test.file')
        self.assertEqual(epc.annotations['test0'], [1, 2])
        self.assertEqual(epc.annotations['test1'], 1.1)
        self.assertEqual(epc.annotations['test2'], 'y1')
        self.assertTrue(epc.annotations['test3'])

    def test_Epoch_repr(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = ('<Epoch: test epoch 1@1.1 ms for 20.0 ns, ' +
                'test epoch 2@1.5 ms for 40.0 ns, ' +
                'test epoch 3@1.7 ms for 60.0 ns>')

        res = repr(epc)

        self.assertEqual(targ, res)

    def test_Epoch_merge(self):
        params1 = {'test2': 'y1', 'test3': True}
        params2 = {'test2': 'no', 'test4': False}
        paramstarg = {'test2': 'yes;no',
                      'test3': True,
                      'test4': False}
        epc1 = Epoch([1.1, 1.5, 1.7]*un.ms,
                     durations=[20, 40, 60]*un.us,
                     labels=np.array(['test epoch 1 1',
                                      'test epoch 1 2',
                                      'test epoch 1 3'], dtype='S'),
                     name='test', description='tester 1',
                     file_origin='test.file',
                     test1=1, **params1)
        epc2 = Epoch([2.1, 2.5, 2.7]*un.us,
                     durations=[3, 5, 7]*un.ms,
                     labels=np.array(['test epoch 2 1',
                                      'test epoch 2 2',
                                      'test epoch 2 3'], dtype='S'),
                     name='test', description='tester 2',
                     file_origin='test.file',
                     test1=1, **params2)
        epctarg = Epoch([1.1, 1.5, 1.7, .0021, .0025, .0027]*un.ms,
                        durations=[20, 40, 60, 3000, 5000, 7000]*un.ns,
                        labels=np.array(['test epoch 1 1',
                                         'test epoch 1 2',
                                         'test epoch 1 3',
                                         'test epoch 2 1',
                                         'test epoch 2 2',
                                         'test epoch 2 3'], dtype='S'),
                        name='test',
                        description='merge(tester 1, tester 2)',
                        file_origin='test.file',
                        test1=1, **paramstarg)
        assert_neo_object_is_compliant(epc1)
        assert_neo_object_is_compliant(epc2)
        assert_neo_object_is_compliant(epctarg)

        epcres = epc1.merge(epc2)
        assert_neo_object_is_compliant(epcres)
        assert_same_sub_schema(epctarg, epcres)

    def test__children(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        segment = Segment(name='seg1')
        segment.epochs = [epc]
        segment.create_many_to_one_relationship()

        self.assertEqual(epc._single_parent_objects, ('Segment',))
        self.assertEqual(epc._multi_parent_objects, ())

        self.assertEqual(epc._single_parent_containers, ('segment',))
        self.assertEqual(epc._multi_parent_containers, ())

        self.assertEqual(epc._parent_objects, ('Segment',))
        self.assertEqual(epc._parent_containers, ('segment',))

        self.assertEqual(len(epc.parents), 1)
        self.assertEqual(epc.parents[0].name, 'seg1')

        assert_neo_object_is_compliant(epc)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file')
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        prepr = pretty(epc)
        targ = ("Epoch\nname: '%s'\ndescription: '%s'\nannotations: %s" %
                (epc.name, epc.description, pretty(epc.annotations)))

        self.assertEqual(prepr, targ)

    def test__time_slice(self):
        epc = Epoch(times=[10, 20, 30] * un.s, durations=[10, 5, 7] * un.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'),
                    foo='bar')

        epc2 = epc.time_slice(10 * un.s, 20 * un.s)
        assert_arrays_equal(epc2.times, [10, 20] * un.s)
        assert_arrays_equal(epc2.durations, [10, 5] * un.ms)
        assert_arrays_equal(epc2.labels, np.array(['btn0','btn1'], dtype='S'))
        self.assertEqual(epc.annotations, epc2.annotations)

    def test_time_slice2(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.5]*un.ms, durations=[40]*un.ns,
                    labels=np.array(['test epoch 2'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
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

    def test_time_slice_out_of_boundries(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
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


    def test_time_slice_empty(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([]*un.ms, durations=[]*un.ns,
                    labels=np.array([], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([]*un.ms, durations=[]*un.ns,
                    labels=np.array([], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
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

    def test_time_slice_none_stop(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.5, 1.7]*un.ms, durations=[40, 60]*un.ns,
                    labels=np.array(['test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
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
        
    def test_time_slice_none_start(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.1, 1.5]*un.ms, durations=[20, 40]*un.ns,
                    labels=np.array(['test epoch 1', 'test epoch 2'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
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
        
    def test_time_slice_none_both(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
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
       
    def test_time_slice_differnt_units(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch([1.1, 1.5, 1.7]*un.ms, durations=[20, 40, 60]*un.ns,
                    labels=np.array(['test epoch 1',
                                     'test epoch 2',
                                     'test epoch 3'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        targ = Epoch([1.5]*un.ms, durations=[40]*un.ns,
                    labels=np.array(['test epoch 2'], dtype='S'),
                    name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        targ.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(targ)
        
        t_start = 0.0012  * un.s
        t_stop = 0.0016 * un.s
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
    



class TestDuplicateWithNewData(unittest.TestCase):
    def setUp(self):
        self.data = np.array([0.1, 0.5, 1.2, 3.3, 6.4, 7])
        self.durations = np.array([0.2, 0.4, 1.1, 2.4, 0.2, 2.0])
        self.quant = un.ms
        self.epoch = Epoch(self.data*self.quant,
                           durations=self.durations*self.quant)

    def test_duplicate_with_new_data(self):
        signal1 = self.epoch
        new_data = np.sort(np.random.uniform(0, 100, (self.epoch))) * un.ms
        signal1b = signal1.duplicate_with_new_data(new_data)
        assert_arrays_almost_equal(np.asarray(signal1b),
                                   np.asarray(new_data), 1e-12)
        assert_arrays_almost_equal(np.asarray(signal1b.durations),
                                   np.asarray(signal1.durations), 1e-12)


if __name__ == "__main__":
    unittest.main()
