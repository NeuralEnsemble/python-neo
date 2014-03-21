# -*- coding: utf-8 -*-
"""
Tests of the neo.core.epoch.Epoch class
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

from neo.core.epoch import Epoch
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
        duration = get_fake_value('duration', pq.Quantity, seed=1)
        label = get_fake_value('label', str, seed=2)
        name = get_fake_value('name', str, seed=3, obj=Epoch)
        description = get_fake_value('description', str, seed=4, obj='Epoch')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'label': label,
                  'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Epoch, annotate=False, seed=0)
        res12 = get_fake_values('Epoch', annotate=False, seed=0)
        res21 = get_fake_values(Epoch, annotate=True, seed=0)
        res22 = get_fake_values('Epoch', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('time'), time)
        assert_arrays_equal(res12.pop('time'), time)
        assert_arrays_equal(res21.pop('time'), time)
        assert_arrays_equal(res22.pop('time'), time)

        assert_arrays_equal(res11.pop('duration'), duration)
        assert_arrays_equal(res12.pop('duration'), duration)
        assert_arrays_equal(res21.pop('duration'), duration)
        assert_arrays_equal(res22.pop('duration'), duration)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'Epoch'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Epoch))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = Epoch
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Epoch))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestEpoch(unittest.TestCase):
    def test_epoch_creation(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch(1.5*pq.ms, duration=20*pq.ns,
                    label='test epoch', name='test', description='tester',
                    file_origin='test.file',
                    test1=1, **params)
        epc.annotate(test1=1.1, test0=[1, 2])
        assert_neo_object_is_compliant(epc)

        self.assertEqual(epc.time, 1.5*pq.ms)
        self.assertEqual(epc.duration, 20*pq.ns)
        self.assertEqual(epc.label, 'test epoch')
        self.assertEqual(epc.name, 'test')
        self.assertEqual(epc.description, 'tester')
        self.assertEqual(epc.file_origin, 'test.file')
        self.assertEqual(epc.annotations['test0'], [1, 2])
        self.assertEqual(epc.annotations['test1'], 1.1)
        self.assertEqual(epc.annotations['test2'], 'y1')
        self.assertTrue(epc.annotations['test3'])

    def test_epoch_merge_NotImplementedError(self):
        epc1 = Epoch(1.5*pq.ms, duration=20*pq.ns,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file')
        epc2 = Epoch(1.5*pq.ms, duration=20*pq.ns,
                     label='test epoch', name='test', description='tester',
                     file_origin='test.file')
        self.assertRaises(NotImplementedError, epc1.merge, epc2)

    def test__children(self):
        params = {'test2': 'y1', 'test3': True}
        epc = Epoch(1.5*pq.ms, duration=20*pq.ns,
                    label='test epoch', name='test', description='tester',
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
        epc = Epoch(1.5*pq.ms, duration=20*pq.ns,
                    label='test epoch', name='test', description='tester',
                    file_origin='test.file')
        epc.annotate(targ1=1.1, targ0=[1])
        assert_neo_object_is_compliant(epc)

        prepr = pretty(epc)
        targ = ("Epoch\nname: '%s'\ndescription: '%s'\nannotations: %s" %
                (epc.name, epc.description, pretty(epc.annotations)))

        self.assertEqual(prepr, targ)

if __name__ == "__main__":
    unittest.main()
