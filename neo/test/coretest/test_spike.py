# -*- coding: utf-8 -*-
"""
Tests of the neo.core.spike.Spike class
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

from neo.core.spike import Spike
from neo.core import Segment, Unit
from neo.test.tools import assert_arrays_equal, assert_neo_object_is_compliant
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
        waveform = get_fake_value('waveform', pq.Quantity, seed=1, dim=2)
        left_sweep = get_fake_value('left_sweep', pq.Quantity, seed=2, dim=0)
        sampling_rate = get_fake_value('sampling_rate', pq.Quantity,
                                       seed=3, dim=0)
        name = get_fake_value('name', str, seed=4, obj=Spike)
        description = get_fake_value('description', str, seed=5, obj='Spike')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Spike, annotate=False, seed=0)
        res12 = get_fake_values('Spike', annotate=False, seed=0)
        res21 = get_fake_values(Spike, annotate=True, seed=0)
        res22 = get_fake_values('Spike', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('time'), time)
        assert_arrays_equal(res12.pop('time'), time)
        assert_arrays_equal(res21.pop('time'), time)
        assert_arrays_equal(res22.pop('time'), time)

        assert_arrays_equal(res11.pop('waveform'), waveform)
        assert_arrays_equal(res12.pop('waveform'), waveform)
        assert_arrays_equal(res21.pop('waveform'), waveform)
        assert_arrays_equal(res22.pop('waveform'), waveform)

        assert_arrays_equal(res11.pop('left_sweep'), left_sweep)
        assert_arrays_equal(res12.pop('left_sweep'), left_sweep)
        assert_arrays_equal(res21.pop('left_sweep'), left_sweep)
        assert_arrays_equal(res22.pop('left_sweep'), left_sweep)

        assert_arrays_equal(res11.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res12.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res21.pop('sampling_rate'), sampling_rate)
        assert_arrays_equal(res22.pop('sampling_rate'), sampling_rate)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = Spike
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Spike))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'Spike'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Spike))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)


class TestSpike(unittest.TestCase):
    def setUp(self):
        params = {'test2': 'y1', 'test3': True}
        self.sampling_rate1 = .1*pq.Hz
        self.left_sweep1 = 2.*pq.s
        self.spike1 = Spike(1.5*pq.ms,  waveform=[[1.1, 1.5, 1.7],
                                                  [2.2, 2.6, 2.8]]*pq.mV,
                            sampling_rate=self.sampling_rate1,
                            left_sweep=self.left_sweep1,
                            name='test', description='tester',
                            file_origin='test.file',
                            test1=1, **params)
        self.spike1.annotate(test1=1.1, test0=[1, 2])

    def test_spike_creation(self):
        assert_neo_object_is_compliant(self.spike1)

        self.assertEqual(self.spike1.time, 1.5*pq.ms)
        assert_arrays_equal(self.spike1.waveform, [[1.1, 1.5, 1.7],
                                                   [2.2, 2.6, 2.8]]*pq.mV)
        self.assertEqual(self.spike1.sampling_rate, .1*pq.Hz)
        self.assertEqual(self.spike1.left_sweep, 2.*pq.s)
        self.assertEqual(self.spike1.description, 'tester')
        self.assertEqual(self.spike1.file_origin, 'test.file')
        self.assertEqual(self.spike1.annotations['test0'], [1, 2])
        self.assertEqual(self.spike1.annotations['test1'], 1.1)
        self.assertEqual(self.spike1.annotations['test2'], 'y1')
        self.assertTrue(self.spike1.annotations['test3'])

    def test__duration(self):
        result1 = self.spike1.duration

        self.spike1.sampling_rate = None
        assert_neo_object_is_compliant(self.spike1)
        result2 = self.spike1.duration

        self.spike1.sampling_rate = self.sampling_rate1
        self.spike1.waveform = None
        assert_neo_object_is_compliant(self.spike1)
        result3 = self.spike1.duration

        self.assertEqual(result1, 30./pq.Hz)
        self.assertEqual(result1.units, 1./pq.Hz)
        self.assertEqual(result2, None)
        self.assertEqual(result3, None)

    def test__sampling_period(self):
        result1 = self.spike1.sampling_period

        self.spike1.sampling_rate = None
        assert_neo_object_is_compliant(self.spike1)
        result2 = self.spike1.sampling_period

        self.spike1.sampling_rate = self.sampling_rate1
        self.spike1.sampling_period = 10.*pq.ms
        assert_neo_object_is_compliant(self.spike1)
        result3a = self.spike1.sampling_period
        result3b = self.spike1.sampling_rate

        self.spike1.sampling_period = None
        result4a = self.spike1.sampling_period
        result4b = self.spike1.sampling_rate

        self.assertEqual(result1, 10./pq.Hz)
        self.assertEqual(result1.units, 1./pq.Hz)
        self.assertEqual(result2, None)
        self.assertEqual(result3a, 10.*pq.ms)
        self.assertEqual(result3a.units, 1.*pq.ms)
        self.assertEqual(result3b, .1/pq.ms)
        self.assertEqual(result3b.units, 1./pq.ms)
        self.assertEqual(result4a, None)
        self.assertEqual(result4b, None)

    def test__right_sweep(self):
        result1 = self.spike1.right_sweep

        self.spike1.left_sweep = None
        assert_neo_object_is_compliant(self.spike1)
        result2 = self.spike1.right_sweep

        self.spike1.left_sweep = self.left_sweep1
        self.spike1.sampling_rate = None
        assert_neo_object_is_compliant(self.spike1)
        result3 = self.spike1.right_sweep

        self.spike1.sampling_rate = self.sampling_rate1
        self.spike1.waveform = None
        assert_neo_object_is_compliant(self.spike1)
        result4 = self.spike1.right_sweep

        self.assertEqual(result1, 32.*pq.s)
        self.assertEqual(result1.units, 1.*pq.s)
        self.assertEqual(result2, None)
        self.assertEqual(result3, None)
        self.assertEqual(result4, None)

    def test__children(self):
        segment = Segment(name='seg1')
        segment.spikes = [self.spike1]
        segment.create_many_to_one_relationship()

        unit = Unit(name='unit1')
        unit.spikes = [self.spike1]
        unit.create_many_to_one_relationship()

        self.assertEqual(self.spike1._single_parent_objects,
                         ('Segment', 'Unit'))
        self.assertEqual(self.spike1._multi_parent_objects, ())

        self.assertEqual(self.spike1._single_parent_containers,
                         ('segment', 'unit'))
        self.assertEqual(self.spike1._multi_parent_containers, ())

        self.assertEqual(self.spike1._parent_objects,
                         ('Segment', 'Unit'))
        self.assertEqual(self.spike1._parent_containers,
                         ('segment', 'unit'))

        self.assertEqual(len(self.spike1.parents), 2)
        self.assertEqual(self.spike1.parents[0].name, 'seg1')
        self.assertEqual(self.spike1.parents[1].name, 'unit1')

        assert_neo_object_is_compliant(self.spike1)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        ann = {'targ0': self.spike1.annotations['test0']}
        self.spike1.annotations = ann
        res = pretty(self.spike1)
        targ = ("Spike " +
                "name: '%s' description: '%s' annotations: %s" %
                (self.spike1.name, self.spike1.description, ann))
        self.assertEqual(res, targ)


if __name__ == "__main__":
    unittest.main()
