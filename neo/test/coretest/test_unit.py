# -*- coding: utf-8 -*-
"""
Tests of the neo.core.unit.Unit class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

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

from neo.core.unit import Unit
from neo.core import SpikeTrain, Spike, RecordingChannelGroup
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal,
                            assert_same_sub_schema)
from neo.test.generate_datasets import (fake_neo, get_fake_value,
                                        get_fake_values, get_annotations,
                                        clone_object, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__get_fake_values(self):
        self.annotations['seed'] = 0
        channel_indexes = get_fake_value('channel_indexes', np.ndarray, seed=0,
                                         dim=1, dtype='i')
        name = get_fake_value('name', str, seed=1, obj=Unit)
        description = get_fake_value('description', str, seed=2, obj='Unit')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Unit, annotate=False, seed=0)
        res12 = get_fake_values('Unit', annotate=False, seed=0)
        res21 = get_fake_values(Unit, annotate=True, seed=0)
        res22 = get_fake_values('Unit', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('channel_indexes'), channel_indexes)
        assert_arrays_equal(res12.pop('channel_indexes'), channel_indexes)
        assert_arrays_equal(res21.pop('channel_indexes'), channel_indexes)
        assert_arrays_equal(res22.pop('channel_indexes'), channel_indexes)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'Unit'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Unit))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.spiketrains), 1)
        self.assertEqual(len(res.spikes), 1)

        for child in res.children:
            del child.annotations['i']
            del child.annotations['j']
        self.assertEqual(res.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(res.spikes[0].annotations,
                         self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = Unit
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Unit))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.spikes), 0)


class TestUnit(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        self.seed1 = 0
        self.seed2 = 10000
        self.unit1 = fake_neo(Unit, seed=self.seed1, n=self.nchildren)
        self.unit2 = fake_neo(Unit, seed=self.seed2, n=self.nchildren)
        self.targobj = self.unit1

        self.spikes1 = self.unit1.spikes
        self.spikes2 = self.unit2.spikes
        self.trains1 = self.unit1.spiketrains
        self.trains2 = self.unit2.spiketrains

        self.spikes1a = clone_object(self.spikes1)
        self.trains1a = clone_object(self.trains1)

    def check_creation(self, unit):
        assert_neo_object_is_compliant(unit)

        seed = unit.annotations['seed']

        targ0 = get_fake_value('channel_indexes', np.ndarray, dim=1, dtype='i',
                               seed=seed+0)
        assert_arrays_equal(unit.channel_indexes, targ0)

        targ1 = get_fake_value('name', str, seed=seed+1, obj=Unit)
        self.assertEqual(unit.name, targ1)

        targ2 = get_fake_value('description', str,
                               seed=seed+2, obj=Unit)
        self.assertEqual(unit.description, targ2)

        targ3 = get_fake_value('file_origin', str)
        self.assertEqual(unit.file_origin, targ3)

        targ4 = get_annotations()
        targ4['seed'] = seed
        self.assertEqual(unit.annotations, targ4)

        self.assertTrue(hasattr(unit, 'spikes'))
        self.assertTrue(hasattr(unit, 'spiketrains'))

        self.assertEqual(len(unit.spikes), self.nchildren)
        self.assertEqual(len(unit.spiketrains), self.nchildren)

    def test__creation(self):
        self.check_creation(self.unit1)
        self.check_creation(self.unit2)

    def test__merge(self):
        unit1a = fake_neo(Unit, seed=self.seed1, n=self.nchildren)
        assert_same_sub_schema(self.unit1, unit1a)
        unit1a.annotate(seed=self.seed2)
        unit1a.spikes.append(self.spikes2[0])
        unit1a.merge(self.unit2)
        self.check_creation(self.unit2)

        assert_same_sub_schema(self.spikes1a + self.spikes2[:1] + self.spikes2,
                               unit1a.spikes)
        assert_same_sub_schema(self.trains1a + self.trains2,
                               unit1a.spiketrains)

    def test__children(self):
        rcg = RecordingChannelGroup(name='rcg1')
        rcg.units = [self.unit1]
        rcg.create_many_to_one_relationship()
        assert_neo_object_is_compliant(self.unit1)
        assert_neo_object_is_compliant(rcg)
        targ = self.unit1

        self.assertEqual(self.unit1._container_child_objects, ())
        self.assertEqual(self.unit1._data_child_objects,
                         ('Spike', 'SpikeTrain'))
        self.assertEqual(self.unit1._single_parent_objects,
                         ('RecordingChannelGroup',))
        self.assertEqual(self.unit1._multi_child_objects, ())
        self.assertEqual(self.unit1._multi_parent_objects, ())
        self.assertEqual(self.unit1._child_properties, ())

        self.assertEqual(self.unit1._single_child_objects,
                         ('Spike', 'SpikeTrain'))

        self.assertEqual(self.unit1._container_child_containers, ())
        self.assertEqual(self.unit1._data_child_containers,
                         ('spikes', 'spiketrains'))
        self.assertEqual(self.unit1._single_child_containers,
                         ('spikes', 'spiketrains'))
        self.assertEqual(self.unit1._single_parent_containers,
                         ('recordingchannelgroup',))
        self.assertEqual(self.unit1._multi_child_containers, ())
        self.assertEqual(self.unit1._multi_parent_containers, ())

        self.assertEqual(self.unit1._child_objects,
                         ('Spike', 'SpikeTrain'))
        self.assertEqual(self.unit1._child_containers,
                         ('spikes', 'spiketrains'))
        self.assertEqual(self.unit1._parent_objects,
                         ('RecordingChannelGroup',))
        self.assertEqual(self.unit1._parent_containers,
                         ('recordingchannelgroup',))

        self.assertEqual(len(self.unit1.children), self.nchildren*2)

        assert_same_sub_schema(list(self.unit1.children),
                               self.spikes1a+self.trains1a)

        self.assertEqual(len(self.unit1.parents), 1)
        self.assertEqual(self.unit1.parents[0].name, 'rcg1')


if __name__ == "__main__":
    unittest.main()
