# -*- coding: utf-8 -*-
"""
Tests of the neo.core.recordingchannelgroup.RecordingChannelGroup class
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

from neo.core.recordingchannelgroup import RecordingChannelGroup
from neo.core import AnalogSignalArray, RecordingChannel, Unit, Block
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
        channel_names = get_fake_value('channel_names', np.ndarray, seed=1,
                                       dim=1, dtype=np.dtype('S'))
        name = get_fake_value('name', str, seed=2, obj=RecordingChannelGroup)
        description = get_fake_value('description', str, seed=3,
                                     obj='RecordingChannelGroup')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(RecordingChannelGroup, annotate=False, seed=0)
        res12 = get_fake_values('RecordingChannelGroup',
                                annotate=False, seed=0)
        res21 = get_fake_values(RecordingChannelGroup, annotate=True, seed=0)
        res22 = get_fake_values('RecordingChannelGroup', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('channel_indexes'), channel_indexes)
        assert_arrays_equal(res12.pop('channel_indexes'), channel_indexes)
        assert_arrays_equal(res21.pop('channel_indexes'), channel_indexes)
        assert_arrays_equal(res22.pop('channel_indexes'), channel_indexes)

        assert_arrays_equal(res11.pop('channel_names'), channel_names)
        assert_arrays_equal(res12.pop('channel_names'), channel_names)
        assert_arrays_equal(res21.pop('channel_names'), channel_names)
        assert_arrays_equal(res22.pop('channel_names'), channel_names)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'RecordingChannelGroup'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannelGroup))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        for child in res.children:
            del child.annotations['i']
            del child.annotations['j']
            for subchild in child.children:
                del subchild.annotations['i']
                del subchild.annotations['j']

        self.assertEqual(len(res.recordingchannels), 1)
        rchan = res.recordingchannels[0]
        self.assertEqual(rchan.annotations, self.annotations)

        self.assertEqual(len(res.units), 1)
        unit = res.units[0]
        self.assertEqual(unit.annotations, self.annotations)

        self.assertEqual(len(res.analogsignalarrays), 1)
        self.assertEqual(res.analogsignalarrays[0].annotations,
                         self.annotations)

        self.assertEqual(len(rchan.analogsignals), 1)
        self.assertEqual(len(rchan.irregularlysampledsignals), 1)
        self.assertEqual(rchan.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(rchan.irregularlysampledsignals[0].annotations,
                         self.annotations)

        self.assertEqual(len(unit.spiketrains), 1)
        self.assertEqual(len(unit.spikes), 1)
        self.assertEqual(unit.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(unit.spikes[0].annotations,
                         self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = RecordingChannelGroup
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannelGroup))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.recordingchannels), 0)
        self.assertEqual(len(res.units), 0)
        self.assertEqual(len(res.analogsignalarrays), 0)


class TestRecordingChannelGroup(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        self.seed1 = 0
        self.seed2 = 10000
        self.rcg1 = fake_neo(RecordingChannelGroup,
                             seed=self.seed1, n=self.nchildren)
        self.rcg2 = fake_neo(RecordingChannelGroup,
                             seed=self.seed2, n=self.nchildren)
        self.targobj = self.rcg1

        self.rchans1 = self.rcg1.recordingchannels
        self.rchans2 = self.rcg2.recordingchannels
        self.units1 = self.rcg1.units
        self.units2 = self.rcg2.units
        self.sigarrs1 = self.rcg1.analogsignalarrays
        self.sigarrs2 = self.rcg2.analogsignalarrays

        self.rchans1a = clone_object(self.rchans1)
        self.units1a = clone_object(self.units1)
        self.sigarrs1a = clone_object(self.sigarrs1, n=2)

        self.targobj = self.rcg1

    def test__recordingchannelgroup__init_defaults(self):
        rcg = RecordingChannelGroup()
        assert_neo_object_is_compliant(rcg)
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.file_origin, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        assert_arrays_equal(rcg.channel_names, np.array([], dtype='S'))
        assert_arrays_equal(rcg.channel_indexes, np.array([]))

    def test_recordingchannelgroup__init(self):
        rcg = RecordingChannelGroup(file_origin='temp.dat',
                                    channel_indexes=np.array([1]))
        assert_neo_object_is_compliant(rcg)
        self.assertEqual(rcg.file_origin, 'temp.dat')
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        assert_arrays_equal(rcg.channel_names, np.array([], dtype='S'))
        assert_arrays_equal(rcg.channel_indexes, np.array([1]))

    def check_creation(self, rcg):
        assert_neo_object_is_compliant(rcg)

        seed = rcg.annotations['seed']

        for i, rchan in enumerate(rcg.recordingchannels):
            self.assertEqual(rchan.name, rcg.channel_names[i].astype(str))
            self.assertEqual(rchan.index, rcg.channel_indexes[i])
        for i, unit in enumerate(rcg.units):
            for sigarr in rcg.analogsignalarrays:
                self.assertEqual(unit.channel_indexes[0],
                                 sigarr.channel_index[i])

        targ2 = get_fake_value('name', str, seed=seed+2,
                               obj=RecordingChannelGroup)
        self.assertEqual(rcg.name, targ2)

        targ3 = get_fake_value('description', str,
                               seed=seed+3, obj=RecordingChannelGroup)
        self.assertEqual(rcg.description, targ3)

        targ4 = get_fake_value('file_origin', str)
        self.assertEqual(rcg.file_origin, targ4)

        targ5 = get_annotations()
        targ5['seed'] = seed
        self.assertEqual(rcg.annotations, targ5)

        self.assertTrue(hasattr(rcg, 'recordingchannels'))
        self.assertTrue(hasattr(rcg, 'units'))
        self.assertTrue(hasattr(rcg, 'analogsignalarrays'))

        self.assertEqual(len(rcg.recordingchannels), self.nchildren)
        self.assertEqual(len(rcg.units), self.nchildren)
        self.assertEqual(len(rcg.analogsignalarrays), self.nchildren)

    def test__creation(self):
        self.check_creation(self.rcg1)
        self.check_creation(self.rcg2)

    def test__merge(self):
        rcg1a = fake_neo(RecordingChannelGroup,
                         seed=self.seed1, n=self.nchildren)
        assert_same_sub_schema(self.rcg1, rcg1a)
        rcg1a.annotate(seed=self.seed2)
        rcg1a.analogsignalarrays.append(self.sigarrs2[0])
        rcg1a.merge(self.rcg2)
        self.check_creation(self.rcg2)
        sigarrm = self.sigarrs2[0].merge(self.sigarrs2[0])

        assert_same_sub_schema(self.sigarrs1a + [sigarrm] + self.sigarrs2[1:],
                               rcg1a.analogsignalarrays,
                               exclude=['channel_index'])
        assert_same_sub_schema(self.units1a + self.units2,
                               rcg1a.units)
        assert_same_sub_schema(self.rchans1a + self.rchans2,
                               rcg1a.recordingchannels,
                               exclude=['channel_index'])

    def test__children(self):
        blk = Block(name='block1')
        blk.recordingchannelgroups = [self.rcg1]
        blk.create_many_to_one_relationship()

        self.assertEqual(self.rcg1._container_child_objects, ('Unit',))
        self.assertEqual(self.rcg1._data_child_objects, ('AnalogSignalArray',))
        self.assertEqual(self.rcg1._single_parent_objects, ('Block',))
        self.assertEqual(self.rcg1._multi_child_objects, ('RecordingChannel',))
        self.assertEqual(self.rcg1._multi_parent_objects, ())
        self.assertEqual(self.rcg1._child_properties, ())

        self.assertEqual(self.rcg1._single_child_objects,
                         ('Unit', 'AnalogSignalArray',))

        self.assertEqual(self.rcg1._container_child_containers, ('units',))
        self.assertEqual(self.rcg1._data_child_containers,
                         ('analogsignalarrays',))
        self.assertEqual(self.rcg1._single_child_containers,
                         ('units', 'analogsignalarrays'))
        self.assertEqual(self.rcg1._single_parent_containers, ('block',))
        self.assertEqual(self.rcg1._multi_child_containers,
                         ('recordingchannels',))
        self.assertEqual(self.rcg1._multi_parent_containers, ())

        self.assertEqual(self.rcg1._child_objects,
                         ('Unit', 'AnalogSignalArray', 'RecordingChannel'))
        self.assertEqual(self.rcg1._child_containers,
                         ('units', 'analogsignalarrays', 'recordingchannels'))
        self.assertEqual(self.rcg1._parent_objects, ('Block',))
        self.assertEqual(self.rcg1._parent_containers, ('block',))

        self.assertEqual(len(self.rcg1.children), self.nchildren*3)

        assert_same_sub_schema(list(self.rcg1.children),
                               self.units1a+self.sigarrs1a+self.rchans1a,
                               exclude=['channel_index'])

        self.assertEqual(len(self.rcg1.parents), 1)
        self.assertEqual(self.rcg1.parents[0].name, 'block1')


if __name__ == '__main__':
    unittest.main()
