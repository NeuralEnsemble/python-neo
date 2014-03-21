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

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.recordingchannelgroup import RecordingChannelGroup
from neo.core.container import filterdata
from neo.core import Block, Segment, SpikeTrain
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

        for child in res.children_recur:
            del child.annotations['i']
            del child.annotations['j']

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

        self.spikes1 = [[spike for spike in unit.spikes]
                        for unit in self.units1]
        self.spikes2 = [[spike for spike in unit.spikes]
                        for unit in self.units2]
        self.trains1 = [[train for train in unit.spiketrains]
                        for unit in self.units1]
        self.trains2 = [[train for train in unit.spiketrains]
                        for unit in self.units2]
        self.sigs1 = [[sig for sig in rchan.analogsignals]
                      for rchan in self.rchans1]
        self.sigs2 = [[sig for sig in rchan.analogsignals]
                      for rchan in self.rchans2]
        self.irsigs1 = [[irsig for irsig in rchan.irregularlysampledsignals]
                        for rchan in self.rchans1]
        self.irsigs2 = [[irsig for irsig in rchan.irregularlysampledsignals]
                        for rchan in self.rchans2]

        self.spikes1 = sum(self.spikes1, [])
        self.spikes2 = sum(self.spikes2, [])
        self.trains1 = sum(self.trains1, [])
        self.trains2 = sum(self.trains2, [])
        self.sigs1 = sum(self.sigs1, [])
        self.sigs2 = sum(self.sigs2, [])
        self.irsigs1 = sum(self.irsigs1, [])
        self.irsigs2 = sum(self.irsigs2, [])

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

        assert_same_sub_schema(self.sigarrs1a + self.sigarrs2,
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

        self.assertEqual(len(self.rcg1._single_children), 2*self.nchildren)
        self.assertEqual(len(self.rcg1._multi_children), self.nchildren)
        self.assertEqual(len(self.rcg1.data_children), self.nchildren)
        self.assertEqual(len(self.rcg1.data_children_recur),
                         self.nchildren + 4*self.nchildren**2)
        self.assertEqual(len(self.rcg1.container_children), 2*self.nchildren)
        self.assertEqual(len(self.rcg1.container_children_recur),
                         2*self.nchildren)
        self.assertEqual(len(self.rcg1.children), 3*self.nchildren)
        self.assertEqual(len(self.rcg1.children_recur),
                         3*self.nchildren + 4*self.nchildren**2)

        assert_same_sub_schema(list(self.rcg1._multi_children), self.rchans1)
        assert_same_sub_schema(list(self.rcg1._single_children),
                               self.units1a + self.sigarrs1a,
                               exclude=['channel_index'])

        assert_same_sub_schema(list(self.rcg1.container_children),
                               self.units1a + self.rchans1)
        assert_same_sub_schema(list(self.rcg1.container_children_recur),
                               self.units1a + self.rchans1)

        assert_same_sub_schema(list(self.rcg1.data_children), self.sigarrs1a,
                               exclude=['channel_index'])
        assert_same_sub_schema(list(self.rcg1.data_children_recur),
                               self.sigarrs1a +
                               self.spikes1[:2] + self.trains1[:2] +
                               self.spikes1[2:] + self.trains1[2:] +
                               self.sigs1[:2] + self.irsigs1[:2] +
                               self.sigs1[2:] + self.irsigs1[2:],
                               exclude=['channel_index'])

        assert_same_sub_schema(list(self.rcg1.children),
                               self.sigarrs1a + self.units1a + self.rchans1a,
                               exclude=['channel_index'])
        assert_same_sub_schema(list(self.rcg1.children_recur),
                               self.sigarrs1a +
                               self.spikes1[:2] + self.trains1[:2] +
                               self.spikes1[2:] + self.trains1[2:] +
                               self.sigs1[:2] + self.irsigs1[:2] +
                               self.sigs1[2:] + self.irsigs1[2:] +
                               self.units1a + self.rchans1a,
                               exclude=['channel_index'])

        self.assertEqual(len(self.rcg1.parents), 1)
        self.assertEqual(self.rcg1.parents[0].name, 'block1')

    def test__size(self):
        targ = {'analogsignalarrays': self.nchildren,
                'units': self.nchildren,
                'recordingchannels': self.nchildren}
        self.assertEqual(self.targobj.size, targ)

    def test__filter_none(self):
        targ = []

        res1 = self.targobj.filter()
        res2 = self.targobj.filter({})
        res3 = self.targobj.filter([])
        res4 = self.targobj.filter([{}])
        res5 = self.targobj.filter([{}, {}])
        res6 = self.targobj.filter([{}, {}])
        res7 = self.targobj.filter(targdict={})
        res8 = self.targobj.filter(targdict=[])
        res9 = self.targobj.filter(targdict=[{}])
        res10 = self.targobj.filter(targdict=[{}, {}])

        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)
        assert_same_sub_schema(res6, targ)
        assert_same_sub_schema(res7, targ)
        assert_same_sub_schema(res8, targ)
        assert_same_sub_schema(res9, targ)
        assert_same_sub_schema(res10, targ)

    def test__filter_annotation_single(self):
        targ = [self.sigarrs1[1],
                self.spikes1[1], self.trains1[1],
                self.spikes1[3], self.trains1[3],
                self.sigs1[1], self.irsigs1[1],
                self.sigs1[3], self.irsigs1[3]]

        res0 = self.targobj.filter(j=1)
        res1 = self.targobj.filter({'j': 1})
        res2 = self.targobj.filter(targdict={'j': 1})
        res3 = self.targobj.filter([{'j': 1}])
        res4 = self.targobj.filter(targdict=[{'j': 1}])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_single_annotation_nores(self):
        targ = []

        res0 = self.targobj.filter(j=5)
        res1 = self.targobj.filter({'j': 5})
        res2 = self.targobj.filter(targdict={'j': 5})
        res3 = self.targobj.filter([{'j': 5}])
        res4 = self.targobj.filter(targdict=[{'j': 5}])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_attribute_single(self):
        targ = [self.spikes1[0]]

        name = self.spikes1[0].name
        res0 = self.targobj.filter(name=name)
        res1 = self.targobj.filter({'name': name})
        res2 = self.targobj.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_attribute_single_nores(self):
        targ = []

        name = self.spikes2[0].name
        res0 = self.targobj.filter(name=name)
        res1 = self.targobj.filter({'name': name})
        res2 = self.targobj.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi(self):
        targ = [self.sigarrs1[1],
                self.spikes1[1], self.trains1[1],
                self.spikes1[3], self.trains1[3],
                self.sigs1[1], self.irsigs1[1],
                self.sigs1[3], self.irsigs1[3],
                self.spikes1[0]]

        name = self.spikes1[0].name
        res0 = self.targobj.filter(name=name, j=1)
        res1 = self.targobj.filter({'name': name, 'j': 1})
        res2 = self.targobj.filter(targdict={'name': name, 'j': 1})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_nores(self):
        targ = []

        name0 = self.sigarrs2[0].name
        res0 = self.targobj.filter([{'j': 5}, {}])
        res1 = self.targobj.filter({}, j=0)
        res2 = self.targobj.filter([{}], i=0)
        res3 = self.targobj.filter({'name': name0}, j=1)
        res4 = self.targobj.filter(targdict={'name': name0}, j=1)
        res5 = self.targobj.filter(name=name0, targdict={'j': 1})
        res6 = self.targobj.filter(name=name0, j=5)
        res7 = self.targobj.filter({'name': name0, 'j': 5})
        res8 = self.targobj.filter(targdict={'name': name0, 'j': 5})
        res9 = self.targobj.filter({'name': name0}, j=5)
        res10 = self.targobj.filter(targdict={'name': name0}, j=5)
        res11 = self.targobj.filter(name=name0, targdict={'j': 5})
        res12 = self.targobj.filter({'name': name0}, j=5)
        res13 = self.targobj.filter(targdict={'name': name0}, j=5)
        res14 = self.targobj.filter(name=name0, targdict={'j': 5})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)
        assert_same_sub_schema(res6, targ)
        assert_same_sub_schema(res7, targ)
        assert_same_sub_schema(res8, targ)
        assert_same_sub_schema(res9, targ)
        assert_same_sub_schema(res10, targ)
        assert_same_sub_schema(res11, targ)
        assert_same_sub_schema(res12, targ)
        assert_same_sub_schema(res13, targ)
        assert_same_sub_schema(res14, targ)

    def test__filter_multi_partres_annotation_attribute(self):
        targ = [self.spikes1[0]]

        name = self.spikes1[0].name
        res0 = self.targobj.filter(name=name, j=9)
        res1 = self.targobj.filter({'name': name, 'j': 9})
        res2 = self.targobj.filter(targdict={'name': name, 'j': 9})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_partres_annotation_annotation(self):
        targ = [self.spikes1[0], self.spikes1[2],
                self.sigs1[0], self.sigs1[2]]

        res0 = self.targobj.filter([{'j': 0}, {'i': 0}])
        res1 = self.targobj.filter({'j': 0}, i=0)
        res2 = self.targobj.filter([{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_obj_single(self):
        targ = [self.trains1[1], self.trains1[3]]

        res0 = self.targobj.filter(j=1, objects='SpikeTrain')
        res1 = self.targobj.filter(j=1, objects=SpikeTrain)
        res2 = self.targobj.filter(j=1, objects=['SpikeTrain'])
        res3 = self.targobj.filter(j=1, objects=[SpikeTrain])
        res4 = self.targobj.filter(j=1, objects=[SpikeTrain,
                                                 RecordingChannelGroup])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_single_annotation_obj_multi(self):
        targ = [self.spikes1[1], self.trains1[1],
                self.spikes1[3], self.trains1[3]]
        res0 = self.targobj.filter(j=1, objects=['Spike', SpikeTrain])
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_obj_none(self):
        targ = []

        res0 = self.targobj.filter(j=1, objects=Segment)
        res1 = self.targobj.filter(j=1, objects='Segment')
        res2 = self.targobj.filter(j=1, objects=[])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_norecur(self):
        targ = [self.sigarrs1[1]]
        res0 = self.targobj.filter(j=1, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_norecur(self):
        targ = [self.sigarrs1[0]]
        res0 = self.targobj.filter(name=self.sigarrs1a[0].name,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata(self):
        targ = []
        res0 = self.targobj.filter(j=1, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigarrs1a[0].name, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=1,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigarrs1a[0].name,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        targ = [self.sigarrs1[1],
                self.spikes1[1], self.trains1[1],
                self.spikes1[3], self.trains1[3],
                self.sigs1[1], self.irsigs1[1],
                self.sigs1[3], self.irsigs1[3],
                self.units1[1], self.rchans1[1]]

        res0 = self.targobj.filter(j=1, container=True)

        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_data(self):
        targ = [self.spikes1[0]]
        res0 = self.targobj.filter(name=self.spikes1[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_container(self):
        targ = [self.rchans1[0]]
        res0 = self.targobj.filter(name=self.rchans1a[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container_norecur(self):
        targ = [self.sigarrs1[1], self.units1[1], self.rchans1[1]]

        res0 = self.targobj.filter(j=1, container=True, recursive=False)

        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur(self):
        targ = [self.rchans1[0]]
        res0 = self.targobj.filter(name=self.rchans1a[0].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur_nores(self):
        targ = []
        res0 = self.targobj.filter(name=self.spikes1[0].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container(self):
        targ = [self.units1[1], self.rchans1[1]]
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container(self):
        targ = [self.rchans1[0]]
        res0 = self.targobj.filter(name=self.rchans1[0].name,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_nores(self):
        targ = []
        res0 = self.targobj.filter(name=self.spikes1[0].name,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container_norecur(self):
        targ = [self.units1[1], self.rchans1[1]]
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_norecur(self):
        targ = [self.rchans1[0]]
        res0 = self.targobj.filter(name=self.rchans1[0].name,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_norecur_nores(self):
        targ = []
        res0 = self.targobj.filter(name=self.spikes1[0].name,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filterdata_multi(self):
        data = self.targobj.children_recur

        targ = [self.sigarrs1[1],
                self.spikes1[1], self.trains1[1],
                self.spikes1[3], self.trains1[3],
                self.sigs1[1], self.irsigs1[1],
                self.sigs1[3], self.irsigs1[3],
                self.units1[1], self.rchans1[1],
                self.spikes1[0]]

        name = self.spikes1[0].name
        res0 = filterdata(data, name=name, j=1)
        res1 = filterdata(data, {'name': name, 'j': 1})
        res2 = filterdata(data, targdict={'name': name, 'j': 1})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_nores(self):
        data = self.targobj.children_recur

        targ = []

        name1 = self.sigarrs1a[0].name
        name2 = self.sigarrs2[0].name
        res0 = filterdata(data, [{'j': 0}, {}])
        res1 = filterdata(data, {}, i=0)
        res2 = filterdata(data, [{}], i=0)
        res3 = filterdata(data, name=name1, targdict={'j': 1})
        res4 = filterdata(data, {'name': name1}, j=1)
        res5 = filterdata(data, targdict={'name': name1}, j=1)
        res6 = filterdata(data, name=name2, j=5)
        res7 = filterdata(data, {'name': name2, 'j': 5})
        res8 = filterdata(data, targdict={'name': name2, 'j': 5})
        res9 = filterdata(data, {'name': name2}, j=5)
        res10 = filterdata(data, targdict={'name': name2}, j=5)
        res11 = filterdata(data, name=name2, targdict={'j': 5})
        res12 = filterdata(data, {'name': name1}, j=5)
        res13 = filterdata(data, targdict={'name': name1}, j=5)
        res14 = filterdata(data, name=name1, targdict={'j': 5})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)
        assert_same_sub_schema(res6, targ)
        assert_same_sub_schema(res7, targ)
        assert_same_sub_schema(res8, targ)
        assert_same_sub_schema(res9, targ)
        assert_same_sub_schema(res10, targ)
        assert_same_sub_schema(res11, targ)
        assert_same_sub_schema(res12, targ)
        assert_same_sub_schema(res13, targ)
        assert_same_sub_schema(res14, targ)

    def test__filterdata_multi_partres_annotation_attribute(self):
        data = self.targobj.children_recur

        targ = [self.spikes1[0]]

        name = self.spikes1[0].name
        res0 = filterdata(data, name=name, j=5)
        res1 = filterdata(data, {'name': name, 'j': 5})
        res2 = filterdata(data, targdict={'name': name, 'j': 5})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_partres_annotation_annotation(self):
        data = self.targobj.children_recur

        targ = [self.spikes1[0], self.spikes1[2],
                self.sigs1[0], self.sigs1[2],
                self.units1[0]]

        res0 = filterdata(data, [{'j': 0}, {'i': 0}])
        res1 = filterdata(data, {'j': 0}, i=0)
        res2 = filterdata(data, [{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        res = pretty(self.rcg1)
        ann = get_annotations()
        ann['seed'] = self.seed1
        ann = pretty(ann).replace('\n ', '\n  ')
        targ = ("RecordingChannelGroup with " +
                ("%s units, %s analogsignalarrays, %s recordingchannels\n" %
                 (len(self.units1a),
                  len(self.sigarrs1a),
                  len(self.rchans1a))) +
                ("name: '%s'\ndescription: '%s'\n" % (self.rcg1.name,
                                                      self.rcg1.description)
                 ) +
                ("annotations: %s" % ann))

        self.assertEqual(res, targ)


if __name__ == '__main__':
    unittest.main()
