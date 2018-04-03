# -*- coding: utf-8 -*-
"""
Tests of the neo.core.channelindex.ChannelIndex class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.channelindex import ChannelIndex
from neo.core.container import filterdata
from neo.core import Block, Segment, SpikeTrain, AnalogSignal
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

    # def test__get_fake_values(self):
    #     self.annotations['seed'] = 0
    #     channel_indexes = get_fake_value('channel_indexes', np.ndarray, seed=0,
    #                                      dim=1, dtype='i')
    #     channel_names = get_fake_value('channel_names', np.ndarray, seed=1,
    #                                    dim=1, dtype=np.dtype('S'))
    #     name = get_fake_value('name', str, seed=3, obj=ChannelIndex)
    #     description = get_fake_value('description', str, seed=4,
    #                                  obj='ChannelIndex')
    #     file_origin = get_fake_value('file_origin', str)
    #     #coordinates = get_fake_value('coordinates', np.ndarray, seed=2, dim=2, dtype='f')
    #     attrs1 = {'name': name,
    #               'description': description,
    #               'file_origin': file_origin,}
    #     #          'coordinates': coordinates}
    #     attrs2 = attrs1.copy()
    #     attrs2.update(self.annotations)
    #
    #     res11 = get_fake_values(ChannelIndex, annotate=False, seed=0)
    #     res12 = get_fake_values('ChannelIndex',
    #                             annotate=False, seed=0)
    #     res21 = get_fake_values(ChannelIndex, annotate=True, seed=0)
    #     res22 = get_fake_values('ChannelIndex', annotate=True, seed=0)
    #
    #     assert_arrays_equal(res11.pop('channel_indexes'), channel_indexes)
    #     assert_arrays_equal(res12.pop('channel_indexes'), channel_indexes)
    #     assert_arrays_equal(res21.pop('channel_indexes'), channel_indexes)
    #     assert_arrays_equal(res22.pop('channel_indexes'), channel_indexes)
    #
    #     assert_arrays_equal(res11.pop('channel_names'), channel_names)
    #     assert_arrays_equal(res12.pop('channel_names'), channel_names)
    #     assert_arrays_equal(res21.pop('channel_names'), channel_names)
    #     assert_arrays_equal(res22.pop('channel_names'), channel_names)
    #
    #     for obj in (res11, res12, res21, res22):
    #         obj.pop("coordinates")
    #     self.assertEqual(res11, attrs1)
    #     self.assertEqual(res12, attrs1)
    #     self.assertEqual(res21, attrs2)
    #     self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'ChannelIndex'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, ChannelIndex))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        for child in res.children_recur:
            del child.annotations['i']
            del child.annotations['j']

        self.assertEqual(len(res.units), 1)
        unit = res.units[0]
        self.assertEqual(unit.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 1)
        self.assertEqual(res.analogsignals[0].annotations,
                         self.annotations)

        self.assertEqual(len(unit.spiketrains), 1)
        self.assertEqual(unit.spiketrains[0].annotations,
                         self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = ChannelIndex
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, ChannelIndex))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.units), 0)
        self.assertEqual(len(res.analogsignals), 0)


class TestChannelIndex(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        self.seed1 = 0
        self.seed2 = 10000
        self.chx1 = fake_neo(ChannelIndex,
                             seed=self.seed1, n=self.nchildren)
        self.chx2 = fake_neo(ChannelIndex,
                             seed=self.seed2, n=self.nchildren)
        self.targobj = self.chx1

        self.units1 = self.chx1.units
        self.units2 = self.chx2.units
        self.sigarrs1 = self.chx1.analogsignals
        self.sigarrs2 = self.chx2.analogsignals
        self.irrsig1 = self.chx1.irregularlysampledsignals
        self.irrsig2 = self.chx2.irregularlysampledsignals

        self.units1a = clone_object(self.units1)
        self.sigarrs1a = clone_object(self.sigarrs1, n=2)
        self.irrsig1a = clone_object(self.irrsig1, n=2)

        self.trains1 = [[train for train in unit.spiketrains]
                        for unit in self.units1]
        self.trains2 = [[train for train in unit.spiketrains]
                        for unit in self.units2]

        self.trains1 = sum(self.trains1, [])
        self.trains2 = sum(self.trains2, [])

    def test__channelindex__init_defaults(self):
        chx = ChannelIndex(index=np.array([1]))
        assert_neo_object_is_compliant(chx)
        self.assertEqual(chx.name, None)
        self.assertEqual(chx.file_origin, None)
        self.assertEqual(chx.analogsignals, [])
        assert_arrays_equal(chx.channel_names, np.array([], dtype='S'))
        assert_arrays_equal(chx.index, np.array([1]))

    def test_channelindex__init(self):
        chx = ChannelIndex(file_origin='temp.dat',
                           index=np.array([1]))
        assert_neo_object_is_compliant(chx)
        self.assertEqual(chx.file_origin, 'temp.dat')
        self.assertEqual(chx.name, None)
        self.assertEqual(chx.analogsignals, [])
        assert_arrays_equal(chx.channel_names, np.array([], dtype='S'))
        assert_arrays_equal(chx.index, np.array([1]))

    def check_creation(self, chx):
        assert_neo_object_is_compliant(chx)

        seed = chx.annotations['seed']

        # for i, unit in enumerate(chx.units):
        #     for sigarr in chx.analogsignals:
        #         self.assertEqual(unit.channel_indexes[0],
        #                          sigarr.channel_index[i])

        targ2 = get_fake_value('name', str, seed=seed + 4,
                               obj=ChannelIndex)
        self.assertEqual(chx.name, targ2)

        targ3 = get_fake_value('description', str,
                               seed=seed + 5, obj=ChannelIndex)
        self.assertEqual(chx.description, targ3)

        targ4 = get_fake_value('file_origin', str)
        self.assertEqual(chx.file_origin, targ4)

        targ5 = get_annotations()
        targ5['seed'] = seed
        self.assertEqual(chx.annotations, targ5)

        self.assertTrue(hasattr(chx, 'units'))
        self.assertTrue(hasattr(chx, 'analogsignals'))

        self.assertEqual(len(chx.units), self.nchildren)
        self.assertEqual(len(chx.analogsignals), self.nchildren)

    def test__creation(self):
        self.check_creation(self.chx1)
        self.check_creation(self.chx2)

    def test__merge(self):
        chx1a = fake_neo(ChannelIndex,
                         seed=self.seed1, n=self.nchildren)
        assert_same_sub_schema(self.chx1, chx1a)
        chx1a.annotate(seed=self.seed2)
        chx1a.analogsignals.append(self.sigarrs2[0])
        chx1a.merge(self.chx2)
        self.check_creation(self.chx2)

        assert_same_sub_schema(self.sigarrs1a + self.sigarrs2,
                               chx1a.analogsignals,
                               exclude=['channel_index'])
        assert_same_sub_schema(self.units1a + self.units2,
                               chx1a.units)

    def test__children(self):
        blk = Block(name='block1')
        blk.channel_indexes = [self.chx1]
        blk.create_many_to_one_relationship()

        self.assertEqual(self.chx1._container_child_objects, ('Unit',))
        self.assertEqual(self.chx1._data_child_objects,
                         ('AnalogSignal', 'IrregularlySampledSignal'))
        self.assertEqual(self.chx1._single_parent_objects, ('Block',))
        self.assertEqual(self.chx1._multi_child_objects, tuple())
        self.assertEqual(self.chx1._multi_parent_objects, ())
        self.assertEqual(self.chx1._child_properties, ())

        self.assertEqual(self.chx1._single_child_objects,
                         ('Unit', 'AnalogSignal', 'IrregularlySampledSignal'))

        self.assertEqual(self.chx1._container_child_containers, ('units',))
        self.assertEqual(self.chx1._data_child_containers,
                         ('analogsignals', 'irregularlysampledsignals'))
        self.assertEqual(self.chx1._single_child_containers,
                         ('units', 'analogsignals', 'irregularlysampledsignals'))
        self.assertEqual(self.chx1._single_parent_containers, ('block',))
        self.assertEqual(self.chx1._multi_child_containers,
                         tuple())
        self.assertEqual(self.chx1._multi_parent_containers, ())

        self.assertEqual(self.chx1._child_objects,
                         ('Unit', 'AnalogSignal', 'IrregularlySampledSignal'))
        self.assertEqual(self.chx1._child_containers,
                         ('units', 'analogsignals', 'irregularlysampledsignals'))
        self.assertEqual(self.chx1._parent_objects, ('Block',))
        self.assertEqual(self.chx1._parent_containers, ('block',))

        self.assertEqual(len(self.chx1._single_children), 3 * self.nchildren)
        self.assertEqual(len(self.chx1._multi_children), 0)
        self.assertEqual(len(self.chx1.data_children), 2 * self.nchildren)
        self.assertEqual(len(self.chx1.data_children_recur),
                         2 * self.nchildren + 1 * self.nchildren ** 2)
        self.assertEqual(len(self.chx1.container_children), 1 * self.nchildren)
        self.assertEqual(len(self.chx1.container_children_recur),
                         1 * self.nchildren)
        self.assertEqual(len(self.chx1.children), 3 * self.nchildren)
        self.assertEqual(len(self.chx1.children_recur),
                         3 * self.nchildren + 1 * self.nchildren ** 2)

        assert_same_sub_schema(list(self.chx1._single_children),
                               self.units1a + self.sigarrs1a + self.irrsig1a,
                               exclude=['channel_index'])

        assert_same_sub_schema(list(self.chx1.data_children), self.sigarrs1a + self.irrsig1a,
                               exclude=['channel_index'])
        assert_same_sub_schema(list(self.chx1.data_children_recur),
                               self.sigarrs1a + self.irrsig1a +
                               self.trains1[:2] + self.trains1[2:],
                               exclude=['channel_index'])

        assert_same_sub_schema(list(self.chx1.children),
                               self.sigarrs1a + self.irrsig1a + self.units1a,
                               exclude=['channel_index'])
        assert_same_sub_schema(list(self.chx1.children_recur),
                               self.sigarrs1a + self.irrsig1a +
                               self.trains1[:2] + self.trains1[2:] +
                               self.units1a,
                               exclude=['channel_index'])

        self.assertEqual(len(self.chx1.parents), 1)
        self.assertEqual(self.chx1.parents[0].name, 'block1')

    def test__size(self):
        targ = {'analogsignals': self.nchildren,
                'units': self.nchildren,
                'irregularlysampledsignals': self.nchildren}
        self.assertEqual(self.targobj.size, targ)

    def test__filter_none(self):
        targ = []
        # collecting all data objects in target block
        targ.extend(self.targobj.analogsignals)
        targ.extend(self.targobj.irregularlysampledsignals)
        for unit in self.targobj.units:
            targ.extend(unit.spiketrains)

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
        targ = [self.sigarrs1[1], self.irrsig1[1],
                self.trains1[1], self.trains1[3],
                ]

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
        targ = [self.trains1[0]]

        name = self.trains1[0].name
        res0 = self.targobj.filter(name=name)
        res1 = self.targobj.filter({'name': name})
        res2 = self.targobj.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_attribute_single_nores(self):
        targ = []

        name = self.trains2[0].name
        res0 = self.targobj.filter(name=name)
        res1 = self.targobj.filter({'name': name})
        res2 = self.targobj.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi(self):
        targ = [self.sigarrs1[1], self.irrsig1[1],
                self.trains1[1], self.trains1[3],
                self.trains1[0]]

        name = self.trains1[0].name
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
        res1 = self.targobj.filter({}, j=5)
        res2 = self.targobj.filter([{}], i=5)
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
        targ = [self.trains1[0]]

        name = self.trains1[0].name
        res0 = self.targobj.filter(name=name, j=9)
        res1 = self.targobj.filter({'name': name, 'j': 9})
        res2 = self.targobj.filter(targdict={'name': name, 'j': 9})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_partres_annotation_annotation(self):
        targ = [self.trains1[0], self.trains1[2]]

        res0 = self.targobj.filter([{'j': 0}, {'i': 0}])
        res1 = self.targobj.filter({'j': 0}, i=0)
        res2 = self.targobj.filter([{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_no_annotation_but_object(self):
        targ = []
        for unit in self.targobj.units:
            targ.extend(unit.spiketrains)
        res = self.targobj.filter(objects=SpikeTrain)
        assert_same_sub_schema(res, targ)

        targ = self.targobj.analogsignals
        res = self.targobj.filter(objects=AnalogSignal)
        assert_same_sub_schema(res, targ)

        targ = []
        targ.extend(self.targobj.analogsignals)
        for unit in self.targobj.units:
            targ.extend(unit.spiketrains)
        res = self.targobj.filter(objects=[AnalogSignal, SpikeTrain])
        assert_same_sub_schema(res, targ)

    def test__filter_single_annotation_obj_single(self):
        targ = [self.trains1[1], self.trains1[3]]

        res0 = self.targobj.filter(j=1, objects='SpikeTrain')
        res1 = self.targobj.filter(j=1, objects=SpikeTrain)
        res2 = self.targobj.filter(j=1, objects=['SpikeTrain'])
        res3 = self.targobj.filter(j=1, objects=[SpikeTrain])
        res4 = self.targobj.filter(j=1, objects=[SpikeTrain,
                                                 ChannelIndex])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_single_annotation_obj_none(self):
        targ = []

        res0 = self.targobj.filter(j=1, objects=Segment)
        res1 = self.targobj.filter(j=1, objects='Segment')
        res2 = self.targobj.filter(j=1, objects=[])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_norecur(self):
        targ = [self.sigarrs1[1], self.irrsig1[1]]
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
        targ = [self.sigarrs1[1], self.irrsig1[1],
                self.trains1[1], self.trains1[3],
                self.units1[1]]

        res0 = self.targobj.filter(j=1, container=True)

        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_data(self):
        targ = [self.trains1[0]]
        res0 = self.targobj.filter(name=self.trains1[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container_norecur(self):
        targ = [self.sigarrs1[1], self.irrsig1[1], self.units1[1]]

        res0 = self.targobj.filter(j=1, container=True, recursive=False)

        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur_nores(self):
        targ = []
        res0 = self.targobj.filter(name=self.trains1[0].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container(self):
        targ = [self.units1[1]]
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_nores(self):
        targ = []
        res0 = self.targobj.filter(name=self.trains1[0].name,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container_norecur(self):
        targ = [self.units1[1]]
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_norecur_nores(self):
        targ = []
        res0 = self.targobj.filter(name=self.trains1[0].name,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filterdata_multi(self):
        data = self.targobj.children_recur

        targ = [self.sigarrs1[1], self.irrsig1[1],
                self.trains1[1], self.trains1[3],
                self.units1[1],
                self.trains1[0]]

        name = self.trains1[0].name
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
        res0 = filterdata(data, [{'j': 5}, {}])
        res1 = filterdata(data, {}, i=5)
        res2 = filterdata(data, [{}], i=5)
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

        targ = [self.trains1[0]]

        name = self.trains1[0].name
        res0 = filterdata(data, name=name, j=5)
        res1 = filterdata(data, {'name': name, 'j': 5})
        res2 = filterdata(data, targdict={'name': name, 'j': 5})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_partres_annotation_annotation(self):
        data = self.targobj.children_recur

        targ = [self.trains1[0], self.trains1[2],
                self.units1[0]]

        res0 = filterdata(data, [{'j': 0}, {'i': 0}])
        res1 = filterdata(data, {'j': 0}, i=0)
        res2 = filterdata(data, [{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

        # @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
        # def test__pretty(self):
        #     res = pretty(self.chx1)
        #     ann = get_annotations()
        #     ann['seed'] = self.seed1
        #     ann = pretty(ann).replace('\n ', '\n  ')
        #     targ = ("ChannelIndex with " +
        #             ("%s units, %s analogsignals, %s irregularlysampledsignals\n" %
        #              (len(self.units1a),
        #               len(self.irrsig1a),
        #               len(self.sigarrs1a),
        #               )) +
        #             ("name: '%s'\ndescription: '%s'\n" % (self.chx1.name,
        #                                                   self.chx1.description)
        #              ) +
        #             ("annotations: %s" % ann))
        #
        #     self.assertEqual(res, targ)


if __name__ == '__main__':
    unittest.main()
