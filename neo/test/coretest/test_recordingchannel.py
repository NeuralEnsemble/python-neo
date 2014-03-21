# -*- coding: utf-8 -*-
"""
Tests of the neo.core.recordingchannel.RecordingChannel class
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

from neo.core.recordingchannel import RecordingChannel
from neo.core.container import filterdata
from neo.core import IrregularlySampledSignal, RecordingChannelGroup, Unit
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
        index = get_fake_value('index', int, seed=0)
        coordinate = get_fake_value('coordinate', pq.Quantity, seed=1, dim=1)
        name = get_fake_value('name', str, seed=2, obj=RecordingChannel)
        description = get_fake_value('description', str, seed=3,
                                     obj='RecordingChannel')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'index': index,
                  'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(RecordingChannel, annotate=False, seed=0)
        res12 = get_fake_values('RecordingChannel', annotate=False, seed=0)
        res21 = get_fake_values(RecordingChannel, annotate=True, seed=0)
        res22 = get_fake_values('RecordingChannel', annotate=True, seed=0)

        assert_arrays_equal(res11.pop('coordinate'), coordinate)
        assert_arrays_equal(res12.pop('coordinate'), coordinate)
        assert_arrays_equal(res21.pop('coordinate'), coordinate)
        assert_arrays_equal(res22.pop('coordinate'), coordinate)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = RecordingChannel
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannel))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 1)
        self.assertEqual(len(res.irregularlysampledsignals), 1)

        for child in res.children_recur:
            del child.annotations['i']
            del child.annotations['j']
        self.assertEqual(res.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(res.irregularlysampledsignals[0].annotations,
                         self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'RecordingChannel'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannel))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)


class TestRecordingChannel(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        self.seed1 = 0
        self.seed2 = 10000
        self.rchan1 = fake_neo(RecordingChannel,
                               seed=self.seed1, n=self.nchildren)
        self.rchan2 = fake_neo(RecordingChannel,
                               seed=self.seed2, n=self.nchildren)
        self.targobj = self.rchan1

        self.sigs1 = self.rchan1.analogsignals
        self.sigs2 = self.rchan2.analogsignals
        self.irsigs1 = self.rchan1.irregularlysampledsignals
        self.irsigs2 = self.rchan2.irregularlysampledsignals

        self.sigs1a = clone_object(self.sigs1)
        self.irsigs1a = clone_object(self.irsigs1)

    def check_creation(self, rchan):
        assert_neo_object_is_compliant(rchan)

        seed = rchan.annotations['seed']

        targ0 = get_fake_value('index', int, seed=seed+0, obj=RecordingChannel)
        self.assertEqual(rchan.index, targ0)

        targ1 = get_fake_value('coordinate', pq.Quantity, dim=1, seed=seed+1)
        assert_arrays_equal(rchan.coordinate, targ1)

        targ2 = get_fake_value('name', str, seed=seed+2, obj=RecordingChannel)
        self.assertEqual(rchan.name, targ2)

        targ3 = get_fake_value('description', str,
                               seed=seed+3, obj=RecordingChannel)
        self.assertEqual(rchan.description, targ3)

        targ4 = get_fake_value('file_origin', str)
        self.assertEqual(rchan.file_origin, targ4)

        targ5 = get_annotations()
        targ5['seed'] = seed
        self.assertEqual(rchan.annotations, targ5)

        self.assertTrue(hasattr(rchan, 'analogsignals'))
        self.assertTrue(hasattr(rchan, 'irregularlysampledsignals'))

        self.assertEqual(len(rchan.analogsignals), self.nchildren)
        self.assertEqual(len(rchan.irregularlysampledsignals), self.nchildren)

    def test__creation(self):
        self.check_creation(self.rchan1)
        self.check_creation(self.rchan2)

    def test__merge(self):
        rchan1a = fake_neo(RecordingChannel, seed=self.seed1, n=self.nchildren)
        assert_same_sub_schema(self.rchan1, rchan1a)
        rchan1a.annotate(seed=self.seed2)
        rchan1a.analogsignals.append(self.sigs2[0])
        rchan1a.merge(self.rchan2)
        self.check_creation(self.rchan2)

        assert_same_sub_schema(self.sigs1a + self.sigs2,
                               rchan1a.analogsignals)
        assert_same_sub_schema(self.irsigs1a + self.irsigs2,
                               rchan1a.irregularlysampledsignals)

    def test__children(self):
        rcg1 = RecordingChannelGroup(name='rcg1')
        rcg2 = RecordingChannelGroup(name='rcg2')
        rcg1.recordingchannels = [self.rchan1]
        rcg2.recordingchannels = [self.rchan1]
        rcg2.create_relationship()
        rcg1.create_relationship()
        assert_neo_object_is_compliant(self.rchan1)
        assert_neo_object_is_compliant(rcg1)
        assert_neo_object_is_compliant(rcg2)

        self.assertEqual(self.rchan1._container_child_objects, ())
        self.assertEqual(self.rchan1._data_child_objects,
                         ('AnalogSignal', 'IrregularlySampledSignal'))
        self.assertEqual(self.rchan1._single_parent_objects, ())
        self.assertEqual(self.rchan1._multi_child_objects, ())
        self.assertEqual(self.rchan1._multi_parent_objects,
                         ('RecordingChannelGroup',))
        self.assertEqual(self.rchan1._child_properties, ())

        self.assertEqual(self.rchan1._single_child_objects,
                         ('AnalogSignal', 'IrregularlySampledSignal'))

        self.assertEqual(self.rchan1._container_child_containers, ())
        self.assertEqual(self.rchan1._data_child_containers,
                         ('analogsignals', 'irregularlysampledsignals',))
        self.assertEqual(self.rchan1._single_child_containers,
                         ('analogsignals', 'irregularlysampledsignals',))
        self.assertEqual(self.rchan1._single_parent_containers, ())
        self.assertEqual(self.rchan1._multi_child_containers, ())
        self.assertEqual(self.rchan1._multi_parent_containers,
                         ('recordingchannelgroups',))

        self.assertEqual(self.rchan1._child_objects,
                         ('AnalogSignal', 'IrregularlySampledSignal'))
        self.assertEqual(self.rchan1._child_containers,
                         ('analogsignals', 'irregularlysampledsignals',))
        self.assertEqual(self.rchan1._parent_objects,
                         ('RecordingChannelGroup',))
        self.assertEqual(self.rchan1._parent_containers,
                         ('recordingchannelgroups',))

        self.assertEqual(len(self.rchan1._single_children), self.nchildren*2)
        self.assertEqual(len(self.rchan1._multi_children), 0)
        self.assertEqual(len(self.rchan1.data_children), self.nchildren*2)
        self.assertEqual(len(self.rchan1.data_children_recur),
                         self.nchildren*2)
        self.assertEqual(len(self.rchan1.container_children), 0)
        self.assertEqual(len(self.rchan1.container_children_recur), 0)
        self.assertEqual(len(self.rchan1.children), self.nchildren*2)
        self.assertEqual(len(self.rchan1.children_recur), self.nchildren*2)

        self.assertEqual(self.rchan1._multi_children, ())
        self.assertEqual(self.rchan1.container_children, ())
        self.assertEqual(self.rchan1.container_children_recur, ())

        assert_same_sub_schema(list(self.rchan1._single_children),
                               self.sigs1a+self.irsigs1a)

        assert_same_sub_schema(list(self.rchan1.data_children),
                               self.sigs1a+self.irsigs1a)

        assert_same_sub_schema(list(self.rchan1.data_children_recur),
                               self.sigs1a+self.irsigs1a)

        assert_same_sub_schema(list(self.rchan1.children),
                               self.sigs1a+self.irsigs1a)

        assert_same_sub_schema(list(self.rchan1.children_recur),
                               self.sigs1a+self.irsigs1a)

        self.assertEqual(len(self.rchan1.parents), 2)
        self.assertEqual(self.rchan1.parents[0].name, 'rcg2')
        self.assertEqual(self.rchan1.parents[1].name, 'rcg1')

    def test__size(self):
        targ = {'analogsignals': self.nchildren,
                'irregularlysampledsignals': self.nchildren}
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
        targ = [self.sigs1a[1], self.irsigs1a[1]]

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
        targ = [self.sigs1a[0]]

        name = self.sigs1a[0].name
        res0 = self.targobj.filter(name=name)
        res1 = self.targobj.filter({'name': name})
        res2 = self.targobj.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_attribute_single_nores(self):
        targ = []

        name = self.sigs2[0].name
        res0 = self.targobj.filter(name=name)
        res1 = self.targobj.filter({'name': name})
        res2 = self.targobj.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi(self):
        targ = [self.sigs1a[1], self.irsigs1a[1], self.sigs1a[0]]

        name = self.sigs1a[0].name
        res0 = self.targobj.filter(name=name, j=1)
        res1 = self.targobj.filter({'name': name, 'j': 1})
        res2 = self.targobj.filter(targdict={'name': name, 'j': 1})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_nores(self):
        targ = []

        name0 = self.sigs2[0].name
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

    def test__filter_multi_partres(self):
        targ = [self.sigs1a[0]]

        name = self.sigs1a[0].name
        res0 = self.targobj.filter(name=name, j=5)
        res1 = self.targobj.filter({'name': name, 'j': 5})
        res2 = self.targobj.filter(targdict={'name': name, 'j': 5})
        res3 = self.targobj.filter([{'j': 0}, {'i': 0}])
        res4 = self.targobj.filter({'j': 0}, i=0)
        res5 = self.targobj.filter([{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)

    def test__filter_single_annotation_obj_single(self):
        targ = [self.irsigs1a[1]]

        res0 = self.targobj.filter(j=1, objects='IrregularlySampledSignal')
        res1 = self.targobj.filter(j=1, objects=IrregularlySampledSignal)
        res2 = self.targobj.filter(j=1, objects=['IrregularlySampledSignal'])
        res3 = self.targobj.filter(j=1, objects=[IrregularlySampledSignal])
        res4 = self.targobj.filter(j=1, objects=[IrregularlySampledSignal,
                                                 Unit])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_single_annotation_obj_multi(self):
        targ = [self.sigs1a[1], self.irsigs1a[1]]
        res0 = self.targobj.filter(j=1, objects=['AnalogSignal',
                                                 IrregularlySampledSignal])
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_obj_none(self):
        targ = []

        res0 = self.targobj.filter(j=1, objects=Unit)
        res1 = self.targobj.filter(j=1, objects='Unit')
        res2 = self.targobj.filter(j=1, objects=[])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_norecur(self):
        targ = [self.sigs1a[1], self.irsigs1a[1]]
        res0 = self.targobj.filter(j=1, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_norecur(self):
        targ = [self.sigs1a[0]]
        res0 = self.targobj.filter(name=self.sigs1a[0].name, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata(self):
        targ = []
        res0 = self.targobj.filter(j=1, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigs1a[0].name, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=1,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigs1a[0].name,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        targ = [self.sigs1a[1], self.irsigs1a[1]]
        res0 = self.targobj.filter(j=1, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container(self):
        targ = [self.sigs1a[0]]
        res0 = self.targobj.filter(name=self.sigs1a[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container_norecur(self):
        targ = [self.sigs1a[1], self.irsigs1a[1]]
        res0 = self.targobj.filter(j=1, container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur(self):
        targ = [self.sigs1a[0]]
        res0 = self.targobj.filter(name=self.sigs1a[0].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container(self):
        targ = []
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigs1a[0].name,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigs1a[0].name,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filterdata_multi(self):
        data = self.targobj.children_recur

        targ = [self.sigs1a[1], self.irsigs1a[1], self.sigs1a[0]]

        name = self.sigs1a[0].name
        res0 = filterdata(data, name=name, j=1)
        res1 = filterdata(data, {'name': name, 'j': 1})
        res2 = filterdata(data, targdict={'name': name, 'j': 1})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_nores(self):
        data = self.targobj.children_recur

        targ = []

        name1 = self.sigs1a[0].name
        name2 = self.sigs2[0].name
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

    def test__filterdata_multi_partres(self):
        data = self.targobj.children_recur

        targ = [self.sigs1a[0]]

        name = self.sigs1a[0].name
        res0 = filterdata(data, name=name, j=5)
        res1 = filterdata(data, {'name': name, 'j': 5})
        res2 = filterdata(data, targdict={'name': name, 'j': 5})
        res3 = filterdata(data, [{'j': 0}, {'i': 0}])
        res4 = filterdata(data, {'j': 0}, i=0)
        res5 = filterdata(data, [{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        ann = get_annotations()
        ann['seed'] = self.seed1
        ann = pretty(ann).replace('\n ', '\n  ')
        res = pretty(self.rchan1)
        targ = ("RecordingChannel with " +
                ("%s analogsignals, %s irregularlysampledsignals\n" %
                 (len(self.sigs1), len(self.irsigs1))) +
                ("name: '%s'\ndescription: '%s'\n" % (self.rchan1.name,
                                                      self.rchan1.description)
                 ) +
                ("annotations: %s" % ann))

        self.assertEqual(res, targ)


if __name__ == "__main__":
    unittest.main()
