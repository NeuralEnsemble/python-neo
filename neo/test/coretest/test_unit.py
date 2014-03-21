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

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.unit import Unit
from neo.core.container import filterdata
from neo.core import SpikeTrain, RecordingChannelGroup
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

        for child in res.children_recur:
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

        assert_same_sub_schema(self.spikes1a + self.spikes2,
                               unit1a.spikes)
        assert_same_sub_schema(self.trains1a + self.trains2,
                               unit1a.spiketrains)

    def test__children(self):
        rcg = RecordingChannelGroup(name='rcg1')
        rcg.units = [self.unit1]
        rcg.create_many_to_one_relationship()
        assert_neo_object_is_compliant(self.unit1)
        assert_neo_object_is_compliant(rcg)

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

        self.assertEqual(len(self.unit1._single_children), self.nchildren*2)
        self.assertEqual(len(self.unit1._multi_children), 0)
        self.assertEqual(len(self.unit1.data_children), self.nchildren*2)
        self.assertEqual(len(self.unit1.data_children_recur), self.nchildren*2)
        self.assertEqual(len(self.unit1.container_children), 0)
        self.assertEqual(len(self.unit1.container_children_recur), 0)
        self.assertEqual(len(self.unit1.children), self.nchildren*2)
        self.assertEqual(len(self.unit1.children_recur), self.nchildren*2)

        self.assertEqual(self.unit1._multi_children, ())
        self.assertEqual(self.unit1.container_children, ())
        self.assertEqual(self.unit1.container_children_recur, ())

        assert_same_sub_schema(list(self.unit1._single_children),
                               self.spikes1a+self.trains1a)

        assert_same_sub_schema(list(self.unit1.data_children),
                               self.spikes1a+self.trains1a)

        assert_same_sub_schema(list(self.unit1.data_children_recur),
                               self.spikes1a+self.trains1a)

        assert_same_sub_schema(list(self.unit1.children),
                               self.spikes1a+self.trains1a)

        assert_same_sub_schema(list(self.unit1.children_recur),
                               self.spikes1a+self.trains1a)

        self.assertEqual(len(self.unit1.parents), 1)
        self.assertEqual(self.unit1.parents[0].name, 'rcg1')

    def test__size(self):
        targ = {'spikes': self.nchildren, 'spiketrains': self.nchildren}
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
        targ = [self.spikes1a[1], self.trains1a[1]]

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
        targ = [self.spikes1a[0]]

        name = self.spikes1a[0].name
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
        targ = [self.spikes1a[1], self.trains1a[1], self.spikes1a[0]]

        name = self.spikes1a[0].name
        res0 = self.targobj.filter(name=name, j=1)
        res1 = self.targobj.filter({'name': name, 'j': 1})
        res2 = self.targobj.filter(targdict={'name': name, 'j': 1})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_nores(self):
        targ = []

        name0 = self.spikes2[0].name
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
        targ = [self.spikes1a[0]]

        name = self.spikes1a[0].name
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
        targ = [self.trains1a[1]]

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
        targ = [self.spikes1a[1], self.trains1a[1]]
        res0 = self.targobj.filter(j=1, objects=['Spike', SpikeTrain])
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_obj_none(self):
        targ = []

        res0 = self.targobj.filter(j=1, objects=RecordingChannelGroup)
        res1 = self.targobj.filter(j=1, objects='RecordingChannelGroup')
        res2 = self.targobj.filter(j=1, objects=[])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_norecur(self):
        targ = [self.spikes1a[1], self.trains1a[1]]
        res0 = self.targobj.filter(j=1, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_norecur(self):
        targ = [self.spikes1a[0]]
        res0 = self.targobj.filter(name=self.spikes1a[0].name, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata(self):
        targ = []
        res0 = self.targobj.filter(j=1, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata(self):
        targ = []
        res0 = self.targobj.filter(name=self.spikes1a[0].name, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=1,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.spikes1a[0].name,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        targ = [self.spikes1a[1], self.trains1a[1]]
        res0 = self.targobj.filter(j=1, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container(self):
        targ = [self.spikes1a[0]]
        res0 = self.targobj.filter(name=self.spikes1a[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container_norecur(self):
        targ = [self.spikes1a[1], self.trains1a[1]]
        res0 = self.targobj.filter(j=1, container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur(self):
        targ = [self.spikes1a[0]]
        res0 = self.targobj.filter(name=self.spikes1a[0].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container(self):
        targ = []
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container(self):
        targ = []
        res0 = self.targobj.filter(name=self.spikes1a[0].name,
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
        res0 = self.targobj.filter(name=self.spikes1a[0].name,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filterdata_multi(self):
        data = self.targobj.children_recur

        targ = [self.spikes1a[1], self.trains1a[1], self.spikes1a[0]]

        name = self.spikes1a[0].name
        res0 = filterdata(data, name=name, j=1)
        res1 = filterdata(data, {'name': name, 'j': 1})
        res2 = filterdata(data, targdict={'name': name, 'j': 1})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_nores(self):
        data = self.targobj.children_recur

        targ = []

        name1 = self.spikes1a[0].name
        name2 = self.spikes2[0].name
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

        targ = [self.spikes1a[0]]

        name = self.spikes1a[0].name
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
        res = pretty(self.unit1)
        ann = get_annotations()
        ann['seed'] = self.seed1
        ann = pretty(ann).replace('\n ', '\n  ')
        targ = ("Unit with " +
                ("%s spikes, %s spiketrains\n" %
                 (len(self.spikes1a), len(self.trains1a))) +
                ("name: '%s'\ndescription: '%s'\n" % (self.unit1.name,
                                                      self.unit1.description)
                 ) +
                ("annotations: %s" % ann))

        self.assertEqual(res, targ)


if __name__ == "__main__":
    unittest.main()
