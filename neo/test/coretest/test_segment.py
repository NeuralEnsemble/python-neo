# -*- coding: utf-8 -*-
"""
Tests of the neo.core.segment.Segment class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from datetime import datetime

import unittest

import numpy as np
import quantities as pq

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.segment import Segment
from neo.core import (AnalogSignal, Block,
                      Epoch, ChannelIndex, SpikeTrain, Unit)
from neo.core.container import filterdata
from neo.test.tools import (assert_neo_object_is_compliant,
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
        file_datetime = get_fake_value('file_datetime', datetime, seed=0)
        rec_datetime = get_fake_value('rec_datetime', datetime, seed=1)
        index = get_fake_value('index', int, seed=2)
        name = get_fake_value('name', str, seed=3, obj=Segment)
        description = get_fake_value('description', str, seed=4, obj='Segment')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'file_datetime': file_datetime,
                  'rec_datetime': rec_datetime,
                  'index': index,
                  'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Segment, annotate=False, seed=0)
        res12 = get_fake_values('Segment', annotate=False, seed=0)
        res21 = get_fake_values(Segment, annotate=True, seed=0)
        res22 = get_fake_values('Segment', annotate=True, seed=0)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = Segment
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 1)
        self.assertEqual(len(res.irregularlysampledsignals), 1)
        self.assertEqual(len(res.spiketrains), 1)
        self.assertEqual(len(res.events), 1)
        self.assertEqual(len(res.epochs), 1)
        for child in res.children:
            del child.annotations['i']
            del child.annotations['j']

        self.assertEqual(res.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(res.irregularlysampledsignals[0].annotations,
                         self.annotations)
        self.assertEqual(res.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(res.events[0].annotations,
                         self.annotations)
        self.assertEqual(res.epochs[0].annotations,
                         self.annotations)

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
        obj_type = 'Segment'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)


class TestSegment(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        blk = fake_neo(Block, seed=0, n=self.nchildren)
        self.unit1, self.unit2, self.unit3, self.unit4 = blk.list_units
        self.seg1, self.seg2 = blk.segments
        self.targobj = self.seg1
        self.seed1 = self.seg1.annotations['seed']
        self.seed2 = self.seg2.annotations['seed']

        del self.seg1.annotations['i']
        del self.seg2.annotations['i']
        del self.seg1.annotations['j']
        del self.seg2.annotations['j']

        self.sigarrs1 = self.seg1.analogsignals
        self.sigarrs2 = self.seg2.analogsignals
        self.irsigs1 = self.seg1.irregularlysampledsignals
        self.irsigs2 = self.seg2.irregularlysampledsignals

        self.trains1 = self.seg1.spiketrains
        self.trains2 = self.seg2.spiketrains

        self.epcs1 = self.seg1.epochs
        self.epcs2 = self.seg2.epochs
        self.evts1 = self.seg1.events
        self.evts2 = self.seg2.events

        self.sigarrs1a = clone_object(self.sigarrs1, n=2)
        self.irsigs1a = clone_object(self.irsigs1)

        self.trains1a = clone_object(self.trains1)

        self.epcs1a = clone_object(self.epcs1)
        self.evts1a = clone_object(self.evts1)

    def test_init(self):
        seg = Segment(name='a segment', index=3)
        assert_neo_object_is_compliant(seg)
        self.assertEqual(seg.name, 'a segment')
        self.assertEqual(seg.file_origin, None)
        self.assertEqual(seg.index, 3)

    def check_creation(self, seg):
        assert_neo_object_is_compliant(seg)

        seed = seg.annotations['seed']

        targ0 = get_fake_value('file_datetime', datetime, seed=seed + 0)
        self.assertEqual(seg.file_datetime, targ0)

        targ1 = get_fake_value('rec_datetime', datetime, seed=seed + 1)
        self.assertEqual(seg.rec_datetime, targ1)

        targ2 = get_fake_value('index', int, seed=seed + 2)
        self.assertEqual(seg.index, targ2)

        targ3 = get_fake_value('name', str, seed=seed + 3, obj=Segment)
        self.assertEqual(seg.name, targ3)

        targ4 = get_fake_value('description', str,
                               seed=seed + 4, obj=Segment)
        self.assertEqual(seg.description, targ4)

        targ5 = get_fake_value('file_origin', str)
        self.assertEqual(seg.file_origin, targ5)

        targ6 = get_annotations()
        targ6['seed'] = seed
        self.assertEqual(seg.annotations, targ6)

        self.assertTrue(hasattr(seg, 'analogsignals'))
        self.assertTrue(hasattr(seg, 'irregularlysampledsignals'))

        self.assertTrue(hasattr(seg, 'epochs'))
        self.assertTrue(hasattr(seg, 'events'))

        self.assertTrue(hasattr(seg, 'spiketrains'))

        self.assertEqual(len(seg.analogsignals), self.nchildren)
        self.assertEqual(len(seg.irregularlysampledsignals), self.nchildren)

        self.assertEqual(len(seg.epochs), self.nchildren)
        self.assertEqual(len(seg.events), self.nchildren)

        self.assertEqual(len(seg.spiketrains), self.nchildren ** 2)

    def test__creation(self):
        self.check_creation(self.seg1)
        self.check_creation(self.seg2)

    def test_times(self):

        for seg in [self.seg1, self.seg2]:
            # calculate target values for t_start and t_stop
            t_starts, t_stops = [], []
            for children in [seg.analogsignals,
                             seg.epochs,
                             seg.events,
                             seg.irregularlysampledsignals,
                             seg.spiketrains]:
                for child in children:
                    if hasattr(child, 't_start'):
                        t_starts.append(child.t_start)
                    if hasattr(child, 't_stop'):
                        t_stops.append(child.t_stop)
                    if hasattr(child, 'time'):
                        t_starts.append(child.time)
                        t_stops.append(child.time)
                    if hasattr(child, 'times'):
                        t_starts.append(child.times[0])
                        t_stops.append(child.times[-1])
            targ_t_start = min(t_starts)
            targ_t_stop = max(t_stops)

            self.assertEqual(seg.t_start, targ_t_start)
            self.assertEqual(seg.t_stop, targ_t_stop)

    def test__merge(self):
        seg1a = fake_neo(Block, seed=self.seed1, n=self.nchildren).segments[0]
        assert_same_sub_schema(self.seg1, seg1a)
        seg1a.epochs.append(self.epcs2[0])
        seg1a.annotate(seed=self.seed2)
        seg1a.merge(self.seg2)
        self.check_creation(self.seg2)

        assert_same_sub_schema(self.sigarrs1a + self.sigarrs2,
                               seg1a.analogsignals)
        assert_same_sub_schema(self.irsigs1a + self.irsigs2,
                               seg1a.irregularlysampledsignals)

        assert_same_sub_schema(self.epcs1 + self.epcs2, seg1a.epochs)
        assert_same_sub_schema(self.evts1 + self.evts2, seg1a.events)

        assert_same_sub_schema(self.trains1 + self.trains2, seg1a.spiketrains)

    def test__children(self):
        blk = Block(name='block1')
        blk.segments = [self.seg1]
        blk.create_many_to_one_relationship(force=True)
        assert_neo_object_is_compliant(self.seg1)
        assert_neo_object_is_compliant(blk)

        childobjs = ('AnalogSignal',
                     'Epoch', 'Event',
                     'IrregularlySampledSignal',
                     'SpikeTrain')
        childconts = ('analogsignals',
                      'epochs', 'events',
                      'irregularlysampledsignals',
                      'spiketrains')
        self.assertEqual(self.seg1._container_child_objects, ())
        self.assertEqual(self.seg1._data_child_objects, childobjs)
        self.assertEqual(self.seg1._single_parent_objects, ('Block',))
        self.assertEqual(self.seg1._multi_child_objects, ())
        self.assertEqual(self.seg1._multi_parent_objects, ())
        self.assertEqual(self.seg1._child_properties, ())

        self.assertEqual(self.seg1._single_child_objects, childobjs)
        self.assertEqual(self.seg1._container_child_containers, ())
        self.assertEqual(self.seg1._data_child_containers, childconts)
        self.assertEqual(self.seg1._single_child_containers, childconts)
        self.assertEqual(self.seg1._single_parent_containers, ('block',))
        self.assertEqual(self.seg1._multi_child_containers, ())
        self.assertEqual(self.seg1._multi_parent_containers, ())

        self.assertEqual(self.seg1._child_objects, childobjs)
        self.assertEqual(self.seg1._child_containers, childconts)
        self.assertEqual(self.seg1._parent_objects, ('Block',))
        self.assertEqual(self.seg1._parent_containers, ('block',))

        totchildren = (self.nchildren * 2 +  # epoch/event
                       self.nchildren +  # analogsignal
                       self.nchildren ** 2 +  # spiketrain
                       self.nchildren)  # irregsignal
        self.assertEqual(len(self.seg1._single_children), totchildren)
        self.assertEqual(len(self.seg1.data_children), totchildren)
        self.assertEqual(len(self.seg1.children), totchildren)
        self.assertEqual(len(self.seg1.data_children_recur), totchildren)
        self.assertEqual(len(self.seg1.children_recur), totchildren)

        self.assertEqual(len(self.seg1._multi_children), 0)
        self.assertEqual(len(self.seg1.container_children), 0)
        self.assertEqual(len(self.seg1.container_children_recur), 0)

        children = (self.sigarrs1a +
                    self.epcs1a + self.evts1a +
                    self.irsigs1a +
                    self.trains1a)
        assert_same_sub_schema(list(self.seg1._single_children), children)
        assert_same_sub_schema(list(self.seg1.data_children), children)
        assert_same_sub_schema(list(self.seg1.data_children_recur), children)
        assert_same_sub_schema(list(self.seg1.children), children)
        assert_same_sub_schema(list(self.seg1.children_recur), children)

        self.assertEqual(len(self.seg1.parents), 1)
        self.assertEqual(self.seg1.parents[0].name, 'block1')

    def test__size(self):
        targ1 = {"epochs": self.nchildren, "events": self.nchildren,
                 "irregularlysampledsignals": self.nchildren,
                 "spiketrains": self.nchildren ** 2,
                 "analogsignals": self.nchildren}
        self.assertEqual(self.targobj.size, targ1)

    def test__filter_none(self):
        targ = []
        # collecting all data objects in target block
        targ.extend(self.targobj.analogsignals)
        targ.extend(self.targobj.epochs)
        targ.extend(self.targobj.events)
        targ.extend(self.targobj.irregularlysampledsignals)
        targ.extend(self.targobj.spiketrains)

        res0 = self.targobj.filter()
        res1 = self.targobj.filter({})
        res2 = self.targobj.filter([])
        res3 = self.targobj.filter([{}])
        res4 = self.targobj.filter([{}, {}])
        res5 = self.targobj.filter([{}, {}])
        res6 = self.targobj.filter(targdict={})
        res7 = self.targobj.filter(targdict=[])
        res8 = self.targobj.filter(targdict=[{}])
        res9 = self.targobj.filter(targdict=[{}, {}])

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

    def test__filter_annotation_single(self):
        targ = (self.sigarrs1a +
                [self.epcs1a[0]] +
                [self.evts1a[0]] +
                self.irsigs1a +
                self.trains1a)

        res0 = self.targobj.filter(j=0)
        res1 = self.targobj.filter({'j': 0})
        res2 = self.targobj.filter(targdict={'j': 0})
        res3 = self.targobj.filter([{'j': 0}])
        res4 = self.targobj.filter(targdict=[{'j': 0}])

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
        targ = [self.epcs1a[1]]

        res0 = self.targobj.filter(name=self.epcs1a[1].name)
        res1 = self.targobj.filter({'name': self.epcs1a[1].name})
        res2 = self.targobj.filter(targdict={'name': self.epcs1a[1].name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_attribute_single_nores(self):
        targ = []

        res0 = self.targobj.filter(name=self.epcs2[0].name)
        res1 = self.targobj.filter({'name': self.epcs2[0].name})
        res2 = self.targobj.filter(targdict={'name': self.epcs2[0].name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi(self):
        targ = (self.sigarrs1a +
                [self.epcs1a[0]] +
                [self.evts1a[0]] +
                self.irsigs1a +
                self.trains1a +
                [self.epcs1a[1]])

        res0 = self.targobj.filter(name=self.epcs1a[1].name, j=0)
        res1 = self.targobj.filter({'name': self.epcs1a[1].name, 'j': 0})
        res2 = self.targobj.filter(targdict={'name': self.epcs1a[1].name,
                                             'j': 0})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_nores(self):
        targ = []

        res0 = self.targobj.filter([{'j': 5}, {}])
        res1 = self.targobj.filter({}, ttype=6)
        res2 = self.targobj.filter([{}], ttype=6)
        res3 = self.targobj.filter({'name': self.epcs1a[1].name}, j=0)
        res4 = self.targobj.filter(targdict={'name': self.epcs1a[1].name},
                                   j=0)
        res5 = self.targobj.filter(name=self.epcs1a[1].name,
                                   targdict={'j': 0})
        res6 = self.targobj.filter(name=self.epcs2[0].name, j=5)
        res7 = self.targobj.filter({'name': self.epcs2[1].name, 'j': 5})
        res8 = self.targobj.filter(targdict={'name': self.epcs2[1].name,
                                             'j': 5})
        res9 = self.targobj.filter({'name': self.epcs2[1].name}, j=5)
        res10 = self.targobj.filter(targdict={'name': self.epcs2[1].name},
                                    j=5)
        res11 = self.targobj.filter(name=self.epcs2[1].name,
                                    targdict={'j': 5})
        res12 = self.targobj.filter({'name': self.epcs1a[1].name}, j=5)
        res13 = self.targobj.filter(targdict={'name': self.epcs1a[1].name},
                                    j=5)
        res14 = self.targobj.filter(name=self.epcs1a[1].name,
                                    targdict={'j': 5})

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
        targ = [self.epcs1a[1]]

        res0 = self.targobj.filter(name=self.epcs1a[1].name, j=5)
        res1 = self.targobj.filter({'name': self.epcs1a[1].name, 'j': 5})
        res2 = self.targobj.filter(targdict={'name': self.epcs1a[1].name,
                                             'j': 5})
        res3 = self.targobj.filter([{'j': 1}, {'i': 1}])
        res4 = self.targobj.filter({'j': 1}, i=1)
        res5 = self.targobj.filter([{'j': 1}], i=1)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)

    def test__filter_no_annotation_but_object(self):
        targ = self.targobj.spiketrains
        res = self.targobj.filter(objects=SpikeTrain)
        assert_same_sub_schema(res, targ)

        targ = self.targobj.analogsignals
        res = self.targobj.filter(objects=AnalogSignal)
        assert_same_sub_schema(res, targ)

        targ = self.targobj.analogsignals + self.targobj.spiketrains
        res = self.targobj.filter(objects=[AnalogSignal, SpikeTrain])
        assert_same_sub_schema(res, targ)
        assert_same_sub_schema(res, targ)

    def test__filter_single_annotation_obj_single(self):
        targ = [self.epcs1a[1]]

        res0 = self.targobj.filter(j=1, objects='Epoch')
        res1 = self.targobj.filter(j=1, objects=Epoch)
        res2 = self.targobj.filter(j=1, objects=['Epoch'])
        res3 = self.targobj.filter(j=1, objects=[Epoch])
        res4 = self.targobj.filter(j=1, objects=[Epoch,
                                                 ChannelIndex])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_single_annotation_obj_multi(self):
        targ = [self.epcs1a[1], self.evts1a[1]]

        res0 = self.targobj.filter(j=1, objects=['Event', Epoch])

        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_obj_none(self):
        targ = []

        res0 = self.targobj.filter(j=1, objects=ChannelIndex)
        res1 = self.targobj.filter(j=1, objects='ChannelIndex')
        res2 = self.targobj.filter(j=1, objects=[])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_norecur(self):
        targ = [self.epcs1a[1], self.evts1a[1]]
        res0 = self.targobj.filter(j=1,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_norecur(self):
        targ = [self.epcs1a[1]]
        res0 = self.targobj.filter(name=self.epcs1a[1].name,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata(self):
        targ = []
        res0 = self.targobj.filter(j=0,
                                   data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata(self):
        targ = []
        res0 = self.targobj.filter(name=self.epcs1a[1].name,
                                   data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=0,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.epcs1a[1].name,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        targ = [self.epcs1a[1], self.evts1a[1]]
        res0 = self.targobj.filter(j=1,
                                   container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container(self):
        targ = [self.epcs1a[1]]
        res0 = self.targobj.filter(name=self.epcs1a[1].name,
                                   container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container_norecur(self):
        targ = [self.epcs1a[1], self.evts1a[1]]
        res0 = self.targobj.filter(j=1,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur(self):
        targ = [self.epcs1a[1]]
        res0 = self.targobj.filter(name=self.epcs1a[1].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container(self):
        targ = []
        res0 = self.targobj.filter(j=0,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container(self):
        targ = []
        res0 = self.targobj.filter(name=self.epcs1a[1].name,
                                   data=False, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=0,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.epcs1a[1].name,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filterdata_multi(self):
        data = self.targobj.children_recur

        targ = (self.sigarrs1a +
                [self.epcs1a[0]] +
                [self.evts1a[0]] +
                self.irsigs1a +
                self.trains1a +
                [self.epcs1a[1]])

        res0 = filterdata(data, name=self.epcs1a[1].name, j=0)
        res1 = filterdata(data, {'name': self.epcs1a[1].name, 'j': 0})
        res2 = filterdata(data, targdict={'name': self.epcs1a[1].name, 'j': 0})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_nores(self):
        data = self.targobj.children_recur

        targ = []

        res0 = filterdata(data, [{'j': 5}, {}])
        res1 = filterdata(data, {}, ttype=0)
        res2 = filterdata(data, [{}], ttype=0)
        res3 = filterdata(data, {'name': self.epcs1a[1].name}, j=0)
        res4 = filterdata(data, targdict={'name': self.epcs1a[1].name}, j=0)
        res5 = filterdata(data, name=self.epcs1a[1].name, targdict={'j': 0})
        res6 = filterdata(data, name=self.epcs2[0].name, j=5)
        res7 = filterdata(data, {'name': self.epcs2[1].name, 'j': 5})
        res8 = filterdata(data, targdict={'name': self.epcs2[1].name, 'j': 5})
        res9 = filterdata(data, {'name': self.epcs2[1].name}, j=5)
        res10 = filterdata(data, targdict={'name': self.epcs2[1].name}, j=5)
        res11 = filterdata(data, name=self.epcs2[1].name, targdict={'j': 5})
        res12 = filterdata(data, {'name': self.epcs1a[1].name}, j=5)
        res13 = filterdata(data, targdict={'name': self.epcs1a[1].name}, j=5)
        res14 = filterdata(data, name=self.epcs1a[1].name, targdict={'j': 5})

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

        targ = [self.epcs1a[1]]

        res0 = filterdata(data, name=self.epcs1a[1].name, j=5)
        res1 = filterdata(data, {'name': self.epcs1a[1].name, 'j': 5})
        res2 = filterdata(data, targdict={'name': self.epcs1a[1].name, 'j': 5})
        res3 = filterdata(data, [{'j': 1}, {'i': 1}])
        res4 = filterdata(data, {'j': 1}, i=1)
        res5 = filterdata(data, [{'j': 1}], i=1)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)

    # @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    # def test__pretty(self):
    #     ann = get_annotations()
    #     ann['seed'] = self.seed1
    #     ann = pretty(ann).replace('\n ', '\n  ')
    #     res = pretty(self.seg1)
    #
    #     sigarr0 = pretty(self.sigarrs1[0])
    #     sigarr1 = pretty(self.sigarrs1[1])
    #     sigarr0 = sigarr0.replace('\n', '\n   ')
    #     sigarr1 = sigarr1.replace('\n', '\n   ')
    #
    #     targ = ("Segment with " +
    #             ("%s analogsignals, " %
    #              (len(self.sigarrs1a),)) +
    #             ("%s epochs, " % len(self.epcs1a)) +
    #             ("%s events, " % len(self.evts1a)) +
    #             ("%s irregularlysampledsignals, " %
    #              len(self.irsigs1a)) +
    #             ("%s spiketrains\n" % len(self.trains1a)) +
    #             ("name: '%s'\ndescription: '%s'\n" %
    #              (self.seg1.name, self.seg1.description)
    #              ) +
    #
    #             ("annotations: %s\n" % ann) +
    #
    #             ("# analogsignals (N=%s)\n" % len(self.sigarrs1a)) +
    #
    #             ('%s: %s\n' % (0, sigarr0)) +
    #             ('%s: %s' % (1, sigarr1)))
    #
    #     self.assertEqual(res, targ)

    def test__construct_subsegment_by_unit(self):
        nb_seg = 3
        nb_unit = 7
        unit_with_sig = np.array([0, 2, 5])
        signal_types = ['Vm', 'Conductances']
        sig_len = 100

        # channelindexes
        chxs = [ChannelIndex(name='Vm',
                             index=unit_with_sig),
                ChannelIndex(name='Conductance',
                             index=unit_with_sig)]

        # Unit
        all_unit = []
        for u in range(nb_unit):
            un = Unit(name='Unit #%d' % u, channel_indexes=np.array([u]))
            assert_neo_object_is_compliant(un)
            all_unit.append(un)

        blk = Block()
        blk.channel_indexes = chxs
        for s in range(nb_seg):
            seg = Segment(name='Simulation %s' % s)
            for j in range(nb_unit):
                st = SpikeTrain([1, 2], units='ms',
                                t_start=0., t_stop=10)
                st.unit = all_unit[j]

            for t in signal_types:
                anasigarr = AnalogSignal(np.zeros((sig_len,
                                                   len(unit_with_sig))),
                                         units='nA',
                                         sampling_rate=1000. * pq.Hz,
                                         channel_indexes=unit_with_sig)
                seg.analogsignals.append(anasigarr)

        blk.create_many_to_one_relationship()
        for unit in all_unit:
            assert_neo_object_is_compliant(unit)
        for chx in chxs:
            assert_neo_object_is_compliant(chx)
        assert_neo_object_is_compliant(blk)

        # what you want
        newseg = seg.construct_subsegment_by_unit(all_unit[:4])
        assert_neo_object_is_compliant(newseg)

    def test_segment_take_spiketrains_by_unit(self):
        result1 = self.seg1.take_spiketrains_by_unit()
        result21 = self.seg1.take_spiketrains_by_unit([self.unit1])
        result22 = self.seg1.take_spiketrains_by_unit([self.unit2])

        self.assertEqual(result1, [])

        assert_same_sub_schema(result21, [self.trains1a[0]])
        assert_same_sub_schema(result22, [self.trains1a[1]])

    # to remove
    # def test_segment_take_analogsignal_by_unit(self):
    #     result1 = self.seg1.take_analogsignal_by_unit()
    #     result21 = self.seg1.take_analogsignal_by_unit([self.unit1])
    #     result22 = self.seg1.take_analogsignal_by_unit([self.unit2])
    #
    #     self.assertEqual(result1, [])
    #
    #     assert_same_sub_schema(result21, [self.sigs1a[0]])
    #     assert_same_sub_schema(result22, [self.sigs1a[1]])
    #
    # def test_segment_take_analogsignal_by_channelindex(self):
    #     ind1 = self.unit1.channel_indexes[0]
    #     ind2 = self.unit2.channel_indexes[0]
    #     result1 = self.seg1.take_analogsignal_by_channelindex()
    #     result21 = self.seg1.take_analogsignal_by_channelindex([ind1])
    #     result22 = self.seg1.take_analogsignal_by_channelindex([ind2])
    #
    #     self.assertEqual(result1, [])
    #
    #     assert_same_sub_schema(result21, [self.sigs1a[0]])
    #     assert_same_sub_schema(result22, [self.sigs1a[1]])

    # commenting out temporarily
    # def test_seg_take_slice_of_analogsignalarray_by_unit(self):
    #     seg = self.seg1
    #     result1 = seg.take_slice_of_analogsignalarray_by_unit()
    #     result21 = seg.take_slice_of_analogsignalarray_by_unit([self.unit1])
    #     result23 = seg.take_slice_of_analogsignalarray_by_unit([self.unit3])
    #
    #     self.assertEqual(result1, [])
    #
    #     targ1 = [self.sigarrs1a[0][:, np.array([True])],
    #              self.sigarrs1a[1][:, np.array([False])]]
    #     targ3 = [self.sigarrs1a[0][:, np.array([False])],
    #              self.sigarrs1a[1][:, np.array([True])]]
    #     assert_same_sub_schema(result21, targ1)
    #     assert_same_sub_schema(result23, targ3)
    #
    # def test_seg_take_slice_of_analogsignalarray_by_channelindex(self):
    #     seg = self.seg1
    #     ind1 = self.unit1.channel_indexes[0]
    #     ind3 = self.unit3.channel_indexes[0]
    #     result1 = seg.take_slice_of_analogsignalarray_by_channelindex()
    #     result21 = seg.take_slice_of_analogsignalarray_by_channelindex([ind1])
    #     result23 = seg.take_slice_of_analogsignalarray_by_channelindex([ind3])
    #
    #     self.assertEqual(result1, [])
    #
    #     targ1 = [self.sigarrs1a[0][:, np.array([True])],
    #              self.sigarrs1a[1][:, np.array([False])]]
    #     targ3 = [self.sigarrs1a[0][:, np.array([False])],
    #              self.sigarrs1a[1][:, np.array([True])]]
    #     assert_same_sub_schema(result21, targ1)
    #     assert_same_sub_schema(result23, targ3)


if __name__ == "__main__":
    unittest.main()
