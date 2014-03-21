# -*- coding: utf-8 -*-
"""
Tests of the neo.core.block.Block class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from datetime import datetime

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

from neo.core.block import Block
from neo.core.container import filterdata
from neo.core import SpikeTrain, Unit
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_same_sub_schema)
from neo.test.generate_datasets import (get_fake_value, get_fake_values,
                                        fake_neo, clone_object,
                                        get_annotations, TEST_ANNOTATIONS)


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
        name = get_fake_value('name', str, seed=3, obj=Block)
        description = get_fake_value('description', str, seed=4, obj='Block')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'file_datetime': file_datetime,
                  'rec_datetime': rec_datetime,
                  'index': index,
                  'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Block, annotate=False, seed=0)
        res12 = get_fake_values('Block', annotate=False, seed=0)
        res21 = get_fake_values(Block, annotate=True, seed=0)
        res22 = get_fake_values('Block', annotate=True, seed=0)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = 'Block'

        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        for child in res.children_recur:
            del child.annotations['i']
            del child.annotations['j']

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.segments), 1)
        seg = res.segments[0]
        self.assertEqual(seg.annotations, self.annotations)

        self.assertEqual(len(res.recordingchannelgroups), 1)
        rcg = res.recordingchannelgroups[0]
        self.assertEqual(rcg.annotations, self.annotations)

        self.assertEqual(len(seg.analogsignalarrays), 1)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(len(seg.irregularlysampledsignals), 1)
        self.assertEqual(len(seg.spiketrains), 1)
        self.assertEqual(len(seg.spikes), 1)
        self.assertEqual(len(seg.events), 1)
        self.assertEqual(len(seg.epochs), 1)
        self.assertEqual(len(seg.eventarrays), 1)
        self.assertEqual(len(seg.epocharrays), 1)
        self.assertEqual(seg.analogsignalarrays[0].annotations,
                         self.annotations)
        self.assertEqual(seg.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(seg.irregularlysampledsignals[0].annotations,
                         self.annotations)
        self.assertEqual(seg.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(seg.spikes[0].annotations,
                         self.annotations)
        self.assertEqual(seg.events[0].annotations,
                         self.annotations)
        self.assertEqual(seg.epochs[0].annotations,
                         self.annotations)
        self.assertEqual(seg.eventarrays[0].annotations,
                         self.annotations)
        self.assertEqual(seg.epocharrays[0].annotations,
                         self.annotations)

        self.assertEqual(len(rcg.recordingchannels), 1)
        rchan = rcg.recordingchannels[0]
        self.assertEqual(rchan.annotations, self.annotations)

        self.assertEqual(len(rcg.units), 1)
        unit = rcg.units[0]
        self.assertEqual(unit.annotations, self.annotations)

        self.assertEqual(len(rcg.analogsignalarrays), 1)
        self.assertEqual(rcg.analogsignalarrays[0].annotations,
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
        obj_type = Block
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.segments), 0)
        self.assertEqual(len(res.recordingchannelgroups), 0)


class TestBlock(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        self.seed1 = 0
        self.seed2 = 10000
        self.blk1 = fake_neo(Block, seed=self.seed1, n=self.nchildren)
        self.blk2 = fake_neo(Block, seed=self.seed2, n=self.nchildren)
        self.targobj = self.blk1

        self.segs1 = self.blk1.segments
        self.segs2 = self.blk2.segments
        self.rcgs1 = self.blk1.recordingchannelgroups
        self.rcgs2 = self.blk2.recordingchannelgroups

        self.units1 = [[unit for unit in rcg.units] for rcg in self.rcgs1]
        self.units2 = [[unit for unit in rcg.units] for rcg in self.rcgs2]
        self.rchans1 = [[rchan for rchan in rcg.recordingchannels]
                        for rcg in self.rcgs1]
        self.rchans2 = [[rchan for rchan in rcg.recordingchannels]
                        for rcg in self.rcgs2]

        self.units1 = sum(self.units1, [])
        self.units2 = sum(self.units2, [])
        self.rchans1 = sum(self.rchans1, [])
        self.rchans2 = sum(self.rchans2, [])

        self.sigarrs1 = [[sigarr for sigarr in rcg.analogsignalarrays]
                         for rcg in self.rcgs1]
        self.sigarrs2 = [[sigarr for sigarr in rcg.analogsignalarrays]
                         for rcg in self.rcgs2]

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

        self.epcs1 = [[epc for epc in seg.epochs]
                      for seg in self.segs1]
        self.epcs2 = [[epc for epc in seg.epochs]
                      for seg in self.segs2]
        self.epcas1 = [[epca for epca in seg.epocharrays]
                       for seg in self.segs1]
        self.epcas2 = [[epca for epca in seg.epocharrays]
                       for seg in self.segs2]
        self.evts1 = [[evt for evt in seg.events]
                      for seg in self.segs1]
        self.evts2 = [[evt for evt in seg.events]
                      for seg in self.segs2]
        self.evtas1 = [[evta for evta in seg.eventarrays]
                       for seg in self.segs1]
        self.evtas2 = [[evta for evta in seg.eventarrays]
                       for seg in self.segs2]

        self.sigarrs1 = sum(self.sigarrs1, [])
        self.sigarrs2 = sum(self.sigarrs2, [])

        self.spikes1 = sum(self.spikes1, [])
        self.spikes2 = sum(self.spikes2, [])
        self.trains1 = sum(self.trains1, [])
        self.trains2 = sum(self.trains2, [])
        self.sigs1 = sum(self.sigs1, [])
        self.sigs2 = sum(self.sigs2, [])
        self.irsigs1 = sum(self.irsigs1, [])
        self.irsigs2 = sum(self.irsigs2, [])

        self.epcs1 = sum(self.epcs1, [])
        self.epcs2 = sum(self.epcs2, [])
        self.epcas1 = sum(self.epcas1, [])
        self.epcas2 = sum(self.epcas2, [])
        self.evts1 = sum(self.evts1, [])
        self.evts2 = sum(self.evts2, [])
        self.evtas1 = sum(self.evtas1, [])
        self.evtas2 = sum(self.evtas2, [])

    def test_block_init(self):
        blk = Block(name='a block')
        assert_neo_object_is_compliant(blk)
        self.assertEqual(blk.name, 'a block')
        self.assertEqual(blk.file_origin, None)

    def check_creation(self, blk):
        assert_neo_object_is_compliant(blk)

        seed = blk.annotations['seed']

        targ0 = get_fake_value('file_datetime', datetime, seed=seed+0)
        self.assertEqual(blk.file_datetime, targ0)

        targ1 = get_fake_value('rec_datetime', datetime, seed=seed+1)
        self.assertEqual(blk.rec_datetime, targ1)

        targ2 = get_fake_value('index', int, seed=seed+2, obj=Block)
        self.assertEqual(blk.index, targ2)

        targ3 = get_fake_value('name', str, seed=seed+3, obj=Block)
        self.assertEqual(blk.name, targ3)

        targ4 = get_fake_value('description', str, seed=seed+4, obj=Block)
        self.assertEqual(blk.description, targ4)

        targ5 = get_fake_value('file_origin', str)
        self.assertEqual(blk.file_origin, targ5)

        targ6 = get_annotations()
        targ6['seed'] = seed
        self.assertEqual(blk.annotations, targ6)

        self.assertTrue(hasattr(blk, 'recordingchannelgroups'))
        self.assertTrue(hasattr(blk, 'segments'))

        self.assertEqual(len(blk.recordingchannelgroups), self.nchildren)
        self.assertEqual(len(blk.segments), self.nchildren)

    def test__creation(self):
        self.check_creation(self.blk1)
        self.check_creation(self.blk2)

    def test__merge(self):
        blk1a = fake_neo(Block,
                         seed=self.seed1, n=self.nchildren)
        assert_same_sub_schema(self.blk1, blk1a)
        blk1a.annotate(seed=self.seed2)
        blk1a.segments.append(self.segs2[0])
        blk1a.merge(self.blk2)

        segs1a = clone_object(self.blk1).segments
        rcgs1a = clone_object(self.rcgs1)

        assert_same_sub_schema(rcgs1a + self.rcgs2,
                               blk1a.recordingchannelgroups)
        assert_same_sub_schema(segs1a + self.segs2,
                               blk1a.segments)

    def test__children(self):
        segs1a = clone_object(self.blk1).segments
        rcgs1a = clone_object(self.rcgs1)

        self.assertEqual(self.blk1._container_child_objects,
                         ('Segment', 'RecordingChannelGroup'))
        self.assertEqual(self.blk1._data_child_objects, ())
        self.assertEqual(self.blk1._single_parent_objects, ())
        self.assertEqual(self.blk1._multi_child_objects, ())
        self.assertEqual(self.blk1._multi_parent_objects, ())
        self.assertEqual(self.blk1._child_properties,
                         ('Unit', 'RecordingChannel'))

        self.assertEqual(self.blk1._single_child_objects,
                         ('Segment', 'RecordingChannelGroup'))

        self.assertEqual(self.blk1._container_child_containers,
                         ('segments', 'recordingchannelgroups'))
        self.assertEqual(self.blk1._data_child_containers, ())
        self.assertEqual(self.blk1._single_child_containers,
                         ('segments', 'recordingchannelgroups'))
        self.assertEqual(self.blk1._single_parent_containers, ())
        self.assertEqual(self.blk1._multi_child_containers, ())
        self.assertEqual(self.blk1._multi_parent_containers, ())

        self.assertEqual(self.blk1._child_objects,
                         ('Segment', 'RecordingChannelGroup'))
        self.assertEqual(self.blk1._child_containers,
                         ('segments', 'recordingchannelgroups'))
        self.assertEqual(self.blk1._parent_objects, ())
        self.assertEqual(self.blk1._parent_containers, ())

        self.assertEqual(len(self.blk1._single_children), 2*self.nchildren)
        self.assertEqual(len(self.blk1._multi_children), 0)
        self.assertEqual(len(self.blk1.data_children), 0)
        self.assertEqual(len(self.blk1.data_children_recur),
                         4*self.nchildren**3 + 5*self.nchildren**2)
        self.assertEqual(len(self.blk1.container_children), 2*self.nchildren)
        self.assertEqual(len(self.blk1.container_children_recur),
                         2*self.nchildren + 2*self.nchildren**2)
        self.assertEqual(len(self.blk1.children), 2*self.nchildren)
        self.assertEqual(len(self.blk1.children_recur),
                         2*self.nchildren +
                         2*self.nchildren**2 +
                         4*self.nchildren**3 + 5*self.nchildren**2)

        self.assertEqual(self.blk1._multi_children, ())
        assert_same_sub_schema(list(self.blk1._single_children),
                               self.segs1 + self.rcgs1)

        assert_same_sub_schema(list(self.blk1.container_children),
                               self.segs1 + self.rcgs1)
        assert_same_sub_schema(list(self.blk1.container_children_recur),
                               self.segs1 + self.rcgs1 +
                               self.units1[:2] + self.rchans1[:2] +
                               self.units1[2:] + self.rchans1[2:])

        assert_same_sub_schema(list(self.blk1.data_children_recur),
                               self.sigs1[::2] + self.sigarrs1[::2] +
                               self.epcs1[:2] + self.epcas1[:2] +
                               self.evts1[:2] + self.evtas1[:2] +
                               self.irsigs1[::2] + self.spikes1[::2] +
                               self.trains1[::2] +
                               self.sigs1[1::2] + self.sigarrs1[1::2] +
                               self.epcs1[2:] + self.epcas1[2:] +
                               self.evts1[2:] + self.evtas1[2:] +
                               self.irsigs1[1::2] +
                               self.spikes1[1::2] + self.trains1[1::2],
                               exclude=['channel_index'])

        assert_same_sub_schema(list(self.blk1.children),
                               segs1a + rcgs1a)
        assert_same_sub_schema(list(self.blk1.children_recur),
                               self.sigs1[::2] + self.sigarrs1[::2] +
                               self.epcs1[:2] + self.epcas1[:2] +
                               self.evts1[:2] + self.evtas1[:2] +
                               self.irsigs1[::2] + self.spikes1[::2] +
                               self.trains1[::2] +
                               self.sigs1[1::2] + self.sigarrs1[1::2] +
                               self.epcs1[2:] + self.epcas1[2:] +
                               self.evts1[2:] + self.evtas1[2:] +
                               self.irsigs1[1::2] +
                               self.spikes1[1::2] + self.trains1[1::2] +
                               self.segs1 + self.rcgs1 +
                               self.units1[:2] + self.rchans1[:2] +
                               self.units1[2:] + self.rchans1[2:],
                               exclude=['channel_index'])

    def test__size(self):
        targ = {'segments': self.nchildren,
                'recordingchannelgroups': self.nchildren}
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
        targ = ([self.epcs1[1], self.epcas1[1],
                 self.evts1[1], self.evtas1[1]] +
                self.sigs1[1::2] + self.sigarrs1[1::2] +
                [self.epcs1[3], self.epcas1[3],
                 self.evts1[3], self.evtas1[3]] +
                self.irsigs1[1::2] +
                self.spikes1[1::2] + self.trains1[1::2])

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
        targ = ([self.epcs1[1], self.epcas1[1],
                 self.evts1[1], self.evtas1[1]] +
                self.sigs1[1::2] + self.sigarrs1[1::2] +
                [self.epcs1[3], self.epcas1[3],
                 self.evts1[3], self.evtas1[3]] +
                self.irsigs1[1::2] +
                self.spikes1[1::2] + self.trains1[1::2] +
                [self.spikes1[0]])

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
        res0 = self.targobj.filter(name=name, j=90)
        res1 = self.targobj.filter({'name': name, 'j': 90})
        res2 = self.targobj.filter(targdict={'name': name, 'j': 90})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_partres_annotation_annotation(self):
        targ = self.sigs1[::2] + self.spikes1[::2]

        res0 = self.targobj.filter([{'j': 0}, {'i': 0}])
        res1 = self.targobj.filter({'j': 0}, i=0)
        res2 = self.targobj.filter([{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_obj_single(self):
        targ = self.trains1[1::2]

        res0 = self.targobj.filter(j=1, objects='SpikeTrain')
        res1 = self.targobj.filter(j=1, objects=SpikeTrain)
        res2 = self.targobj.filter(j=1, objects=['SpikeTrain'])
        res3 = self.targobj.filter(j=1, objects=[SpikeTrain])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)

    def test__filter_single_annotation_obj_multi(self):
        targ = self.spikes1[1::2] + self.trains1[1::2]
        res0 = self.targobj.filter(j=1, objects=['Spike', SpikeTrain])
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=1, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigarrs1[0].name,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata(self):
        targ = []
        res0 = self.targobj.filter(j=1, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigarrs1[0].name, data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(j=1,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_norecur(self):
        targ = []
        res0 = self.targobj.filter(name=self.sigarrs1[0].name,
                                   data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        targ = ([self.epcs1[1], self.epcas1[1],
                 self.evts1[1], self.evtas1[1]] +
                self.sigs1[1::2] + self.sigarrs1[1::2] +
                [self.epcs1[3], self.epcas1[3],
                 self.evts1[3], self.evtas1[3]] +
                self.irsigs1[1::2] +
                self.spikes1[1::2] + self.trains1[1::2] +
                [self.segs1[1], self.rcgs1[1],
                 self.units1[1], self.rchans1[1],
                 self.units1[3], self.rchans1[3]])

        res0 = self.targobj.filter(j=1, container=True)

        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_data(self):
        targ = [self.spikes1[0]]
        res0 = self.targobj.filter(name=self.spikes1[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_container(self):
        targ = [self.rchans1[0]]
        res0 = self.targobj.filter(name=self.rchans1[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container_norecur(self):
        targ = [self.segs1[1], self.rcgs1[1]]

        res0 = self.targobj.filter(j=1, container=True, recursive=False)

        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur(self):
        targ = [self.segs1[0]]
        res0 = self.targobj.filter(name=self.segs1[0].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur_nores(self):
        targ = []
        res0 = self.targobj.filter(name=self.spikes1[0].name,
                                   container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container(self):
        targ = [self.segs1[1], self.rcgs1[1],
                self.units1[1], self.rchans1[1],
                self.units1[3], self.rchans1[3]]
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
        targ = [self.segs1[1], self.rcgs1[1]]
        res0 = self.targobj.filter(j=1,
                                   data=False, container=True,
                                   recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_norecur(self):
        targ = [self.segs1[0]]
        res0 = self.targobj.filter(name=self.segs1[0].name,
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

        targ = ([self.epcs1[1], self.epcas1[1],
                 self.evts1[1], self.evtas1[1]] +
                self.sigs1[1::2] + self.sigarrs1[1::2] +
                [self.epcs1[3], self.epcas1[3],
                 self.evts1[3], self.evtas1[3]] +
                self.irsigs1[1::2] +
                self.spikes1[1::2] + self.trains1[1::2] +
                [self.segs1[1], self.rcgs1[1],
                 self.units1[1], self.rchans1[1],
                 self.units1[3], self.rchans1[3],
                 self.spikes1[0]])

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

        name1 = self.sigarrs1[0].name
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
        res0 = filterdata(data, name=name, j=90)
        res1 = filterdata(data, {'name': name, 'j': 90})
        res2 = filterdata(data, targdict={'name': name, 'j': 90})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_partres_annotation_annotation(self):
        data = self.targobj.children_recur

        targ = (self.sigs1[::2] + self.spikes1[::2] +
                self.segs1[:1] + self.units1[::2])

        res0 = filterdata(data, [{'j': 0}, {'i': 0}])
        res1 = filterdata(data, {'j': 0}, i=0)
        res2 = filterdata(data, [{'j': 0}], i=0)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    def test__pretty(self):
        res = pretty(self.blk1)
        ann = get_annotations()
        ann['seed'] = self.seed1
        ann = pretty(ann).replace('\n ', '\n  ')

        seg0 = pretty(self.segs1[0])
        seg1 = pretty(self.segs1[1])
        seg0 = seg0.replace('\n', '\n   ')
        seg1 = seg1.replace('\n', '\n   ')

        targ = ("Block with " +
                ("%s segments, %s recordingchannelgroups\n" %
                 (len(self.segs1), len(self.rcgs1))) +
                ("name: '%s'\ndescription: '%s'\n" % (self.blk1.name,
                                                      self.blk1.description)) +
                ("annotations: %s\n" % ann) +
                ("file_origin: '%s'\n" % self.blk1.file_origin) +
                ("file_datetime: %s\n" % repr(self.blk1.file_datetime)) +
                ("rec_datetime: %s\n" % repr(self.blk1.rec_datetime)) +
                ("index: %s\n" % self.blk1.index) +


                ("# segments (N=%s)\n" % len(self.segs1)) +
                ('%s: %s\n' % (0, seg0)) +
                ('%s: %s' % (1, seg1)))

        self.assertEqual(res, targ)

    def test_block_list_units(self):
        assert_same_sub_schema(self.units1, self.blk1.list_units)
        assert_same_sub_schema(self.units2, self.blk2.list_units)
        assert_same_sub_schema(self.units1,
                               self.blk1.list_children_by_class(Unit))
        assert_same_sub_schema(self.units2,
                               self.blk2.list_children_by_class(Unit))
        assert_same_sub_schema(self.units1,
                               self.blk1.list_children_by_class('Unit'))
        assert_same_sub_schema(self.units2,
                               self.blk2.list_children_by_class('Unit'))
        assert_same_sub_schema(self.units1,
                               self.blk1.list_children_by_class('units'))
        assert_same_sub_schema(self.units2,
                               self.blk2.list_children_by_class('units'))

    def test_block_list_recordingchannels(self):
        assert_same_sub_schema(self.rchans1, self.blk1.list_recordingchannels)
        assert_same_sub_schema(self.rchans2, self.blk2.list_recordingchannels)


if __name__ == "__main__":
    unittest.main()
