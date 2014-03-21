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
from neo.core import RecordingChannelGroup, RecordingChannel, Segment, Unit
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal,
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

        for child in res.children:
            del child.annotations['i']
            del child.annotations['j']
            for subchild in child.children:
                for subsubchild in subchild.children:
                    if 'i' not in subsubchild.annotations:
                        continue
                    del subsubchild.annotations['i']
                    del subsubchild.annotations['j']
                if 'i' not in subchild.annotations:
                    continue
                del subchild.annotations['i']
                del subchild.annotations['j']

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

        self.assertEqual(len(self.blk1.children), self.nchildren*2)

        assert_same_sub_schema(list(self.blk1.children),
                               segs1a + rcgs1a)

    def test_block_list_units(self):
        assert_same_sub_schema(self.units1, self.blk1.list_units)
        assert_same_sub_schema(self.units2, self.blk2.list_units)

    def test_block_list_recordingchannels(self):
        assert_same_sub_schema(self.rchans1, self.blk1.list_recordingchannels)
        assert_same_sub_schema(self.rchans2, self.blk2.list_recordingchannels)


if __name__ == "__main__":
    unittest.main()
