# -*- coding: utf-8 -*-
"""
Tests of the neo.core.block.Block class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.core.block import Block
from neo.core.recordingchannelgroup import RecordingChannelGroup
from neo.core.recordingchannel import RecordingChannel
from neo.core.segment import Segment
from neo.core.unit import Unit
from neo.io.tools import create_many_to_one_relationship
from neo.test.tools import assert_neo_object_is_compliant


class TestBlock(unittest.TestCase):
    def setUp(self):
        unitname11 = 'unit 1 1'
        unitname12 = 'unit 1 2'
        unitname21 = 'unit 2 1'
        unitname22 = 'unit 2 2'

        channame11 = 'chan 1 1'
        channame12 = 'chan 1 2'
        channame21 = 'chan 2 1'
        channame22 = 'chan 2 2'

        segname11 = 'seg 1 1'
        segname12 = 'seg 1 2'
        segname21 = 'seg 2 1'
        segname22 = 'seg 2 2'

        self.rcgname1 = 'rcg 1'
        self.rcgname2 = 'rcg 2'

        self.unitnames1 = [unitname11, unitname12]
        self.unitnames2 = [unitname21, unitname22, unitname11]
        self.unitnames = [unitname11, unitname12, unitname21, unitname22]

        self.channames1 = [channame11, channame12]
        self.channames2 = [channame21, channame22, channame11]
        self.channames = [channame11, channame12, channame21, channame22]

        self.segnames1 = [segname11, segname12]
        self.segnames2 = [segname21, segname22, segname11]
        self.segnames = [segname11, segname12, segname21, segname22]

        unit11 = Unit(name=unitname11)
        unit12 = Unit(name=unitname12)
        unit21 = Unit(name=unitname21)
        unit22 = Unit(name=unitname22)
        unit23 = unit11

        chan11 = RecordingChannel(name=channame11)
        chan12 = RecordingChannel(name=channame12)
        chan21 = RecordingChannel(name=channame21)
        chan22 = RecordingChannel(name=channame22)
        chan23 = chan11

        seg11 = Segment(name=segname11)
        seg12 = Segment(name=segname12)
        seg21 = Segment(name=segname21)
        seg22 = Segment(name=segname22)
        seg23 = seg11

        self.units1 = [unit11, unit12]
        self.units2 = [unit21, unit22, unit23]
        self.units = [unit11, unit12, unit21, unit22]

        self.chan1 = [chan11, chan12]
        self.chan2 = [chan21, chan22, chan23]
        self.chan = [chan11, chan12, chan21, chan22]

        self.seg1 = [seg11, seg12]
        self.seg2 = [seg21, seg22, seg23]
        self.seg = [seg11, seg12, seg21, seg22]

        self.rcg1 = RecordingChannelGroup(name=self.rcgname1)
        self.rcg2 = RecordingChannelGroup(name=self.rcgname2)

        self.rcg1.units = self.units1
        self.rcg2.units = self.units2
        self.rcg1.recordingchannels = self.chan1
        self.rcg2.recordingchannels = self.chan2

    def test_block_init(self):
        blk = Block(name='a block')
        assert_neo_object_is_compliant(blk)
        self.assertEqual(blk.name, 'a block')
        self.assertEqual(blk.file_origin, None)

    def test_block_list_units(self):
        blk = Block(name='a block')
        blk.recordingchannelgroups = [self.rcg1, self.rcg2]
        create_many_to_one_relationship(blk)
        #assert_neo_object_is_compliant(blk)

        unitres1 = [unit.name for unit in blk.recordingchannelgroups[0].units]
        unitres2 = [unit.name for unit in blk.recordingchannelgroups[1].units]
        unitres = [unit.name for unit in blk.list_units]

        self.assertEqual(self.unitnames1, unitres1)
        self.assertEqual(self.unitnames2, unitres2)
        self.assertEqual(self.unitnames, unitres)

    def test_block_list_recordingchannel(self):
        blk = Block(name='a block')
        blk.recordingchannelgroups = [self.rcg1, self.rcg2]
        create_many_to_one_relationship(blk)
        #assert_neo_object_is_compliant(blk)

        chanres1 = [chan.name for chan in
                    blk.recordingchannelgroups[0].recordingchannels]
        chanres2 = [chan.name for chan in
                    blk.recordingchannelgroups[1].recordingchannels]
        chanres = [chan.name for chan in blk.list_recordingchannels]

        self.assertEqual(self.channames1, chanres1)
        self.assertEqual(self.channames2, chanres2)
        self.assertEqual(self.channames, chanres)

    def test_block_merge(self):
        blk1 = Block(name='block 1')
        blk2 = Block(name='block 2')

        rcg3 = RecordingChannelGroup(name=self.rcgname1)
        rcg3.units = self.units1 + [self.units2[0]]
        rcg3.recordingchannels = self.chan1 + [self.chan2[1]]

        blk1.recordingchannelgroups = [self.rcg1]
        blk2.recordingchannelgroups = [self.rcg2, rcg3]
        blk1.segments = self.seg1
        blk2.segments = self.seg2

        blk1.merge(blk2)

        rcgres1 = [rcg.name for rcg in blk1.recordingchannelgroups]
        rcgres2 = [rcg.name for rcg in blk2.recordingchannelgroups]

        segres1 = [seg.name for seg in blk1.segments]
        segres2 = [seg.name for seg in blk2.segments]

        chanres1 = [chan.name for chan in blk1.list_recordingchannels]
        chanres2 = [chan.name for chan in blk2.list_recordingchannels]

        unitres1 = [unit.name for unit in blk1.list_units]
        unitres2 = [unit.name for unit in blk2.list_units]

        self.assertEqual(rcgres1, [self.rcgname1, self.rcgname2])
        self.assertEqual(rcgres2, [self.rcgname2, self.rcgname1])

        self.assertEqual(segres1, self.segnames)
        self.assertEqual(segres2, self.segnames2)

        self.assertEqual(chanres1, self.channames1 + self.channames2[-2::-1])
        self.assertEqual(chanres2, self.channames2[:-1] + self.channames1)

        self.assertEqual(unitres1, self.unitnames)
        self.assertEqual(unitres2, self.unitnames2[:-1] + self.unitnames1)


if __name__ == "__main__":
    unittest.main()
