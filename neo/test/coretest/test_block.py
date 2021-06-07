"""
Tests of the neo.core.block.Block class
"""

from datetime import datetime
from copy import deepcopy
from neo.core.view import ChannelView

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
from neo.core import SpikeTrain, AnalogSignal, Event
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_same_sub_schema)
from neo.test.generate_datasets import random_block, simple_block


N_EXAMPLES = 5


class TestBlock(unittest.TestCase):

    def setUp(self):
        self.blocks = [random_block() for i in range(N_EXAMPLES)]

    def test_block_init(self):
        blk = Block(name='a block')
        assert_neo_object_is_compliant(blk)
        self.assertEqual(blk.name, 'a block')
        self.assertEqual(blk.file_origin, None)

    def test__merge(self):
        blk1 = self.blocks[0]
        blk2 = self.blocks[1]

        orig_blk1 = deepcopy(blk1)
        blk1.merge(blk2)

        assert_same_sub_schema(orig_blk1.segments + blk2.segments,
                               blk1.segments)

    def test__size(self):
        for block in self.blocks:
            targ = {
                'segments': len(block.segments),
                'groups': len(block.groups),  # only counts the top-level groups?
            }
            self.assertEqual(block.size, targ)

    def test__filter_none(self):
        for block in self.blocks:
            targ = []
            # collecting all data objects in target block
            for seg in block.segments:
                targ.extend(seg.analogsignals)
                targ.extend(seg.epochs)
                targ.extend(seg.events)
                targ.extend(seg.irregularlysampledsignals)
                targ.extend(seg.spiketrains)
                targ.extend(seg.imagesequences)
            chv_names = set([])
            for grp in block.groups:
                for grp1 in grp.walk():
                    for chv in grp1.channelviews:
                        if chv.name not in chv_names:
                            targ.append(chv)
                            chv_names.add(chv.name)

            res1 = block.filter()
            res2 = block.filter({})
            res3 = block.filter([])
            res4 = block.filter([{}])
            res5 = block.filter([{}, {}])
            res6 = block.filter([{}, {}])
            res7 = block.filter(targdict={})
            res8 = block.filter(targdict=[])
            res9 = block.filter(targdict=[{}])
            res10 = block.filter(targdict=[{}, {}])

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
        block = simple_block()

        targ = [
            block.segments[0].analogsignals[0],
            block.segments[0].spiketrains[1],
            block.segments[1].events[0]
        ]

        res0 = block.filter(thing="wotsit")
        res1 = block.filter({'thing': "wotsit"})
        res2 = block.filter(targdict={'thing': "wotsit"})
        res3 = block.filter([{'thing': "wotsit"}])
        res4 = block.filter(targdict=[{'thing': "wotsit"}])

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_single_annotation_nores(self):
        block = simple_block()

        res0 = block.filter(j=5)
        res1 = block.filter({'j': 5})
        res2 = block.filter(targdict={'j': 5})
        res3 = block.filter([{'j': 5}])
        res4 = block.filter(targdict=[{'j': 5}])

        self.assertEqual(len(res0), 0)
        self.assertEqual(len(res1), 0)
        self.assertEqual(len(res2), 0)
        self.assertEqual(len(res3), 0)
        self.assertEqual(len(res4), 0)

    def test__filter_attribute_single(self):
        block = simple_block()

        targ = [block.segments[1].analogsignals[0], block.segments[1].irregularlysampledsignals[0]]

        name = targ[0].name
        res0 = block.filter(name=name)
        res1 = block.filter({'name': name})
        res2 = block.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_attribute_single_nores(self):
        block = simple_block()

        name = "potato"
        res0 = block.filter(name=name)
        res1 = block.filter({'name': name})
        res2 = block.filter(targdict={'name': name})

        self.assertEqual(len(res0), 0)
        self.assertEqual(len(res1), 0)
        self.assertEqual(len(res2), 0)

    def test__filter_multi(self):

        block = simple_block()
        targ = [block.segments[1].analogsignals[0],
                block.segments[1].irregularlysampledsignals[0]]

        filter = {
            "name": targ[0].name,
            "thing": targ[0].annotations["thing"]
        }

        res0 = block.filter(**filter)
        res1 = block.filter(filter)
        res2 = block.filter(targdict=filter)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_nores(self):
        block = simple_block()
        targ = []

        filter = {
            "name": "carrot",
            "thing": "another thing"
        }

        res0 = block.filter(**filter)
        res1 = block.filter(filter)
        res2 = block.filter(targdict=filter)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_no_annotation_but_object(self):
        block = simple_block()
        targ = []
        for seg in block.segments:
            targ.extend(seg.events)
        res = block.filter(objects=Event)
        assert_same_sub_schema(res, targ)

        targ = []
        for seg in block.segments:
            targ.extend(seg.analogsignals)
        res = block.filter(objects=AnalogSignal)
        assert_same_sub_schema(res, targ)

        targ = []
        for seg in block.segments:
            targ.extend(seg.analogsignals)
            targ.extend(seg.events)
        res = block.filter(objects=[AnalogSignal, Event])
        assert_same_sub_schema(res, targ)

    def test__filter_single_annotation_obj_single(self):
        block = simple_block()
        targ = [block.segments[0].analogsignals[1]]

        res0 = block.filter(thing="frooble", objects='AnalogSignal')
        res1 = block.filter(thing="frooble", objects=AnalogSignal)
        res2 = block.filter(thing="frooble", objects=['AnalogSignal'])
        res3 = block.filter(thing="frooble", objects=[AnalogSignal])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)

    def test__filter_single_annotation_norecur(self):
        block = simple_block()
        targ = []
        res0 = block.filter(thing="frooble", recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_norecur(self):
        block = simple_block()
        targ = []
        res0 = block.filter(name=block.segments[1].analogsignals[0].name,
                            recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata(self):
        block = simple_block()
        targ = []
        res0 = block.filter(thing="frooble", data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata(self):
        block = simple_block()
        targ = []
        res0 = block.filter(name=block.segments[0].analogsignals[0], data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_norecur(self):
        block = simple_block()
        targ = []
        res0 = block.filter(thing="frooble",
                            data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_norecur(self):
        block = simple_block()
        targ = []
        res0 = block.filter(name=block.segments[0].analogsignals[0],
                            data=False, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        block = simple_block()

        targ = [
            block.segments[1].analogsignals[0],
            block.segments[1].irregularlysampledsignals[0],
            block.segments[1]
        ]

        res0 = block.filter(thing="amajig", container=True)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_data(self):
        block = simple_block()
        targ = [block.segments[1]]
        res0 = block.filter(name=targ[0].name, container=True)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container_norecur(self):
        block = simple_block()
        targ = [
            block.segments[1]
        ]
        res0 = block.filter(thing="amajig", container=True, recursive=False)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur(self):
        block = simple_block()
        targ = [
            block.segments[1]
        ]
        res0 = block.filter(name=targ[0].name, container=True, recursive=False)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container_norecur_nores(self):
        block = simple_block()
        targ = []
        res0 = block.filter(name="penguin",
                            container=True, recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_nodata_container(self):
        block = simple_block()

        targ = [
            block.segments[1]
        ]

        res0 = block.filter(thing="amajig", container=True, data=False)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_nores(self):
        block = simple_block()
        targ = []
        res0 = block.filter(name="narwhal",
                            data=False, container=True)
        assert_same_sub_schema(res0, targ)

    # def test__filter_single_annotation_nodata_container_norecur(self):
    #     targ = [self.segs1[1], self.chxs1[1]]
    #     res0 = block.filter(j=1,
    #                                data=False, container=True,
    #                                recursive=False)
    #     assert_same_sub_schema(res0, targ)

    # def test__filter_single_attribute_nodata_container_norecur(self):
    #     targ = [self.segs1[0]]
    #     res0 = block.filter(name=self.segs1[0].name,
    #                                data=False, container=True,
    #                                recursive=False)
    #     assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container_norecur_nores(self):
        block = simple_block()
        targ = []
        res0 = block.filter(name="puffin",
                            data=False, container=True,
                            recursive=False)
        assert_same_sub_schema(res0, targ)

    def test__filterdata_multi(self):
        block = simple_block()
        targ = [block.segments[1].analogsignals[0],
                block.segments[1].irregularlysampledsignals[0],
                block.segments[1]]
        data = block.children_recur

        filter = {
            "name": targ[0].name,
            "thing": targ[0].annotations["thing"]
        }

        res0 = filterdata(data, **filter)
        res1 = filterdata(data, filter)
        res2 = filterdata(data, targdict=filter)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filterdata_multi_nores(self):
        block = simple_block()
        targ = []
        data = block.children_recur

        name1 = block.segments[0].analogsignals[0].name,

        res0 = filterdata(data, [{"thing": "a good thing"}, {}])
        res1 = filterdata(data, {}, thing="a good thing")
        res2 = filterdata(data, [{}], thing="a good thing")
        res3 = filterdata(data, name=name1, targdict={"thing": "a good thing"})
        res4 = filterdata(data, {'name': name1}, thing="a good thing")
        res5 = filterdata(data, targdict={'name': name1}, thing="a good thing")
        res12 = filterdata(data, {'name': name1}, thing="a good thing")
        res13 = filterdata(data, targdict={'name': name1}, thing="a good thing")
        res14 = filterdata(data, name=name1, targdict={"thing": "a good thing"})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)
        assert_same_sub_schema(res5, targ)
        assert_same_sub_schema(res12, targ)
        assert_same_sub_schema(res13, targ)
        assert_same_sub_schema(res14, targ)

    # @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    # def test__pretty(self):
    #     res = pretty(self.blk1)
    #     ann = get_annotations()
    #     ann['seed'] = self.seed1
    #     ann = pretty(ann).replace('\n ', '\n  ')
    #
    #     seg0 = pretty(self.segs1[0])
    #     seg1 = pretty(self.segs1[1])
    #     seg0 = seg0.replace('\n', '\n   ')
    #     seg1 = seg1.replace('\n', '\n   ')
    #
    #     targ = ("Block with " +
    #             ("%s segments, %s channel_indexes\n" %
    #              (len(self.segs1), len(self.chxs1))) +
    #             ("name: '%s'\ndescription: '%s'\n" % (self.blk1.name,
    #                                                   self.blk1.description)) +
    #             ("annotations: %s\n" % ann) +
    #             ("file_origin: '%s'\n" % self.blk1.file_origin) +
    #             ("file_datetime: %s\n" % repr(self.blk1.file_datetime)) +
    #             ("rec_datetime: %s\n" % repr(self.blk1.rec_datetime)) +
    #             ("index: %s\n" % self.blk1.index) +
    #
    #
    #             ("# segments (N=%s)\n" % len(self.segs1)) +
    #             ('%s: %s\n' % (0, seg0)) +
    #             ('%s: %s' % (1, seg1)))
    #
    #     self.assertEqual(res, targ)

    def test__deepcopy(self):
        blk1 = self.blocks[0]
        blk1_copy = deepcopy(blk1)

        # Check links from parents to children and object attributes
        assert_same_sub_schema(blk1_copy, blk1)

        # Check links from children to parents
        for segment in blk1_copy.segments:
            self.assertEqual(id(segment.block), id(blk1_copy))
            for sig in segment.analogsignals:
                self.assertEqual(id(sig.segment), id(segment))
            for sptr in segment.spiketrains:
                self.assertEqual(id(sptr.segment), id(segment))


if __name__ == "__main__":
    unittest.main()
