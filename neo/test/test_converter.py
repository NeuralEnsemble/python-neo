"""
Tests of the neo.conversion module
"""

import unittest
import copy
import numpy as np

from neo.io.proxyobjects import (AnalogSignalProxy, SpikeTrainProxy,
                EventProxy, EpochProxy)

from neo.core import (Epoch, Event, SpikeTrain)
from neo.core.basesignal import BaseSignal

from neo.test.tools import (assert_arrays_equal, assert_same_attributes)
from neo.test.generate_datasets import fake_neo
from neo.converter import convert_channelindex_to_view_group

class ConversionTest(unittest.TestCase):
    def setUp(self):
        block = fake_neo(n=3)
        self.old_block = copy.deepcopy(block)
        self.new_block = convert_channelindex_to_view_group(block)

    def test_no_deprecated_attributes(self):
        self.assertFalse(hasattr(self.new_block, 'channel_indexes'))
        # collecting data objects
        objs = []
        for seg in self.new_block.segments:
            objs.extend(seg.analogsignals)
            objs.extend(seg.irregularlysampledsignals)
            objs.extend(seg.events)
            objs.extend(seg.epochs)
            objs.extend(seg.spiketrains)
            objs.extend(seg.imagesequences)

        for obj in objs:
            if isinstance(obj, BaseSignal):
                self.assertFalse(hasattr(obj, 'channel_index'))
            elif isinstance(obj, SpikeTrain):
                self.assertFalse(hasattr(obj, 'unit'))
            elif isinstance(obj, (Event, Epoch)):
                pass
            else:
                raise TypeError(f'Unexpected data type object {type(obj)}')

    def test_block_conversion(self):
        # verify that all previous data is present in new structure
        groups = self.new_block.groups
        for channel_index in self.old_block.channel_indexes:
            # check existence of objects and attributes
            self.assertIn(channel_index.name, [g.name for g in groups])
            group = groups[[g.name for g in groups].index(channel_index.name)]

            # comparing group attributes to channel_index attributes
            assert_same_attributes(group, channel_index)
            self.assertDictEqual(channel_index.annotations, group.annotations)

            # comparing views and their attributes
            view_names = np.asarray([v.name for v in group.channelviews])
            matching_views = np.asarray(group.channelviews)[view_names == channel_index.name]
            for view in matching_views:
                self.assertIn('channel_ids', view.array_annotations)
                self.assertIn('channel_names', view.array_annotations)
                self.assertIn('coordinates_dim0', view.array_annotations)
                self.assertIn('coordinates_dim1', view.array_annotations)

                # check content of attributes
                assert_arrays_equal(channel_index.index, view.index)
                assert_arrays_equal(channel_index.channel_ids, view.array_annotations['channel_ids'])
                assert_arrays_equal(channel_index.channel_names,
                                    view.array_annotations['channel_names'])
                view_coordinates = np.vstack((view.array_annotations['coordinates_dim0'],
                                              view.array_annotations['coordinates_dim1'])).T
                # readd unit lost during stacking of arrays
                units = view.array_annotations['coordinates_dim0'].units
                view_coordinates = view_coordinates.magnitude * units
                assert_arrays_equal(channel_index.coordinates, view_coordinates)
                self.assertDictEqual(channel_index.annotations, view.annotations)

            # check linking between objects
            self.assertEqual(len(channel_index.data_children), len(matching_views))

            # check linking between objects
            for child in channel_index.data_children:
                # comparing names instead of objects as attributes differ
                self.assertIn(child.name, [v.obj.name for v in matching_views])
            group_names = np.asarray([g.name for g in group.groups])
            for unit in channel_index.units:
                self.assertIn(unit.name, group_names)

            unit_names = np.asarray([u.name for u in channel_index.units])
            matching_groups = np.isin(group_names, unit_names)
            self.assertEqual(len(channel_index.units), len(matching_groups))
