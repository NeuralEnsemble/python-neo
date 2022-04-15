# -*- coding: utf-8 -*-
"""
Tests of the neo.core.spiketrainlist.SpikeTrainList class
"""

import sys

import unittest
import warnings
from copy import deepcopy

import numpy as np
from numpy.testing import assert_array_equal
import quantities as pq

from neo.core.spiketrain import SpikeTrain
from neo.core.spiketrainlist import SpikeTrainList
from neo.io.proxyobjects import SpikeTrainProxy


class MockRawIO(object):
    raw_annotations = {
        "blocks": [{
            "segments": [{
                "spikes": [{
                    "__array_annotations__": {}
                }]
            }]
        }]
    }
    header = {
        "spike_channels": [{
            'wf_sampling_rate': 5,
            'wf_left_sweep': 3,
            'wf_units': "mV"
        }],
    }

    def source_name(self):
        return "name_of_source"

    def segment_t_start(self, block_index=0, seg_index=0):
        return 0

    def segment_t_stop(self, block_index=0, seg_index=0):
        return 100.0

    def spike_count(self, block_index=0, seg_index=0, spike_channel_index=0):
        return 2

    def get_spike_timestamps(self, block_index=0, seg_index=0, spike_channel_index=0,
                             t_start=None, t_stop=None):
        return np.array([0.0011, 0.0885])

    def rescale_spike_timestamp(self, spike_timestamps, dtype='float64'):
        return spike_timestamps * pq.s


class TestSpikeTrainList(unittest.TestCase):

    def setUp(self):
        spike_time_array = np.array([0.5, 0.6, 0.7, 1.1, 11.2, 23.6, 88.5, 99.2])
        channel_id_array = np.array([0, 0, 1, 2, 1, 0, 2, 0])
        all_channel_ids = (0, 1, 2, 3)
        self.stl_from_array = SpikeTrainList.from_spike_time_array(
            spike_time_array,
            channel_id_array,
            all_channel_ids=all_channel_ids,
            units='ms',
            t_start=0 * pq.ms,
            t_stop=100.0 * pq.ms,
            identifier=["A", "B", "C", "D"],     # separate annotation for each SpikeTrain
            global_str="some string annotation",  # global annotations, same for each SpikeTrain
            global_int=42
        )

        self.stl_from_obj_list = SpikeTrainList(items=(
            SpikeTrain([0.5, 0.6, 23.6, 99.2], units="ms",
                       t_start=0 * pq.ms, t_stop=100.0 * pq.ms, channel_id=101),
            SpikeTrain([0.0007, 0.0112], units="s", t_start=0 * pq.ms, t_stop=100.0 * pq.ms,
                       channel_id=102),
            SpikeTrain([1100, 88500], units="us", t_start=0 * pq.ms, t_stop=100.0 * pq.ms,
                       channel_id=103),
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms,
                       channel_id=104),
        ))

        self.stl_from_obj_list_incl_proxy = SpikeTrainList(items=(
            SpikeTrain([0.5, 0.6, 23.6, 99.2], units="ms",
                       t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([0.0007, 0.0112], units="s", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrainProxy(rawio=MockRawIO(), spike_channel_index=0),
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
        ))

    def test_create_from_spiketrain_array(self):
        self.assertEqual(type(self.stl_from_array._spike_time_array), pq.Quantity)
        as_list = list(self.stl_from_array)
        assert_array_equal(as_list[0].times.magnitude,
                           np.array([0.5, 0.6, 23.6, 99.2]))
        assert_array_equal(as_list[1].times.magnitude,
                           np.array([0.7, 11.2]))
        assert_array_equal(as_list[2].times.magnitude,
                           np.array([1.1, 88.5]))
        assert_array_equal(as_list[3].times.magnitude,
                           np.array([]))
        self.assertEqual(as_list[0].annotations["identifier"], "A")
        self.assertEqual(as_list[1].annotations["identifier"], "B")
        self.assertEqual(as_list[2].annotations["identifier"], "C")
        self.assertEqual(as_list[3].annotations["identifier"], "D")
        self.assertEqual(as_list[0].annotations["global_str"], "some string annotation")
        self.assertEqual(as_list[3].annotations["global_str"], "some string annotation")
        self.assertEqual(as_list[2].annotations["global_int"], 42)
        self.assertEqual(as_list[1].annotations["global_int"], 42)
        self.assertEqual(self.stl_from_array.t_stop, 100.0 * pq.ms)
        self.assertEqual(self.stl_from_array.all_channel_ids, (0, 1, 2, 3))


    def test_create_from_spiketrain_list(self):
        as_list = list(self.stl_from_obj_list)
        assert_array_equal(as_list[0].times.rescale(pq.ms).magnitude,
                           np.array([0.5, 0.6, 23.6, 99.2]))
        assert_array_equal(as_list[1].times.rescale(pq.ms).magnitude,
                           np.array([0.7, 11.2]))
        assert_array_equal(as_list[2].times.rescale(pq.ms).magnitude,
                           np.array([1.1, 88.5]))
        assert_array_equal(as_list[3].times.rescale(pq.ms).magnitude,
                           np.array([]))
        self.assertAlmostEqual(self.stl_from_obj_list.t_stop, 100.0 * pq.ms)
        self.assertEqual(self.stl_from_obj_list.all_channel_ids, [101, 102, 103, 104])


    def test_create_from_spiketrain_list_incl_proxy(self):
        as_list = list(self.stl_from_obj_list_incl_proxy)
        assert_array_equal(as_list[0].times.rescale(pq.ms).magnitude,
                           np.array([0.5, 0.6, 23.6, 99.2]))
        assert_array_equal(as_list[1].times.rescale(pq.ms).magnitude,
                           np.array([0.7, 11.2]))
        assert isinstance(as_list[2], SpikeTrainProxy)
        assert_array_equal(as_list[3].times.rescale(pq.ms).magnitude,
                           np.array([]))
        self.assertAlmostEqual(self.stl_from_obj_list_incl_proxy.t_stop, 100.0 * pq.ms)
        self.assertEqual(self.stl_from_obj_list_incl_proxy.all_channel_ids, [0, 1, 2, 3])

    def test_str(self):
        target = "SpikeTrainList containing 8 spikes from 4 neurons"
        self.assertEqual(target, str(self.stl_from_array))
        target = ("[<SpikeTrain(array([ 0.5,  0.6, 23.6, 99.2]) * ms, [0.0 ms, 100.0 ms])>,"
                  " <SpikeTrain(array([0.0007, 0.0112]) * s, [0.0 s, 0.1 s])>,"
                  " <SpikeTrain(array([ 1100., 88500.]) * us, [0.0 us, 100000.00000000001 us])>,"
                  " <SpikeTrain(array([], dtype=float64) * ms, [0.0 ms, 100.0 ms])>]"
                  )
        self.assertEqual(target, str(self.stl_from_obj_list))

    def test_get_single_item(self):
        """Indexing a SpikeTrainList with a single integer should return a SpikeTrain"""
        for stl in (self.stl_from_obj_list, self.stl_from_array):
            st = stl[1]
            assert isinstance(st, SpikeTrain)
            assert_array_equal(st.times.rescale(pq.ms).magnitude, np.array([0.7, 11.2]))

    def test_get_slice(self):
        """Slicing a SpikeTrainList should return a SpikeTrainList"""
        for stl in (self.stl_from_obj_list, self.stl_from_array):
            new_stl = stl[1:3]
            self.assertIsInstance(new_stl, SpikeTrainList)
            self.assertEqual(len(new_stl), 2)

    def test_len(self):
        for stl in (self.stl_from_obj_list, self.stl_from_array):
            self.assertEqual(len(stl), 4)

    def test_add_spiketrainlists(self):
        """Adding two SpikeTrainLists should return a new SpikeTrainList object,
        whatever the internal representation being used by the two SpikeTrainLists."""
        a = self.stl_from_array
        b = self.stl_from_obj_list_incl_proxy
        c = a + b
        self.assertEqual(len(c), 8)
        self.assertEqual(len(a), 4)
        self.assertNotEqual(id(c), id(a))

        c = b + a
        self.assertEqual(len(c), 8)
        self.assertEqual(len(b), 4)
        self.assertNotEqual(id(c), id(b))

        b = deepcopy(a)
        b._all_channel_ids = [5, 6, 7, 8]
        c = a + b
        self.assertEqual(len(c), 8)
        self.assertEqual(len(a), 4)
        self.assertNotEqual(id(c), id(a))

    def test_iadd_spiketrainlists(self):
        """Adding a SpikeTrainLists to another in place should
        return the first SpikeTrainList object"""
        a = deepcopy(self.stl_from_array)
        b = self.stl_from_obj_list_incl_proxy
        c = a
        c += b
        self.assertEqual(len(c), 8)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(b), 4)
        self.assertEqual(id(c), id(a))

        a = self.stl_from_array
        b = deepcopy(self.stl_from_obj_list_incl_proxy)
        c = b
        c += a
        self.assertEqual(len(c), 8)
        self.assertEqual(len(b), 8)
        self.assertEqual(len(a), 4)
        self.assertEqual(id(c), id(b))

        a = deepcopy(self.stl_from_array)
        b = deepcopy(a)
        b._all_channel_ids = [5, 6, 7, 8]
        c = a
        c += b
        self.assertEqual(len(c), 8)
        self.assertEqual(len(a), 8)
        self.assertEqual(id(c), id(a))

    def test_add_list_of_spiketrains(self):
        """Adding a list of SpikeTrains to a SpikeTrainList should return a new SpikeTrainList"""
        extended_stl = self.stl_from_array + [
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([22.2, 33.3], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms), ]
        self.assertIsInstance(extended_stl, SpikeTrainList)
        self.assertEqual(len(extended_stl), 7)
        self.assertNotEqual(id(extended_stl), id(self.stl_from_array))

        extended_stl = self.stl_from_obj_list_incl_proxy + [
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([22.2, 33.3], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms)]
        self.assertIsInstance(extended_stl, SpikeTrainList)
        self.assertEqual(len(extended_stl), 7)

    def test_iadd_list_of_spiketrains(self):
        """Adding a list of SpikeTrains to a SpikeTrainList in place
        should return the original SpikeTrainList"""
        extended_stl = deepcopy(self.stl_from_array)
        extended_stl += [
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([22.2, 33.3], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms), ]
        self.assertIsInstance(extended_stl, SpikeTrainList)
        self.assertEqual(len(extended_stl), 7)

    def test_add_list_of_something_else(self):
        """Adding something that is not a list of SpikeTrains to a SpikeTrainList
        should return a plain list"""
        bag = self.stl_from_array + ["apples", "bananas"]
        self.assertIsInstance(bag, list)

    def test_radd_list_of_spiketrains(self):
        """ """
        extended_stl = [
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([22.2, 33.3], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms)
        ] + self.stl_from_array
        self.assertIsInstance(extended_stl, SpikeTrainList)
        self.assertEqual(len(extended_stl), 7)

        extended_stl = [
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([22.2, 33.3], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
            SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms)
        ] + self.stl_from_obj_list_incl_proxy
        self.assertIsInstance(extended_stl, SpikeTrainList)
        self.assertEqual(len(extended_stl), 7)

    def test_radd_list_of_something_else(self):
        """Adding a SpikeTrainList to something that is not a list of SpikeTrains
        should return a plain list"""
        bag = ["apples", "bananas"] + self.stl_from_array
        self.assertIsInstance(bag, list)

    def test_append(self):
        """Appending a SpikeTrain to a SpikeTrainList should make the STL longer"""
        for stl in (self.stl_from_obj_list, self.stl_from_array):
            stl.append(SpikeTrain([22.2, 33.3], units="ms",
                                  t_start=0 * pq.ms, t_stop=100.0 * pq.ms))
        self.assertEqual(len(stl), 5)

    def test_append_something_else(self):
        """Trying to append something other than a SpikeTrain should raise an Exception"""
        for stl in (self.stl_from_obj_list, self.stl_from_array):
            self.assertRaises(ValueError, stl.append, None)

    def test_multiplexed(self):
        """The multiplexed property should return a pair of arrays"""
        channel_id_array, spike_time_array = self.stl_from_array.multiplexed
        assert type(spike_time_array) == pq.Quantity
        assert type(channel_id_array) == np.ndarray
        assert_array_equal(channel_id_array, np.array([0, 0, 1, 2, 1, 0, 2, 0]))
        assert_array_equal(spike_time_array, np.array(
            [0.5, 0.6, 0.7, 1.1, 11.2, 23.6, 88.5, 99.2]) * pq.ms)

        channel_id_array, spike_time_array = self.stl_from_obj_list.multiplexed
        assert type(spike_time_array) == pq.Quantity
        assert type(channel_id_array) == np.ndarray
        assert_array_equal(channel_id_array, np.array([101, 101, 101, 101, 102, 102, 103, 103]))
        assert_array_equal(spike_time_array, np.array(
            [0.5, 0.6, 23.6, 99.2, 0.7, 11.2, 1.1, 88.5]) * pq.ms)
