# -*- coding: utf-8 -*-
# Copyright (c) 2016, German Neuroinformatics Node (G-Node)
#                     Achilleas Koutsou <achilleas.k@gmail.com>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted under the terms of the BSD License. See
# LICENSE file in the root of the Project.
"""
Tests for neo.io.nixio
"""

import os
from datetime import datetime

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock
import string
import itertools

import numpy as np
import quantities as pq

from neo.core import (Block, Segment, ChannelIndex, AnalogSignal,
                      IrregularlySampledSignal, Unit, SpikeTrain, Event, Epoch)
from neo.test.iotest.common_io_test import BaseTestIO

try:
    import nixio as nix
    HAVE_NIX = True
except ImportError:
    HAVE_NIX = False

from neo.io.nixio import NixIO
from neo.io.nixio import string_types


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOTest(unittest.TestCase):

    filename = None
    io = None

    def compare_blocks(self, neoblocks, nixblocks):
        for neoblock, nixblock in zip(neoblocks, nixblocks):
            self.compare_attr(neoblock, nixblock)
            self.assertEqual(len(neoblock.segments), len(nixblock.groups))
            for idx, neoseg in enumerate(neoblock.segments):
                nixgrp = nixblock.groups[neoseg.name]
                self.compare_segment_group(neoseg, nixgrp)
            for idx, neochx in enumerate(neoblock.channel_indexes):
                if neochx.name:
                    nixsrc = nixblock.sources[neochx.name]
                else:
                    nixsrc = nixblock.sources[idx]
                self.compare_chx_source(neochx, nixsrc)
            self.check_refs(neoblock, nixblock)

    def compare_chx_source(self, neochx, nixsrc):
        self.compare_attr(neochx, nixsrc)
        nix_channels = list(src for src in nixsrc.sources
                            if src.type == "neo.channelindex")
        self.assertEqual(len(neochx.index), len(nix_channels))
        for nixchan in nix_channels:
            nixchanidx = nixchan.metadata["index"]
            try:
                neochanpos = list(neochx.index).index(nixchanidx)
            except ValueError:
                self.fail("Channel indexes do not match.")
            if len(neochx.channel_names):
                neochanname = neochx.channel_names[neochanpos]
                if ((not isinstance(neochanname, str)) and
                        isinstance(neochanname, bytes)):
                    neochanname = neochanname.decode()
                nixchanname = nixchan.name
                self.assertEqual(neochanname, nixchanname)
        nix_units = list(src for src in nixsrc.sources
                         if src.type == "neo.unit")
        self.assertEqual(len(neochx.units), len(nix_units))
        for neounit in neochx.units:
            nixunit = nixsrc.sources[neounit.name]
            self.compare_attr(neounit, nixunit)

    def check_refs(self, neoblock, nixblock):
        """
        Checks whether the references between objects that are not nested are
        mapped correctly (e.g., SpikeTrains referenced by a Unit).

        :param neoblock: A Neo block
        :param nixblock: The corresponding NIX block
        """
        for idx, neochx in enumerate(neoblock.channel_indexes):
            if neochx.name:
                nixchx = nixblock.sources[neochx.name]
            else:
                nixchx = nixblock.sources[idx]
            # AnalogSignals referencing CHX
            neoasigs = list(sig.name for sig in neochx.analogsignals)
            nixasigs = list(set(da.metadata.name for da in nixblock.data_arrays
                                if da.type == "neo.analogsignal" and
                                nixchx in da.sources))

            self.assertEqual(len(neoasigs), len(nixasigs))

            # IrregularlySampledSignals referencing CHX
            neoisigs = list(sig.name for sig in
                            neochx.irregularlysampledsignals)
            nixisigs = list(
                set(da.metadata.name for da in nixblock.data_arrays
                    if da.type == "neo.irregularlysampledsignal" and
                    nixchx in da.sources)
            )
            self.assertEqual(len(neoisigs), len(nixisigs))
            # SpikeTrains referencing CHX and Units
            for sidx, neounit in enumerate(neochx.units):
                if neounit.name:
                    nixunit = nixchx.sources[neounit.name]
                else:
                    nixunit = nixchx.sources[sidx]
                neosts = list(st.name for st in neounit.spiketrains)
                nixsts = list(mt for mt in nixblock.multi_tags
                              if mt.type == "neo.spiketrain" and
                              nixunit.name in mt.sources)
                # SpikeTrains must also reference CHX
                for nixst in nixsts:
                    self.assertIn(nixchx.name, nixst.sources)
                nixsts = list(st.name for st in nixsts)
                self.assertEqual(len(neosts), len(nixsts))
                for neoname in neosts:
                    if neoname:
                        self.assertIn(neoname, nixsts)

        # Events and Epochs must reference all Signals in the Group (NIX only)
        for nixgroup in nixblock.groups:
            nixevep = list(mt for mt in nixgroup.multi_tags
                           if mt.type in ["neo.event", "neo.epoch"])
            nixsigs = list(da.name for da in nixgroup.data_arrays
                           if da.type in ["neo.analogsignal",
                                          "neo.irregularlysampledsignal"])
            for nee in nixevep:
                for ns in nixsigs:
                    self.assertIn(ns, nee.references)

    def compare_segment_group(self, neoseg, nixgroup):
        self.compare_attr(neoseg, nixgroup)
        neo_signals = neoseg.analogsignals + neoseg.irregularlysampledsignals
        self.compare_signals_das(neo_signals, nixgroup.data_arrays)
        neo_eests = neoseg.epochs + neoseg.events + neoseg.spiketrains
        self.compare_eests_mtags(neo_eests, nixgroup.multi_tags)

    def compare_signals_das(self, neosignals, data_arrays):
        for sig in neosignals:
            if self.io._find_lazy_loaded(sig) is not None:
                sig = self.io.load_lazy_object(sig)
            dalist = list()
            for idx in itertools.count():
                nixname = "{}.{}".format(sig.name, idx)
                if nixname in data_arrays:
                    dalist.append(data_arrays[nixname])
                else:
                    break
            _, nsig = np.shape(sig)
            self.assertEqual(nsig, len(dalist))
            self.compare_signal_dalist(sig, dalist)

    def compare_signal_dalist(self, neosig, nixdalist):
        """
        Check if a Neo Analog or IrregularlySampledSignal matches a list of
        NIX DataArrays.

        :param neosig: Neo Analog or IrregularlySampledSignal
        :param nixdalist: List of DataArrays
        """
        nixmd = nixdalist[0].metadata
        self.assertTrue(all(nixmd == da.metadata for da in nixdalist))
        neounit = str(neosig.dimensionality)
        for sig, da in zip(np.transpose(neosig),
                           sorted(nixdalist, key=lambda d: d.name)):
            self.compare_attr(neosig, da)
            np.testing.assert_almost_equal(sig.magnitude, da)
            self.assertEqual(neounit, da.unit)
            timedim = da.dimensions[0]
            if isinstance(neosig, AnalogSignal):
                self.assertIsInstance(timedim, nix.pycore.SampledDimension)
                self.assertEqual(
                    pq.Quantity(timedim.sampling_interval, timedim.unit),
                    neosig.sampling_period
                )
                self.assertEqual(timedim.offset, neosig.t_start.magnitude)
                if "t_start.units" in da.metadata.props:
                    self.assertEqual(da.metadata["t_start.units"],
                                     str(neosig.t_start.dimensionality))
            elif isinstance(neosig, IrregularlySampledSignal):
                self.assertIsInstance(timedim, nix.pycore.RangeDimension)
                np.testing.assert_almost_equal(neosig.times.magnitude,
                                               timedim.ticks)
                self.assertEqual(timedim.unit,
                                 str(neosig.times.dimensionality))

    def compare_eests_mtags(self, eestlist, mtaglist):
        self.assertEqual(len(eestlist), len(mtaglist))
        for eest in eestlist:
            if self.io._find_lazy_loaded(eest) is not None:
                eest = self.io.load_lazy_object(eest)
            mtag = mtaglist[eest.name]
            if isinstance(eest, Epoch):
                self.compare_epoch_mtag(eest, mtag)
            elif isinstance(eest, Event):
                self.compare_event_mtag(eest, mtag)
            elif isinstance(eest, SpikeTrain):
                self.compare_spiketrain_mtag(eest, mtag)

    def compare_epoch_mtag(self, epoch, mtag):
        self.assertEqual(mtag.type, "neo.epoch")
        self.compare_attr(epoch, mtag)
        np.testing.assert_almost_equal(epoch.times.magnitude, mtag.positions)

        np.testing.assert_almost_equal(epoch.durations.magnitude, mtag.extents)
        self.assertEqual(mtag.positions.unit,
                         str(epoch.times.units.dimensionality))
        self.assertEqual(mtag.extents.unit,
                         str(epoch.durations.units.dimensionality))
        for neol, nixl in zip(epoch.labels,
                              mtag.positions.dimensions[0].labels):
            # Dirty. Should find the root cause instead
            if isinstance(neol, bytes):
                neol = neol.decode()
            if isinstance(nixl, bytes):
                nixl = nixl.decode()
            self.assertEqual(neol, nixl)

    def compare_event_mtag(self, event, mtag):
        self.assertEqual(mtag.type, "neo.event")
        self.compare_attr(event, mtag)
        np.testing.assert_almost_equal(event.times.magnitude, mtag.positions)
        self.assertEqual(mtag.positions.unit, str(event.units.dimensionality))
        for neol, nixl in zip(event.labels,
                              mtag.positions.dimensions[0].labels):
            # Dirty. Should find the root cause instead
            # Only happens in 3.2
            if isinstance(neol, bytes):
                neol = neol.decode()
            if isinstance(nixl, bytes):
                nixl = nixl.decode()
            self.assertEqual(neol, nixl)

    def compare_spiketrain_mtag(self, spiketrain, mtag):
        self.assertEqual(mtag.type, "neo.spiketrain")
        self.compare_attr(spiketrain, mtag)
        np.testing.assert_almost_equal(spiketrain.times.magnitude,
                                       mtag.positions)
        if len(mtag.features):
            neowf = spiketrain.waveforms
            nixwf = mtag.features[0].data
            self.assertEqual(np.shape(neowf), np.shape(nixwf))
            self.assertEqual(nixwf.unit, str(neowf.units.dimensionality))
            np.testing.assert_almost_equal(neowf.magnitude, nixwf)
            self.assertIsInstance(nixwf.dimensions[0], nix.pycore.SetDimension)
            self.assertIsInstance(nixwf.dimensions[1], nix.pycore.SetDimension)
            self.assertIsInstance(nixwf.dimensions[2],
                                  nix.pycore.SampledDimension)

    def compare_attr(self, neoobj, nixobj):
        if neoobj.name:
            if isinstance(neoobj, (AnalogSignal, IrregularlySampledSignal)):
                nix_name = ".".join(nixobj.name.split(".")[:-1])
            else:
                nix_name = nixobj.name
            self.assertEqual(neoobj.name, nix_name)
        self.assertEqual(neoobj.description, nixobj.definition)
        if hasattr(neoobj, "rec_datetime") and neoobj.rec_datetime:
            self.assertEqual(neoobj.rec_datetime,
                             datetime.fromtimestamp(nixobj.created_at))
        if hasattr(neoobj, "file_datetime") and neoobj.file_datetime:
            self.assertEqual(neoobj.file_datetime,
                             datetime.fromtimestamp(
                                 nixobj.metadata["file_datetime"]))
        if neoobj.annotations:
            nixmd = nixobj.metadata
            for k, v, in neoobj.annotations.items():
                if isinstance(v, pq.Quantity):
                    self.assertEqual(nixmd.props[str(k)].unit,
                                     str(v.dimensionality))
                    np.testing.assert_almost_equal(nixmd[str(k)],
                                                   v.magnitude)
                else:
                    self.assertEqual(nixmd[str(k)], v)

    @classmethod
    def create_full_nix_file(cls, filename):
        nixfile = nix.File.open(filename, nix.FileMode.Overwrite,
                                backend="h5py")

        nix_block_a = nixfile.create_block(cls.rword(10), "neo.block")
        nix_block_a.definition = cls.rsentence(5, 10)
        nix_block_b = nixfile.create_block(cls.rword(10), "neo.block")
        nix_block_b.definition = cls.rsentence(3, 3)

        nix_block_a.metadata = nixfile.create_section(
            nix_block_a.name, nix_block_a.name+".metadata"
        )

        nix_block_b.metadata = nixfile.create_section(
            nix_block_b.name, nix_block_b.name+".metadata"
        )

        nix_blocks = [nix_block_a, nix_block_b]

        for blk in nix_blocks:
            for ind in range(3):
                group = blk.create_group(cls.rword(), "neo.segment")
                group.definition = cls.rsentence(10, 15)

                group_md = blk.metadata.create_section(group.name,
                                                       group.name+".metadata")
                group.metadata = group_md

        blk = nix_blocks[0]
        group = blk.groups[0]
        allspiketrains = list()
        allsignalgroups = list()

        # analogsignals
        for n in range(3):
            siggroup = list()
            asig_name = "{}_asig{}".format(cls.rword(10), n)
            asig_definition = cls.rsentence(5, 5)
            asig_md = group.metadata.create_section(asig_name,
                                                    asig_name+".metadata")
            for idx in range(3):
                da_asig = blk.create_data_array(
                    "{}.{}".format(asig_name, idx),
                    "neo.analogsignal",
                    data=cls.rquant(100, 1)
                )
                da_asig.definition = asig_definition
                da_asig.unit = "mV"

                da_asig.metadata = asig_md

                timedim = da_asig.append_sampled_dimension(0.01)
                timedim.unit = "ms"
                timedim.label = "time"
                timedim.offset = 10
                da_asig.append_set_dimension()
                group.data_arrays.append(da_asig)
                siggroup.append(da_asig)
            allsignalgroups.append(siggroup)

        # irregularlysampledsignals
        for n in range(2):
            siggroup = list()
            isig_name = "{}_isig{}".format(cls.rword(10), n)
            isig_definition = cls.rsentence(12, 12)
            isig_md = group.metadata.create_section(isig_name,
                                                    isig_name+".metadata")
            isig_times = cls.rquant(200, 1, True)
            for idx in range(10):
                da_isig = blk.create_data_array(
                    "{}.{}".format(isig_name, idx),
                    "neo.irregularlysampledsignal",
                    data=cls.rquant(200, 1)
                )
                da_isig.definition = isig_definition
                da_isig.unit = "mV"

                da_isig.metadata = isig_md

                timedim = da_isig.append_range_dimension(isig_times)
                timedim.unit = "s"
                timedim.label = "time"
                da_isig.append_set_dimension()
                group.data_arrays.append(da_isig)
                siggroup.append(da_isig)
            allsignalgroups.append(siggroup)

        # SpikeTrains with Waveforms
        for n in range(4):
            stname = "{}-st{}".format(cls.rword(20), n)
            times = cls.rquant(400, 1, True)
            times_da = blk.create_data_array(
                "{}.times".format(stname),
                "neo.spiketrain.times",
                data=times
            )
            times_da.unit = "ms"
            mtag_st = blk.create_multi_tag(stname,
                                           "neo.spiketrain",
                                           times_da)
            group.multi_tags.append(mtag_st)
            mtag_st.definition = cls.rsentence(20, 30)
            mtag_st_md = group.metadata.create_section(
                mtag_st.name, mtag_st.name+".metadata"
            )
            mtag_st.metadata = mtag_st_md
            mtag_st_md.create_property(
                "t_stop", nix.Value(max(times_da).item()+1)
            )

            waveforms = cls.rquant((10, 8, 5), 1)
            wfname = "{}.waveforms".format(mtag_st.name)
            wfda = blk.create_data_array(wfname, "neo.waveforms",
                                         data=waveforms)
            wfda.unit = "mV"
            mtag_st.create_feature(wfda, nix.LinkType.Indexed)
            wfda.append_set_dimension()  # spike dimension
            wfda.append_set_dimension()  # channel dimension
            wftimedim = wfda.append_sampled_dimension(0.1)
            wftimedim.unit = "ms"
            wftimedim.label = "time"
            wfda.metadata = mtag_st_md.create_section(
                wfname, "neo.waveforms.metadata"
            )
            wfda.metadata.create_property("left_sweep",
                                          [nix.Value(20)]*5)
            allspiketrains.append(mtag_st)

        # Epochs
        for n in range(3):
            epname = "{}-ep{}".format(cls.rword(5), n)
            times = cls.rquant(5, 1, True)
            times_da = blk.create_data_array(
                "{}.times".format(epname),
                "neo.epoch.times",
                data=times
            )
            times_da.unit = "s"

            extents = cls.rquant(5, 1)
            extents_da = blk.create_data_array(
                "{}.durations".format(epname),
                "neo.epoch.durations",
                data=extents
            )
            extents_da.unit = "s"

            mtag_ep = blk.create_multi_tag(
                epname, "neo.epoch", times_da
            )
            group.multi_tags.append(mtag_ep)
            mtag_ep.definition = cls.rsentence(2)
            mtag_ep.extents = extents_da
            label_dim = mtag_ep.positions.append_set_dimension()
            label_dim.labels = cls.rsentence(5).split(" ")
            # reference all signals in the group
            for siggroup in allsignalgroups:
                mtag_ep.references.extend(siggroup)

        # Events
        for n in range(2):
            evname = "{}-ev{}".format(cls.rword(5), n)
            times = cls.rquant(5, 1, True)
            times_da = blk.create_data_array(
                "{}.times".format(evname),
                "neo.event.times",
                data=times
            )
            times_da.unit = "s"

            mtag_ev = blk.create_multi_tag(
                evname, "neo.event", times_da
            )
            group.multi_tags.append(mtag_ev)
            mtag_ev.definition = cls.rsentence(2)
            label_dim = mtag_ev.positions.append_set_dimension()
            label_dim.labels = cls.rsentence(5).split(" ")
            # reference all signals in the group
            for siggroup in allsignalgroups:
                mtag_ev.references.extend(siggroup)

        # CHX
        nixchx = blk.create_source(cls.rword(10),
                                   "neo.channelindex")
        nixchx.metadata = nix_blocks[0].metadata.create_section(
            nixchx.name, "neo.channelindex.metadata"
        )
        chantype = "neo.channelindex"
        # 3 channels
        for idx in [2, 5, 9]:
            channame = cls.rword(20)
            nixrc = nixchx.create_source(channame, chantype)
            nixrc.definition = cls.rsentence(13)
            nixrc.metadata = nixchx.metadata.create_section(
                nixrc.name, "neo.channelindex.metadata"
            )
            nixrc.metadata.create_property("index", nix.Value(idx))
            dims = tuple(map(nix.Value, cls.rquant(3, 1)))
            nixrc.metadata.create_property("coordinates", dims)
            nixrc.metadata.create_property("coordinates.units",
                                           nix.Value("um"))

        nunits = 1
        stsperunit = np.array_split(allspiketrains, nunits)
        for idx in range(nunits):
            unitname = "{}-unit{}".format(cls.rword(5), idx)
            nixunit = nixchx.create_source(unitname, "neo.unit")
            nixunit.definition = cls.rsentence(4, 10)
            for st in stsperunit[idx]:
                st.sources.append(nixchx)
                st.sources.append(nixunit)

        # pick a few signal groups to reference this CHX
        randsiggroups = np.random.choice(allsignalgroups, 5, False)
        for siggroup in randsiggroups:
            for sig in siggroup:
                sig.sources.append(nixchx)

        return nixfile

    @staticmethod
    def rdate():
        return datetime(year=np.random.randint(1980, 2020),
                        month=np.random.randint(1, 13),
                        day=np.random.randint(1, 29))

    @classmethod
    def populate_dates(cls, obj):
        obj.file_datetime = cls.rdate()
        obj.rec_datetime = cls.rdate()

    @staticmethod
    def rword(n=10):
        return "".join(np.random.choice(list(string.ascii_letters), n))

    @classmethod
    def rsentence(cls, n=3, maxwl=10):
        return " ".join(cls.rword(np.random.randint(1, maxwl))
                        for _ in range(n))

    @classmethod
    def rdict(cls, nitems):
        rd = dict()
        for _ in range(nitems):
            key = cls.rword()
            value = cls.rword() if np.random.choice((0, 1)) \
                else np.random.uniform()
            rd[key] = value
        return rd

    @staticmethod
    def rquant(shape, unit, incr=False):
        try:
            dim = len(shape)
        except TypeError:
            dim = 1
        if incr and dim > 1:
            raise TypeError("Shape of quantity array may only be "
                            "one-dimensional when incremental values are "
                            "requested.")
        arr = np.random.random(shape)
        if incr:
            arr = np.array(np.cumsum(arr))
        return arr*unit

    @classmethod
    def create_all_annotated(cls):
        times = cls.rquant(1, pq.s)
        signal = cls.rquant(1, pq.V)
        blk = Block()
        blk.annotate(**cls.rdict(3))

        seg = Segment()
        seg.annotate(**cls.rdict(4))
        blk.segments.append(seg)

        asig = AnalogSignal(signal=signal, sampling_rate=pq.Hz)
        asig.annotate(**cls.rdict(2))
        seg.analogsignals.append(asig)

        isig = IrregularlySampledSignal(times=times, signal=signal,
                                        time_units=pq.s)
        isig.annotate(**cls.rdict(2))
        seg.irregularlysampledsignals.append(isig)

        epoch = Epoch(times=times, durations=times)
        epoch.annotate(**cls.rdict(4))
        seg.epochs.append(epoch)

        event = Event(times=times)
        event.annotate(**cls.rdict(4))
        seg.events.append(event)

        spiketrain = SpikeTrain(times=times, t_stop=pq.s, units=pq.s)
        d = cls.rdict(6)
        d["quantity"] = pq.Quantity(10, "mV")
        d["qarray"] = pq.Quantity(range(10), "mA")
        spiketrain.annotate(**d)
        seg.spiketrains.append(spiketrain)

        chx = ChannelIndex(name="achx", index=[1, 2])
        chx.annotate(**cls.rdict(5))
        blk.channel_indexes.append(chx)

        unit = Unit()
        unit.annotate(**cls.rdict(2))
        chx.units.append(unit)

        return blk


class NixIOWriteTest(NixIOTest):

    def setUp(self):
        self.filename = "nixio_testfile_write.h5"
        self.writer = NixIO(self.filename, "ow")
        self.io = self.writer
        self.reader = nix.File.open(self.filename,
                                    nix.FileMode.ReadOnly,
                                    backend="h5py")

    def tearDown(self):
        self.writer.close()
        self.reader.close()
        os.remove(self.filename)

    def write_and_compare(self, blocks):
        self.writer.write_all_blocks(blocks)
        self.compare_blocks(self.writer.read_all_blocks(), self.reader.blocks)

    def test_block_write(self):
        block = Block(name=self.rword(),
                      description=self.rsentence())
        self.write_and_compare([block])

        block.annotate(**self.rdict(5))
        self.write_and_compare([block])

    def test_segment_write(self):
        block = Block(name=self.rword())
        segment = Segment(name=self.rword(), description=self.rword())
        block.segments.append(segment)
        self.write_and_compare([block])

        segment.annotate(**self.rdict(2))
        self.write_and_compare([block])

    def test_channel_index_write(self):
        block = Block(name=self.rword())
        chx = ChannelIndex(name=self.rword(),
                           description=self.rsentence(),
                           index=[1, 2, 3, 5, 8, 13])
        block.channel_indexes.append(chx)
        self.write_and_compare([block])

        chx.annotate(**self.rdict(3))
        self.write_and_compare([block])

    def test_signals_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        asig = AnalogSignal(signal=self.rquant((10, 3), pq.mV),
                            sampling_rate=pq.Quantity(10, "Hz"))
        seg.analogsignals.append(asig)
        self.write_and_compare([block])

        anotherblock = Block("ir signal block")
        seg = Segment("ir signal seg")
        anotherblock.segments.append(seg)
        irsig = IrregularlySampledSignal(
            signal=np.random.random((20, 3)),
            times=self.rquant(20, pq.ms, True),
            units=pq.A
        )
        seg.irregularlysampledsignals.append(irsig)
        self.write_and_compare([anotherblock])

        block.segments[0].analogsignals.append(
            AnalogSignal(signal=[10.0, 1.0, 3.0], units=pq.S,
                         sampling_period=pq.Quantity(3, "s"),
                         dtype=np.double, name="signal42",
                         description="this is an analogsignal",
                         t_start=45 * pq.ms),
        )
        self.write_and_compare([block, anotherblock])

        block.segments[0].irregularlysampledsignals.append(
            IrregularlySampledSignal(times=np.random.random(10),
                                     signal=np.random.random((10, 3)),
                                     units="mV", time_units="s",
                                     dtype=np.float,
                                     name="some sort of signal",
                                     description="the signal is described")
        )
        self.write_and_compare([block, anotherblock])

    def test_epoch_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        epoch = Epoch(times=[1, 1, 10, 3]*pq.ms, durations=[3, 3, 3, 1]*pq.ms,
                      labels=np.array(["one", "two", "three", "four"]),
                      name="test epoch", description="an epoch for testing")

        seg.epochs.append(epoch)
        self.write_and_compare([block])

    def test_event_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        event = Event(times=np.arange(0, 30, 10)*pq.s,
                      labels=np.array(["0", "1", "2"]),
                      name="event name",
                      description="event description")
        seg.events.append(event)
        self.write_and_compare([block])

    def test_spiketrain_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        spiketrain = SpikeTrain(times=[3, 4, 5]*pq.s, t_stop=10.0,
                                name="spikes!", description="sssssspikes")
        seg.spiketrains.append(spiketrain)
        self.write_and_compare([block])

        waveforms = self.rquant((3, 5, 10), pq.mV)
        spiketrain = SpikeTrain(times=[1, 1.1, 1.2]*pq.ms, t_stop=1.5*pq.s,
                                name="spikes with wf",
                                description="spikes for waveform test",
                                waveforms=waveforms)

        seg.spiketrains.append(spiketrain)
        self.write_and_compare([block])

        spiketrain.left_sweep = np.random.random(10)*pq.ms
        self.write_and_compare([block])

    def test_metadata_structure_write(self):
        neoblk = self.create_all_annotated()
        self.io.write_block(neoblk)
        blk = self.io.nix_file.blocks[0]

        blkmd = blk.metadata
        self.assertEqual(blk.name, blkmd.name)

        grp = blk.groups[0]  # segment
        self.assertIn(grp.name, blkmd.sections)

        grpmd = blkmd.sections[grp.name]
        for da in grp.data_arrays:  # signals
            name = ".".join(da.name.split(".")[:-1])
            self.assertIn(name, grpmd.sections)
        for mtag in grp.multi_tags:  # spiketrains, events, and epochs
            self.assertIn(mtag.name, grpmd.sections)

        srcchx = blk.sources[0]  # chx
        self.assertIn(srcchx.name, blkmd.sections)

        for srcunit in blk.sources:  # units
            self.assertIn(srcunit.name, blkmd.sections)

        self.write_and_compare([neoblk])

    def test_anonymous_objects_write(self):
        nblocks = 2
        nsegs = 2
        nanasig = 4
        nirrseg = 2
        nepochs = 3
        nevents = 4
        nspiketrains = 3
        nchx = 5
        nunits = 10

        times = self.rquant(1, pq.s)
        signal = self.rquant(1, pq.V)
        blocks = []
        for blkidx in range(nblocks):
            blk = Block()
            blocks.append(blk)
            for segidx in range(nsegs):
                seg = Segment()
                blk.segments.append(seg)
                for anaidx in range(nanasig):
                    seg.analogsignals.append(AnalogSignal(signal=signal,
                                                          sampling_rate=pq.Hz))
                for irridx in range(nirrseg):
                    seg.irregularlysampledsignals.append(
                        IrregularlySampledSignal(times=times,
                                                 signal=signal,
                                                 time_units=pq.s)
                    )
                for epidx in range(nepochs):
                    seg.epochs.append(Epoch(times=times, durations=times))
                for evidx in range(nevents):
                    seg.events.append(Event(times=times))
                for stidx in range(nspiketrains):
                    seg.spiketrains.append(SpikeTrain(times=times, t_stop=pq.s,
                                                      units=pq.s))
            for chidx in range(nchx):
                chx = ChannelIndex(name="chx{}".format(chidx),
                                   index=[1, 2])
                blk.channel_indexes.append(chx)
                for unidx in range(nunits):
                    unit = Unit()
                    chx.units.append(unit)
        self.writer.write_all_blocks(blocks)
        self.compare_blocks(blocks, self.reader.blocks)

    def test_to_value(self):
        section = self.io.nix_file.create_section("Metadata value test",
                                                  "Test")
        writeprop = self.io._write_property

        # quantity
        qvalue = pq.Quantity(10, "mV")
        writeprop(section, "qvalue", qvalue)
        self.assertEqual(section["qvalue"], 10)
        self.assertEqual(section.props["qvalue"].unit, "mV")

        # datetime
        dt = self.rdate()
        writeprop(section, "dt", dt)
        self.assertEqual(datetime.fromtimestamp(section["dt"]), dt)

        # string
        randstr = self.rsentence()
        writeprop(section, "randstr", randstr)
        self.assertEqual(section["randstr"], randstr)

        # bytes
        bytestring = b"bytestring"
        writeprop(section, "randbytes", bytestring)
        self.assertEqual(section["randbytes"], bytestring.decode())

        # iterables
        randlist = np.random.random(10).tolist()
        writeprop(section, "randlist", randlist)
        self.assertEqual(randlist, section["randlist"])

        randarray = np.random.random(10)
        writeprop(section, "randarray", randarray)
        np.testing.assert_almost_equal(randarray, section["randarray"])

        # numpy item
        npval = np.float64(2398)
        writeprop(section, "npval", npval)
        self.assertEqual(npval, section["npval"])

        # number
        val = 42
        writeprop(section, "val", val)
        self.assertEqual(val, section["val"])


class NixIOReadTest(NixIOTest):

    filename = "testfile_readtest.h5"
    nixfile = None
    nix_blocks = None
    original_methods = dict()

    @classmethod
    def setUpClass(cls):
        if HAVE_NIX:
            cls.nixfile = cls.create_full_nix_file(cls.filename)

    def setUp(self):
        self.io = NixIO(self.filename, "ro")
        self.original_methods["_read_cascade"] = self.io._read_cascade
        self.original_methods["_update_maps"] = self.io._update_maps

    @classmethod
    def tearDownClass(cls):
        if HAVE_NIX:
            cls.nixfile.close()

    def tearDown(self):
        self.io.close()

    def test_all_read(self):
        neo_blocks = self.io.read_all_blocks(cascade=True, lazy=False)
        nix_blocks = self.io.nix_file.blocks
        self.compare_blocks(neo_blocks, nix_blocks)

    def test_lazyload_fullcascade_read(self):
        neo_blocks = self.io.read_all_blocks(cascade=True, lazy=True)
        nix_blocks = self.io.nix_file.blocks
        # data objects should be empty
        for block in neo_blocks:
            for seg in block.segments:
                for asig in seg.analogsignals:
                    self.assertEqual(len(asig), 0)
                for isig in seg.irregularlysampledsignals:
                    self.assertEqual(len(isig), 0)
                for epoch in seg.epochs:
                    self.assertEqual(len(epoch), 0)
                for event in seg.events:
                    self.assertEqual(len(event), 0)
                for st in seg.spiketrains:
                    self.assertEqual(len(st), 0)
        self.compare_blocks(neo_blocks, nix_blocks)

    def test_lazyload_lazycascade_read(self):
        neo_blocks = self.io.read_all_blocks(cascade="lazy", lazy=True)
        nix_blocks = self.io.nix_file.blocks
        self.compare_blocks(neo_blocks, nix_blocks)

    def test_lazycascade_read(self):
        def getitem(self, index):
            return self._data.__getitem__(index)
        from neo.io.nixio import LazyList
        getitem_original = LazyList.__getitem__
        LazyList.__getitem__ = getitem
        neo_blocks = self.io.read_all_blocks(cascade="lazy", lazy=False)
        for block in neo_blocks:
            self.assertIsInstance(block.segments, LazyList)
            self.assertIsInstance(block.channel_indexes, LazyList)
            for seg in block.segments:
                self.assertIsInstance(seg, string_types)
            for chx in block.channel_indexes:
                self.assertIsInstance(chx, string_types)
        LazyList.__getitem__ = getitem_original

    def test_load_lazy_cascade(self):
        from neo.io.nixio import LazyList
        neo_blocks = self.io.read_all_blocks(cascade="lazy", lazy=False)
        for block in neo_blocks:
            self.assertIsInstance(block.segments, LazyList)
            self.assertIsInstance(block.channel_indexes, LazyList)
            name = block.name
            block = self.io.load_lazy_cascade("/" + name, lazy=False)
            self.assertIsInstance(block.segments, list)
            self.assertIsInstance(block.channel_indexes, list)
            for seg in block.segments:
                self.assertIsInstance(seg.analogsignals, list)
                self.assertIsInstance(seg.irregularlysampledsignals, list)
                self.assertIsInstance(seg.epochs, list)
                self.assertIsInstance(seg.events, list)
                self.assertIsInstance(seg.spiketrains, list)

    def test_nocascade_read(self):
        self.io._read_cascade = mock.Mock()
        neo_blocks = self.io.read_all_blocks(cascade=False)
        self.io._read_cascade.assert_not_called()
        for block in neo_blocks:
            self.assertEqual(len(block.segments), 0)
            nix_block = self.io.nix_file.blocks[block.name]
            self.compare_attr(block, nix_block)

    def test_lazy_load_subschema(self):
        blk = self.io.nix_file.blocks[0]
        segpath = "/" + blk.name + "/segments/" + blk.groups[0].name
        segment = self.io.load_lazy_cascade(segpath, lazy=True)
        self.assertIsInstance(segment, Segment)
        self.assertEqual(segment.name, blk.groups[0].name)
        self.assertIs(segment.block, None)
        self.assertEqual(len(segment.analogsignals[0]), 0)
        segment = self.io.load_lazy_cascade(segpath, lazy=False)
        self.assertEqual(np.shape(segment.analogsignals[0]), (100, 3))


class NixIOHashTest(NixIOTest):

    def setUp(self):
        self.hash = NixIO._hash_object

    def _hash_test(self, objtype, argfuncs):
        attr = {}
        for arg, func in argfuncs.items():
            attr[arg] = func()

        obj_one = objtype(**attr)
        obj_two = objtype(**attr)
        hash_one = self.hash(obj_one)
        hash_two = self.hash(obj_two)
        self.assertEqual(hash_one, hash_two)

        for arg, func in argfuncs.items():
            chattr = attr.copy()
            chattr[arg] = func()
            obj_two = objtype(**chattr)
            hash_two = self.hash(obj_two)
            self.assertNotEqual(
                hash_one, hash_two,
                "Hash test failed with different '{}'".format(arg)
            )

    def test_block_seg_hash(self):
        argfuncs = {"name": self.rword,
                    "description": self.rsentence,
                    "rec_datetime": self.rdate,
                    "file_datetime": self.rdate,
                    # annotations
                    self.rword(): self.rword,
                    self.rword(): lambda: self.rquant((10, 10), pq.mV)}
        self._hash_test(Block, argfuncs)
        self._hash_test(Segment, argfuncs)
        self._hash_test(Unit, argfuncs)

    def test_chx_hash(self):
        argfuncs = {"name": self.rword,
                    "description": self.rsentence,
                    "index": lambda: np.random.random(10).tolist(),
                    "channel_names": lambda: self.rsentence(10).split(" "),
                    "coordinates": lambda: [(np.random.random() * pq.cm,
                                             np.random.random() * pq.cm,
                                             np.random.random() * pq.cm)]*10,
                    # annotations
                    self.rword(): self.rword,
                    self.rword(): lambda: self.rquant((10, 10), pq.mV)}
        self._hash_test(ChannelIndex, argfuncs)

    def test_analogsignal_hash(self):
        argfuncs = {"name": self.rword,
                    "description": self.rsentence,
                    "signal": lambda: self.rquant((10, 10), pq.mV),
                    "sampling_rate": lambda: np.random.random() * pq.Hz,
                    "t_start": lambda: np.random.random() * pq.sec,
                    "t_stop": lambda: np.random.random() * pq.sec,
                    # annotations
                    self.rword(): self.rword,
                    self.rword(): lambda: self.rquant((10, 10), pq.mV)}
        self._hash_test(AnalogSignal, argfuncs)

    def test_irregularsignal_hash(self):
        argfuncs = {"name": self.rword,
                    "description": self.rsentence,
                    "signal": lambda: self.rquant((10, 10), pq.mV),
                    "times": lambda: self.rquant(10, pq.ms, True),
                    # annotations
                    self.rword(): self.rword,
                    self.rword(): lambda: self.rquant((10, 10), pq.mV)}
        self._hash_test(IrregularlySampledSignal, argfuncs)

    def test_event_hash(self):
        argfuncs = {"name": self.rword,
                    "description": self.rsentence,
                    "times": lambda: self.rquant(10, pq.ms),
                    "durations": lambda: self.rquant(10, pq.ms),
                    "labels": lambda: self.rsentence(10).split(" "),
                    # annotations
                    self.rword(): self.rword,
                    self.rword(): lambda: self.rquant((10, 10), pq.mV)}
        self._hash_test(Event, argfuncs)
        self._hash_test(Epoch, argfuncs)

    def test_spiketrain_hash(self):
        argfuncs = {"name": self.rword,
                    "description": self.rsentence,
                    "times": lambda: self.rquant(10, pq.ms, True),
                    "t_start": lambda: -np.random.random() * pq.sec,
                    "t_stop": lambda: np.random.random() * 100 * pq.sec,
                    "waveforms": lambda: self.rquant((10, 10, 20), pq.mV),
                    # annotations
                    self.rword(): self.rword,
                    self.rword(): lambda: self.rquant((10, 10), pq.mV)}
        self._hash_test(SpikeTrain, argfuncs)


class NixIOPartialWriteTest(NixIOTest):

    filename = "testfile_partialwrite.h5"
    nixfile = None
    neo_blocks = None
    original_methods = dict()

    @classmethod
    def setUpClass(cls):
        if HAVE_NIX:
            cls.nixfile = cls.create_full_nix_file(cls.filename)

    def setUp(self):
        self.io = NixIO(self.filename, "rw")
        self.neo_blocks = self.io.read_all_blocks()
        self.original_methods["_write_attr_annotations"] =\
            self.io._write_attr_annotations

    @classmethod
    def tearDownClass(cls):
        if HAVE_NIX:
            cls.nixfile.close()

    def tearDown(self):
        self.restore_methods()
        self.io.close()

    def restore_methods(self):
        for name, method in self.original_methods.items():
            setattr(self.io, name, self.original_methods[name])

    def _mock_write_attr(self, objclass):
        typestr = str(objclass.__name__).lower()
        self.io._write_attr_annotations = mock.Mock(
            wraps=self.io._write_attr_annotations,
            side_effect=self.check_obj_type("neo.{}".format(typestr))
        )
        neo_blocks = self.neo_blocks
        self.modify_objects(neo_blocks, excludes=[objclass])
        self.io.write_all_blocks(neo_blocks)
        self.restore_methods()

    def check_obj_type(self, typestring):
        neq = self.assertNotEqual

        def side_effect_func(*args, **kwargs):
            obj = kwargs.get("nixobj", args[0])
            if isinstance(obj, list):
                for sig in obj:
                    neq(sig.type, typestring)
            else:
                neq(obj.type, typestring)
        return side_effect_func

    @classmethod
    def modify_objects(cls, objs, excludes=()):
        excludes = tuple(excludes)
        for obj in objs:
            if not (excludes and isinstance(obj, excludes)):
                obj.description = cls.rsentence()
            for container in getattr(obj, "_child_containers", []):
                children = getattr(obj, container)
                cls.modify_objects(children, excludes)

    def test_partial(self):
        for objclass in NixIO.supported_objects:
            self._mock_write_attr(objclass)
            self.compare_blocks(self.neo_blocks, self.io.nix_file.blocks)

    def test_no_modifications(self):
        self.io._write_attr_annotations = mock.Mock()

        self.io.write_all_blocks(self.neo_blocks)
        self.io._write_attr_annotations.assert_not_called()
        self.compare_blocks(self.neo_blocks, self.io.nix_file.blocks)

        # clearing hashes and checking again
        for k in self.io._object_hashes.keys():
            self.io._object_hashes[k] = None
        self.io.write_all_blocks(self.neo_blocks)
        self.io._write_attr_annotations.assert_not_called()

        # changing hashes to force rewrite
        for k in self.io._object_hashes.keys():
            self.io._object_hashes[k] = "_"
        self.io.write_all_blocks(self.neo_blocks)
        callcount = self.io._write_attr_annotations.call_count
        self.assertEqual(callcount, len(self.io._object_hashes))
        self.compare_blocks(self.neo_blocks, self.io.nix_file.blocks)


class NixIOContextTests(NixIOTest):

    filename = "context_test.h5"

    def test_context_write(self):
        neoblock = Block(name=self.rword(), description=self.rsentence())
        with NixIO(self.filename, "ow") as iofile:
            iofile.write_block(neoblock)

        nixfile = nix.File.open(self.filename, nix.FileMode.ReadOnly,
                                backend="h5py")
        self.compare_blocks([neoblock], nixfile.blocks)
        nixfile.close()

        neoblock.annotate(**self.rdict(5))
        with NixIO(self.filename, "rw") as iofile:
            iofile.write_block(neoblock)
        nixfile = nix.File.open(self.filename, nix.FileMode.ReadOnly,
                                backend="h5py")
        self.compare_blocks([neoblock], nixfile.blocks)
        nixfile.close()

    def test_context_read(self):
        nixfile = nix.File.open(self.filename, nix.FileMode.Overwrite,
                                backend="h5py")
        name_one = self.rword()
        name_two = self.rword()
        nixfile.create_block(name_one, "neo.block")
        nixfile.create_block(name_two, "neo.block")
        nixfile.close()

        with NixIO(self.filename, "ro") as iofile:
            blocks = iofile.read_all_blocks()

        self.assertEqual(blocks[0].name, name_one)
        self.assertEqual(blocks[1].name, name_two)


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class CommonTests(BaseTestIO, unittest.TestCase):

    ioclass = NixIO
    read_and_write_is_bijective = False
