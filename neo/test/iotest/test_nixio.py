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
Tests for NixIO
"""

import os
import shutil
from collections import Iterable
from datetime import datetime

from tempfile import mkdtemp

import unittest
import string
import numpy as np
import quantities as pq

from neo.core import (Block, Segment, ChannelIndex, AnalogSignal,
                      IrregularlySampledSignal, Unit, SpikeTrain, Event, Epoch)
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.nixio import NixIO, create_quantity, units_to_string, neover

try:
    import nixio as nix

    HAVE_NIX = True
except ImportError:
    HAVE_NIX = False


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOTest(unittest.TestCase):
    io = None
    tempdir = None
    filename = None

    def compare_blocks(self, neoblocks, nixblocks):
        for neoblock, nixblock in zip(neoblocks, nixblocks):
            self.compare_attr(neoblock, nixblock)
            self.assertEqual(len(neoblock.segments), len(nixblock.groups))
            for idx, neoseg in enumerate(neoblock.segments):
                nixgrp = nixblock.groups[neoseg.annotations["nix_name"]]
                self.compare_segment_group(neoseg, nixgrp)
            self.assertEqual(len(neoblock.channel_indexes),
                             len(nixblock.sources))
            for idx, neochx in enumerate(neoblock.channel_indexes):
                nixsrc = nixblock.sources[neochx.annotations["nix_name"]]
                self.compare_chx_source(neochx, nixsrc)
            self.check_refs(neoblock, nixblock)

    def compare_chx_source(self, neochx, nixsrc):
        self.compare_attr(neochx, nixsrc)
        nix_channels = list(src for src in nixsrc.sources
                            if src.type == "neo.channelindex")
        self.assertEqual(len(neochx.index), len(nix_channels))

        if len(neochx.channel_ids):
            nix_chanids = list(src.metadata["channel_id"] for src
                               in nixsrc.sources
                               if src.type == "neo.channelindex")
            self.assertEqual(len(neochx.channel_ids), len(nix_chanids))

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
                nixchanname = nixchan.metadata["neo_name"]
                self.assertEqual(neochanname, nixchanname)
            if len(neochx.channel_ids):
                neochanid = neochx.channel_ids[neochanpos]
                nixchanid = nixchan.metadata["channel_id"]
                self.assertEqual(neochanid, nixchanid)
            elif "channel_id" in nixchan.metadata:
                self.fail("Channel ID not loaded")
        nix_units = list(src for src in nixsrc.sources
                         if src.type == "neo.unit")
        self.assertEqual(len(neochx.units), len(nix_units))
        for neounit in neochx.units:
            nixunit = nixsrc.sources[neounit.annotations["nix_name"]]
            self.compare_attr(neounit, nixunit)

    def check_refs(self, neoblock, nixblock):
        """
        Checks whether the references between objects that are not nested are
        mapped correctly (e.g., SpikeTrains referenced by a Unit).

        :param neoblock: A Neo block
        :param nixblock: The corresponding NIX block
        """
        for idx, neochx in enumerate(neoblock.channel_indexes):
            nixchx = nixblock.sources[neochx.annotations["nix_name"]]
            # AnalogSignals referencing CHX
            neoasigs = list(sig.annotations["nix_name"]
                            for sig in neochx.analogsignals)
            nixasigs = list(set(da.metadata.name for da in nixblock.data_arrays
                                if da.type == "neo.analogsignal" and
                                nixchx in da.sources))

            self.assertEqual(len(neoasigs), len(nixasigs))

            # IrregularlySampledSignals referencing CHX
            neoisigs = list(sig.annotations["nix_name"] for sig in
                            neochx.irregularlysampledsignals)
            nixisigs = list(
                set(da.metadata.name for da in nixblock.data_arrays
                    if da.type == "neo.irregularlysampledsignal" and
                    nixchx in da.sources)
            )
            self.assertEqual(len(neoisigs), len(nixisigs))
            # SpikeTrains referencing CHX and Units
            for sidx, neounit in enumerate(neochx.units):
                nixunit = nixchx.sources[neounit.annotations["nix_name"]]
                neosts = list(st.annotations["nix_name"]
                              for st in neounit.spiketrains)
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
        totalsignals = 0
        for sig in neosignals:
            dalist = list()
            nixname = sig.annotations["nix_name"]
            for da in data_arrays:
                if da.metadata.name == nixname:
                    dalist.append(da)
            _, nsig = np.shape(sig)
            totalsignals += nsig
            self.assertEqual(nsig, len(dalist))
            self.compare_signal_dalist(sig, dalist)
        self.assertEqual(totalsignals, len(data_arrays))

    def compare_signal_dalist(self, neosig, nixdalist):
        """
        Check if a Neo Analog or IrregularlySampledSignal matches a list of
        NIX DataArrays.

        :param neosig: Neo Analog or IrregularlySampledSignal
        :param nixdalist: List of DataArrays
        """
        nixmd = nixdalist[0].metadata
        self.assertTrue(all(nixmd == da.metadata for da in nixdalist))
        neounit = neosig.units
        for sig, da in zip(np.transpose(neosig),
                           sorted(nixdalist, key=lambda d: d.name)):
            self.compare_attr(neosig, da)
            daquant = create_quantity(da[:], da.unit)
            np.testing.assert_almost_equal(sig, daquant)
            nixunit = create_quantity(1, da.unit)
            self.assertEqual(neounit, nixunit)
            timedim = da.dimensions[0]
            if isinstance(neosig, AnalogSignal):
                self.assertEqual(timedim.dimension_type,
                                 nix.DimensionType.Sample)
                neosp = neosig.sampling_period
                nixsp = create_quantity(timedim.sampling_interval,
                                        timedim.unit)
                self.assertEqual(neosp, nixsp)
                tsunit = timedim.unit
                if "t_start.units" in da.metadata.props:
                    tsunit = da.metadata["t_start.units"]
                neots = neosig.t_start
                nixts = create_quantity(timedim.offset, tsunit)
                self.assertEqual(neots, nixts)
            elif isinstance(neosig, IrregularlySampledSignal):
                self.assertEqual(timedim.dimension_type,
                                 nix.DimensionType.Range)
                np.testing.assert_almost_equal(neosig.times.magnitude,
                                               timedim.ticks)
                self.assertEqual(timedim.unit,
                                 units_to_string(neosig.times.units))

    def compare_eests_mtags(self, eestlist, mtaglist):
        self.assertEqual(len(eestlist), len(mtaglist))
        for eest in eestlist:
            mtag = mtaglist[eest.annotations["nix_name"]]
            if isinstance(eest, Epoch):
                self.compare_epoch_mtag(eest, mtag)
            elif isinstance(eest, Event):
                self.compare_event_mtag(eest, mtag)
            elif isinstance(eest, SpikeTrain):
                self.compare_spiketrain_mtag(eest, mtag)

    def compare_epoch_mtag(self, epoch, mtag):
        self.assertEqual(mtag.type, "neo.epoch")
        self.compare_attr(epoch, mtag)
        pos = mtag.positions
        posquant = create_quantity(pos[:], pos.unit)
        ext = mtag.extents
        extquant = create_quantity(ext[:], ext.unit)
        np.testing.assert_almost_equal(epoch.as_quantity(), posquant)
        np.testing.assert_almost_equal(epoch.durations, extquant)
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
        pos = mtag.positions
        posquant = create_quantity(pos[:], pos.unit)
        np.testing.assert_almost_equal(event.as_quantity(), posquant)
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
        pos = mtag.positions
        posquant = create_quantity(pos[:], pos.unit)
        np.testing.assert_almost_equal(spiketrain.as_quantity(), posquant)
        if len(mtag.features):
            neowfs = spiketrain.waveforms
            nixwfs = mtag.features[0].data
            self.assertEqual(np.shape(neowfs), np.shape(nixwfs))
            for nixwf, neowf in zip(nixwfs, neowfs):
                for nixrow, neorow in zip(nixwf, neowf):
                    for nixv, neov in zip(nixrow, neorow):
                        self.assertEqual(create_quantity(nixv, nixwfs.unit),
                                         neov)
            self.assertEqual(nixwfs.dimensions[0].dimension_type,
                             nix.DimensionType.Set)
            self.assertEqual(nixwfs.dimensions[1].dimension_type,
                             nix.DimensionType.Set)
            self.assertEqual(nixwfs.dimensions[2].dimension_type,
                             nix.DimensionType.Sample)

    def compare_attr(self, neoobj, nixobj):
        if isinstance(neoobj, (AnalogSignal, IrregularlySampledSignal)):
            nix_name = ".".join(nixobj.name.split(".")[:-1])
        else:
            nix_name = nixobj.name
        self.assertEqual(neoobj.annotations["nix_name"], nix_name)
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
                if k == "nix_name":
                    continue
                if isinstance(v, pq.Quantity):
                    nixunit = nixmd.props[str(k)].unit
                    self.assertEqual(nixunit, units_to_string(v.units))
                    nixvalue = nixmd[str(k)]
                    if isinstance(nixvalue, Iterable):
                        nixvalue = np.array(nixvalue)
                    np.testing.assert_almost_equal(nixvalue, v.magnitude)
                else:
                    self.assertEqual(nixmd[str(k)], v,
                                     "Property value mismatch: {}".format(k))

    @classmethod
    def create_full_nix_file(cls, filename):
        nixfile = nix.File.open(filename, nix.FileMode.Overwrite,
                                backend="h5py")

        nix_block_a = nixfile.create_block(cls.rword(10), "neo.block")
        nix_block_a.definition = cls.rsentence(5, 10)
        nix_block_b = nixfile.create_block(cls.rword(10), "neo.block")
        nix_block_b.definition = cls.rsentence(3, 3)

        nix_block_a.metadata = nixfile.create_section(
            nix_block_a.name, nix_block_a.name + ".metadata"
        )
        nix_block_a.metadata["neo_name"] = cls.rword(5)

        nix_block_b.metadata = nixfile.create_section(
            nix_block_b.name, nix_block_b.name + ".metadata"
        )
        nix_block_b.metadata["neo_name"] = cls.rword(5)

        nix_blocks = [nix_block_a, nix_block_b]

        for blk in nix_blocks:
            for ind in range(3):
                group = blk.create_group(cls.rword(), "neo.segment")
                group.definition = cls.rsentence(10, 15)

                group_md = blk.metadata.create_section(
                    group.name, group.name + ".metadata"
                )
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
                                                    asig_name + ".metadata")
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
            asig_md["t_start.dim"] = "ms"
            allsignalgroups.append(siggroup)

        # irregularlysampledsignals
        for n in range(2):
            siggroup = list()
            isig_name = "{}_isig{}".format(cls.rword(10), n)
            isig_definition = cls.rsentence(12, 12)
            isig_md = group.metadata.create_section(isig_name,
                                                    isig_name + ".metadata")
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
            times = cls.rquant(40, 1, True)
            times_da = blk.create_data_array(
                "{}.times".format(stname),
                "neo.spiketrain.times",
                data=times
            )
            times_da.unit = "ms"
            mtag_st = blk.create_multi_tag(stname, "neo.spiketrain", times_da)
            group.multi_tags.append(mtag_st)
            mtag_st.definition = cls.rsentence(20, 30)
            mtag_st_md = group.metadata.create_section(
                mtag_st.name, mtag_st.name + ".metadata"
            )
            mtag_st.metadata = mtag_st_md
            mtag_st_md.create_property("t_stop", nix.Value(times[-1] + 1.0))

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
                                          [nix.Value(20)] * 5)
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
            mtag_ep.metadata = group.metadata.create_section(
                epname, epname + ".metadata"
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
            mtag_ev.metadata = group.metadata.create_section(
                evname, evname + ".metadata"
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
        for idx, chan in enumerate([2, 5, 9]):
            channame = "{}.ChannelIndex{}".format(nixchx.name, idx)
            nixrc = nixchx.create_source(channame, chantype)
            nixrc.definition = cls.rsentence(13)
            nixrc.metadata = nixchx.metadata.create_section(
                nixrc.name, "neo.channelindex.metadata"
            )
            nixrc.metadata.create_property("index", nix.Value(chan))
            nixrc.metadata.create_property("channel_id", nix.Value(chan + 1))
            dims = tuple(map(nix.Value, cls.rquant(3, 3)))
            coordprop = nixrc.metadata.create_property("coordinates", dims)
            coordprop.unit = "pm"

        nunits = 1
        stsperunit = np.array_split(allspiketrains, nunits)
        for idx in range(nunits):
            unitname = "{}-unit{}".format(cls.rword(5), idx)
            nixunit = nixchx.create_source(unitname, "neo.unit")
            nixunit.metadata = nixchx.metadata.create_section(
                unitname, unitname + ".metadata"
            )
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
        return arr * unit

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

        chx = ChannelIndex(name="achx", index=[1, 2], channel_ids=[0, 10])
        chx.annotate(**cls.rdict(5))
        blk.channel_indexes.append(chx)

        unit = Unit()
        unit.annotate(**cls.rdict(2))
        chx.units.append(unit)

        return blk


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOWriteTest(NixIOTest):
    def setUp(self):
        self.tempdir = mkdtemp(prefix="nixiotest")
        self.filename = os.path.join(self.tempdir, "testnixio.nix")
        self.writer = NixIO(self.filename, "ow")
        self.io = self.writer
        self.reader = nix.File.open(self.filename,
                                    nix.FileMode.ReadOnly,
                                    backend="h5py")

    def tearDown(self):
        self.writer.close()
        self.reader.close()
        shutil.rmtree(self.tempdir)

    def write_and_compare(self, blocks):
        self.writer.write_all_blocks(blocks)
        self.compare_blocks(blocks, self.reader.blocks)
        self.compare_blocks(self.writer.read_all_blocks(), self.reader.blocks)
        self.compare_blocks(blocks, self.reader.blocks)

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
                           channel_ids=[10, 20, 30, 50, 80, 130],
                           index=[1, 2, 3, 5, 8, 13])
        block.channel_indexes.append(chx)
        self.write_and_compare([block])

        chx.annotate(**self.rdict(3))
        self.write_and_compare([block])

        chx.channel_names = ["one", "two", "three", "five",
                             "eight", "xiii"]

        chx.coordinates = self.rquant((6, 3), pq.um)
        self.write_and_compare([block])

        # add an empty channel index and check again
        newchx = ChannelIndex(np.array([]))
        block.channel_indexes.append(newchx)
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
        self.write_and_compare([block, anotherblock])

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

    def test_signals_compound_units(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        units = pq.CompoundUnit("1/30000*V")
        srate = pq.Quantity(10, pq.CompoundUnit("1.0/10 * Hz"))
        asig = AnalogSignal(signal=self.rquant((10, 3), units),
                            sampling_rate=srate)
        seg.analogsignals.append(asig)

        self.write_and_compare([block])

        anotherblock = Block("ir signal block")
        seg = Segment("ir signal seg")
        anotherblock.segments.append(seg)
        irsig = IrregularlySampledSignal(
            signal=np.random.random((20, 3)),
            times=self.rquant(20, pq.CompoundUnit("0.1 * ms"), True),
            units=pq.CompoundUnit("10 * V / s")
        )
        seg.irregularlysampledsignals.append(irsig)
        self.write_and_compare([block, anotherblock])

        block.segments[0].analogsignals.append(
            AnalogSignal(signal=[10.0, 1.0, 3.0], units=pq.S,
                         sampling_period=pq.Quantity(3, "s"),
                         dtype=np.double, name="signal42",
                         description="this is an analogsignal",
                         t_start=45 * pq.CompoundUnit("3.14 * s")),
        )
        self.write_and_compare([block, anotherblock])

        times = self.rquant(10, pq.CompoundUnit("3 * year"), True)
        block.segments[0].irregularlysampledsignals.append(
            IrregularlySampledSignal(times=times,
                                     signal=np.random.random((10, 3)),
                                     units="mV", dtype=np.float,
                                     name="some sort of signal",
                                     description="the signal is described")
        )

        self.write_and_compare([block, anotherblock])

    def test_epoch_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        epoch = Epoch(times=[1, 1, 10, 3] * pq.ms,
                      durations=[3, 3, 3, 1] * pq.ms,
                      labels=np.array(["one", "two", "three", "four"]),
                      name="test epoch", description="an epoch for testing")

        seg.epochs.append(epoch)
        self.write_and_compare([block])

    def test_event_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        event = Event(times=np.arange(0, 30, 10) * pq.s,
                      labels=np.array(["0", "1", "2"]),
                      name="event name",
                      description="event description")
        seg.events.append(event)
        self.write_and_compare([block])

    def test_spiketrain_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        spiketrain = SpikeTrain(times=[3, 4, 5] * pq.s, t_stop=10.0,
                                name="spikes!", description="sssssspikes")
        seg.spiketrains.append(spiketrain)
        self.write_and_compare([block])

        waveforms = self.rquant((3, 5, 10), pq.mV)
        spiketrain = SpikeTrain(times=[1, 1.1, 1.2] * pq.ms, t_stop=1.5 * pq.s,
                                name="spikes with wf",
                                description="spikes for waveform test",
                                waveforms=waveforms)

        seg.spiketrains.append(spiketrain)
        self.write_and_compare([block])

        spiketrain.left_sweep = np.random.random(10) * pq.ms
        self.write_and_compare([block])

        spiketrain.left_sweep = pq.Quantity(-10, "ms")
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
                    seg.spiketrains.append(SpikeTrain(times=times,
                                                      t_stop=times[-1] + pq.s,
                                                      units=pq.s))
            for chidx in range(nchx):
                chx = ChannelIndex(name="chx{}".format(chidx),
                                   index=[1, 2],
                                   channel_ids=[11, 22])
                blk.channel_indexes.append(chx)
                for unidx in range(nunits):
                    unit = Unit()
                    chx.units.append(unit)
        self.writer.write_all_blocks(blocks)
        self.compare_blocks(blocks, self.reader.blocks)

    def test_multiref_write(self):
        blk = Block("blk1")
        signal = AnalogSignal(name="sig1", signal=[0, 1, 2], units="mV",
                              sampling_period=pq.Quantity(1, "ms"))
        othersignal = IrregularlySampledSignal(name="i1", signal=[0, 0, 0],
                                               units="mV", times=[1, 2, 3],
                                               time_units="ms")
        event = Event(name="Evee", times=[0.3, 0.42], units="year")
        epoch = Epoch(name="epoche", times=[0.1, 0.2] * pq.min,
                      durations=[0.5, 0.5] * pq.min)
        st = SpikeTrain(name="the train of spikes", times=[0.1, 0.2, 10.3],
                        t_stop=11, units="us")

        for idx in range(3):
            segname = "seg" + str(idx)
            seg = Segment(segname)
            blk.segments.append(seg)
            seg.analogsignals.append(signal)
            seg.irregularlysampledsignals.append(othersignal)
            seg.events.append(event)
            seg.epochs.append(epoch)
            seg.spiketrains.append(st)

        chidx = ChannelIndex([10, 20, 29])
        seg = blk.segments[0]
        st = SpikeTrain(name="choochoo", times=[10, 11, 80], t_stop=1000,
                        units="s")
        seg.spiketrains.append(st)
        blk.channel_indexes.append(chidx)
        for idx in range(6):
            unit = Unit("unit" + str(idx))
            chidx.units.append(unit)
            unit.spiketrains.append(st)

        self.writer.write_block(blk)
        self.compare_blocks([blk], self.reader.blocks)

    def test_no_segment_write(self):
        # Tests storing AnalogSignal, IrregularlySampledSignal, and SpikeTrain
        # objects in the secondary (ChannelIndex) substructure without them
        # being attached to a Segment.
        blk = Block("segmentless block")
        signal = AnalogSignal(name="sig1", signal=[0, 1, 2], units="mV",
                              sampling_period=pq.Quantity(1, "ms"))
        othersignal = IrregularlySampledSignal(name="i1", signal=[0, 0, 0],
                                               units="mV", times=[1, 2, 3],
                                               time_units="ms")
        sta = SpikeTrain(name="the train of spikes", times=[0.1, 0.2, 10.3],
                         t_stop=11, units="us")
        stb = SpikeTrain(name="the train of spikes b", times=[1.1, 2.2, 10.1],
                         t_stop=100, units="ms")

        chidx = ChannelIndex([8, 13, 21])
        blk.channel_indexes.append(chidx)
        chidx.analogsignals.append(signal)
        chidx.irregularlysampledsignals.append(othersignal)

        unit = Unit()
        chidx.units.append(unit)
        unit.spiketrains.extend([sta, stb])
        self.writer.write_block(blk)

        self.compare_blocks([blk], self.reader.blocks)

        self.writer.close()
        reader = NixIO(self.filename, "ro")
        blk = reader.read_block(neoname="segmentless block")
        chx = blk.channel_indexes[0]
        self.assertEqual(len(chx.analogsignals), 1)
        self.assertEqual(len(chx.irregularlysampledsignals), 1)
        self.assertEqual(len(chx.units[0].spiketrains), 2)

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

        # empty string
        writeprop(section, "emptystring", "")
        self.assertEqual("", section["emptystring"])

    def test_annotations_special_cases(self):
        # Special cases for annotations: empty list, list of strings,
        # multidimensional lists/arrays
        # These are handled differently on read, so we test them on a block
        # instead of just checking the property writer method
        # empty value

        # empty list
        wblock = Block("block with empty list", an_empty_list=list())
        self.writer.write_block(wblock)
        rblock = self.writer.read_block(neoname="block with empty list")
        self.assertEqual(rblock.annotations["an_empty_list"], list())

        # empty tuple (gets read out as list)
        wblock = Block("block with empty tuple", an_empty_tuple=tuple())
        self.writer.write_block(wblock)
        rblock = self.writer.read_block(neoname="block with empty tuple")
        self.assertEqual(rblock.annotations["an_empty_tuple"], list())

        # list of strings
        losval = ["one", "two", "one million"]
        wblock = Block("block with list of strings",
                       los=losval)
        self.writer.write_block(wblock)
        rblock = self.writer.read_block(neoname="block with list of strings")
        self.assertEqual(rblock.annotations["los"], losval)

        # TODO: multi dimensional value (GH Issue #501)


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOReadTest(NixIOTest):
    nixfile = None
    nix_blocks = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = mkdtemp(prefix="nixiotest")
        cls.filename = os.path.join(cls.tempdir, "testnixio.nix")
        if HAVE_NIX:
            cls.nixfile = cls.create_full_nix_file(cls.filename)

    def setUp(self):
        self.io = NixIO(self.filename, "ro")

    @classmethod
    def tearDownClass(cls):
        if HAVE_NIX:
            cls.nixfile.close()
        shutil.rmtree(cls.tempdir)

    def tearDown(self):
        self.io.close()

    def test_all_read(self):
        neo_blocks = self.io.read_all_blocks()
        nix_blocks = self.io.nix_file.blocks
        self.compare_blocks(neo_blocks, nix_blocks)

    def test_iter_read(self):
        blocknames = [blk.name for blk in self.nixfile.blocks]
        for blk, nixname in zip(self.io.iter_blocks(), blocknames):
            self.assertEqual(blk.annotations["nix_name"], nixname)

    def test_nix_name_read(self):
        for nixblock in self.nixfile.blocks:
            nixname = nixblock.name
            neoblock = self.io.read_block(nixname=nixname)
            self.assertEqual(neoblock.annotations["nix_name"], nixname)

    def test_index_read(self):
        for idx, nixblock in enumerate(self.nixfile.blocks):
            neoblock = self.io.read_block(index=idx)
            self.assertEqual(neoblock.annotations["nix_name"], nixblock.name)

    def test_auto_index_read(self):
        for nixblock in self.nixfile.blocks:
            neoblock = self.io.read_block()  # don't specify index
            self.assertEqual(neoblock.annotations["nix_name"], nixblock.name)

        # No more blocks - should return None
        self.assertIs(self.io.read_block(), None)
        self.assertIs(self.io.read_block(), None)
        self.assertIs(self.io.read_block(), None)

    def test_neo_name_read(self):
        for nixblock in self.nixfile.blocks:
            neoname = nixblock.metadata["neo_name"]
            neoblock = self.io.read_block(neoname=neoname)
            self.assertEqual(neoblock.annotations["nix_name"], nixblock.name)


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOContextTests(NixIOTest):
    def setUp(self):
        self.tempdir = mkdtemp(prefix="nixiotest")
        self.filename = os.path.join(self.tempdir, "testnixio.nix")

    def tearDown(self):
        shutil.rmtree(self.tempdir)

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

        self.assertEqual(blocks[0].annotations["nix_name"], name_one)
        self.assertEqual(blocks[1].annotations["nix_name"], name_two)


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOVerTests(NixIOTest):
    def setUp(self):
        self.tempdir = mkdtemp(prefix="nixiotest")
        self.filename = os.path.join(self.tempdir, "testnixio.nix")

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_new_file(self):
        with NixIO(self.filename, "ow") as iofile:
            self.assertEqual(iofile._file_version, neover)

        nixfile = nix.File.open(self.filename, nix.FileMode.ReadOnly)
        filever = nixfile.sections["neo"]["version"]
        self.assertEqual(filever, neover)
        nixfile.close()

    def test_oldfile_nover(self):
        nixfile = nix.File.open(self.filename, nix.FileMode.Overwrite)
        nixfile.close()
        with NixIO(self.filename, "ro") as iofile:
            self.assertEqual(iofile._file_version, '0.5.2')  # compat version

        nixfile = nix.File.open(self.filename, nix.FileMode.ReadOnly)
        self.assertNotIn("neo", nixfile.sections)
        nixfile.close()

        with NixIO(self.filename, "rw") as iofile:
            self.assertEqual(iofile._file_version, '0.5.2')  # compat version

        # section should have been created now
        nixfile = nix.File.open(self.filename, nix.FileMode.ReadOnly)
        self.assertIn("neo", nixfile.sections)
        self.assertEqual(nixfile.sections["neo"]["version"], '0.5.2')
        nixfile.close()

    def test_file_with_ver(self):
        someversion = '0.100.10'
        nixfile = nix.File.open(self.filename, nix.FileMode.Overwrite)
        filemd = nixfile.create_section("neo", "neo.metadata")
        filemd["version"] = someversion
        nixfile.close()

        with NixIO(self.filename, "ro") as iofile:
            self.assertEqual(iofile._file_version, someversion)

        with NixIO(self.filename, "rw") as iofile:
            self.assertEqual(iofile._file_version, someversion)

        with NixIO(self.filename, "ow") as iofile:
            self.assertEqual(iofile._file_version, neover)


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class CommonTests(BaseTestIO, unittest.TestCase):
    ioclass = NixIO
    read_and_write_is_bijective = False


if __name__ == "__main__":
    unittest.main()
