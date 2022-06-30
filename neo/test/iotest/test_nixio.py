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

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from datetime import date, time, datetime

from tempfile import mkdtemp
from itertools import chain
import unittest
import string
import numpy as np
import quantities as pq

from neo.core import (Block, Segment, AnalogSignal,
                      IrregularlySampledSignal, SpikeTrain,
                      Event, Epoch, ImageSequence, Group, ChannelView)
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.nixio import (NixIO, create_quantity, units_to_string, neover,
                          dt_from_nix, dt_to_nix, DATETIMEANNOTATION)
from neo.io.nixio_fr import NixIO as NixIO_lazy
from neo.io.proxyobjects import (AnalogSignalProxy, SpikeTrainProxy,
                                 EventProxy, EpochProxy)

try:
    import nixio as nix

    HAVE_NIX = True
except ImportError:
    HAVE_NIX = False

try:
    from unittest import mock

    SKIPMOCK = False
except ImportError:
    SKIPMOCK = True


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOTest(unittest.TestCase):
    io = None
    tempdir = None
    filename = None

    def compare_blocks(self, neoblocks, nixblocks):
        for neoblock, nixblock in zip(neoblocks, nixblocks):
            self.compare_attr(neoblock, nixblock)
            self.assertEqual(len(neoblock.segments),
                             len([grp for grp in nixblock.groups if grp.type == "neo.segment"]))
            self.assertEqual(len(neoblock.groups),
                             len([grp for grp in nixblock.groups if grp.type == "neo.group"]))
            for idx, neoseg in enumerate(neoblock.segments):
                nixgrp = nixblock.groups[neoseg.annotations["nix_name"]]
                self.compare_segment_group(neoseg, nixgrp)
            self.check_refs(neoblock, nixblock)

    def check_refs(self, neoblock, nixblock):
        """
        Checks whether the references between objects that are not nested are
        mapped correctly (e.g., SpikeTrains referenced by a Unit).

        :param neoblock: A Neo block
        :param nixblock: The corresponding NIX block
        """

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
        neo_signals = neoseg.analogsignals + neoseg.irregularlysampledsignals \
            + neoseg.imagesequences
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
            nsig = np.shape(sig)[-1]
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
        if isinstance(neosig, AnalogSignalProxy):
            neosig = neosig.load()
        for sig, da in zip(np.transpose(neosig), nixdalist):
            self.compare_attr(neosig, da)
            daquant = create_quantity(da[:], da.unit)
            np.testing.assert_almost_equal(sig.view(pq.Quantity), daquant)
            nixunit = create_quantity(1, da.unit)
            self.assertEqual(neounit, nixunit)

            if isinstance(neosig, AnalogSignal):
                timedim = da.dimensions[0]
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
                timedim = da.dimensions[0]
                self.assertEqual(timedim.dimension_type,
                                 nix.DimensionType.Range)
                np.testing.assert_almost_equal(neosig.times.magnitude,
                                               timedim.ticks)
                self.assertEqual(timedim.unit,
                                 units_to_string(neosig.times.units))
            elif isinstance(neosig, ImageSequence):
                rate = da.metadata["sampling_rate"]
                unit = da.metadata.props["sampling_rate"].unit
                sampling_rate = create_quantity(rate, unit)
                neosr = neosig.sampling_rate
                self.assertEqual(sampling_rate, neosr)
                scale = da.metadata["spatial_scale"]
                unit = da.metadata.props["spatial_scale"].unit
                spatial_scale = create_quantity(scale, unit)
                neosps = neosig.spatial_scale
                self.assertEqual(spatial_scale, neosps)

    def compare_eests_mtags(self, eestlist, mtaglist):
        self.assertEqual(len(eestlist), len(mtaglist))
        for eest in eestlist:
            if isinstance(eest, (EventProxy, EpochProxy)):
                eest = eest.load()
            elif isinstance(eest, SpikeTrainProxy):
                eest = eest.load(load_waveforms=True)
            mtag = mtaglist[eest.annotations["nix_name"]]
            if isinstance(eest, Epoch):
                self.compare_epoch_mtag(eest, mtag)
            elif isinstance(eest, Event):
                self.compare_event_mtag(eest, mtag)
            elif isinstance(eest, SpikeTrain):
                self.compare_spiketrain_mtag(eest, mtag)
            else:
                self.fail("Stray object")

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
            self.assertEqual(neol, nixl)

    def compare_event_mtag(self, event, mtag):
        self.assertEqual(mtag.type, "neo.event")
        self.compare_attr(event, mtag)
        pos = mtag.positions
        posquant = create_quantity(pos[:], pos.unit)
        np.testing.assert_almost_equal(event.as_quantity(), posquant)
        for neol, nixl in zip(event.labels,
                              mtag.positions.dimensions[0].labels):
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
        if isinstance(neoobj, (AnalogSignal, IrregularlySampledSignal,
                               ImageSequence)):
            nix_name = ".".join(nixobj.name.split(".")[:-1])
        else:
            nix_name = nixobj.name

        self.assertEqual(neoobj.annotations["nix_name"], nix_name)
        self.assertEqual(neoobj.description, nixobj.definition)
        if hasattr(neoobj, "rec_datetime") and neoobj.rec_datetime:
            self.assertEqual(neoobj.rec_datetime,
                             datetime.fromtimestamp(nixobj.created_at))
        if hasattr(neoobj, "file_datetime") and neoobj.file_datetime:
            nixdt = dt_from_nix(nixobj.metadata["file_datetime"],
                                DATETIMEANNOTATION)
            assert neoobj.file_datetime == nixdt
            self.assertEqual(neoobj.file_datetime, nixdt)
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
        if hasattr(neoobj, 'array_annotations'):
            if neoobj.array_annotations:
                nixmd = nixobj.metadata
                for k, v, in neoobj.array_annotations.items():
                    if k in ['labels', 'durations']:
                        continue
                    if isinstance(v, pq.Quantity):
                        nixunit = nixmd.props[str(k)].unit
                        self.assertEqual(nixunit, units_to_string(v.units))
                        nixvalue = nixmd[str(k)]
                        if isinstance(nixvalue, Iterable):
                            nixvalue = np.array(nixvalue)
                        np.testing.assert_almost_equal(nixvalue, v.magnitude)
                    if isinstance(v, np.ndarray):
                        self.assertTrue(np.all(v == nixmd[str(k)]))
                    else:
                        msg = "Property value mismatch: {}".format(k)
                        self.assertEqual(nixmd[str(k)], v, msg)

    @classmethod
    def create_full_nix_file(cls, filename):
        nixfile = nix.File.open(filename, nix.FileMode.Overwrite)

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
        for n in range(5):
            siggroup = list()
            asig_name = "{}_asig{}".format(cls.rword(10), n)
            asig_definition = cls.rsentence(5, 5)
            asig_md = group.metadata.create_section(asig_name,
                                                    asig_name + ".metadata")

            arr_ann_name, arr_ann_val = 'anasig_arr_ann', cls.rquant(10, pq.uV)
            asig_md.create_property(arr_ann_name,
                                    arr_ann_val.magnitude.flatten())
            asig_md.props[arr_ann_name].unit = str(arr_ann_val.dimensionality)
            asig_md.props[arr_ann_name].type = 'ARRAYANNOTATION'

            for idx in range(10):
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
        # imagesequence
        for n in range(5):
            imgseqgroup = list()
            imgseq_name = "{}_imgs{}".format(cls.rword(10), n)
            imgseq_definition = cls.rsentence(5, 5)
            imgseq_md = group.metadata.create_section(imgseq_name,
                                                      imgseq_name + ".metadata")

            arr_ann_name, arr_ann_val = 'imgseq_arr_ann', cls.rquant(10, pq.V)
            imgseq_md.create_property(arr_ann_name,
                                      arr_ann_val.magnitude.flatten())
            imgseq_md.props[arr_ann_name].unit = str(arr_ann_val.dimensionality)
            imgseq_md.props[arr_ann_name].type = 'ARRAYANNOTATION'

            for idx in range(10):
                da_imgseq = blk.create_data_array(
                    "{}.{}".format(imgseq_name, idx),
                    "neo.imagesequence",
                    data=cls.rquant((20, 10), 1)
                )
                da_imgseq.definition = imgseq_definition
                da_imgseq.unit = "mV"

                da_imgseq.metadata = imgseq_md
                imgseq_md["sampling_rate"] = 10
                imgseq_md.props["sampling_rate"].unit = units_to_string(pq.V)
                imgseq_md["spatial_scale"] = 10
                imgseq_md.props["spatial_scale"].unit = units_to_string(pq.micrometer)

                group.data_arrays.append(da_imgseq)
                imgseqgroup.append(da_imgseq)

            allsignalgroups.append(imgseqgroup)
        # irregularlysampledsignals
        for n in range(2):
            siggroup = list()
            isig_name = "{}_isig{}".format(cls.rword(10), n)
            isig_definition = cls.rsentence(12, 12)
            isig_md = group.metadata.create_section(isig_name,
                                                    isig_name + ".metadata")
            isig_times = cls.rquant(200, 1, True)
            arr_ann_name, arr_ann_val = 'irrsig_arr_ann', cls.rquant(7, pq.uV)
            isig_md.create_property(arr_ann_name,
                                    arr_ann_val.magnitude.flatten())
            isig_md.props[arr_ann_name].unit = str(arr_ann_val.dimensionality)
            isig_md.props[arr_ann_name].type = 'ARRAYANNOTATION'
            for idx in range(7):
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
            mtag_st_md.create_property("t_stop", times[-1] + 1.0)

            arr_ann_name, arr_ann_val = 'st_arr_ann', cls.rquant(40, pq.uV)
            mtag_st_md.create_property(arr_ann_name,
                                       arr_ann_val.magnitude.flatten())
            mtag_st_md.props[arr_ann_name].unit = str(arr_ann_val.dimensionality)
            mtag_st_md.props[arr_ann_name].type = 'ARRAYANNOTATION'

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
                                          [20] * 5)
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

            arr_ann_name, arr_ann_val = 'ep_arr_ann', cls.rquant(5, pq.uV)
            mtag_ep.metadata.create_property(arr_ann_name,
                                             arr_ann_val.magnitude.flatten())
            mtag_ep.metadata.props[arr_ann_name].unit = str(arr_ann_val.dimensionality)
            mtag_ep.metadata.props[arr_ann_name].type = 'ARRAYANNOTATION'

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

            arr_ann_name, arr_ann_val = 'ev_arr_ann',\
                                        cls.rquant(5, pq.uV)
            mtag_ev.metadata.create_property(arr_ann_name,
                                             arr_ann_val.magnitude.flatten())
            mtag_ev.metadata.props[arr_ann_name].unit = str(arr_ann_val.dimensionality)
            mtag_ev.metadata.props[arr_ann_name].type = 'ARRAYANNOTATION'

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
            nixrc.metadata.create_property("index", chan)
            nixrc.metadata.create_property("channel_id", chan + 1)
            dims = cls.rquant(3, 1)
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
        rand_idxs = np.random.choice(range(len(allsignalgroups)), 5, False)
        randsiggroups = [allsignalgroups[idx] for idx in rand_idxs]
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
        times = cls.rquant(10, pq.s, incr=True)
        times_ann = {cls.rword(6): cls.rquant(10, pq.ms)}
        signal = cls.rquant((10, 10), pq.V)
        signal_ann = {cls.rword(6): cls.rquant(10, pq.uV)}
        blk = Block()
        blk.annotate(**cls.rdict(3))
        cls.populate_dates(blk)

        seg = Segment()
        seg.annotate(**cls.rdict(4))
        cls.populate_dates(seg)
        blk.segments.append(seg)

        asig = AnalogSignal(signal=signal, sampling_rate=pq.Hz,
                            array_annotations=signal_ann)
        asig.annotate(**cls.rdict(2))
        seg.analogsignals.append(asig)

        isig = IrregularlySampledSignal(times=times, signal=signal,
                                        time_units=pq.s,
                                        array_annotations=signal_ann)
        isig.annotate(**cls.rdict(2))
        seg.irregularlysampledsignals.append(isig)

        epoch = Epoch(times=times, durations=times,
                      array_annotations=times_ann)
        epoch.annotate(**cls.rdict(4))
        seg.epochs.append(epoch)

        event = Event(times=times, array_annotations=times_ann)
        event.annotate(**cls.rdict(4))
        seg.events.append(event)

        spiketrain = SpikeTrain(times=times, t_stop=10 * pq.s,
                        units=pq.s, array_annotations=times_ann)
        d = cls.rdict(6)
        d["quantity"] = pq.Quantity(10, "mV")
        d["qarray"] = pq.Quantity(range(10), "mA")
        spiketrain.annotate(**d)
        seg.spiketrains.append(spiketrain)

        chx = Group(name="achx", index=[1, 2], channel_ids=[0, 10])
        chx.annotate(**cls.rdict(5))
        blk.groups.append(chx)

        unit = Group()
        unit.annotate(**cls.rdict(2))
        chx.add(unit)

        return blk


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class NixIOWriteTest(NixIOTest):
    def setUp(self):
        self.tempdir = mkdtemp(prefix="nixiotest")
        self.filename = os.path.join(self.tempdir, "testnixio.nix")
        self.writer = NixIO(self.filename, "ow")
        self.io = self.writer
        self.reader = nix.File.open(self.filename, nix.FileMode.ReadOnly)

    def tearDown(self):
        self.writer.close()
        self.reader.close()
        shutil.rmtree(self.tempdir)

    def write_and_compare(self, blocks, use_obj_names=False):
        self.writer.write_all_blocks(blocks, use_obj_names)
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

    def test_signals_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        asig = AnalogSignal(signal=self.rquant((19, 15), pq.mV),
                            sampling_rate=pq.Quantity(10, "Hz"))
        seg.analogsignals.append(asig)
        self.write_and_compare([block])

        anotherblock = Block("ir signal block")
        seg = Segment("ir signal seg")
        anotherblock.segments.append(seg)
        irsig = IrregularlySampledSignal(
            signal=np.random.random((20, 30)),
            times=self.rquant(20, pq.ms, incr=True),
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
            IrregularlySampledSignal(times=np.sort(np.random.random(10)),
                                     signal=np.random.random((10, 13)),
                                     units="mV", time_units="s",
                                     dtype=np.float32,
                                     name="some sort of signal",
                                     description="the signal is described")
        )
        self.write_and_compare([block, anotherblock])

    def test_imagesequence_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        imgseq = ImageSequence(image_data=self.rquant((19, 10, 15), 1),
                               sampling_rate=pq.Quantity(10, "Hz"),
                               spatial_scale=pq.Quantity(10, "micrometer"),
                               units=pq.V)
        seg.imagesequences.append(imgseq)
        self.write_and_compare([block])

    def test_signals_compound_units(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        units = pq.CompoundUnit("1/30000*V")
        srate = pq.Quantity(10, pq.CompoundUnit("1.0/10 * Hz"))
        asig = AnalogSignal(signal=self.rquant((10, 23), units),
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
                                     units="mV", dtype=float,
                                     name="some sort of signal",
                                     description="the signal is described")
        )

        self.write_and_compare([block, anotherblock])

    def test_imagesequence_compound_units(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        units = pq.CompoundUnit("1/30000*V")
        srate = pq.Quantity(10, pq.CompoundUnit("1.0/10 * Hz"))
        size = pq.Quantity(10, pq.CompoundUnit("1.0/10 * micrometer"))
        imgseq = ImageSequence(image_data=self.rquant((10, 20, 10), units),
                               sampling_rate=srate, spatial_scale=size)
        seg.imagesequences.append(imgseq)

        self.write_and_compare([block])

    def test_epoch_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        epoch = Epoch(times=[1, 1, 10, 3] * pq.ms,
                      durations=[3, 3, 3, 1] * pq.ms,
                      labels=np.array(["one", "two", "three", "four"], dtype='U'),
                      name="test epoch", description="an epoch for testing")

        seg.epochs.append(epoch)
        self.write_and_compare([block])

    def test_event_write(self):
        block = Block()
        seg = Segment()
        block.segments.append(seg)

        event = Event(times=np.arange(0, 30, 10) * pq.s,
                      labels=np.array(["0", "1", "2"], dtype='U'),
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

    def test_group_write(self):
        signals = [
            AnalogSignal(np.random.random(size=(1000, 5)) * pq.mV,
                         sampling_period=1 * pq.ms, name="sig1"),
            AnalogSignal(np.random.random(size=(1000, 3)) * pq.mV,
                         sampling_period=1 * pq.ms, name="sig2"),
        ]
        spiketrains = [
            SpikeTrain([0.1, 54.3, 76.6, 464.2], units=pq.ms,
                       t_stop=1000.0 * pq.ms, t_start=0.0 * pq.ms),
            SpikeTrain([30.1, 154.3, 276.6, 864.2], units=pq.ms,
                       t_stop=1000.0 * pq.ms, t_start=0.0 * pq.ms),
            SpikeTrain([120.1, 454.3, 576.6, 764.2], units=pq.ms,
                       t_stop=1000.0 * pq.ms, t_start=0.0 * pq.ms),
        ]
        epochs =  [
            Epoch(times=[0, 500], durations=[100, 100], units=pq.ms, labels=["A", "B"])
        ]

        seg = Segment(name="seg1")
        seg.analogsignals.extend(signals)
        seg.spiketrains.extend(spiketrains)
        seg.epochs.extend(epochs)
        for obj in chain(signals, spiketrains, epochs):
            obj.segment = seg

        views = [ChannelView(index=np.array([0, 3, 4]), obj=signals[0], name="view_of_sig1")]
        groups = [
            Group(objects=(signals[0:1] + spiketrains[0:2] + epochs + views), name="group1"),
            Group(objects=(signals[1:2] + spiketrains[1:] + epochs), name="group2")
        ]

        block = Block(name="block1")
        block.segments.append(seg)
        block.groups.extend(groups)
        for obj in chain([seg], groups):
            obj.block = block

        self.write_and_compare([block])

    def test_group_write_nested(self):
        signals = [
            AnalogSignal(np.random.random(size=(1000, 5)) * pq.mV,
                         sampling_period=1 * pq.ms, name="sig1"),
            AnalogSignal(np.random.random(size=(1000, 3)) * pq.mV,
                         sampling_period=1 * pq.ms, name="sig2"),
        ]
        spiketrains = [
            SpikeTrain([0.1, 54.3, 76.6, 464.2], units=pq.ms,
                       t_stop=1000.0 * pq.ms, t_start=0.0 * pq.ms),
            SpikeTrain([30.1, 154.3, 276.6, 864.2], units=pq.ms,
                       t_stop=1000.0 * pq.ms, t_start=0.0 * pq.ms),
            SpikeTrain([120.1, 454.3, 576.6, 764.2], units=pq.ms,
                       t_stop=1000.0 * pq.ms, t_start=0.0 * pq.ms),
        ]
        epochs =  [
            Epoch(times=[0, 500], durations=[100, 100], units=pq.ms, labels=["A", "B"])
        ]

        seg = Segment(name="seg1")
        seg.analogsignals.extend(signals)
        seg.spiketrains.extend(spiketrains)
        seg.epochs.extend(epochs)
        for obj in chain(signals, spiketrains, epochs):
            obj.segment = seg

        views = [ChannelView(index=np.array([0, 3, 4]), obj=signals[0], name="view_of_sig1")]

        subgroup = Group(objects=(signals[0:1] + views), name="subgroup")
        groups = [
            Group(objects=([subgroup] + spiketrains[0:2] + epochs), name="group1"),
            Group(objects=(signals[1:2] + spiketrains[1:] + epochs), name="group2")
        ]

        block = Block(name="block1")
        block.segments.append(seg)
        block.groups.extend(groups)
        for obj in chain([seg], groups):
            obj.block = block

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

        self.write_and_compare([neoblk])

    def test_anonymous_objects_write(self):
        nblocks = 2
        nsegs = 2
        nanasig = 4
        nimgseq = 4
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
                for imgseqdx in range(nimgseq):
                    seg.imagesequences.append(ImageSequence(image_data=self.rquant(
                                                            (10, 20, 10), pq.V),
                                                            sampling_rate=pq.Hz,
                                                            spatial_scale=pq.micrometer))
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
                chx = Group(index=[1, 2],
                            channel_ids=[11, 22])
                blk.groups.append(chx)
                for unidx in range(nunits):
                    unit = Group()
                    chx.add(unit)
        self.writer.write_all_blocks(blocks)
        self.compare_blocks(blocks, self.reader.blocks)

        with self.assertRaises(ValueError):
            self.writer.write_all_blocks(blocks, use_obj_names=True)

    def test_name_objects_write(self):
        nblocks = 2
        nsegs = 2
        nanasig = 4
        nimgseq = 2
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
            blk = Block(name="block{}".format(blkidx))
            blocks.append(blk)
            for segidx in range(nsegs):
                seg = Segment(name="seg{}".format(segidx))
                blk.segments.append(seg)
                for anaidx in range(nanasig):
                    asig = AnalogSignal(
                        name="{}:as{}".format(seg.name, anaidx),
                        signal=signal, sampling_rate=pq.Hz
                    )
                    seg.analogsignals.append(asig)
                # imagesequence
                for imgseqdx in range(nimgseq):
                    imseq = ImageSequence(
                        name="{}:imgs{}".format(seg.name, imgseqdx),
                        image_data=np.random.rand(20, 10, 10), units=pq.mV,
                        sampling_rate=pq.Hz, spatial_scale=pq.micrometer
                    )
                    seg.imagesequences.append(imseq)
                for irridx in range(nirrseg):
                    isig = IrregularlySampledSignal(
                        name="{}:is{}".format(seg.name, irridx),
                        times=times,
                        signal=signal,
                        time_units=pq.s
                    )
                    seg.irregularlysampledsignals.append(isig)
                for epidx in range(nepochs):
                    seg.epochs.append(
                        Epoch(name="{}:ep{}".format(seg.name, epidx),
                              times=times, durations=times)
                    )
                for evidx in range(nevents):
                    seg.events.append(
                        Event(name="{}:ev{}".format(seg.name, evidx),
                              times=times)
                    )
                for stidx in range(nspiketrains):
                    seg.spiketrains.append(
                        SpikeTrain(name="{}:st{}".format(seg.name, stidx),
                                   times=times,
                                   t_stop=times[-1] + pq.s,
                                   units=pq.s)
                    )
            for chidx in range(nchx):
                chx = Group(name="chx{}".format(chidx),
                                   index=[1, 2],
                                   channel_ids=[11, 22])
                blk.groups.append(chx)
                for unidx in range(nunits):
                    unit = Group(name="chx{}-unit{}".format(chidx, unidx))
                    chx.add(unit)

        # put guard on _generate_nix_name
        if not SKIPMOCK:
            nixgenmock = mock.Mock(name="_generate_nix_name",
                                   wraps=self.io._generate_nix_name)
            self.io._generate_nix_name = nixgenmock
        self.writer.write_block(blocks[0], use_obj_names=True)
        self.compare_blocks([blocks[0]], self.reader.blocks)
        self.compare_blocks(self.writer.read_all_blocks(), self.reader.blocks)
        self.compare_blocks(blocks, self.reader.blocks)
        if not SKIPMOCK:
            nixgenmock.assert_not_called()

        self.write_and_compare(blocks, use_obj_names=True)
        if not SKIPMOCK:
            nixgenmock.assert_not_called()

        self.assertEqual(self.reader.blocks[0].name, "block0")

        blocks[0].name = blocks[1].name  # name conflict
        with self.assertRaises(ValueError):
            self.writer.write_all_blocks(blocks, use_obj_names=True)
        blocks[0].name = "new name"
        self.assertEqual(blocks[0].segments[1].spiketrains[1].name, "seg1:st1")
        st0 = blocks[0].segments[0].spiketrains[0].name
        blocks[0].segments[0].spiketrains[1].name = st0  # name conflict
        with self.assertRaises(ValueError):
            self.writer.write_all_blocks(blocks, use_obj_names=True)
        with self.assertRaises(ValueError):
            self.writer.write_block(blocks[0], use_obj_names=True)
        if not SKIPMOCK:
            nixgenmock.assert_not_called()

    def test_name_conflicts(self):
        # anon block
        blk = Block()
        with self.assertRaises(ValueError):
            self.io.write_block(blk, use_obj_names=True)

        # two anon blocks
        blocks = [Block(), Block()]
        with self.assertRaises(ValueError):
            self.io.write_all_blocks(blocks, use_obj_names=True)

        # same name blocks
        blocks = [Block(name="one"), Block(name="one")]
        with self.assertRaises(ValueError):
            self.io.write_all_blocks(blocks, use_obj_names=True)

        # one block, two same name segments
        blk = Block("new")
        seg = Segment("I am the segment", a="a annoation")
        blk.segments.append(seg)
        seg = Segment("I am the segment", a="b annotation")
        blk.segments.append(seg)
        with self.assertRaises(ValueError):
            self.io.write_block(blk, use_obj_names=True)

        times = self.rquant(1, pq.s)
        signal = self.rquant(1, pq.V)
        # name conflict: analog + irregular signals
        seg.analogsignals.append(
            AnalogSignal(name="signal", signal=signal, sampling_rate=pq.Hz)
        )
        seg.imagesequences.append(
            ImageSequence(name='signal',
                          image_data=self.rquant((10, 20, 10), pq.V),
                          sampling_rate=pq.Hz,
                          spatial_scale=pq.micrometer))

        seg.irregularlysampledsignals.append(
            IrregularlySampledSignal(name="signal", signal=signal, times=times)
        )
        blk = Block(name="Signal conflict Block")
        blk.segments.append(seg)
        with self.assertRaises(ValueError):
            self.io.write_block(blk, use_obj_names=True)

        # name conflict: event + spiketrain
        blk = Block(name="Event+SpikeTrain conflict Block")
        seg = Segment(name="Event+SpikeTrain conflict Segment")
        blk.segments.append(seg)
        seg.events.append(Event(name="TimeyStuff", times=times))
        seg.spiketrains.append(SpikeTrain(name="TimeyStuff", times=times,
                                          t_stop=pq.s))
        with self.assertRaises(ValueError):
            self.io.write_block(blk, use_obj_names=True)

        # make spiketrain anon
        blk.segments[0].spiketrains[0].name = None
        with self.assertRaises(ValueError):
            self.io.write_block(blk, use_obj_names=True)

        # name conflict in groups
        blk = Block(name="Group conflict Block")
        blk.groups.append(Group(name="chax", index=[1]))
        blk.groups.append(Group(name="chax", index=[2]))
        with self.assertRaises(ValueError):
            self.io.write_block(blk, use_obj_names=True)

        # name conflict in sub-groups
        blk = Block(name="unitconf")
        chx = Group(name="ok", index=[100])
        blk.groups.append(chx)
        chx.add(Group(name="IHAVEATWIN"))
        chx.add(Group(name="IHAVEATWIN"))
        with self.assertRaises(ValueError):
            self.io.write_block(blk, use_obj_names=True)

    def test_multiref_write(self):
        blk = Block("blk1")
        signal = AnalogSignal(name="sig1", signal=[0, 1, 2], units="mV",
                              sampling_period=pq.Quantity(1, "ms"))
        othersignal = IrregularlySampledSignal(name="i1", signal=[0, 0, 0],
                                               units="mV", times=[1, 2, 3],
                                               time_units="ms")
        imgseq = ImageSequence(name="img1", image_data=self.rquant((10, 20, 10), pq.mV),
                               frame_duration=pq.Quantity(1, "ms"),
                               spatial_scale=pq.meter)
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
            seg.imagesequences.append(imgseq)
            seg.irregularlysampledsignals.append(othersignal)
            seg.events.append(event)
            seg.epochs.append(epoch)
            seg.spiketrains.append(st)

        chidx = Group(index=[10, 20, 29])
        seg = blk.segments[0]
        st = SpikeTrain(name="choochoo", times=[10, 11, 80], t_stop=1000,
                        units="s")
        seg.spiketrains.append(st)
        blk.groups.append(chidx)
        for idx in range(6):
            unit = Group(name="unit" + str(idx))
            chidx.add(unit)
            unit.add(st)

        self.writer.write_block(blk)
        self.compare_blocks([blk], self.reader.blocks)

    # NOTE: storing data objects that are not within a segment is currently
    #       disallowed. Leaving this test commented out until this policy
    #       is properly discussed.
    # def test_no_segment_write(self):
    #     # Tests storing AnalogSignal, IrregularlySampledSignal, and SpikeTrain
    #     # objects in the secondary (Group) substructure without them
    #     # being attached to a Segment.
    #     blk = Block("segmentless block")
    #     signal = AnalogSignal(name="sig1", signal=[0, 1, 2], units="mV",
    #                           sampling_period=pq.Quantity(1, "ms"))
    #     othersignal = IrregularlySampledSignal(name="i1", signal=[0, 0, 0],
    #                                            units="mV", times=[1, 2, 3],
    #                                            time_units="ms")
    #     sta = SpikeTrain(name="the train of spikes", times=[0.1, 0.2, 10.3],
    #                      t_stop=11, units="us")
    #     stb = SpikeTrain(name="the train of spikes b", times=[1.1, 2.2, 10.1],
    #                      t_stop=100, units="ms")

    #     chidx = Group(index=[8, 13, 21])
    #     blk.groups.append(chidx)
    #     chidx.add(signal)
    #     chidx.add(othersignal)

    #     unit = Group()
    #     chidx.add(unit)
    #     unit.add(sta, stb)
    #     self.writer.write_block(blk)
    #     self.writer.close()

    #     self.compare_blocks([blk], self.reader.blocks)

    #     reader = NixIO(self.filename, "ro")
    #     blk = reader.read_block(neoname="segmentless block")
    #     chx = blk.groups[0]
    #     self.assertEqual(len(chx.analogsignals), 1)
    #     self.assertEqual(len(chx.irregularlysampledsignals), 1)
    #     self.assertEqual(len(chx.units[0].spiketrains), 2)

    def test_rewrite_refs(self):

        def checksignalcounts(fname):
            with NixIO(fname, "ro") as r:
                blk = r.read_block()
            chidx = blk.groups[0]
            seg = blk.segments[0]
            self.assertEqual(len(chidx.analogsignals), 2)
            self.assertEqual(len(chidx.groups[0].spiketrains), 3)
            self.assertEqual(len(seg.analogsignals), 3)
            self.assertEqual(len(seg.spiketrains), 4)

        blk = Block()
        seg = Segment()
        blk.segments.append(seg)

        # Group replacing previous ChannelIndex
        chidx = Group(index=[1])
        blk.groups.append(chidx)

        # Two signals in Group
        for idx in range(2):
            asigchx = AnalogSignal(signal=[idx], units="mV",
                                   sampling_rate=pq.Hz)
            chidx.add(asigchx)
            seg.analogsignals.append(asigchx)

        # Group replacing previous Unit
        unit = Group()
        chidx.add(unit)

        # Three SpikeTrains on Unit
        for idx in range(3):
            st = SpikeTrain([idx], units="ms", t_stop=40)
            unit.add(st)
            seg.spiketrains.append(st)

        # One signal in Segment but not in Group
        asigseg = AnalogSignal(signal=[2], units="uA",
                               sampling_rate=pq.Hz)
        seg.analogsignals.append(asigseg)

        # One spiketrain in Segment but not in Group
        stseg = SpikeTrain([10], units="ms", t_stop=40)
        seg.spiketrains.append(stseg)

        # Write, compare, and check counts
        self.writer.write_block(blk)
        self.compare_blocks([blk], self.reader.blocks)
        self.assertEqual(len(chidx.analogsignals), 2)
        self.assertEqual(len(seg.analogsignals), 3)
        self.assertEqual(len(chidx.groups[0].spiketrains), 3)
        self.assertEqual(len(seg.spiketrains), 4)

        # Check counts with separate reader
        checksignalcounts(self.filename)

        # Write again and check counts
        secondwrite = os.path.join(self.tempdir, "testnixio-2.nix")
        with NixIO(secondwrite, "ow") as w:
            w.write_block(blk)

        self.compare_blocks([blk], self.reader.blocks)

        # Read back and check counts
        scndreader = nix.File.open(secondwrite, mode=nix.FileMode.ReadOnly)
        self.compare_blocks([blk], scndreader.blocks)
        checksignalcounts(secondwrite)

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
        self.assertEqual(section["dt"], dt_to_nix(dt)[0])

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

        # empty string (gets stored as empty list)
        writeprop(section, "emptystring", "")
        self.assertEqual(list(), section["emptystring"])

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

    def test_empty_array_annotations(self):
        wblock = Block("block with spiketrain")
        wseg = Segment()
        empty_array_annotations = {'emptylist': [],
                                   'emptyarray': np.array([]),
                                   'quantitylist': [] * pq.s,
                                   'quantityarray': np.array([]) * pq.s}
        expected_array_annotations = {'emptylist': np.array([]),
                                   'emptyarray': np.array([]),
                                   'quantitylist': np.array([]) * pq.s,
                                   'quantityarray': np.array([]) * pq.s}
        wseg.spiketrains = [SpikeTrain(times=[] * pq.s, t_stop=1 * pq.s,
                                       array_annotations=empty_array_annotations)]
        wblock.segments = [wseg]
        self.writer.write_block(wblock)
        try:
            rblock = self.writer.read_block(neoname="block with spiketrain")
        except Exception as exc:
            self.fail('The following exception was raised when'
                      + ' reading the block with an empty array annotation:\n'
                      + str(exc))
        rst = rblock.segments[0].spiketrains[0]
        for k, v in expected_array_annotations.items():
            self.assertIn(k, rst.array_annotations)
            np.testing.assert_array_equal(rst.array_annotations[k], v)
            if hasattr(v, 'units'):
                self.assertEqual(rst.array_annotations[k].units, v.units)
            else:
                self.assertFalse(hasattr(rst.array_annotations[k], 'units'))

    def test_write_proxyobjects(self):

        def generate_complete_block():
            block = Block()
            seg = Segment()
            block.segments.append(seg)

            # add spiketrain
            waveforms = self.rquant((3, 5, 10), pq.mV)
            spiketrain = SpikeTrain(times=[1, 1.1, 1.2] * pq.ms,
                                    t_stop=1.5 * pq.s,
                                    name="spikes with wf",
                                    description="spikes for waveform test",
                                    waveforms=waveforms)
            seg.spiketrains.append(spiketrain)
            # add imagesequence
            imgseq = ImageSequence(name="img1",
                                   image_data=self.rquant((10, 20, 10), pq.mV),
                                   frame_duration=pq.Quantity(1, "ms"),
                                   spatial_scale=pq.meter)

            seg.imagesequences.append(imgseq)
            # add signals
            asig = AnalogSignal(signal=self.rquant((19, 15), pq.mV),
                                sampling_rate=pq.Quantity(10, "Hz"))
            seg.analogsignals.append(asig)
            irsig = IrregularlySampledSignal(signal=np.random.random((20, 30)),
                                             times=self.rquant(20, pq.ms, True),
                                             units=pq.A)
            seg.irregularlysampledsignals.append(irsig)

            # add events and epochs
            epoch = Epoch(times=[1, 1, 10, 3] * pq.ms,
                          durations=[3, 3, 3, 1] * pq.ms,
                          labels=np.array(["one", "two", "three", "four"]),
                          name="test epoch", description="an epoch for testing")
            seg.epochs.append(epoch)
            event = Event(times=np.arange(0, 30, 10) * pq.s,
                          labels=np.array(["0", "1", "2"]),
                          name="event name",
                          description="event description")
            seg.events.append(event)

            # add channel index and unit
            channel = Group(index=[0], channel_names=['mychannelname'],
                                   channel_ids=[4],
                                   name=['testname'])
            block.groups.append(channel)
            unit = Group(name='myunit', description='blablabla',
                        file_origin='fileA.nix',
                        myannotation='myannotation')
            channel.add(unit)
            unit.add(spiketrain)

            # make sure everything is linked properly
            block.create_relationship()

            return block

        block = generate_complete_block()

        basename, ext = os.path.splitext(self.filename)
        filename2 = basename + '-2.' + ext

        # writing block to file 1
        with NixIO(filename2, 'ow') as io:
            io.write_block(block)

        # reading data as lazy objects from file 1
        with NixIO_lazy(filename2) as io:
            block_lazy = io.read_block(lazy=True)

            self.write_and_compare([block_lazy])

    def test_annotation_types(self):
        annotations = {
            "somedate": self.rdate(),
            "now": datetime.now(),
            "today": date.today(),
            "sometime": time(13, 37, 42),
            "somequantity": self.rquant(10, pq.ms),
            "somestring": self.rsentence(3),
            "npfloat": np.float64(10),
            "nparray": np.array([1, 2, 400]),
            "emptystr": "",
        }
        wblock = Block("annotation_block", **annotations)
        self.writer.write_block(wblock)
        rblock = self.writer.read_block(neoname="annotation_block")
        for k in annotations:
            orig = annotations[k]
            readval = rblock.annotations[k]
            if isinstance(orig, np.ndarray):
                np.testing.assert_almost_equal(orig, readval)
            else:
                self.assertEqual(annotations[k], rblock.annotations[k])


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
            self.assertEqual(neoblock.annotations["nix_name"],
                             self.nixfile.blocks[idx].name)

    def test_auto_index_read(self):
        for nixblock in self.nixfile.blocks:
            neoblock = self.io.read_block()  # don't specify index
            self.assertEqual(neoblock.annotations["nix_name"], nixblock.name)

        # No more blocks - should return None
        self.assertIs(self.io.read_block(), None)
        self.assertIs(self.io.read_block(), None)
        self.assertIs(self.io.read_block(), None)

        with NixIO(self.filename, "ro") as nf:
            neoblock = nf.read_block(index=1)
            self.assertEqual(self.nixfile.blocks[1].name,
                             neoblock.annotations["nix_name"])

            neoblock = nf.read_block()  # should start again from 0
            self.assertEqual(self.nixfile.blocks[0].name,
                             neoblock.annotations["nix_name"])

    def test_neo_name_read(self):
        for nixblock in self.nixfile.blocks:
            neoname = nixblock.metadata["neo_name"]
            neoblock = self.io.read_block(neoname=neoname)
            self.assertEqual(neoblock.annotations["nix_name"], nixblock.name)

    def test_array_annotations_read(self):
        for bl in self.io.read_all_blocks():
            nix_block = self.nixfile.blocks[bl.annotations['nix_name']]
            for seg in bl.segments:

                for anasig in seg.analogsignals:
                    da = nix_block.data_arrays[anasig.annotations['nix_name'] + '.0']
                    self.assertIn('anasig_arr_ann', da.metadata)
                    self.assertIn('anasig_arr_ann', anasig.array_annotations)
                    nix_ann = da.metadata['anasig_arr_ann']
                    neo_ann = anasig.array_annotations['anasig_arr_ann']
                    self.assertTrue(np.all(nix_ann == neo_ann.magnitude))
                    self.assertEqual(da.metadata.props['anasig_arr_ann'].unit,
                                     units_to_string(neo_ann.units))
                for irrsig in seg.irregularlysampledsignals:
                    da = nix_block.data_arrays[irrsig.annotations['nix_name'] + '.0']
                    self.assertIn('irrsig_arr_ann', da.metadata)
                    self.assertIn('irrsig_arr_ann', irrsig.array_annotations)
                    nix_ann = da.metadata['irrsig_arr_ann']
                    neo_ann = irrsig.array_annotations['irrsig_arr_ann']
                    self.assertTrue(np.all(nix_ann == neo_ann.magnitude))
                    self.assertEqual(da.metadata.props['irrsig_arr_ann'].unit,
                                     units_to_string(neo_ann.units))
                for imgseq in seg.imagesequences:
                    da = nix_block.data_arrays[imgseq.annotations['nix_name'] + '.0']
                    self.assertIn('imgseq_arr_ann', da.metadata)
                    self.assertIn('imgseq_arr_ann', imgseq.array_annotations)
                    nix_ann = da.metadata['imgseq_arr_ann']
                    neo_ann = imgseq.array_annotations['imgseq_arr_ann']
                    self.assertTrue(np.all(nix_ann == neo_ann.magnitude))
                    self.assertEqual(da.metadata.props['imgseq_arr_ann'].unit,
                                     units_to_string(neo_ann.units))
                for ev in seg.events:
                    da = nix_block.multi_tags[ev.annotations['nix_name']]
                    self.assertIn('ev_arr_ann', da.metadata)
                    self.assertIn('ev_arr_ann', ev.array_annotations)
                    nix_ann = da.metadata['ev_arr_ann']
                    neo_ann = ev.array_annotations['ev_arr_ann']
                    self.assertTrue(np.all(nix_ann == neo_ann.magnitude))
                    self.assertEqual(da.metadata.props['ev_arr_ann'].unit,
                                     units_to_string(neo_ann.units))
                for ep in seg.epochs:
                    da = nix_block.multi_tags[ep.annotations['nix_name']]
                    self.assertIn('ep_arr_ann', da.metadata)
                    self.assertIn('ep_arr_ann', ep.array_annotations)
                    nix_ann = da.metadata['ep_arr_ann']
                    neo_ann = ep.array_annotations['ep_arr_ann']
                    self.assertTrue(np.all(nix_ann == neo_ann.magnitude))
                    self.assertEqual(da.metadata.props['ep_arr_ann'].unit,
                                     units_to_string(neo_ann.units))
                for st in seg.spiketrains:
                    da = nix_block.multi_tags[st.annotations['nix_name']]
                    self.assertIn('st_arr_ann', da.metadata)
                    self.assertIn('st_arr_ann', st.array_annotations)
                    nix_ann = da.metadata['st_arr_ann']
                    neo_ann = st.array_annotations['st_arr_ann']
                    self.assertTrue(np.all(nix_ann == neo_ann.magnitude))
                    self.assertEqual(da.metadata.props['st_arr_ann'].unit,
                                     units_to_string(neo_ann.units))

    def test_read_blocks_are_writable(self):
        filename = os.path.join(self.tempdir, "testnixio_out.nix")
        writer = NixIO(filename, "ow")

        blocks = self.io.read_all_blocks()

        try:
            writer.write_all_blocks(blocks)
        except Exception as exc:
            self.fail('The following exception was raised when'
                      + ' writing the blocks loaded with NixIO:\n'
                      + str(exc))


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

        nixfile = nix.File.open(self.filename, nix.FileMode.ReadOnly)
        self.compare_blocks([neoblock], nixfile.blocks)
        nixfile.close()

        neoblock.annotate(**self.rdict(5))
        with NixIO(self.filename, "rw") as iofile:
            iofile.write_block(neoblock)
        nixfile = nix.File.open(self.filename, nix.FileMode.ReadOnly)
        self.compare_blocks([neoblock], nixfile.blocks)
        nixfile.close()

    def test_context_read(self):
        nixfile = nix.File.open(self.filename, nix.FileMode.Overwrite)
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
    entities_to_download = []
    entities_to_est = []


if __name__ == "__main__":
    unittest.main()
