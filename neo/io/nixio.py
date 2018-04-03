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
Module for reading data from files in the NIX format.

Author: Achilleas Koutsou

This IO supports both writing and reading of NIX files. Reading is supported
only if the NIX file was created using this IO.

Details on how the Neo object tree is mapped to NIX, as well as details on
behaviours specific to this IO, can be found on the wiki of the G-Node fork of
Neo: https://github.com/G-Node/python-neo/wiki
"""

from __future__ import absolute_import

import time
from datetime import datetime
from collections import Iterable
import itertools
from uuid import uuid4

import quantities as pq
import numpy as np

from .baseio import BaseIO
from ..core import (Block, Segment, ChannelIndex, AnalogSignal,
                    IrregularlySampledSignal, Epoch, Event, SpikeTrain, Unit)
from ..version import version as neover

try:
    import nixio as nix

    HAVE_NIX = True
except ImportError:
    HAVE_NIX = False

try:
    string_types = basestring
except NameError:
    string_types = str

EMPTYANNOTATION = "EMPTYLIST"


def stringify(value):
    if value is None:
        return value
    if isinstance(value, bytes):
        value = value.decode()
    return str(value)


def create_quantity(values, unitstr):
    if "*" in unitstr:
        unit = pq.CompoundUnit(stringify(unitstr))
    else:
        unit = unitstr
    return pq.Quantity(values, unit)


def units_to_string(pqunit):
    dim = str(pqunit.dimensionality)
    if dim.startswith("(") and dim.endswith(")"):
        return dim.strip("()")
    return dim


def calculate_timestamp(dt):
    if isinstance(dt, datetime):
        return int(time.mktime(dt.timetuple()))
    return int(dt)


class NixIO(BaseIO):
    """
    Class for reading and writing NIX files.
    """

    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, ChannelIndex,
                         AnalogSignal, IrregularlySampledSignal,
                         Epoch, Event, SpikeTrain, Unit]
    readable_objects = [Block]
    writeable_objects = [Block]

    name = "NIX"
    extensions = ["h5", "nix"]
    mode = "file"

    nix_version = nix.__version__ if HAVE_NIX else "NIX NOT FOUND"

    def __init__(self, filename, mode="rw"):
        """
        Initialise IO instance and NIX file.

        :param filename: Full path to the file
        """

        if not HAVE_NIX:
            raise Exception("Failed to import NIX. "
                            "The NixIO requires the Python bindings for NIX "
                            "(nixio on PyPi). Try `pip install nixio`.")

        BaseIO.__init__(self, filename)
        self.filename = filename
        if mode == "ro":
            filemode = nix.FileMode.ReadOnly
        elif mode == "rw":
            filemode = nix.FileMode.ReadWrite
        elif mode == "ow":
            filemode = nix.FileMode.Overwrite
        else:
            raise ValueError("Invalid mode specified '{}'. "
                             "Valid modes: 'ro' (ReadOnly)', 'rw' (ReadWrite),"
                             " 'ow' (Overwrite).".format(mode))
        self.nix_file = nix.File.open(self.filename, filemode, backend="h5py")

        if self.nix_file.mode == nix.FileMode.ReadOnly:
            self._file_version = '0.5.2'
            if "neo" in self.nix_file.sections:
                self._file_version = self.nix_file.sections["neo"]["version"]
        elif self.nix_file.mode == nix.FileMode.ReadWrite:
            if "neo" in self.nix_file.sections:
                self._file_version = self.nix_file.sections["neo"]["version"]
            else:
                self._file_version = '0.5.2'
                filemd = self.nix_file.create_section("neo", "neo.metadata")
                filemd["version"] = self._file_version
        else:
            # new file
            filemd = self.nix_file.create_section("neo", "neo.metadata")
            filemd["version"] = neover
            self._file_version = neover

        self._block_read_counter = 0

        # helper maps
        self._neo_map = dict()
        self._ref_map = dict()
        self._signal_map = dict()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_all_blocks(self):
        return list(self._nix_to_neo_block(blk)
                    for blk in self.nix_file.blocks)

    def read_block(self, index=None, nixname=None, neoname=None, lazy=False):
        """
        Loads a Block from the NIX file along with all contained child objects
        and returns the equivalent Neo Block.

        The Block to read can be specified in one of three ways:
        - Index (position) in the file
        - Name of the NIX Block (see [...] for details on the naming)
        - Name of the original Neo Block

        If no arguments are specified, the first Block is returned and
        consecutive calls to the function return the next Block in the file.
        After all Blocks have been loaded this way, the function returns None.

        If more than one argument is specified, the precedence order is:
        index, nixname, neoname

        Note that Neo objects can be anonymous or have non-unique names,
        so specifying a Neo name may be ambiguous.

        See also :meth:`NixIO.iter_blocks`.

        :param index: The position of the Block to be loaded (creation order)
        :param nixname: The name of the Block in NIX
        :param neoname: The name of the original Neo Block
        """
        assert not lazy, "Lazy loading not supported"
        nix_block = None
        if index is not None:
            nix_block = self.nix_file.blocks[index]
        elif nixname is not None:
            nix_block = self.nix_file.blocks[nixname]
        elif neoname is not None:
            for blk in self.nix_file.blocks:
                if ("neo_name" in blk.metadata
                    and blk.metadata["neo_name"] == neoname):
                    nix_block = blk
                    break
            else:
                raise KeyError(
                    "Block with Neo name '{}' does not exist".format(neoname)
                )
        else:
            index = self._block_read_counter
            if index >= len(self.nix_file.blocks):
                return None
            nix_block = self.nix_file.blocks[index]

        nix_block = self.nix_file.blocks[self._block_read_counter]
        self._block_read_counter += 1
        return self._nix_to_neo_block(nix_block)

    def iter_blocks(self):
        """
        Returns an iterator which can be used to consecutively load and convert
        all Blocks from the NIX File.
        """
        for blk in self.nix_file.blocks:
            yield self._nix_to_neo_block(blk)

    def _nix_to_neo_block(self, nix_block):
        neo_attrs = self._nix_attr_to_neo(nix_block)
        neo_block = Block(**neo_attrs)
        neo_block.rec_datetime = datetime.fromtimestamp(
            nix_block.created_at
        )

        # descend into Groups
        for grp in nix_block.groups:
            newseg = self._nix_to_neo_segment(grp)
            neo_block.segments.append(newseg)
            # parent reference
            newseg.block = neo_block

        # find free floating (Groupless) signals and spiketrains
        blockdas = self._group_signals(nix_block.data_arrays)
        for name, das in blockdas.items():
            if name not in self._neo_map:
                if das[0].type == "neo.analogsignal":
                    self._nix_to_neo_analogsignal(das)
                elif das[0].type == "neo.irregularlysampledsignal":
                    self._nix_to_neo_irregularlysampledsignal(das)
        for mt in nix_block.multi_tags:
            if mt.type == "neo.spiketrain" and mt.name not in self._neo_map:
                self._nix_to_neo_spiketrain(mt)

        # descend into Sources
        for src in nix_block.sources:
            newchx = self._nix_to_neo_channelindex(src)
            neo_block.channel_indexes.append(newchx)
            # parent reference
            newchx.block = neo_block

        # reset maps
        self._neo_map = dict()
        self._ref_map = dict()
        self._signal_map = dict()

        return neo_block

    def _nix_to_neo_segment(self, nix_group):
        neo_attrs = self._nix_attr_to_neo(nix_group)
        neo_segment = Segment(**neo_attrs)
        neo_segment.rec_datetime = datetime.fromtimestamp(
            nix_group.created_at
        )

        self._neo_map[nix_group.name] = neo_segment

        # this will probably get all the DAs anyway, but if we change any part
        # of the mapping to add other kinds of DataArrays to a group, such as
        # MultiTag positions and extents, this filter will be necessary
        dataarrays = list(filter(
            lambda da: da.type in ("neo.analogsignal",
                                   "neo.irregularlysampledsignal"),
            nix_group.data_arrays))
        dataarrays = self._group_signals(dataarrays)
        # descend into DataArrays
        for name, das in dataarrays.items():
            if das[0].type == "neo.analogsignal":
                newasig = self._nix_to_neo_analogsignal(das)
                neo_segment.analogsignals.append(newasig)
                # parent reference
                newasig.segment = neo_segment
            elif das[0].type == "neo.irregularlysampledsignal":
                newisig = self._nix_to_neo_irregularlysampledsignal(das)
                neo_segment.irregularlysampledsignals.append(newisig)
                # parent reference
                newisig.segment = neo_segment

        # descend into MultiTags
        for mtag in nix_group.multi_tags:
            if mtag.type == "neo.event":
                newevent = self._nix_to_neo_event(mtag)
                neo_segment.events.append(newevent)
                # parent reference
                newevent.segment = neo_segment
            elif mtag.type == "neo.epoch":
                newepoch = self._nix_to_neo_epoch(mtag)
                neo_segment.epochs.append(newepoch)
                # parent reference
                newepoch.segment = neo_segment
            elif mtag.type == "neo.spiketrain":
                newst = self._nix_to_neo_spiketrain(mtag)
                neo_segment.spiketrains.append(newst)
                # parent reference
                newst.segment = neo_segment

        return neo_segment

    def _nix_to_neo_channelindex(self, nix_source):
        neo_attrs = self._nix_attr_to_neo(nix_source)
        channels = list(self._nix_attr_to_neo(c)
                        for c in nix_source.sources
                        if c.type == "neo.channelindex")
        neo_attrs["index"] = np.array([c["index"]
                                       for c in channels])
        if len(channels):
            chan_names = list(c["neo_name"]
                              for c in channels if "neo_name" in c)
            chan_ids = list(c["channel_id"]
                            for c in channels if "channel_id" in c)
            if chan_names:
                neo_attrs["channel_names"] = chan_names
            if chan_ids:
                neo_attrs["channel_ids"] = chan_ids
            if "coordinates" in channels[0]:
                neo_attrs["coordinates"] = list(c["coordinates"]
                                                for c in channels)

        neo_chx = ChannelIndex(**neo_attrs)
        self._neo_map[nix_source.name] = neo_chx

        # create references to Signals
        signals = self._ref_map.get(nix_source.name, list())
        for sig in signals:
            if isinstance(sig, AnalogSignal):
                neo_chx.analogsignals.append(sig)
            elif isinstance(sig, IrregularlySampledSignal):
                neo_chx.irregularlysampledsignals.append(sig)
                # else error?

        # descend into Sources
        for src in nix_source.sources:
            if src.type == "neo.unit":
                newunit = self._nix_to_neo_unit(src)
                neo_chx.units.append(newunit)
                # parent reference
                newunit.channel_index = neo_chx

        return neo_chx

    def _nix_to_neo_unit(self, nix_source):
        neo_attrs = self._nix_attr_to_neo(nix_source)
        neo_unit = Unit(**neo_attrs)
        self._neo_map[nix_source.name] = neo_unit

        # create references to SpikeTrains
        neo_unit.spiketrains.extend(self._ref_map.get(nix_source.name, list()))
        return neo_unit

    def _nix_to_neo_analogsignal(self, nix_da_group):
        """
        Convert a group of NIX DataArrays to a Neo AnalogSignal. This method
        expects a list of data arrays that all represent the same,
        multidimensional Neo AnalogSignal object.

        :param nix_da_group: a list of NIX DataArray objects
        :return: a Neo AnalogSignal object
        """
        nix_da_group = sorted(nix_da_group, key=lambda d: d.name)
        neo_attrs = self._nix_attr_to_neo(nix_da_group[0])
        metadata = nix_da_group[0].metadata
        neo_attrs["nix_name"] = metadata.name  # use the common base name

        unit = nix_da_group[0].unit
        signaldata = np.array([d[:] for d in nix_da_group]).transpose()
        signaldata = create_quantity(signaldata, unit)
        timedim = self._get_time_dimension(nix_da_group[0])
        sampling_period = create_quantity(timedim.sampling_interval,
                                          timedim.unit)
        # t_start should have been added to neo_attrs via the NIX
        # object's metadata. This may not be present since in older
        # versions, we didn't store t_start in the metadata when it
        # wasn't necessary, such as when the timedim.offset and unit
        # did not require rescaling.
        if "t_start" in neo_attrs:
            t_start = neo_attrs["t_start"]
            del neo_attrs["t_start"]
        else:
            t_start = create_quantity(timedim.offset, timedim.unit)

        neo_signal = AnalogSignal(
            signal=signaldata, sampling_period=sampling_period,
            t_start=t_start, **neo_attrs
        )
        self._neo_map[neo_attrs["nix_name"]] = neo_signal
        # all DAs reference the same sources
        srcnames = list(src.name for src in nix_da_group[0].sources)
        for n in srcnames:
            if n not in self._ref_map:
                self._ref_map[n] = list()
            self._ref_map[n].append(neo_signal)
        return neo_signal

    def _nix_to_neo_irregularlysampledsignal(self, nix_da_group):
        """
        Convert a group of NIX DataArrays to a Neo IrregularlySampledSignal.
        This method expects a list of data arrays that all represent the same,
        multidimensional Neo IrregularlySampledSignal object.

        :param nix_da_group: a list of NIX DataArray objects
        :return: a Neo IrregularlySampledSignal object
        """
        nix_da_group = sorted(nix_da_group, key=lambda d: d.name)
        neo_attrs = self._nix_attr_to_neo(nix_da_group[0])
        metadata = nix_da_group[0].metadata
        neo_attrs["nix_name"] = metadata.name  # use the common base name

        unit = nix_da_group[0].unit
        signaldata = np.array([d[:] for d in nix_da_group]).transpose()
        signaldata = create_quantity(signaldata, unit)
        timedim = self._get_time_dimension(nix_da_group[0])
        times = create_quantity(timedim.ticks, timedim.unit)
        neo_signal = IrregularlySampledSignal(
            signal=signaldata, times=times, **neo_attrs
        )
        self._neo_map[neo_attrs["nix_name"]] = neo_signal
        # all DAs reference the same sources
        srcnames = list(src.name for src in nix_da_group[0].sources)
        for n in srcnames:
            if n not in self._ref_map:
                self._ref_map[n] = list()
            self._ref_map[n].append(neo_signal)
        return neo_signal

    def _nix_to_neo_event(self, nix_mtag):
        neo_attrs = self._nix_attr_to_neo(nix_mtag)
        time_unit = nix_mtag.positions.unit
        times = create_quantity(nix_mtag.positions, time_unit)
        labels = np.array(nix_mtag.positions.dimensions[0].labels,
                          dtype="S")
        neo_event = Event(times=times, labels=labels, **neo_attrs)
        self._neo_map[nix_mtag.name] = neo_event
        return neo_event

    def _nix_to_neo_epoch(self, nix_mtag):
        neo_attrs = self._nix_attr_to_neo(nix_mtag)
        time_unit = nix_mtag.positions.unit
        times = create_quantity(nix_mtag.positions, time_unit)
        durations = create_quantity(nix_mtag.extents,
                                    nix_mtag.extents.unit)
        labels = np.array(nix_mtag.positions.dimensions[0].labels,
                          dtype="S")
        neo_epoch = Epoch(times=times, durations=durations, labels=labels,
                          **neo_attrs)
        self._neo_map[nix_mtag.name] = neo_epoch
        return neo_epoch

    def _nix_to_neo_spiketrain(self, nix_mtag):
        neo_attrs = self._nix_attr_to_neo(nix_mtag)
        time_unit = nix_mtag.positions.unit
        times = create_quantity(nix_mtag.positions, time_unit)
        neo_spiketrain = SpikeTrain(times=times, **neo_attrs)
        if nix_mtag.features:
            wfda = nix_mtag.features[0].data
            wftime = self._get_time_dimension(wfda)
            neo_spiketrain.waveforms = create_quantity(wfda, wfda.unit)
            interval_units = wftime.unit
            neo_spiketrain.sampling_period = create_quantity(
                wftime.sampling_interval, interval_units
            )
            left_sweep_units = wftime.unit
            if "left_sweep" in wfda.metadata:
                neo_spiketrain.left_sweep = create_quantity(
                    wfda.metadata["left_sweep"], left_sweep_units
                )
        self._neo_map[nix_mtag.name] = neo_spiketrain

        srcnames = list(src.name for src in nix_mtag.sources)
        for n in srcnames:
            if n not in self._ref_map:
                self._ref_map[n] = list()
            self._ref_map[n].append(neo_spiketrain)
        return neo_spiketrain

    def write_all_blocks(self, neo_blocks):
        """
        Convert all ``neo_blocks`` to the NIX equivalent and write them to the
        file.

        :param neo_blocks: List (or iterable) containing Neo blocks
        """
        for bl in neo_blocks:
            self.write_block(bl)

    def write_block(self, block):
        """
        Convert the provided Neo Block to a NIX Block and write it to
        the NIX file.

        :param block: Neo Block to be written
        """
        if "nix_name" in block.annotations:
            nix_name = block.annotations["nix_name"]
        else:
            nix_name = "neo.block.{}".format(self._generate_nix_name())
            block.annotate(nix_name=nix_name)

        if nix_name in self.nix_file.blocks:
            nixblock = self.nix_file.blocks[nix_name]
            del self.nix_file.blocks[nix_name]
            del self.nix_file.sections[nix_name]

        nixblock = self.nix_file.create_block(nix_name, "neo.block")
        nixblock.metadata = self.nix_file.create_section(
            nix_name, "neo.block.metadata"
        )
        metadata = nixblock.metadata
        neoname = block.name if block.name is not None else ""
        metadata["neo_name"] = neoname
        nixblock.definition = block.description
        if block.rec_datetime:
            nixblock.force_created_at(
                calculate_timestamp(block.rec_datetime)
            )
        if block.file_datetime:
            metadata["file_datetime"] = block.file_datetime
        if block.annotations:
            for k, v in block.annotations.items():
                self._write_property(metadata, k, v)

        # descend into Segments
        for seg in block.segments:
            self._write_segment(seg, nixblock)

        # descend into ChannelIndexes
        for chx in block.channel_indexes:
            self._write_channelindex(chx, nixblock)

        self._create_source_links(block, nixblock)

    def _write_channelindex(self, chx, nixblock):
        """
        Convert the provided Neo ChannelIndex to a NIX Source and write it to
        the NIX file. For each index in the ChannelIndex object, a child
        NIX Source is also created.

        :param chx: The Neo ChannelIndex to be written
        :param nixblock: NIX Block where the Source will be created
        """
        if "nix_name" in chx.annotations:
            nix_name = chx.annotations["nix_name"]
        else:
            nix_name = "neo.channelindex.{}".format(self._generate_nix_name())
            chx.annotate(nix_name=nix_name)
        nixsource = nixblock.create_source(nix_name, "neo.channelindex")
        nixsource.metadata = nixblock.metadata.create_section(
            nix_name, "neo.channelindex.metadata"
        )

        metadata = nixsource.metadata
        neoname = chx.name if chx.name is not None else ""
        metadata["neo_name"] = neoname
        nixsource.definition = chx.description
        if chx.annotations:
            for k, v in chx.annotations.items():
                self._write_property(metadata, k, v)

        for idx, channel in enumerate(chx.index):
            channame = "{}.ChannelIndex{}".format(nix_name, idx)
            nixchan = nixsource.create_source(channame, "neo.channelindex")
            nixchan.metadata = nixsource.metadata.create_section(
                nixchan.name, "neo.channelindex.metadata"
            )
            nixchan.definition = nixsource.definition
            chanmd = nixchan.metadata
            chanmd["index"] = nix.Value(int(channel))
            if len(chx.channel_names):
                neochanname = stringify(chx.channel_names[idx])
                chanmd["neo_name"] = nix.Value(neochanname)
            if len(chx.channel_ids):
                chanid = chx.channel_ids[idx]
                chanmd["channel_id"] = nix.Value(chanid)
            if chx.coordinates is not None:
                coords = chx.coordinates[idx]
                coordunits = stringify(coords[0].dimensionality)
                nixcoords = tuple(nix.Value(c.magnitude.item())
                                  for c in coords)
                chanprop = chanmd.create_property("coordinates", nixcoords)
                chanprop.unit = coordunits

        # Descend into Units
        for unit in chx.units:
            self._write_unit(unit, nixsource)

    def _write_segment(self, segment, nixblock):
        """
        Convert the provided Neo Segment to a NIX Group and write it to the
        NIX file.

        :param segment: Neo Segment to be written
        :param nixblock: NIX Block where the Group will be created
        """
        if "nix_name" in segment.annotations:
            nix_name = segment.annotations["nix_name"]
        else:
            nix_name = "neo.segment.{}".format(self._generate_nix_name())
            segment.annotate(nix_name=nix_name)

        nixgroup = nixblock.create_group(nix_name, "neo.segment")
        nixgroup.metadata = nixblock.metadata.create_section(
            nix_name, "neo.segment.metadata"
        )
        metadata = nixgroup.metadata
        neoname = segment.name if segment.name is not None else ""
        metadata["neo_name"] = neoname
        nixgroup.definition = segment.description
        if segment.rec_datetime:
            nixgroup.force_created_at(
                calculate_timestamp(segment.rec_datetime)
            )
        if segment.file_datetime:
            metadata["file_datetime"] = segment.file_datetime
        if segment.annotations:
            for k, v in segment.annotations.items():
                self._write_property(metadata, k, v)

        # write signals, events, epochs, and spiketrains
        for asig in segment.analogsignals:
            self._write_analogsignal(asig, nixblock, nixgroup)
        for isig in segment.irregularlysampledsignals:
            self._write_irregularlysampledsignal(isig, nixblock, nixgroup)
        for event in segment.events:
            self._write_event(event, nixblock, nixgroup)
        for epoch in segment.epochs:
            self._write_epoch(epoch, nixblock, nixgroup)
        for spiketrain in segment.spiketrains:
            self._write_spiketrain(spiketrain, nixblock, nixgroup)

    def _write_analogsignal(self, anasig, nixblock, nixgroup):
        """
        Convert the provided ``anasig`` (AnalogSignal) to a list of NIX
        DataArray objects and write them to the NIX file. All DataArray objects
        created from the same AnalogSignal have their metadata section point to
        the same object.

        :param anasig: The Neo AnalogSignal to be written
        :param nixblock: NIX Block where the DataArrays will be created
        :param nixgroup: NIX Group where the DataArrays will be attached
        """
        if "nix_name" in anasig.annotations:
            nix_name = anasig.annotations["nix_name"]
        else:
            nix_name = "neo.analogsignal.{}".format(self._generate_nix_name())
            anasig.annotate(nix_name=nix_name)

        if "{}.0".format(nix_name) in nixblock.data_arrays and nixgroup:
            # AnalogSignal is in multiple Segments.
            # Append DataArrays to Group and return.
            dalist = list()
            for idx in itertools.count():
                daname = "{}.{}".format(nix_name, idx)
                if daname in nixblock.data_arrays:
                    dalist.append(nixblock.data_arrays[daname])
                else:
                    break
            nixgroup.data_arrays.extend(dalist)
            return

        data = np.transpose(anasig[:].magnitude)
        parentmd = nixgroup.metadata if nixgroup else nixblock.metadata
        metadata = parentmd.create_section(nix_name,
                                           "neo.analogsignal.metadata")
        nixdas = list()
        for idx, row in enumerate(data):
            daname = "{}.{}".format(nix_name, idx)
            da = nixblock.create_data_array(daname, "neo.analogsignal",
                                            data=row)
            da.metadata = metadata
            da.definition = anasig.description
            da.unit = units_to_string(anasig.units)

            timedim = da.append_sampled_dimension(
                anasig.sampling_period.magnitude.item()
            )
            timedim.unit = units_to_string(anasig.sampling_period.units)
            tstart = anasig.t_start
            metadata["t_start"] = tstart.magnitude.item()
            metadata.props["t_start"].unit = units_to_string(tstart.units)
            timedim.offset = tstart.rescale(timedim.unit).magnitude.item()
            timedim.label = "time"

            nixdas.append(da)
            if nixgroup:
                nixgroup.data_arrays.append(da)

        neoname = anasig.name if anasig.name is not None else ""
        metadata["neo_name"] = neoname
        if anasig.annotations:
            for k, v in anasig.annotations.items():
                self._write_property(metadata, k, v)

        self._signal_map[nix_name] = nixdas

    def _write_irregularlysampledsignal(self, irsig, nixblock, nixgroup):
        """
        Convert the provided ``irsig`` (IrregularlySampledSignal) to a list of
        NIX DataArray objects and write them to the NIX file at the location.
        All DataArray objects created from the same IrregularlySampledSignal
        have their metadata section point to the same object.

        :param irsig: The Neo IrregularlySampledSignal to be written
        :param nixblock: NIX Block where the DataArrays will be created
        :param nixgroup: NIX Group where the DataArrays will be attached
        """
        if "nix_name" in irsig.annotations:
            nix_name = irsig.annotations["nix_name"]
        else:
            nix_name = "neo.irregularlysampledsignal.{}".format(
                self._generate_nix_name()
            )
            irsig.annotate(nix_name=nix_name)

        if "{}.0".format(nix_name) in nixblock.data_arrays and nixgroup:
            # IrregularlySampledSignal is in multiple Segments.
            # Append DataArrays to Group and return.
            dalist = list()
            for idx in itertools.count():
                daname = "{}.{}".format(nix_name, idx)
                if daname in nixblock.data_arrays:
                    dalist.append(nixblock.data_arrays[daname])
                else:
                    break
            nixgroup.data_arrays.extend(dalist)
            return

        data = np.transpose(irsig[:].magnitude)
        parentmd = nixgroup.metadata if nixgroup else nixblock.metadata
        metadata = parentmd.create_section(
            nix_name, "neo.irregularlysampledsignal.metadata"
        )
        nixdas = list()
        for idx, row in enumerate(data):
            daname = "{}.{}".format(nix_name, idx)
            da = nixblock.create_data_array(
                daname, "neo.irregularlysampledsignal", data=row
            )
            da.metadata = metadata
            da.definition = irsig.description
            da.unit = units_to_string(irsig.units)

            timedim = da.append_range_dimension(irsig.times.magnitude)
            timedim.unit = units_to_string(irsig.times.units)
            timedim.label = "time"

            nixdas.append(da)
            if nixgroup:
                nixgroup.data_arrays.append(da)

        neoname = irsig.name if irsig.name is not None else ""
        metadata["neo_name"] = neoname
        if irsig.annotations:
            for k, v in irsig.annotations.items():
                self._write_property(metadata, k, v)

        self._signal_map[nix_name] = nixdas

    def _write_event(self, event, nixblock, nixgroup):
        """
        Convert the provided Neo Event to a NIX MultiTag and write it to the
        NIX file.

        :param event: The Neo Event to be written
        :param nixblock: NIX Block where the MultiTag will be created
        :param nixgroup: NIX Group where the MultiTag will be attached
        """
        if "nix_name" in event.annotations:
            nix_name = event.annotations["nix_name"]
        else:
            nix_name = "neo.event.{}".format(self._generate_nix_name())
            event.annotate(nix_name=nix_name)

        if nix_name in nixblock.multi_tags:
            # Event is in multiple Segments. Append to Group and return.
            mt = nixblock.multi_tags[nix_name]
            nixgroup.multi_tags.append(mt)
            return

        times = event.times.magnitude
        units = units_to_string(event.times.units)
        timesda = nixblock.create_data_array(
            "{}.times".format(nix_name), "neo.event.times", data=times
        )
        timesda.unit = units
        nixmt = nixblock.create_multi_tag(nix_name, "neo.event",
                                          positions=timesda)

        nixmt.metadata = nixgroup.metadata.create_section(
            nix_name, "neo.event.metadata"
        )
        metadata = nixmt.metadata

        labeldim = timesda.append_set_dimension()
        labeldim.labels = event.labels

        neoname = event.name if event.name is not None else ""
        metadata["neo_name"] = neoname
        nixmt.definition = event.description
        if event.annotations:
            for k, v in event.annotations.items():
                self._write_property(metadata, k, v)

        nixgroup.multi_tags.append(nixmt)

        # reference all AnalogSignals and IrregularlySampledSignals in Group
        for da in nixgroup.data_arrays:
            if da.type in ("neo.analogsignal", "neo.irregularlysampledsignal"):
                nixmt.references.append(da)

    def _write_epoch(self, epoch, nixblock, nixgroup):
        """
        Convert the provided Neo Epoch to a NIX MultiTag and write it to the
        NIX file.

        :param epoch: The Neo Epoch to be written
        :param nixblock: NIX Block where the MultiTag will be created
        :param nixgroup: NIX Group where the MultiTag will be attached
        """
        if "nix_name" in epoch.annotations:
            nix_name = epoch.annotations["nix_name"]
        else:
            nix_name = "neo.epoch.{}".format(self._generate_nix_name())
            epoch.annotate(nix_name=nix_name)

        if nix_name in nixblock.multi_tags:
            # Epoch is in multiple Segments. Append to Group and return.
            mt = nixblock.multi_tags[nix_name]
            nixgroup.multi_tags.append(mt)
            return

        times = epoch.times.magnitude
        tunits = units_to_string(epoch.times.units)
        durations = epoch.durations.magnitude
        dunits = units_to_string(epoch.durations.units)

        timesda = nixblock.create_data_array(
            "{}.times".format(nix_name), "neo.epoch.times", data=times
        )
        timesda.unit = tunits
        nixmt = nixblock.create_multi_tag(nix_name, "neo.epoch",
                                          positions=timesda)

        durada = nixblock.create_data_array(
            "{}.durations".format(nix_name), "neo.epoch.durations",
            data=durations
        )
        durada.unit = dunits
        nixmt.extents = durada

        nixmt.metadata = nixgroup.metadata.create_section(
            nix_name, "neo.epoch.metadata"
        )
        metadata = nixmt.metadata

        labeldim = timesda.append_set_dimension()
        labeldim.labels = epoch.labels

        neoname = epoch.name if epoch.name is not None else ""
        metadata["neo_name"] = neoname
        nixmt.definition = epoch.description
        if epoch.annotations:
            for k, v in epoch.annotations.items():
                self._write_property(metadata, k, v)

        nixgroup.multi_tags.append(nixmt)

        # reference all AnalogSignals and IrregularlySampledSignals in Group
        for da in nixgroup.data_arrays:
            if da.type in ("neo.analogsignal", "neo.irregularlysampledsignal"):
                nixmt.references.append(da)

    def _write_spiketrain(self, spiketrain, nixblock, nixgroup):
        """
        Convert the provided Neo SpikeTrain to a NIX MultiTag and write it to
        the NIX file.

        :param spiketrain: The Neo SpikeTrain to be written
        :param nixblock: NIX Block where the MultiTag will be created
        :param nixgroup: NIX Group where the MultiTag will be attached
        """
        if "nix_name" in spiketrain.annotations:
            nix_name = spiketrain.annotations["nix_name"]
        else:
            nix_name = "neo.spiketrain.{}".format(self._generate_nix_name())
            spiketrain.annotate(nix_name=nix_name)

        if nix_name in nixblock.multi_tags and nixgroup:
            # SpikeTrain is in multiple Segments. Append to Group and return.
            mt = nixblock.multi_tags[nix_name]
            nixgroup.multi_tags.append(mt)
            return

        times = spiketrain.times.magnitude
        tunits = units_to_string(spiketrain.times.units)

        timesda = nixblock.create_data_array(
            "{}.times".format(nix_name), "neo.spiketrain.times", data=times
        )
        timesda.unit = tunits
        nixmt = nixblock.create_multi_tag(nix_name, "neo.spiketrain",
                                          positions=timesda)

        parentmd = nixgroup.metadata if nixgroup else nixblock.metadata
        nixmt.metadata = parentmd.create_section(nix_name,
                                                 "neo.spiketrain.metadata")
        metadata = nixmt.metadata

        neoname = spiketrain.name if spiketrain.name is not None else ""
        metadata["neo_name"] = neoname
        nixmt.definition = spiketrain.description

        self._write_property(metadata, "t_start", spiketrain.t_start)
        self._write_property(metadata, "t_stop", spiketrain.t_stop)

        if spiketrain.annotations:
            for k, v in spiketrain.annotations.items():
                self._write_property(metadata, k, v)

        if nixgroup:
            nixgroup.multi_tags.append(nixmt)

        if spiketrain.waveforms is not None:
            wfdata = list(wf.magnitude for wf in
                          list(wfgroup for wfgroup in
                               spiketrain.waveforms))
            wfunits = units_to_string(spiketrain.waveforms.units)
            wfda = nixblock.create_data_array(
                "{}.waveforms".format(nix_name), "neo.waveforms",
                data=wfdata
            )
            wfda.unit = wfunits
            wfda.metadata = nixmt.metadata.create_section(
                wfda.name, "neo.waveforms.metadata"
            )
            nixmt.create_feature(wfda, nix.LinkType.Indexed)
            # TODO: Move time dimension first for PR #457
            # https://github.com/NeuralEnsemble/python-neo/pull/457
            wfda.append_set_dimension()
            wfda.append_set_dimension()
            wftime = wfda.append_sampled_dimension(
                spiketrain.sampling_period.magnitude.item()
            )
            wftime.unit = units_to_string(spiketrain.sampling_period.units)
            wftime.label = "time"

        if spiketrain.left_sweep is not None:
            self._write_property(wfda.metadata, "left_sweep",
                                 spiketrain.left_sweep)

    def _write_unit(self, neounit, nixchxsource):
        """
        Convert the provided Neo Unit to a NIX Source and write it to the
        NIX file.

        :param neounit: The Neo Unit to be written
        :param nixchxsource: NIX Source (ChannelIndex) where the new Source
        (Unit) will be created
        """
        if "nix_name" in neounit.annotations:
            nix_name = neounit.annotations["nix_name"]
        else:
            nix_name = "neo.unit.{}".format(self._generate_nix_name())
            neounit.annotate(nix_name=nix_name)
        nixunitsource = nixchxsource.create_source(nix_name,
                                                   "neo.unit")
        nixunitsource.metadata = nixchxsource.metadata.create_section(
            nix_name, "neo.unit.metadata"
        )
        metadata = nixunitsource.metadata
        neoname = neounit.name if neounit.name is not None else ""
        metadata["neo_name"] = neoname
        nixunitsource.definition = neounit.description
        if neounit.annotations:
            for k, v in neounit.annotations.items():
                self._write_property(metadata, k, v)

    def _create_source_links(self, neoblock, nixblock):
        """
        Creates references between objects in a NIX Block to store the
        references in the Neo ChannelIndex and Unit objects.
        Specifically:
        - If a Neo ChannelIndex references a Neo AnalogSignal or
        IrregularlySampledSignal, the corresponding signal DataArray will
        reference the corresponding NIX Source object which represents the
        Neo ChannelIndex.
        - If a Neo Unit references a Neo SpikeTrain, the corresponding
        MultiTag will reference the NIX Source objects which represent the
        Neo Unit and its parent ChannelIndex.

        The two arguments must represent the same Block in each corresponding
        format.

        Neo objects that have not been converted yet (i.e., AnalogSignal,
        IrregularlySampledSignal, or SpikeTrain objects that are not attached
        to a Segment) are created on the nixblock.

        :param neoblock: A Neo Block object
        :param nixblock: The corresponding NIX Block
        """

        for chx in neoblock.channel_indexes:
            signames = []
            for asig in chx.analogsignals:
                if "nix_name" not in asig.annotations:
                    self._write_analogsignal(asig, nixblock, None)
                signames.append(asig.annotations["nix_name"])
            for isig in chx.irregularlysampledsignals:
                if "nix_name" not in isig.annotations:
                    self._write_irregularlysampledsignal(isig, nixblock, None)
                signames.append(isig.annotations["nix_name"])
            chxsource = nixblock.sources[chx.annotations["nix_name"]]
            for name in signames:
                for da in self._signal_map[name]:
                    da.sources.append(chxsource)

            for unit in chx.units:
                unitsource = chxsource.sources[unit.annotations["nix_name"]]
                for st in unit.spiketrains:
                    if "nix_name" not in st.annotations:
                        self._write_spiketrain(st, nixblock, None)
                    stmt = nixblock.multi_tags[st.annotations["nix_name"]]
                    stmt.sources.append(chxsource)
                    stmt.sources.append(unitsource)

    @staticmethod
    def _generate_nix_name():
        return uuid4().hex

    def _write_property(self, section, name, v):
        """
        Create a metadata property with a given name and value on the provided
        metadata section.

        :param section: The metadata section to hold the new property
        :param name: The name of the property
        :param v: The value to write
        :return: The newly created property
        """

        if isinstance(v, pq.Quantity):
            if len(v.shape):
                section[name] = list(nix.Value(vv) for vv in v.magnitude)
            else:
                section[name] = nix.Value(v.magnitude.item())
            section.props[name].unit = str(v.dimensionality)
        elif isinstance(v, datetime):
            section[name] = nix.Value(calculate_timestamp(v))
        elif isinstance(v, string_types):
            section[name] = nix.Value(v)
        elif isinstance(v, bytes):
            section[name] = nix.Value(v.decode())
        elif isinstance(v, Iterable):
            values = []
            unit = None
            definition = None
            if len(v) == 0:
                # empty list can't be saved in NIX property
                # but we can store an empty string and use the
                # definition to signify that it should be restored
                # as an iterable (list)
                values = ""
                definition = EMPTYANNOTATION
            elif hasattr(v, "ndim") and v.ndim == 0:
                values = v.item()
                if isinstance(v, pq.Quantity):
                    unit = str(v.dimensionality)
            else:
                for item in v:
                    if isinstance(item, string_types):
                        item = nix.Value(item)
                    elif isinstance(item, pq.Quantity):
                        unit = str(item.dimensionality)
                        item = nix.Value(item.magnitude.item())
                    elif isinstance(item, Iterable):
                        self.logger.warn("Multidimensional arrays and nested "
                                         "containers are not currently "
                                         "supported when writing to NIX.")
                        return None
                    else:
                        item = nix.Value(item)
                    values.append(item)
            section[name] = values
            section.props[name].unit = unit
            if definition:
                section.props[name].definition = definition
        elif type(v).__module__ == "numpy":
            section[name] = nix.Value(v.item())
        else:
            section[name] = nix.Value(v)
        return section.props[name]

    @staticmethod
    def _nix_attr_to_neo(nix_obj):
        """
        Reads common attributes and metadata from a NIX object and populates a
        dictionary with Neo-compatible attributes and annotations.

        Common attributes: neo_name, nix_name, description,
                           file_datetime (if applicable).

        Metadata: For properties that specify a 'unit', a Quantity object is
                  created.
        """
        neo_attrs = dict()
        neo_attrs["nix_name"] = nix_obj.name
        neo_attrs["description"] = stringify(nix_obj.definition)
        if nix_obj.metadata:
            for prop in nix_obj.metadata.props:
                values = list(v.value for v in prop.values)
                if prop.unit:
                    units = prop.unit
                    values = create_quantity(values, units)
                if len(values) == 1:
                    values = values[0]
                if values == "" and prop.definition == EMPTYANNOTATION:
                    values = list()
                neo_attrs[prop.name] = values
        neo_attrs["name"] = stringify(neo_attrs.get("neo_name"))

        if "file_datetime" in neo_attrs:
            neo_attrs["file_datetime"] = datetime.fromtimestamp(
                neo_attrs["file_datetime"]
            )
        return neo_attrs

    @staticmethod
    def _group_signals(dataarrays):
        """
        Groups data arrays that were generated by the same Neo Signal object.
        The collection can contain both  AnalogSignals and
        IrregularlySampledSignals.

        :param dataarrays: A collection of DataArray objects to group
        :return: A dictionary mapping a base name to a list of DataArrays which
        belong to the same Signal
        """
        # first sort by name
        dataarrays = sorted(dataarrays, key=lambda x: x.name)

        # now start grouping
        groups = dict()
        for da in dataarrays:
            basename = ".".join(da.name.split(".")[:-1])
            if basename not in groups:
                groups[basename] = list()
            groups[basename].append(da)

        return groups

    @staticmethod
    def _get_time_dimension(obj):
        for dim in obj.dimensions:
            if hasattr(dim, "label") and dim.label == "time":
                return dim
        return None

    def close(self):
        """
        Closes the open nix file and resets maps.
        """
        if (hasattr(self, "nix_file") and
                self.nix_file and self.nix_file.is_open()):
            self.nix_file.close()
            self.nix_file = None
            self._neo_map = None
            self._ref_map = None
            self._signal_map = None
            self._block_read_counter = None

    def __del__(self):
        self.close()
