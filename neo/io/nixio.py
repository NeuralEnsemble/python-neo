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
"""

from __future__ import absolute_import

import time
from datetime import datetime
from collections import Iterable
import itertools
from hashlib import md5
from uuid import uuid4

import quantities as pq
import numpy as np

from .baseio import BaseIO
from ..core import (Block, Segment, ChannelIndex, AnalogSignal,
                    IrregularlySampledSignal, Epoch, Event, SpikeTrain, Unit)
from .tools import LazyList
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


def stringify(value):
    if value is None:
        return value
    if isinstance(value, bytes):
        value = value.decode()
    return str(value)


def create_quantity(values, unitstr):
    if "*" in unitstr:
        unit = pq.CompoundUnit(unitstr)
    else:
        unit = unitstr
    return pq.Quantity(values, unit)


def units_to_string(pqunit):
    dim = str(pqunit.dimensionality)
    scaling = pqunit.magnitude.item()
    if scaling == 1:
        return dim.strip("()") if dim.startswith("(") else dim
    return "{} * {}".format(scaling, dim)


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

    _container_map = {
        "segments": "groups",
        "analogsignals": "data_arrays",
        "irregularlysampledsignals": "data_arrays",
        "events": "multi_tags",
        "epochs": "multi_tags",
        "spiketrains": "multi_tags",
        "channel_indexes": "sources",
        "units": "sources"
    }

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
        if self.nix_file.mode == nix.FileMode.Overwrite:
            filemd = self.nix_file.create_section("neo", "neo.metadata")
            filemd["neo.version"] = neover
        # TODO: Version check for existing files
        self._block_read_counter = 0
        self._neo_map = dict()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_all_blocks(self):
        blocks = list()
        for blk in self.nix_file.blocks:
            blocks.append(self.read_block(blk))
        return blocks

    def read_block(self, nixblock):
        neoblock = self._block_to_neo(nixblock)
        for group in nixblock.groups:
            neoblock.segments.append(
                self.read_segment(group)
            )
        return neoblock

    def read_segment(self, nixgroup):
        neo_segment = self._group_to_neo(nixgroup)
        # nix_parent = self._get_parent(path)
        # neo_parent = self._neo_map.get(nix_parent.name)
        # if neo_parent:
        #     neo_segment.block = neo_parent
        return neo_segment

    def read_channelindex(self, path, lazy=False):
        assert not lazy, 'Do not support lazy'
        nix_source = self._get_object_at(path)
        neo_rcg = self._source_chx_to_neo(nix_source)
        neo_rcg.path = path
        self._read_cascade(nix_source, path, True, lazy)
        self._update_maps(neo_rcg, lazy)
        nix_parent = self._get_parent(path)
        neo_parent = self._neo_map.get(nix_parent.name)
        neo_rcg.block = neo_parent
        return neo_rcg

    def read_signal(self, path, lazy=False):
        nix_data_arrays = list()
        parent_group = self._get_parent(path)
        parent_container = parent_group.data_arrays
        signal_group_name = path.split("/")[-1]
        for idx in itertools.count():
            signal_name = "{}.{}".format(signal_group_name, idx)
            if signal_name in parent_container:
                nix_data_arrays.append(parent_container[signal_name])
            else:
                break
        # check metadata segment
        group_section = nix_data_arrays[0].metadata
        for da in nix_data_arrays:
            assert da.metadata == group_section,\
                "DataArray {} is not a member of signal group {}".format(
                    da.name, group_section.name
                )
        neo_signal = self._signal_da_to_neo(nix_data_arrays, lazy)
        neo_signal.path = path
        if self._find_lazy_loaded(neo_signal) is None:
            self._update_maps(neo_signal, lazy)
            nix_parent = self._get_parent(path)
            neo_parent = self._neo_map.get(nix_parent.name)
            neo_signal.segment = neo_parent
        return neo_signal

    def read_analogsignal(self, path, lazy=False):
        assert not lazy, 'Do not support lazy'
        return self.read_signal(path, lazy)

    def read_irregularlysampledsignal(self, path, lazy=False):
        assert not lazy, 'Do not support lazy'
        return self.read_signal(path, lazy)

    def read_eest(self, path, lazy=False):
        nix_mtag = self._get_object_at(path)
        neo_eest = self._mtag_eest_to_neo(nix_mtag, lazy)
        neo_eest.path = path
        self._update_maps(neo_eest, lazy)
        nix_parent = self._get_parent(path)
        neo_parent = self._neo_map.get(nix_parent.name)
        neo_eest.segment = neo_parent
        return neo_eest

    def read_epoch(self, path, lazy=False):
        assert not lazy, 'Do not support lazy'
        return self.read_eest(path, lazy)

    def read_event(self, path, lazy=False):
        assert not lazy, 'Do not support lazy'
        return self.read_eest(path, lazy)

    def read_spiketrain(self, path, lazy=False):
        assert not lazy, 'Do not support lazy'
        return self.read_eest(path, lazy)

    def read_unit(self, path, lazy=False):
        assert not lazy, 'Do not support lazy'
        nix_source = self._get_object_at(path)
        neo_unit = self._source_unit_to_neo(nix_source)
        neo_unit.path = path
        self._read_cascade(nix_source, path, True, lazy)
        self._update_maps(neo_unit, lazy)
        nix_parent = self._get_parent(path)
        neo_parent = self._neo_map.get(nix_parent.name)
        neo_unit.channel_index = neo_parent
        return neo_unit

    def _block_to_neo(self, nix_block):
        neo_attrs = self._nix_attr_to_neo(nix_block)
        neo_block = Block(**neo_attrs)
        neo_block.rec_datetime = datetime.fromtimestamp(
            nix_block.created_at
        )
        self._neo_map[nix_block.name] = neo_block
        return neo_block

    def _group_to_neo(self, nix_group):
        neo_attrs = self._nix_attr_to_neo(nix_group)
        neo_segment = Segment(**neo_attrs)
        neo_segment.rec_datetime = datetime.fromtimestamp(
            nix_group.created_at
        )
        self._neo_map[nix_group.name] = neo_segment
        return neo_segment

    def _source_chx_to_neo(self, nix_source):
        neo_attrs = self._nix_attr_to_neo(nix_source)
        chx = list(self._nix_attr_to_neo(c)
                   for c in nix_source.sources
                   if c.type == "neo.channelindex")
        chan_names = list(c["neo_name"] for c in chx if "neo_name" in c)
        chan_ids = list(c["channel_id"] for c in chx if "channel_id" in c)
        if chan_names:
            neo_attrs["channel_names"] = chan_names
        if chan_ids:
            neo_attrs["channel_ids"] = chan_ids
        neo_attrs["index"] = np.array([c["index"] for c in chx])
        if "coordinates" in chx[0]:
            coord_units = chx[0]["coordinates.units"]
            coord_values = list(c["coordinates"] for c in chx)
            neo_attrs["coordinates"] = create_quantity(coord_values,
                                                       coord_units)
        rcg = ChannelIndex(**neo_attrs)
        self._neo_map[nix_source.name] = rcg
        return rcg

    def _source_unit_to_neo(self, nix_unit):
        neo_attrs = self._nix_attr_to_neo(nix_unit)
        neo_unit = Unit(**neo_attrs)
        self._neo_map[nix_unit.name] = neo_unit
        return neo_unit

    def _signal_da_to_neo(self, nix_da_group, lazy):
        """
        Convert a group of NIX DataArrays to a Neo signal. This method expects
        a list of data arrays that all represent the same, multidimensional
        Neo Signal object.
        This returns either an AnalogSignal or IrregularlySampledSignal.

        :param nix_da_group: a list of NIX DataArray objects
        :return: a Neo Signal object
        """
        nix_da_group = sorted(nix_da_group, key=lambda d: d.name)
        neo_attrs = self._nix_attr_to_neo(nix_da_group[0])
        metadata = nix_da_group[0].metadata
        neo_type = nix_da_group[0].type
        neo_attrs["nix_name"] = metadata.name  # use the common base name

        unit = nix_da_group[0].unit
        if lazy:
            signaldata = create_quantity(np.empty(0), unit)
            lazy_shape = (len(nix_da_group[0]), len(nix_da_group))
        else:
            signaldata = np.array([d[:] for d in nix_da_group]).transpose()
            signaldata = create_quantity(signaldata, unit)
            lazy_shape = None
        timedim = self._get_time_dimension(nix_da_group[0])
        if (neo_type == "neo.analogsignal" or
                timedim.dimension_type == nix.DimensionType.Sample):
            if lazy:
                sampling_period = create_quantity(1, timedim.unit)
                t_start = create_quantity(0, timedim.unit)
            else:
                sampling_period = create_quantity(timedim.sampling_interval,
                                                  timedim.unit)
                t_start = neo_attrs["t_start"]
            del neo_attrs["t_start"]

            neo_signal = AnalogSignal(
                signal=signaldata, sampling_period=sampling_period,
                t_start=t_start, **neo_attrs
            )
        elif (neo_type == "neo.irregularlysampledsignal"
              or timedim.dimension_type == nix.DimensionType.Range):
            if lazy:
                times = create_quantity(np.empty(0), timedim.unit)
            else:
                times = create_quantity(timedim.ticks, timedim.unit)
            neo_signal = IrregularlySampledSignal(
                signal=signaldata, times=times, **neo_attrs
            )
        else:
            return None
        for da in nix_da_group:
            self._neo_map[da.name] = neo_signal
        if lazy_shape:
            neo_signal.lazy_shape = lazy_shape
        return neo_signal

    def _mtag_eest_to_neo(self, nix_mtag, lazy):
        neo_attrs = self._nix_attr_to_neo(nix_mtag)
        neo_type = nix_mtag.type

        time_unit = nix_mtag.positions.unit
        if lazy:
            times = create_quantity(np.empty(0), time_unit)
            lazy_shape = np.shape(nix_mtag.positions)
        else:
            times = create_quantity(nix_mtag.positions, time_unit)
            lazy_shape = None
        if neo_type == "neo.epoch":
            if lazy:
                durations = create_quantity(np.empty(0), nix_mtag.extents.unit)
                labels = np.empty(0, dtype='S')
            else:
                durations = create_quantity(nix_mtag.extents,
                                            nix_mtag.extents.unit)
                labels = np.array(nix_mtag.positions.dimensions[0].labels,
                                  dtype="S")
            eest = Epoch(times=times, durations=durations, labels=labels,
                         **neo_attrs)
        elif neo_type == "neo.event":
            if lazy:
                labels = np.empty(0, dtype='S')
            else:
                labels = np.array(nix_mtag.positions.dimensions[0].labels,
                                  dtype="S")
            eest = Event(times=times, labels=labels, **neo_attrs)
        elif neo_type == "neo.spiketrain":
            eest = SpikeTrain(times=times, **neo_attrs)
            if nix_mtag.features:
                wfda = nix_mtag.features[0].data
                wftime = self._get_time_dimension(wfda)
                if lazy:
                    eest.waveforms = create_quantity(np.empty((0, 0, 0)),
                                                     wfda.unit)
                    eest.sampling_period = create_quantity(1, wftime.unit)
                    eest.left_sweep = create_quantity(0, wftime.unit)
                else:
                    eest.waveforms = create_quantity(wfda, wfda.unit)
                    interval_units = wftime.unit
                    eest.sampling_period = create_quantity(
                        wftime.sampling_interval, interval_units
                    )
                    left_sweep_units = wftime.unit
                    if "left_sweep" in wfda.metadata:
                        eest.left_sweep = create_quantity(
                            wfda.metadata["left_sweep"], left_sweep_units
                        )
        else:
            return None
        self._neo_map[nix_mtag.name] = eest
        if lazy_shape:
            eest.lazy_shape = lazy_shape
        return eest

    def _read_cascade(self, nix_obj, path, cascade, lazy):
        neo_obj = self._neo_map[nix_obj.name]
        for neocontainer in getattr(neo_obj, "_child_containers", []):
            nixcontainer = self._container_map[neocontainer]
            if not hasattr(nix_obj, nixcontainer):
                continue
            if neocontainer == "channel_indexes":
                neotype = "channelindex"
            else:
                neotype = neocontainer[:-1]
            chpaths = list(path + "/" + neocontainer + "/" + c.name
                           for c in getattr(nix_obj, nixcontainer)
                           if c.type == "neo." + neotype)
            if neocontainer in ("analogsignals",
                                "irregularlysampledsignals"):
                chpaths = self._group_signals(chpaths)
            if cascade != "lazy":
                read_func = getattr(self, "read_" + neotype)
                children = list(read_func(cp, lazy)
                                for cp in chpaths)
            else:
                children = LazyList(self, lazy, chpaths)
            setattr(neo_obj, neocontainer, children)

        if isinstance(neo_obj, ChannelIndex):
            # set references to signals
            parent_block_path = "/" + path.split("/")[1]
            parent_block = self._get_object_at(parent_block_path)
            ref_das = self._get_referers(nix_obj, parent_block.data_arrays)
            ref_signals = list(self._neo_map[da.name] for da in ref_das)
            # deduplicate by name
            ref_signals = list(dict((s.annotations["nix_name"], s)
                                    for s in ref_signals).values())
            for sig in ref_signals:
                if isinstance(sig, AnalogSignal):
                    neo_obj.analogsignals.append(sig)
                elif isinstance(sig, IrregularlySampledSignal):
                    neo_obj.irregularlysampledsignals.append(sig)
                sig.channel_index = neo_obj

        elif isinstance(neo_obj, Unit):
            # set references to spiketrains
            parent_block_path = "/" + path.split("/")[1]
            parent_block = self._get_object_at(parent_block_path)
            ref_mtags = self._get_referers(nix_obj, parent_block.multi_tags)
            ref_sts = list(self._neo_map[mt.name] for mt in ref_mtags)
            for st in ref_sts:
                neo_obj.spiketrains.append(st)
                st.unit = neo_obj

    def get(self, path, lazy):
        parts = path.split("/")
        if len(parts) > 2:
            neotype = parts[-2][:-1]
        else:
            neotype = "block"
        if neotype == "channel_indexe":
            neotype = "channelindex"
        read_func = getattr(self, "read_" + neotype)
        return read_func(path, lazy)

    def load_lazy_cascade(self, path, lazy):
        """
        Loads the object at the location specified by the path and all
        children. Data is loaded if lazy is False.

        :param path: Location of object in file
        :param lazy: Do not load data if True
        :return: The loaded object
        """
        neoobj = self.get(path, lazy=lazy)
        return neoobj

    def write_all_blocks(self, neo_blocks):
        """
        Convert all ``neo_blocks`` to the NIX equivalent and write them to the
        file.

        :param neo_blocks: List (or iterable) containing Neo blocks
        :return: A list containing the new NIX Blocks
        """
        for bl in neo_blocks:
            self.write_block(bl)

    def _write_object(self, obj, loc=""):
        objtype = type(obj).__name__.lower()
        if objtype == "channelindex":
            containerstr = "/channel_indexes/"
        else:
            containerstr = "/" + type(obj).__name__.lower() + "s/"
        if "nix_name" in obj.annotations:
            nix_name = obj.annotations["nix_name"]
        else:
            nix_name = "neo.{}.{}".format(objtype, self._generate_nix_name())
            obj.annotate(nix_name=nix_name)
        objpath = loc + containerstr + nix_name
        oldhash = self._object_hashes.get(nix_name)
        if oldhash is None:
            try:
                oldobj = self.get(objpath, lazy=False)
                oldhash = self._hash_object(oldobj)
            except (KeyError, IndexError):
                oldhash = None
        newhash = self._hash_object(obj)
        if oldhash != newhash:
            attr = self._neo_attr_to_nix(obj)
            attr["name"] = nix_name
            if isinstance(obj, pq.Quantity):
                attr.update(self._neo_data_to_nix(obj))
            if oldhash is None:
                nixobj = self._create_nix_obj(loc, attr)
            else:
                self._update_data(loc, attr)
                if objpath in self._path_map:
                    del self._path_map[objpath]
                nixobj = self._get_object_at(objpath)
            self._write_attr_annotations(nixobj, attr, objpath)
            if isinstance(obj, pq.Quantity):
                self._write_data(nixobj, attr, objpath)
        else:
            nixobj = self._nix_map.get(nix_name)
            if nixobj is None:
                nixobj = self._get_object_at(objpath)
            else:
                # object is already in file but may not be linked at objpath
                objat = self._get_object_at(objpath)
                if not objat:
                    self._link_nix_obj(nixobj, loc, containerstr)
        self._nix_map[nix_name] = nixobj
        self._object_hashes[nix_name] = newhash
        self._write_cascade(obj, objpath)

    def _create_nix_obj(self, loc, attr):
        parentobj = self._get_object_at(loc)
        if attr["type"] == "block":
            nixobj = parentobj.create_block(attr["name"], "neo.block")
            nixobj.metadata = self.nix_file.create_section(
                attr["name"], "neo.block.metadata"
            )
        elif attr["type"] == "segment":
            nixobj = parentobj.create_group(attr["name"], "neo.segment")
            nixobj.metadata = parentobj.metadata.create_section(
                attr["name"], "neo.segment.metadata"
            )
        elif attr["type"] == "channelindex":
            nixobj = parentobj.create_source(attr["name"],
                                             "neo.channelindex")
            nixobj.metadata = parentobj.metadata.create_section(
                attr["name"], "neo.channelindex.metadata"
            )
        elif attr["type"] in ("analogsignal", "irregularlysampledsignal"):
            blockpath = "/" + loc.split("/")[1]
            parentblock = self._get_object_at(blockpath)
            nixobj = list()
            typestr = "neo.{}".format(attr["type"])
            sigmd = parentobj.metadata.create_section(
                attr["name"], "{}.metadata".format(typestr)
            )
            for idx, datarow in enumerate(attr["data"]):
                name = "{}.{}".format(attr["name"], idx)
                da = parentblock.create_data_array(name, typestr, data=datarow)
                da.unit = attr["data.units"]
                da.metadata = sigmd
                nixobj.append(da)
            parentobj.data_arrays.extend(nixobj)
        elif attr["type"] in ("epoch", "event", "spiketrain"):
            blockpath = "/" + loc.split("/")[1]
            parentblock = self._get_object_at(blockpath)
            typestr = "neo.{}".format(attr["type"])
            timesda = parentblock.create_data_array(
                "{}.times".format(attr["name"]), "{}.times".format(typestr),
                data=attr["data"]
            )
            nixobj = parentblock.create_multi_tag(
                attr["name"], typestr, timesda
            )
            nixobj.metadata = parentobj.metadata.create_section(
                attr["name"], "{}.metadata".format(typestr)
            )
            parentobj.multi_tags.append(nixobj)
        elif attr["type"] == "unit":
            nixobj = parentobj.create_source(attr["name"], "neo.unit")
            nixobj.metadata = parentobj.metadata.create_section(
                attr["name"], "neo.unit.metadata"
            )
        else:
            raise ValueError("Unable to create NIX object. Invalid type.")
        return nixobj

    def _update_data(self, loc, attr):
        parentobj = self._get_object_at(loc)
        if attr["type"] in ("analogsignal", "irregularlysampledsignal"):
            blockpath = "/" + loc.split("/")[1]
            parentblock = self._get_object_at(blockpath)
            nixobj = list()
            typestr = "neo.{}".format(attr["type"])
            sigmd = parentobj.metadata[attr["name"]]
            objname = attr["name"]
            if objname in self._nix_map:
                del self._nix_map[objname]
            if objname in self._object_hashes:
                del self._object_hashes[objname]
            for idx, datarow in enumerate(attr["data"]):
                name = "{}.{}".format(objname, idx)
                del parentobj.data_arrays[name]
                del parentblock.data_arrays[name]
                da = parentblock.create_data_array(name, typestr, data=datarow)
                da.metadata = sigmd
                nixobj.append(da)
            parentobj.data_arrays.extend(nixobj)
        elif attr["type"] in ("epoch", "event", "spiketrain"):
            blockpath = "/" + loc.split("/")[1]
            parentblock = self._get_object_at(blockpath)
            typestr = "neo.{}".format(attr["type"])
            tname = "{}.times".format(attr["name"])
            del parentblock.data_arrays[tname]
            timesda = parentblock.create_data_array(
                tname, "{}.times".format(typestr), data=attr["data"]
            )
            nixobj = parentblock.multi_tags[attr["name"]]
            nixobj.positions = timesda

    def _link_nix_obj(self, obj, loc, neocontainer):
        parentobj = self._get_object_at(loc)
        container = getattr(parentobj,
                            self._container_map[neocontainer.strip("/")])
        if isinstance(obj, list):
            container.extend(obj)
        else:
            container.append(obj)

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
        else:
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

        # descend into ChannelIndexes
        for chx in block.channel_indexes:
            self.write_channelindex(chx, nixblock)

        # self._link_nix_obj(nixblock, loc, containerstr)
        # self._create_references(neoblock)

    def write_channelindex(self, chx, loc=""):
        """
        Convert the provided ``chx`` (ChannelIndex) to a NIX Source and write
        it to the NIX file at the location defined by ``loc``.

        :param chx: The Neo ChannelIndex to be written
        :param loc: Path to the parent of the new CHX
        """
        self._write_object(chx, loc)

    def write_segment(self, segment, nixblock):
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
        if nix_name in nixblock.groups:
            nixgroup = nixblock.groups[nix_name]
        else:
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
            self.write_analogsignal(asig, nixblock, nixgroup)
        for isig in segment.irregularlysampledsignals:
            self.write_irregularlysampledsignal(isig, nixblock, nixgroup)
        for event in segment.events:
            self.write_event(event, nixblock, nixgroup)
        for epoch in segment.epochs:
            self.write_epoch(epoch, nixblock, nixgroup)
        for spiketrain in segment.spiketrains:
            self.write_spiketrain(spiketrain, nixblock, nixgroup)

    def write_indices(self, chx, loc=""):
        """
        Create NIX Source objects to represent individual indices based on the
        provided ``chx`` (ChannelIndex) write them to the NIX file at
        the parent ChannelIndex object.

        :param chx: The Neo ChannelIndex
        :param loc: Path to the CHX
        """
        nixsource = self._nix_map[chx.annotations["nix_name"]]
        for idx, channel in enumerate(chx.index):
            channame = "{}.ChannelIndex{}".format(chx.annotations["nix_name"],
                                                  idx)
            if channame in nixsource.sources:
                nixchan = nixsource.sources[channame]
            else:
                nixchan = nixsource.create_source(channame,
                                                  "neo.channelindex")
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
                nixcoords = tuple(
                    nix.Value(c.rescale(coordunits).magnitude.item())
                    for c in coords
                )
                if "coordinates" in chanmd:
                    del chanmd["coordinates"]
                chanprop = chanmd.create_property("coordinates", nixcoords)
                chanprop.unit = coordunits

    def write_analogsignal(self, anasig, nixblock, nixgroup):
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

        def get_all_sigs():
            ret = list()
            for idx in itertools.count():
                name = "{}.{}".format(nix_name, idx)
                if name in nixblock.data_arrays:
                    ret.append(nixblock.data_arrays[name])
                else:
                    return ret

        firstdaname = "{}.0".format(nix_name)
        if firstdaname in nixblock.data_arrays:
            nixdas = get_all_sigs()
            metadata = nixdas[0].metadata
        else:
            data = np.transpose(anasig[:].magnitude)
            metadata = nixgroup.metadata.create_section(
                nix_name, "neo.analogsignal.metadata"
            )
            nixdas = list()
            for idx, row in enumerate(data):
                daname = "{}.{}".format(nix_name, idx)
                da = nixblock.create_data_array(daname,
                                                "neo.analogsignal",
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
                nixgroup.data_arrays.append(da)

        neoname = anasig.name if anasig.name is not None else ""
        metadata["neo_name"] = neoname
        if anasig.annotations:
            for k, v in anasig.annotations.items():
                self._write_property(metadata, k, v)

        # TODO: Dimensions

    def write_irregularlysampledsignal(self, irsig, nixblock, nixgroup):
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

        def get_all_sigs():
            ret = list()
            for idx in itertools.count():
                name = "{}.{}".format(nix_name, idx)
                if name in nixblock.data_arrays:
                    ret.append(nixblock.data_arrays[name])
                else:
                    return ret

        firstdaname = "{}.0".format(nix_name)
        if firstdaname in nixblock.data_arrays:
            nixdas = get_all_sigs()
            metadata = nixdas[0].metadata
        else:
            data = np.transpose(irsig[:].magnitude)
            metadata = nixgroup.metadata.create_section(
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
                tstart = irsig.t_start
                metadata["t_start"] = tstart.magnitude.item()
                metadata.props["t_start"].unit = units_to_string(tstart.units)
                timedim.offset = tstart.rescale(timedim.unit).magnitude.item()
                timedim.label = "time"

                nixdas.append(da)
                nixgroup.data_arrays.append(da)

        neoname = irsig.name if irsig.name is not None else ""
        metadata["neo_name"] = neoname
        if irsig.annotations:
            for k, v in irsig.annotations.items():
                self._write_property(metadata, k, v)

        # TODO: Dimensions

    def write_epoch(self, ep, loc=""):
        """
        Convert the provided ``ep`` (Epoch) to a NIX MultiTag and write it to
        the NIX file at the location defined by ``loc``.

        :param ep: The Neo Epoch to be written
        :param loc: Path to the parent of the new MultiTag
        """
        self._write_object(ep, loc)

    def write_event(self, ev, loc=""):
        """
        Convert the provided ``ev`` (Event) to a NIX MultiTag and write it to
        the NIX file at the location defined by ``loc``.

        :param ev: The Neo Event to be written
        :param loc: Path to the parent of the new MultiTag
        """
        self._write_object(ev, loc)

    def write_spiketrain(self, sptr, loc=""):
        """
        Convert the provided ``sptr`` (SpikeTrain) to a NIX MultiTag and write
        it to the NIX file at the location defined by ``loc``.

        :param sptr: The Neo SpikeTrain to be written
        :param loc: Path to the parent of the new MultiTag
        """
        self._write_object(sptr, loc)

    def write_unit(self, ut, loc=""):
        """
        Convert the provided ``ut`` (Unit) to a NIX Source and write it to the
        NIX file at the parent RCG.

        :param ut: The Neo Unit to be written
        :param loc: Path to the parent of the new Source
        """
        self._write_object(ut, loc)

    def _write_cascade(self, neoobj, path=""):
        if isinstance(neoobj, ChannelIndex):
            containers = ["units"]
            self.write_indices(neoobj, path)
        elif isinstance(neoobj, Unit):
            containers = []
        else:
            containers = getattr(neoobj, "_child_containers", [])
        for neocontainer in containers:
            if neocontainer == "channel_indexes":
                neotype = "channelindex"
            else:
                neotype = neocontainer[:-1]
            children = getattr(neoobj, neocontainer)
            write_func = getattr(self, "write_" + neotype)
            for ch in children:
                write_func(ch, path)

    def _create_references(self, block):
        """
        Create references between NIX objects according to the supplied Neo
        Block.
        MultiTags reference DataArrays of the same Group.
        DataArrays reference ChannelIndexs as sources, based on Neo
         RCG -> Signal relationships.
        MultiTags (SpikeTrains) reference ChannelIndexs and Units as
         sources, based on Neo RCG -> Unit -> SpikeTrain relationships.

        :param block: A Neo Block that has already been converted and mapped to
         NIX objects.
        """
        for seg in block.segments:
            group = self._nix_map[seg.annotations["nix_name"]]
            group_signals = self._get_contained_signals(group)
            for mtag in group.multi_tags:
                if mtag.type in ("neo.epoch", "neo.event"):
                    mtag.references.extend([sig for sig in group_signals
                                            if sig not in mtag.references])
        for rcg in block.channel_indexes:
            chidxsrc = self._nix_map[rcg.annotations["nix_name"]]
            das = list(self._nix_map[sig.annotations["nix_name"]]
                       for sig in rcg.analogsignals +
                       rcg.irregularlysampledsignals)
            # flatten nested lists
            das = [da for dalist in das for da in dalist]
            for da in das:
                if chidxsrc not in da.sources:
                    da.sources.append(chidxsrc)
            for unit in rcg.units:
                unitsource = self._nix_map[unit.annotations["nix_name"]]
                for st in unit.spiketrains:
                    stmtag = self._nix_map[st.annotations["nix_name"]]
                    if chidxsrc not in stmtag.sources:
                        stmtag.sources.append(chidxsrc)
                    if unitsource not in stmtag.sources:
                        stmtag.sources.append(unitsource)

    def _get_object_at(self, path):
        """
        Returns the object at the location defined by the path.
        ``path`` is a '/' delimited string. Each part of the string alternates
        between an object name and a container.

        If the requested object is an AnalogSignal or IrregularlySampledSignal,
        identified by the second-to-last part of the path string, a list of
        (DataArray) objects is returned.

        Example path: /block_1/segments/segment_a/events/event_a1

        :param path: Path string
        :return: The object at the location defined by the path
        """
        if path in self._path_map:
            return self._path_map[path]
        if path in ("", "/"):
            return self.nix_file
        parts = path.split("/")
        if parts[0]:
            ValueError("Invalid object path: {}".format(path))
        if len(parts) == 2:  # root block
            return self.nix_file.blocks[parts[1]]
        parent_obj = self._get_parent(path)
        container_name = self._container_map[parts[-2]]
        parent_container = getattr(parent_obj, container_name)
        objname = parts[-1]
        if parts[-2] in ["analogsignals", "irregularlysampledsignals"]:
            obj = list()
            for idx in itertools.count():
                name = "{}.{}".format(objname, idx)
                if name in parent_container:
                    obj.append(parent_container[name])
                else:
                    break
        else:
            obj = parent_container[objname]
        self._path_map[path] = obj
        return obj

    def _get_parent(self, path):
        parts = path.split("/")
        parent_path = "/".join(parts[:-2])
        parent_obj = self._get_object_at(parent_path)
        return parent_obj

    def _write_attr_annotations(self, nixobj, attr, path):
        if isinstance(nixobj, list):
            metadata = nixobj[0].metadata
            for obj in nixobj:
                obj.definition = attr["definition"]
            self._write_attr_annotations(nixobj[0], attr, path)
            return
        else:
            metadata = nixobj.metadata
            nixobj.definition = attr["definition"]
        if "neo_name" in attr:
            metadata["neo_name"] = attr["neo_name"]
        if "created_at" in attr:
            nixobj.force_created_at(calculate_timestamp(attr["created_at"]))
        if "file_datetime" in attr:
            self._write_property(metadata,
                                 "file_datetime", attr["file_datetime"])
        if attr.get("rec_datetime"):
            self._write_property(metadata,
                                 "rec_datetime", attr["rec_datetime"])

        if "annotations" in attr:
            for k, v in attr["annotations"].items():
                self._write_property(metadata, k, v)

    def _write_data(self, nixobj, attr, path):
        if isinstance(nixobj, list):
            metadata = nixobj[0].metadata
            for obj in nixobj:
                obj.unit = attr["data.units"]
                if attr["type"] == "analogsignal":
                    timedim = obj.append_sampled_dimension(
                        attr["sampling_period"]
                    )
                    timedim.unit = attr["sampling_period.units"]
                    timedim.offset = attr["t_start"]
                elif attr["type"] == "irregularlysampledsignal":
                    timedim = obj.append_range_dimension(attr["times"])
                    timedim.unit = attr["times.units"]
                timedim.label = "time"
                metadata["t_start"] = attr["t_start"]
                metadata.props["t_start"].unit = attr["t_start.units"]
        else:
            metadata = nixobj.metadata
            nixobj.positions.unit = attr["data.units"]
            blockpath = "/" + path.split("/")[1]
            parentblock = self._get_object_at(blockpath)
            if "durations" in attr:
                extname = nixobj.name + ".durations"
                exttype = nixobj.type + ".durations"
                if extname in parentblock.data_arrays:
                    del parentblock.data_arrays[extname]
                extents = parentblock.create_data_array(extname, exttype,
                                                        data=attr["durations"])
                extents.unit = attr["durations.units"]
                nixobj.extents = extents
            if "labels" in attr:
                labeldim = nixobj.positions.append_set_dimension()
                labeldim.labels = attr["labels"]
            if "t_start" in attr:
                metadata["t_start"] = nix.Value(attr["t_start"])
                metadata.props["t_start"].unit = attr["t_start.units"]
            if "t_stop" in attr:
                metadata["t_stop"] = nix.Value(attr["t_stop"])
                metadata.props["t_stop"].unit = attr["t_stop.units"]
            if "waveforms" in attr:
                wfname = nixobj.name + ".waveforms"
                if wfname in parentblock.data_arrays:
                    del metadata.sections[wfname]
                    del parentblock.data_arrays[wfname]
                    del nixobj.features[0]
                wfda = parentblock.create_data_array(
                    wfname, "neo.waveforms",
                    data=attr["waveforms"]
                )
                wfda.metadata = nixobj.metadata.create_section(
                    wfda.name, "neo.waveforms.metadata"
                )
                wfda.unit = attr["waveforms.units"]
                nixobj.create_feature(wfda, nix.LinkType.Indexed)
                wfda.append_set_dimension()
                wfda.append_set_dimension()
                wftime = wfda.append_sampled_dimension(
                    attr["sampling_period"]
                )
                wftime.unit = attr["sampling_period.units"]
                wftime.label = "time"
                if "left_sweep" in attr:
                    wfdamd = self._write_property(wfda.metadata, "left_sweep",
                                                  attr["left_sweep"])
                    wfdamd.unit = attr["left_sweep.units"]

    def _update_maps(self, obj, lazy):
        objidx = self._find_lazy_loaded(obj)
        if lazy and objidx is None:
            self._lazy_loaded.append(obj)
        elif not lazy and objidx is not None:
            self._lazy_loaded.pop(objidx)
        if not lazy:
            nix_name = obj.annotations["nix_name"]
            self._object_hashes[nix_name] = self._hash_object(obj)

    def _find_lazy_loaded(self, obj):
        """
        Finds the index of an object in the _lazy_loaded list by comparing the
        path attribute. Returns None if the object is not in the list.

        :param obj: The object to find
        :return: The index of the object in the _lazy_loaded list or None if it
        was not added
        """
        for idx, llobj in enumerate(self._lazy_loaded):
            if llobj.path == obj.path:
                return idx
        else:
            return None

    @staticmethod
    def _generate_nix_name():
        return uuid4().hex

    @staticmethod
    def _neo_attr_to_nix(neoobj):
        neotype = type(neoobj).__name__
        attrs = dict()
        # NIX metadata does not support None values
        # The property will be excluded to signify 'name is None'
        if neoobj.name is not None:
            attrs["neo_name"] = neoobj.name
        attrs["type"] = neotype.lower()
        attrs["definition"] = neoobj.description
        if isinstance(neoobj, (Block, Segment)):
            attrs["rec_datetime"] = neoobj.rec_datetime
            if neoobj.rec_datetime:
                attrs["created_at"] = neoobj.rec_datetime
            if neoobj.file_datetime:
                attrs["file_datetime"] = neoobj.file_datetime
        if neoobj.annotations:
            attrs["annotations"] = neoobj.annotations
        return attrs

    @classmethod
    def _neo_data_to_nix(cls, neoobj):
        attr = dict()
        attr["data"] = np.transpose(neoobj.magnitude)
        attr["data.units"] = units_to_string(neoobj.units)
        if isinstance(neoobj, IrregularlySampledSignal):
            # times for other objects (Events, Epochs, SpikeTrains) are the
            # data property, so we only need to explicitly take the times for
            # IrregularlySampledSignals
            attr["times.units"] = units_to_string(neoobj.times.units)
            attr["times"] = neoobj.times.magnitude
            tdimunits = attr["times.units"]
        if hasattr(neoobj, "sampling_period"):
            sp = neoobj.sampling_period
            attr["sampling_period"] = sp.magnitude.item()
            attr["sampling_period.units"] = units_to_string(sp.units)
            tdimunits = attr["sampling_period.units"]
        if hasattr(neoobj, "t_start"):
            tstart = neoobj.t_start
            attr["t_start"] = tstart.rescale(tdimunits).magnitude.item()
            attr["t_start.units"] = units_to_string(tstart.units)
        if hasattr(neoobj, "t_stop"):
            tstop = neoobj.t_stop
            attr["t_stop"] = tstop.magnitude.item()
            attr["t_stop.units"] = units_to_string(tstop.units)
        if hasattr(neoobj, "durations"):
            dura = neoobj.durations
            attr["durations"] = dura.magnitude
            attr["durations.units"] = units_to_string(dura.units)
        if hasattr(neoobj, "labels"):
            attr["labels"] = neoobj.labels.tolist()
        if hasattr(neoobj, "waveforms") and neoobj.waveforms is not None:
            attr["waveforms"] = list(wf.magnitude for wf in
                                     list(wfgroup for wfgroup in
                                          neoobj.waveforms))
            attr["waveforms.units"] = units_to_string(neoobj.waveforms.units)
        if hasattr(neoobj, "left_sweep") and neoobj.left_sweep is not None:
            lsweep = neoobj.left_sweep
            attr["left_sweep"] = lsweep.magnitude
            attr["left_sweep"] = units_to_string(lsweep.units)
        return attr

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
            if hasattr(v, "ndim") and v.ndim == 0:
                values = v.item()
                if isinstance(v, pq.Quantity):
                    unit = str(v.dimensionality)
            else:
                for item in v:
                    if isinstance(item, pq.Quantity):
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
        elif type(v).__module__ == "numpy":
            section[name] = nix.Value(v.item())
        else:
            section[name] = nix.Value(v)
        return section.props[name]

    @staticmethod
    def _get_contained_signals(obj):
        return list(
            da for da in obj.data_arrays
            if da.type in ["neo.analogsignal", "neo.irregularlysampledsignal"]
        )

    @staticmethod
    def _nix_attr_to_neo(nix_obj):
        neo_attrs = dict()
        neo_attrs["nix_name"] = nix_obj.name
        neo_attrs["description"] = stringify(nix_obj.definition)
        if nix_obj.metadata:
            for prop in nix_obj.metadata.props:
                values = prop.values
                values = list(v.value for v in values)
                if prop.unit:
                    units = prop.unit
                    values = create_quantity(values, units)
                if len(values) == 1:
                    neo_attrs[prop.name] = values[0]
                else:
                    neo_attrs[prop.name] = values
        neo_attrs["name"] = stringify(neo_attrs.get("neo_name"))

        if "file_datetime" in neo_attrs:
            neo_attrs["file_datetime"] = datetime.fromtimestamp(
                neo_attrs["file_datetime"]
            )
        return neo_attrs

    @staticmethod
    def _group_signals(paths):
        """
        Groups data arrays that were generated by the same Neo Signal object.

        :param paths: A list of paths (strings) of all the signals to be
        grouped :return: A list of paths (strings) of signal groups. The last
        part of each path is the common name of the signals in the group.
        """
        grouppaths = list(".".join(p.split(".")[:-1])
                          for p in paths)
        # deduplicating paths
        uniquepaths = []
        for path in grouppaths:
            if path not in uniquepaths:
                uniquepaths.append(path)
        return uniquepaths

    @staticmethod
    def _get_referers(nix_obj, obj_list):
        ref_list = list()
        for ref in obj_list:
            if nix_obj.name in list(src.name for src in ref.sources):
                ref_list.append(ref)
        return ref_list

    @staticmethod
    def _get_time_dimension(obj):
        for dim in obj.dimensions:
            if hasattr(dim, "label") and dim.label == "time":
                return dim
        return None

    @staticmethod
    def _hash_object(obj):
        """
        Computes an MD5 hash of a Neo object based on its attribute values and
        data objects. Child objects are not counted.

        :param obj: A Neo object
        :return: MD5 sum
        """
        objhash = md5()

        def strupdate(a):
            objhash.update(str(a).encode())

        def dupdate(d):
            if isinstance(d, np.ndarray) and not d.flags["C_CONTIGUOUS"]:
                d = d.copy(order="C")
            objhash.update(d)

        # attributes
        strupdate(obj.name)
        strupdate(obj.description)

        # annotations
        for k, v in sorted(obj.annotations.items()):
            strupdate(k)
            strupdate(v)

        # data objects and type-specific attributes
        if isinstance(obj, (Block, Segment)):
            strupdate(obj.rec_datetime)
            strupdate(obj.file_datetime)
        elif isinstance(obj, ChannelIndex):
            for idx in obj.index:
                strupdate(idx)
            for n in obj.channel_names:
                strupdate(n)
            if obj.coordinates is not None:
                for coord in obj.coordinates:
                    for c in coord:
                        strupdate(c)
        elif isinstance(obj, AnalogSignal):
            dupdate(obj)
            dupdate(obj.units)
            dupdate(obj.t_start)
            dupdate(obj.sampling_rate)
            dupdate(obj.t_stop)
        elif isinstance(obj, IrregularlySampledSignal):
            dupdate(obj)
            dupdate(obj.times)
            dupdate(obj.units)
        elif isinstance(obj, Event):
            dupdate(obj.times)
            for l in obj.labels:
                strupdate(l)
        elif isinstance(obj, Epoch):
            dupdate(obj.times)
            dupdate(obj.durations)
            for l in obj.labels:
                strupdate(l)
        elif isinstance(obj, SpikeTrain):
            dupdate(obj.times)
            dupdate(obj.units)
            dupdate(obj.t_stop)
            dupdate(obj.t_start)
            if obj.waveforms is not None:
                dupdate(obj.waveforms)
            dupdate(obj.sampling_rate)
            if obj.left_sweep is not None:
                strupdate(obj.left_sweep)

        # type
        strupdate(type(obj).__name__)

        return objhash.hexdigest()

    def close(self):
        """
        Closes the open nix file and resets maps.
        """
        if (hasattr(self, "nix_file") and
                self.nix_file and self.nix_file.is_open()):
            self.nix_file.close()
            self.nix_file = None
            self._lazy_loaded = None
            self._object_hashes = None
            self._block_read_counter = None

    def __del__(self):
        self.close()
