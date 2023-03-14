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


from datetime import date, time, datetime
from collections.abc import Iterable
from collections import OrderedDict
import itertools
from uuid import uuid4
import warnings
from packaging.version import Version
from itertools import chain

import quantities as pq
import numpy as np

from .baseio import BaseIO
from ..core import (Block, Segment, AnalogSignal,
                    IrregularlySampledSignal, Epoch, Event, SpikeTrain,
                    ImageSequence, ChannelView, Group)
from ..io.proxyobjects import BaseProxy
from .. import __version__ as neover


datetime_types = (date, time, datetime)

EMPTYANNOTATION = "EMPTYLIST"
ARRAYANNOTATION = "ARRAYANNOTATION"
DATETIMEANNOTATION = "DATETIME"
DATEANNOTATION = "DATE"
TIMEANNOTATION = "TIME"
MIN_NIX_VER = Version("1.5.0")

datefmt = "%Y-%m-%d"
timefmt = "%H:%M:%S.%f"
datetimefmt = datefmt + "T" + timefmt


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


def dt_to_nix(dt):
    """
    Converts date, time, and datetime objects to an ISO string representation
    appropriate for storing in NIX. Returns the converted value and the
    annotation type definition for converting back to the original value
    type.
    """
    if isinstance(dt, datetime):
        return dt.strftime(datetimefmt), DATETIMEANNOTATION
    if isinstance(dt, date):
        return dt.strftime(datefmt), DATEANNOTATION
    if isinstance(dt, time):
        return dt.strftime(timefmt), TIMEANNOTATION
    # Unknown: returning as is
    return dt


def dt_from_nix(nixdt, annotype):
    """
    Inverse function of 'dt_to_nix()'. Requires the stored annotation type to
    distinguish between the three source types (date, time, and datetime).
    """
    if annotype == DATEANNOTATION:
        dt = datetime.strptime(nixdt, datefmt)
        return dt.date()
    if annotype == TIMEANNOTATION:
        dt = datetime.strptime(nixdt, timefmt)
        return dt.time()
    if annotype == DATETIMEANNOTATION:
        dt = datetime.strptime(nixdt, datetimefmt)
        return dt
    # Unknown type: older (or newer) IO version?
    # Returning as is to avoid data loss.
    return nixdt


def check_nix_version():
    try:
        import nixio
    except ImportError:
        raise Exception(
            "Failed to import NIX. "
            "The NixIO requires the Python package for NIX "
            "(nixio on PyPi). Try `pip install nixio`."
        )

    # nixio version numbers have a 'v' prefix which breaks the comparison
    nixverstr = nixio.__version__.lstrip("v")
    try:
        nixver = Version(nixverstr)
    except ValueError:
        warnings.warn(
            f"Could not understand NIX Python version {nixverstr}. "
            f"The NixIO requires version {MIN_NIX_VER} of the Python package for NIX. "
            "The IO may not work correctly."
        )
        return

    if nixver < MIN_NIX_VER:
        raise Exception(
            "NIX version not supported. "
            f"The NixIO requires version {MIN_NIX_VER} or higher of the Python package "
            f"for NIX. Found version {nixverstr}"
        )


class NixIO(BaseIO):
    """
    Class for reading and writing NIX files.
    """

    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, Group, ChannelView,
                         AnalogSignal, IrregularlySampledSignal,
                         Epoch, Event, SpikeTrain]
    readable_objects = [Block]
    writeable_objects = [Block]

    name = "NIX"
    extensions = ["h5", "nix"]
    mode = "file"

    def __init__(self, filename, mode="rw"):
        """
        Initialise IO instance and NIX file.

        :param filename: Full path to the file
        """
        check_nix_version()
        import nixio

        BaseIO.__init__(self, filename)
        self.filename = str(filename)
        if mode == "ro":
            filemode = nixio.FileMode.ReadOnly
        elif mode == "rw":
            filemode = nixio.FileMode.ReadWrite
        elif mode == "ow":
            filemode = nixio.FileMode.Overwrite
        else:
            raise ValueError(f"Invalid mode specified '{mode}'. "
                             "Valid modes: 'ro' (ReadOnly)', 'rw' (ReadWrite),"
                             " 'ow' (Overwrite).")
        self.nix_file = nixio.File.open(self.filename, filemode)

        if self.nix_file.mode == nixio.FileMode.ReadOnly:
            self._file_version = '0.5.2'
            if "neo" in self.nix_file.sections:
                self._file_version = self.nix_file.sections["neo"]["version"]
        elif self.nix_file.mode == nixio.FileMode.ReadWrite:
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
        self._view_map = dict()

        # _names_ok is used to guard against name check duplication
        self._names_ok = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_all_blocks(self, lazy=False):
        if lazy:
            raise Exception("Lazy loading is not supported for NixIO")
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
        if lazy:
            raise Exception("Lazy loading is not supported for NixIO")

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
                raise KeyError(f"Block with Neo name '{neoname}' does not exist")
        else:
            index = self._block_read_counter
            if index >= len(self.nix_file.blocks):
                return None
            nix_block = self.nix_file.blocks[index]
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
        neo_block.rec_datetime = datetime.fromtimestamp(nix_block.created_at)

        # descend into Groups
        groups_to_resolve = []
        for grp in nix_block.groups:
            if grp.type == "neo.segment":
                newseg = self._nix_to_neo_segment(grp)
                neo_block.segments.append(newseg)
                # parent reference
                newseg.block = neo_block
            elif grp.type == "neo.group":
                newgrp, parent_name = self._nix_to_neo_group(grp)
                assert parent_name is None
                neo_block.groups.append(newgrp)
                # parent reference
                newgrp.block = neo_block
            elif grp.type == "neo.subgroup":
                newgrp, parent_name = self._nix_to_neo_group(grp)
                groups_to_resolve.append((newgrp, parent_name))
            else:
                raise Exception("Unexpected group type")

        # link subgroups to parents
        for newgrp, parent_name in groups_to_resolve:
            parent = self._neo_map[parent_name]
            parent.groups.append(newgrp)

        # find free floating (Groupless) signals and spiketrains
        blockdas = self._group_signals(nix_block.data_arrays)
        for name, das in blockdas.items():
            if name not in self._neo_map:
                if das[0].type == "neo.analogsignal":
                    self._nix_to_neo_analogsignal(das)
                elif das[0].type == "neo.irregularlysampledsignal":
                    self._nix_to_neo_irregularlysampledsignal(das)
                elif das[0].type == "neo.imagesequence":
                    self._nix_to_neo_imagesequence(das)
        for mt in nix_block.multi_tags:
            if mt.type == "neo.spiketrain" and mt.name not in self._neo_map:
                self._nix_to_neo_spiketrain(mt)

        # create object links
        neo_block.create_relationship()

        # reset maps
        self._neo_map = dict()
        self._ref_map = dict()
        self._signal_map = dict()
        self._view_map = dict()

        return neo_block

    def _nix_to_neo_segment(self, nix_group):
        neo_attrs = self._nix_attr_to_neo(nix_group)
        neo_segment = Segment(**neo_attrs)
        neo_segment.rec_datetime = datetime.fromtimestamp(nix_group.created_at)
        self._neo_map[nix_group.name] = neo_segment

        # this will probably get all the DAs anyway, but if we change any part
        # of the mapping to add other kinds of DataArrays to a group, such as
        # MultiTag positions and extents, this filter will be necessary
        dataarrays = list(filter(
            lambda da: da.type in ("neo.analogsignal",
                                   "neo.irregularlysampledsignal",
                                   "neo.imagesequence",),
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
            elif das[0].type == "neo.imagesequence":
                new_imgseq = self._nix_to_neo_imagesequence(das)
                neo_segment.imagesequences.append(new_imgseq)
                # parent reference
                new_imgseq.segment = neo_segment

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

    def _nix_to_neo_group(self, nix_group):
        neo_attrs = self._nix_attr_to_neo(nix_group)
        parent_name = neo_attrs.pop("neo_parent", None)
        neo_group = Group(**neo_attrs)
        self._neo_map[nix_group.name] = neo_group
        dataarrays = list(filter(
            lambda da: da.type in ("neo.analogsignal",
                                   "neo.irregularlysampledsignal",
                                   "neo.imagesequence",),
            nix_group.data_arrays))
        dataarrays = self._group_signals(dataarrays)
        # descend into DataArrays
        for name in dataarrays:
            obj = self._neo_map[name]
            neo_group.add(obj)
        # descend into MultiTags
        for mtag in nix_group.multi_tags:
            if mtag.type == "neo.channelview" and mtag.name not in self._neo_map:
                self._nix_to_neo_channelview(mtag)
            obj = self._neo_map[mtag.name]
            neo_group.add(obj)

        return neo_group, parent_name

    def _nix_to_neo_channelview(self, nix_mtag):
        neo_attrs = self._nix_attr_to_neo(nix_mtag)
        index = nix_mtag.positions
        nix_name, = self._group_signals(nix_mtag.references).keys()
        obj = self._neo_map[nix_name]
        neo_chview = ChannelView(obj, index, **neo_attrs)
        self._neo_map[nix_mtag.name] = neo_chview
        return neo_chview

    def _nix_to_neo_analogsignal(self, nix_da_group):
        """
        Convert a group of NIX DataArrays to a Neo AnalogSignal. This method
        expects a list of data arrays that all represent the same,
        multidimensional Neo AnalogSignal object.

        :param nix_da_group: a list of NIX DataArray objects
        :return: a Neo AnalogSignal object
        """
        neo_attrs = self._nix_attr_to_neo(nix_da_group[0])
        metadata = nix_da_group[0].metadata
        neo_attrs["nix_name"] = metadata.name  # use the common base name

        unit = nix_da_group[0].unit
        signaldata = np.array([d[:] for d in nix_da_group]).transpose()
        signaldata = create_quantity(signaldata, unit)
        timedim = self._get_time_dimension(nix_da_group[0])
        sampling_period = create_quantity(timedim.sampling_interval, timedim.unit)
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

        neo_signal = AnalogSignal(signal=signaldata, sampling_period=sampling_period,
                                  t_start=t_start, **neo_attrs)
        self._neo_map[neo_attrs["nix_name"]] = neo_signal
        # all DAs reference the same sources
        srcnames = list(src.name for src in nix_da_group[0].sources)
        for n in srcnames:
            if n not in self._ref_map:
                self._ref_map[n] = list()
            self._ref_map[n].append(neo_signal)
        return neo_signal

    def _nix_to_neo_imagesequence(self, nix_da_group):
        """
        Convert a group of NIX DataArrays to a Neo ImageSequence. This method
        expects a list of data arrays that all represent the same,
        multidimensional Neo ImageSequence object.

        :param nix_da_group: a list of NIX DataArray objects
        :return: a Neo ImageSequence object
        """

        neo_attrs = self._nix_attr_to_neo(nix_da_group[0])
        metadata = nix_da_group[0].metadata
        neo_attrs["nix_name"] = metadata.name  # use the common base name
        unit = nix_da_group[0].unit
        imgseq = np.array([d[:] for d in nix_da_group]).transpose()

        sampling_rate = neo_attrs["sampling_rate"]
        del neo_attrs["sampling_rate"]
        spatial_scale = neo_attrs["spatial_scale"]
        del neo_attrs["spatial_scale"]
        if "t_start" in neo_attrs:
            t_start = neo_attrs["t_start"]
            del neo_attrs["t_start"]
        else:
            t_start = 0.0 * pq.ms

        neo_seq = ImageSequence(image_data=imgseq, sampling_rate=sampling_rate,
                                spatial_scale=spatial_scale, units=unit,
                                t_start=t_start, **neo_attrs)

        self._neo_map[neo_attrs["nix_name"]] = neo_seq
        # all DAs reference the same sources
        srcnames = list(src.name for src in nix_da_group[0].sources)
        for n in srcnames:
            if n not in self._ref_map:
                self._ref_map[n] = list()
            self._ref_map[n].append(neo_seq)
        return neo_seq

    def _nix_to_neo_irregularlysampledsignal(self, nix_da_group):
        """
        Convert a group of NIX DataArrays to a Neo IrregularlySampledSignal.
        This method expects a list of data arrays that all represent the same,
        multidimensional Neo IrregularlySampledSignal object.

        :param nix_da_group: a list of NIX DataArray objects
        :return: a Neo IrregularlySampledSignal object
        """
        neo_attrs = self._nix_attr_to_neo(nix_da_group[0])
        metadata = nix_da_group[0].metadata
        neo_attrs["nix_name"] = metadata.name  # use the common base name

        unit = nix_da_group[0].unit
        signaldata = np.array([d[:] for d in nix_da_group]).transpose()
        signaldata = create_quantity(signaldata, unit)
        timedim = self._get_time_dimension(nix_da_group[0])
        times = create_quantity(timedim.ticks, timedim.unit)

        neo_signal = IrregularlySampledSignal(signal=signaldata, times=times, **neo_attrs)
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
        labels = np.array(nix_mtag.positions.dimensions[0].labels, dtype="U")
        neo_event = Event(times=times, labels=labels, **neo_attrs)
        self._neo_map[nix_mtag.name] = neo_event
        return neo_event

    def _nix_to_neo_epoch(self, nix_mtag):
        neo_attrs = self._nix_attr_to_neo(nix_mtag)
        time_unit = nix_mtag.positions.unit
        times = create_quantity(nix_mtag.positions, time_unit)
        durations = create_quantity(nix_mtag.extents, nix_mtag.extents.unit)

        if len(nix_mtag.positions.dimensions[0].labels) > 0:
            labels = np.array(nix_mtag.positions.dimensions[0].labels, dtype="U")
        else:
            labels = None
        neo_epoch = Epoch(times=times, durations=durations, labels=labels, **neo_attrs)
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
            neo_spiketrain.sampling_period = create_quantity(wftime.sampling_interval,
                                                             interval_units)
            left_sweep_units = wftime.unit
            if "left_sweep" in wfda.metadata:
                neo_spiketrain.left_sweep = create_quantity(wfda.metadata["left_sweep"],
                                                            left_sweep_units)
        self._neo_map[nix_mtag.name] = neo_spiketrain

        srcnames = list(src.name for src in nix_mtag.sources)
        for n in srcnames:
            if n not in self._ref_map:
                self._ref_map[n] = list()
            self._ref_map[n].append(neo_spiketrain)
        return neo_spiketrain

    def write_all_blocks(self, neo_blocks, use_obj_names=False):
        """
        Convert all ``neo_blocks`` to the NIX equivalent and write them to the
        file.

        :param neo_blocks: List (or iterable) containing Neo blocks
        :param use_obj_names: If True, will not generate unique object names
        but will instead try to use the name of each Neo object. If these are
        not unique, an exception will be raised.
        """
        if use_obj_names:
            self._use_obj_names(neo_blocks)
            self._names_ok = True
        for bl in neo_blocks:
            self.write_block(bl, use_obj_names)

    def write_block(self, block, use_obj_names=False):
        """
        Convert the provided Neo Block to a NIX Block and write it to
        the NIX file.

        :param block: Neo Block to be written
        :param use_obj_names: If True, will not generate unique object names
        but will instead try to use the name of each Neo object. If these are
        not unique, an exception will be raised.
        """
        if use_obj_names:
            if not self._names_ok:
                # _names_ok guards against check duplication
                # If it's False, it means write_block() was called directly
                self._use_obj_names([block])
        if "nix_name" in block.annotations:
            nix_name = block.annotations["nix_name"]
        else:
            nix_name = f"neo.block.{self._generate_nix_name()}"
            block.annotate(nix_name=nix_name)

        if nix_name in self.nix_file.blocks:
            nixblock = self.nix_file.blocks[nix_name]
            del self.nix_file.blocks[nix_name]
            del self.nix_file.sections[nix_name]

        nixblock = self.nix_file.create_block(nix_name, "neo.block")
        nixblock.metadata = self.nix_file.create_section(nix_name, "neo.block.metadata")
        metadata = nixblock.metadata
        neoname = block.name if block.name is not None else ""
        metadata["neo_name"] = neoname
        nixblock.definition = block.description
        if block.rec_datetime:
            nix_rec_dt = int(block.rec_datetime.strftime("%s"))
            nixblock.force_created_at(nix_rec_dt)
        if block.file_datetime:
            fdt, annotype = dt_to_nix(block.file_datetime)
            fdtprop = metadata.create_property("file_datetime", fdt)
            fdtprop.definition = annotype
        if block.annotations:
            for k, v in block.annotations.items():
                self._write_property(metadata, k, v)

        # descend into Segments
        for seg in block.segments:
            self._write_segment(seg, nixblock)

        # descend into Neo Groups
        for group in block.groups:
            self._write_group(group, nixblock)

    def _write_channelview(self, chview, nixblock, nixgroup):
        """
        Convert the provided Neo ChannelView to a NIX MultiTag and write it to
        the NIX file.

        :param chx: The Neo ChannelView to be written
        :param nixblock: NIX Block where the MultiTag will be created
        """
        if "nix_name" in chview.annotations:
            nix_name = chview.annotations["nix_name"]
        else:
            nix_name = "neo.channelview.{}".format(self._generate_nix_name())
            chview.annotate(nix_name=nix_name)

        # create a new data array if this channelview was not saved yet
        if not nix_name in self._view_map:
            channels = nixblock.create_data_array(
                "{}.index".format(nix_name), "neo.channelview.index", data=chview.index
            )

            nixmt = nixblock.create_multi_tag(nix_name, "neo.channelview",
                                              positions=channels)

            nixmt.metadata = nixgroup.metadata.create_section(
                nix_name, "neo.channelview.metadata"
            )
            metadata = nixmt.metadata
            neoname = chview.name if chview.name is not None else ""
            metadata["neo_name"] = neoname
            nixmt.definition = chview.description
            if chview.annotations:
                for k, v in chview.annotations.items():
                    self._write_property(metadata, k, v)
            self._view_map[nix_name] = nixmt

            # link tag to the data array for the ChannelView's signal
            if not ("nix_name" in chview.obj.annotations
                    and chview.obj.annotations["nix_name"] in self._signal_map):
                # the following restriction could be relaxed later
                # but for a first pass this simplifies my mental model
                raise Exception("Need to save signals before saving views")
            nix_name = chview.obj.annotations["nix_name"]
            nixmt.references.extend(self._signal_map[nix_name])
        else:
            nixmt = self._view_map[nix_name]

        nixgroup.multi_tags.append(nixmt)

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
            nix_name = f"neo.segment.{self._generate_nix_name()}"
            segment.annotate(nix_name=nix_name)

        nixgroup = nixblock.create_group(nix_name, "neo.segment")
        nixgroup.metadata = nixblock.metadata.create_section(nix_name,
                                                             "neo.segment.metadata")
        metadata = nixgroup.metadata
        neoname = segment.name if segment.name is not None else ""
        metadata["neo_name"] = neoname
        nixgroup.definition = segment.description
        if segment.rec_datetime:
            nix_rec_dt = int(segment.rec_datetime.strftime("%s"))
            nixgroup.force_created_at(nix_rec_dt)
        if segment.file_datetime:
            fdt, annotype = dt_to_nix(segment.file_datetime)
            fdtprop = metadata.create_property("file_datetime", fdt)
            fdtprop.definition = annotype
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

        for imagesequence in segment.imagesequences:
            self._write_imagesequence(imagesequence, nixblock, nixgroup)

    def _write_group(self, neo_group, nixblock, parent=None):
        """
        Convert the provided Neo Group to a NIX Group and write it to the
        NIX file.

        :param neo_group: Neo Group to be written
        :param nixblock: NIX Block where the NIX Group will be created
        :param parent: for sub-groups, the parent Neo Group
        """
        if parent:
            label = "neo.subgroup"
            # note that the use of a different label for top-level groups and sub-groups is not
            # strictly  necessary, the presence of the "neo_parent" annotation is sufficient.
            # However, I think it adds clarity and helps in debugging and testing.
        else:
            label = "neo.group"

        if "nix_name" in neo_group.annotations:
            nix_name = neo_group.annotations["nix_name"]
        else:
            nix_name = "{}.{}".format(label, self._generate_nix_name())
            neo_group.annotate(nix_name=nix_name)

        nixgroup = nixblock.create_group(nix_name, label)
        nixgroup.metadata = nixblock.metadata.create_section(
            nix_name, f"{label}.metadata"
        )
        metadata = nixgroup.metadata
        neoname = neo_group.name if neo_group.name is not None else ""
        metadata["neo_name"] = neoname
        if parent:
            metadata["neo_parent"] = parent.annotations["nix_name"]
        nixgroup.definition = neo_group.description
        if neo_group.annotations:
            for k, v in neo_group.annotations.items():
                self._write_property(metadata, k, v)

        # link signals and image sequences
        objnames = []
        for obj in chain(
            neo_group.analogsignals,
            neo_group.irregularlysampledsignals,
            neo_group.imagesequences,
        ):
            if not ("nix_name" in obj.annotations
                    and obj.annotations["nix_name"] in self._signal_map):
                # the following restriction could be relaxed later
                # but for a first pass this simplifies my mental model
                raise Exception("Orphan signals/image sequences cannot be stored, needs to belong to a Segment")
            objnames.append(obj.annotations["nix_name"])
        for name in objnames:
            for da in self._signal_map[name]:
                nixgroup.data_arrays.append(da)

        # link events, epochs and spiketrains
        objnames = []
        for obj in chain(
            neo_group.events,
            neo_group.epochs,
            neo_group.spiketrains,
        ):
            if not ("nix_name" in obj.annotations
                    and obj.annotations["nix_name"] in nixblock.multi_tags):
                # the following restriction could be relaxed later
                # but for a first pass this simplifies my mental model
                raise Exception("Orphan epochs/events/spiketrains cannot be stored, needs to belong to a Segment")
            objnames.append(obj.annotations["nix_name"])
        for name in objnames:
            mt = nixblock.multi_tags[name]
            nixgroup.multi_tags.append(mt)

        # save channel views
        for chview in neo_group.channelviews:
            self._write_channelview(chview, nixblock, nixgroup)

        # save sub-groups
        for subgroup in neo_group.groups:
            self._write_group(subgroup, nixblock, parent=neo_group)

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
            nix_name = f"neo.analogsignal.{self._generate_nix_name()}"
            anasig.annotate(nix_name=nix_name)

        if f"{nix_name}.0" in nixblock.data_arrays and nixgroup:
            # AnalogSignal is in multiple Segments.
            # Append DataArrays to Group and return.
            dalist = list()
            for idx in itertools.count():
                daname = f"{nix_name}.{idx}"
                if daname in nixblock.data_arrays:
                    dalist.append(nixblock.data_arrays[daname])
                else:
                    break
            nixgroup.data_arrays.extend(dalist)
            return

        if isinstance(anasig, BaseProxy):
            data = np.transpose(anasig.load()[:].magnitude)
        else:
            data = np.transpose(anasig[:].magnitude)

        parentmd = nixgroup.metadata if nixgroup else nixblock.metadata
        metadata = parentmd.create_section(nix_name, "neo.analogsignal.metadata")
        nixdas = list()
        for idx, row in enumerate(data):
            daname = f"{nix_name}.{idx}"
            da = nixblock.create_data_array(daname, "neo.analogsignal", data=row)
            da.metadata = metadata
            da.definition = anasig.description
            da.unit = units_to_string(anasig.units)

            sampling_period = anasig.sampling_period.magnitude.item()
            timedim = da.append_sampled_dimension(sampling_period)
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
        if anasig.array_annotations:
            for k, v in anasig.array_annotations.items():
                p = self._write_property(metadata, k, v)
                p.type = ARRAYANNOTATION

        self._signal_map[nix_name] = nixdas

    def _write_imagesequence(self, imgseq, nixblock, nixgroup):
        """
       Convert the provided ``imgseq`` (ImageSequence) to a list of NIX
       DataArray objects and write them to the NIX file. All DataArray objects
       created from the same ImageSequence have their metadata section point to
       the same object.

       :param anasig: The Neo ImageSequence to be written
       :param nixblock: NIX Block where the DataArrays will be created
       :param nixgroup: NIX Group where the DataArrays will be attached
       """

        if "nix_name" in imgseq.annotations:
            nix_name = imgseq.annotations["nix_name"]
        else:
            nix_name = f"neo.imagesequence.{self._generate_nix_name()}"
            imgseq.annotate(nix_name=nix_name)

        if f"{nix_name}.0" in nixblock.data_arrays and nixgroup:

            dalist = list()
            for idx in itertools.count():
                daname = f"{nix_name}.{idx}"
                if daname in nixblock.data_arrays:
                    dalist.append(nixblock.data_arrays[daname])
                else:
                    break
            nixgroup.data_arrays.extend(dalist)
            return

        if isinstance(imgseq, BaseProxy):
            data = np.transpose(imgseq.load()[:].magnitude)
        else:
            data = np.transpose(imgseq[:].magnitude)

        parentmd = nixgroup.metadata if nixgroup else nixblock.metadata
        metadata = parentmd.create_section(nix_name, "neo.imagesequence.metadata")

        nixdas = list()
        for idx, row in enumerate(data):
            daname = f"{nix_name}.{idx}"
            da = nixblock.create_data_array(daname, "neo.imagesequence", data=row)

            da.metadata = metadata
            da.definition = imgseq.description
            da.unit = units_to_string(imgseq.units)

            metadata["sampling_rate"] = imgseq.sampling_rate.magnitude.item()
            units = imgseq.sampling_rate.units
            metadata.props["sampling_rate"].unit = units_to_string(units)
            metadata["spatial_scale"] = imgseq.spatial_scale.magnitude.item()
            units = imgseq.spatial_scale.units
            metadata.props["spatial_scale"].unit = units_to_string(units)
            metadata["t_start"] = imgseq.t_start.magnitude.item()
            units = imgseq.t_start.units
            metadata.props["t_start"].unit = units_to_string(units)

            nixdas.append(da)
            if nixgroup:
                nixgroup.data_arrays.append(da)

        neoname = imgseq.name if imgseq.name is not None else ""
        metadata["neo_name"] = neoname
        if imgseq.annotations:
            for k, v in imgseq.annotations.items():
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
            nix_name = f"neo.irregularlysampledsignal.{self._generate_nix_name()}"
            irsig.annotate(nix_name=nix_name)

        if f"{nix_name}.0" in nixblock.data_arrays and nixgroup:
            # IrregularlySampledSignal is in multiple Segments.
            # Append DataArrays to Group and return.
            dalist = list()
            for idx in itertools.count():
                daname = f"{nix_name}.{idx}"
                if daname in nixblock.data_arrays:
                    dalist.append(nixblock.data_arrays[daname])
                else:
                    break
            nixgroup.data_arrays.extend(dalist)
            return

        if isinstance(irsig, BaseProxy):
            data = np.transpose(irsig.load()[:].magnitude)
        else:
            data = np.transpose(irsig[:].magnitude)

        parentmd = nixgroup.metadata if nixgroup else nixblock.metadata
        metadata = parentmd.create_section(nix_name, "neo.irregularlysampledsignal.metadata")
        nixdas = list()
        for idx, row in enumerate(data):
            daname = f"{nix_name}.{idx}"
            da = nixblock.create_data_array(daname, "neo.irregularlysampledsignal", data=row)
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
        if irsig.array_annotations:
            for k, v in irsig.array_annotations.items():
                p = self._write_property(metadata, k, v)
                p.type = ARRAYANNOTATION

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
            nix_name = f"neo.event.{self._generate_nix_name()}"
            event.annotate(nix_name=nix_name)

        if nix_name in nixblock.multi_tags:
            # Event is in multiple Segments. Append to Group and return.
            mt = nixblock.multi_tags[nix_name]
            nixgroup.multi_tags.append(mt)
            return

        if isinstance(event, BaseProxy):
            event = event.load()

        times = event.times.magnitude
        units = units_to_string(event.times.units)
        labels = event.labels
        timesda = nixblock.create_data_array(f"{nix_name}.times", "neo.event.times", data=times)
        timesda.unit = units
        nixmt = nixblock.create_multi_tag(nix_name, "neo.event", positions=timesda)

        nixmt.metadata = nixgroup.metadata.create_section(nix_name, "neo.event.metadata")
        metadata = nixmt.metadata

        labeldim = timesda.append_set_dimension()
        labeldim.labels = labels

        neoname = event.name if event.name is not None else ""
        metadata["neo_name"] = neoname
        nixmt.definition = event.description
        if event.annotations:
            for k, v in event.annotations.items():
                self._write_property(metadata, k, v)
        if event.array_annotations:
            for k, v in event.array_annotations.items():
                p = self._write_property(metadata, k, v)
                p.type = ARRAYANNOTATION

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
            nix_name = f"neo.epoch.{self._generate_nix_name()}"
            epoch.annotate(nix_name=nix_name)

        if nix_name in nixblock.multi_tags:
            # Epoch is in multiple Segments. Append to Group and return.
            mt = nixblock.multi_tags[nix_name]
            nixgroup.multi_tags.append(mt)
            return

        if isinstance(epoch, BaseProxy):
            epoch = epoch.load()
        times = epoch.times.magnitude
        tunits = units_to_string(epoch.times.units)
        durations = epoch.durations.magnitude
        dunits = units_to_string(epoch.durations.units)

        timesda = nixblock.create_data_array(f"{nix_name}.times", "neo.epoch.times", data=times)
        timesda.unit = tunits
        nixmt = nixblock.create_multi_tag(nix_name, "neo.epoch", positions=timesda)

        durada = nixblock.create_data_array(f"{nix_name}.durations", "neo.epoch.durations",
                                            data=durations)
        durada.unit = dunits
        nixmt.extents = durada

        nixmt.metadata = nixgroup.metadata.create_section(nix_name, "neo.epoch.metadata")
        metadata = nixmt.metadata

        labeldim = timesda.append_set_dimension()
        labeldim.labels = epoch.labels

        neoname = epoch.name if epoch.name is not None else ""
        metadata["neo_name"] = neoname
        nixmt.definition = epoch.description
        if epoch.annotations:
            for k, v in epoch.annotations.items():
                self._write_property(metadata, k, v)
        if epoch.array_annotations:
            for k, v in epoch.array_annotations.items():
                p = self._write_property(metadata, k, v)
                p.type = ARRAYANNOTATION

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
        import nixio

        if "nix_name" in spiketrain.annotations:
            nix_name = spiketrain.annotations["nix_name"]
        else:
            nix_name = f"neo.spiketrain.{self._generate_nix_name()}"
            spiketrain.annotate(nix_name=nix_name)

        if nix_name in nixblock.multi_tags and nixgroup:
            # SpikeTrain is in multiple Segments. Append to Group and return.
            mt = nixblock.multi_tags[nix_name]
            nixgroup.multi_tags.append(mt)
            return

        if isinstance(spiketrain, BaseProxy):
            spiketrain = spiketrain.load(load_waveforms=True)

        times = spiketrain.times.magnitude
        tunits = units_to_string(spiketrain.times.units)
        waveforms = spiketrain.waveforms

        timesda = nixblock.create_data_array(f"{nix_name}.times", "neo.spiketrain.times",
                                             data=times)
        timesda.unit = tunits
        nixmt = nixblock.create_multi_tag(nix_name, "neo.spiketrain", positions=timesda)

        parentmd = nixgroup.metadata if nixgroup else nixblock.metadata
        nixmt.metadata = parentmd.create_section(nix_name, "neo.spiketrain.metadata")
        metadata = nixmt.metadata

        neoname = spiketrain.name if spiketrain.name is not None else ""
        metadata["neo_name"] = neoname
        nixmt.definition = spiketrain.description

        self._write_property(metadata, "t_start", spiketrain.t_start)
        self._write_property(metadata, "t_stop", spiketrain.t_stop)

        if spiketrain.annotations:
            for k, v in spiketrain.annotations.items():
                self._write_property(metadata, k, v)
        if spiketrain.array_annotations:
            for k, v in spiketrain.array_annotations.items():
                p = self._write_property(metadata, k, v)
                p.type = ARRAYANNOTATION

        if nixgroup:
            nixgroup.multi_tags.append(nixmt)

        if waveforms is not None:
            wfdata = list(wf.magnitude for wf in
                          list(wfgroup for wfgroup in spiketrain.waveforms))
            wfunits = units_to_string(spiketrain.waveforms.units)
            wfda = nixblock.create_data_array(f"{nix_name}.waveforms", "neo.waveforms",
                                              data=wfdata)
            wfda.unit = wfunits
            wfda.metadata = nixmt.metadata.create_section(wfda.name, "neo.waveforms.metadata")
            nixmt.create_feature(wfda, nixio.LinkType.Indexed)
            # TODO: Move time dimension first for PR #457
            # https://github.com/NeuralEnsemble/python-neo/pull/457
            wfda.append_set_dimension()
            wfda.append_set_dimension()
            wftime = wfda.append_sampled_dimension(spiketrain.sampling_period.magnitude.item())
            wftime.unit = units_to_string(spiketrain.sampling_period.units)
            wftime.label = "time"

            if spiketrain.left_sweep is not None:
                self._write_property(wfda.metadata, "left_sweep", spiketrain.left_sweep)

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
        import nixio

        if isinstance(v, datetime_types):
            value, annotype = dt_to_nix(v)
            prop = section.create_property(name, value)
            prop.definition = annotype
        elif isinstance(v, str):
            if len(v):
                section.create_property(name, v)
            else:
                section.create_property(name, nixio.DataType.String)
        elif isinstance(v, bytes):
            section.create_property(name, v.decode())
        elif isinstance(v, Iterable):
            values = []
            unit = None
            definition = None
            # handling (quantity) arrays with only a single element
            if hasattr(v, "ndim") and v.ndim == 0:
                values = v.item()
            # handling empty arrays or lists
            elif (hasattr(v, 'size') and (v.size == 0)) or (len(v) == 0):
                # NIX supports empty properties but dtype must be specified
                # Defaulting to String and using definition to signify empty
                # iterable as opposed to empty string
                values = nixio.DataType.String
                definition = EMPTYANNOTATION
            else:
                for item in v:
                    if isinstance(item, str):
                        item = item
                    elif isinstance(item, pq.Quantity):
                        current_unit = str(item.dimensionality)
                        if unit is None:
                            unit = current_unit
                        elif unit != current_unit:
                            raise ValueError(f'Inconsistent units detected for '
                                             f'property {name}: {v}')
                        item = item.magnitude.item()
                    elif isinstance(item, Iterable):
                        self.logger.warn("Multidimensional arrays and nested "
                                         "containers are not currently "
                                         "supported when writing to NIX.")
                        return None
                    else:
                        item = item
                    values.append(item)
            if hasattr(v, 'dimensionality'):
                unit = str(v.dimensionality)
            section.create_property(name, values)
            section.props[name].unit = unit
            section.props[name].definition = definition
        elif type(v).__module__ == "numpy":
            section.create_property(name, v.item())
        else:
            section.create_property(name, v)
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
        import nixio

        neo_attrs = dict()
        neo_attrs["nix_name"] = nix_obj.name
        neo_attrs["description"] = stringify(nix_obj.definition)
        if nix_obj.metadata:
            for prop in nix_obj.metadata.inherited_properties():
                values = list(prop.values)
                if not len(values):
                    if prop.definition == EMPTYANNOTATION:
                        values = list()
                    elif prop.data_type == nixio.DataType.String:
                        values = ""
                elif len(values) == 1:
                    values = values[0]
                if prop.unit:
                    units = prop.unit
                    values = create_quantity(values, units)
                if prop.definition in (DATEANNOTATION, TIMEANNOTATION, DATETIMEANNOTATION):
                    values = dt_from_nix(values, prop.definition)
                if prop.type == ARRAYANNOTATION:
                    if 'array_annotations' in neo_attrs:
                        neo_attrs['array_annotations'][prop.name] = values
                    else:
                        neo_attrs['array_annotations'] = {prop.name: values}
                else:
                    neo_attrs[prop.name] = values
        # since the 'neo_name' NIX property becomes the actual object's name,
        # there's no reason to keep it in the annotations
        neo_attrs["name"] = stringify(neo_attrs.pop("neo_name", None))

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
        # now start grouping
        groups = OrderedDict()
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

    def _use_obj_names(self, blocks):

        errmsg = "use_obj_names enabled: found conflict or anonymous object"
        allobjs = []

        def check_unique(objs):
            names = list(o.name for o in objs)
            if None in names or "" in names:
                raise ValueError(names)
            if len(names) != len(set(names)):
                self._names_ok = False
                raise ValueError(names)
            # collect objs if ok
            allobjs.extend(objs)

        try:
            check_unique(blocks)
        except ValueError as exc:
            raise ValueError(f"{errmsg} in Blocks") from exc

        for blk in blocks:
            try:
                # Segments
                check_unique(blk.segments)
            except ValueError as exc:
                raise ValueError(f"{errmsg} at Block '{blk.name}' > segments") from exc

            # collect all signals in all segments
            signals = []
            # collect all events, epochs, and spiketrains in all segments
            eests = []
            for seg in blk.segments:
                signals.extend(seg.analogsignals)
                signals.extend(seg.irregularlysampledsignals)
                signals.extend(seg.imagesequences)
                eests.extend(seg.events)
                eests.extend(seg.epochs)
                eests.extend(seg.spiketrains)

            try:
                # AnalogSignals and IrregularlySampledSignals
                check_unique(signals)
            except ValueError as exc:
                raise ValueError(f"{errmsg} in Signal names of Block '{blk.name}'") from exc

            try:
                # Events, Epochs, and SpikeTrains
                check_unique(eests)
            except ValueError as exc:
                raise ValueError(
                    f"{errmsg} in Event, Epoch, and Spiketrain names of Block '{blk.name}'"
                ) from exc

            # groups
            groups = []
            for grp in blk.groups:
                groups.extend(list(grp.walk()))
            try:
                check_unique(groups)
            except ValueError as exc:
                raise ValueError(f"{errmsg} in Group names of Block '{blk.name}'") from exc

        # names are OK: assign annotations
        for o in allobjs:
            o.annotations["nix_name"] = o.name

    def close(self):
        """
        Closes the open nix file and resets maps.
        """
        if (hasattr(self, "nix_file") and self.nix_file and self.nix_file.is_open()):
            self.nix_file.close()
            self.nix_file = None
            self._neo_map = None
            self._ref_map = None
            self._signal_map = None
            self._view_map = None
            self._block_read_counter = None

    def __del__(self):
        self.close()
