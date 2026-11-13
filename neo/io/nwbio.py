"""
NWBIO
=====

IO class for reading data from a Neurodata Without Borders (NWB) dataset

Documentation : https://www.nwb.org/
Depends on: h5py, nwb, dateutil
Supported: Read, Write
Python API -  https://pynwb.readthedocs.io
Sample datasets from CRCNS - https://crcns.org/NWB
Sample datasets from Allen Institute
- http://alleninstitute.github.io/AllenSDK/cell_types.html#neurodata-without-borders
"""

from __future__ import absolute_import, division

import json
import logging
import os
from itertools import chain
from json.decoder import JSONDecodeError

import numpy as np
import quantities as pq

from neo.core import Segment, SpikeTrain, Epoch, Event, AnalogSignal, IrregularlySampledSignal, Block, ImageSequence
from neo.core.baseneo import _check_annotations
from neo.core.dataobject import ArrayDict
from neo.io.baseio import BaseIO
from neo.io.proxyobjects import (
    AnalogSignalProxy as BaseAnalogSignalProxy,
    EventProxy as BaseEventProxy,
    EpochProxy as BaseEpochProxy,
    SpikeTrainProxy as BaseSpikeTrainProxy,
)

logger = logging.getLogger("Neo")

GLOBAL_ANNOTATIONS = (
    "session_start_time",
    "identifier",
    "timestamps_reference_time",
    "experimenter",
    "experiment_description",
    "session_id",
    "institution",
    "keywords",
    "notes",
    "pharmacology",
    "protocol",
    "related_publications",
    "slices",
    "source_script",
    "source_script_file_name",
    "data_collection",
    "surgery",
    "virus",
    "stimulus_notes",
    "lab",
    "session_description",
    "rec_datetime",
)

POSSIBLE_JSON_FIELDS = ("source_script", "description")

# TimeSeries fields that are handled explicitly when writing, so are not taken from `nwb:` annotations.
# The scaling fields are applied to the data when reading, and so must not be applied again.
WRITE_HANDLED_FIELDS = ("conversion", "offset", "channel_conversion", "comments", "description")

prefix_map = {1e9: "giga", 1e6: "mega", 1e3: "kilo", 1: "", 1e-3: "milli", 1e-6: "micro", 1e-9: "nano", 1e-12: "pico"}


def try_json_field(content):
    """
    Try to interpret a string as JSON data.

    If successful, return the JSON data (dict or list)
    If unsuccessful, return the original string
    """
    try:
        return json.loads(content)
    except (JSONDecodeError, TypeError):
        return content


def _parse_json_dict(content):
    """
    Return `content` parsed as a JSON object (dict), or None if `content`
    is not a string containing a JSON object.
    """
    parsed = try_json_field(content)
    if isinstance(parsed, dict):
        return parsed
    return None


# Hierarchy information stored by Neo in the `comments` field of a TimeSeries
# always contains these keys
_HIERARCHY_KEYS = ("block", "segment")

# Maximum number of elements read from a dataset into an annotation
_MAX_ANNOTATION_SIZE = 10000


class _NotAnAnnotation:
    """Sentinel type marking a value that cannot be stored as a Neo annotation."""


NOT_AN_ANNOTATION = _NotAnAnnotation()


def _to_annotation_value(value, max_size=_MAX_ANNOTATION_SIZE):
    """
    Convert a value read from an NWB file to a value that can be stored as a Neo annotation.

    Datasets (h5py, including hdmf's StrDataset) are read into memory, provided they
    contain no more than `max_size` elements, and arrays of Python objects
    (e.g. variable-length strings) are converted to arrays of str.

    Returns `NOT_AN_ANNOTATION` if the value cannot be stored as an annotation,
    e.g. if it is a link to another NWB object (such as a DynamicTableRegion).
    """
    from hdmf.container import AbstractContainer

    if isinstance(value, AbstractContainer):
        return NOT_AN_ANNOTATION
    if hasattr(value, "shape") and hasattr(value, "dtype") and not isinstance(value, (np.ndarray, np.generic)):
        # a dataset that has not yet been read from the file
        size = int(np.prod(value.shape)) if value.shape else 1
        if size > max_size:
            return NOT_AN_ANNOTATION
        value = value[()]
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        try:
            value = value.astype(str)
        except (TypeError, ValueError):
            return NOT_AN_ANNOTATION
    if isinstance(value, (list, tuple)):
        converted = [_to_annotation_value(item, max_size) for item in value]
        if any(item is NOT_AN_ANNOTATION for item in converted):
            return NOT_AN_ANNOTATION
        value = type(value)(converted)
    try:
        _check_annotations(value)
    except (ValueError, TypeError):
        return NOT_AN_ANNOTATION
    return value


def _get_hierarchy(timeseries):
    """
    Return the Neo hierarchy information (block and segment names, and optionally the
    original Neo object name and the NWB comments) that Neo stores as JSON in the `comments`
    field of a TimeSeries, or None if the TimeSeries was not written by Neo.
    """
    hierarchy = _parse_json_dict(timeseries.comments)
    if hierarchy is not None and all(key in hierarchy for key in _HIERARCHY_KEYS):
        return hierarchy
    return None


def _foreign_comments(timeseries):
    """
    Return the `comments` of a TimeSeries, or None if the comments field contains
    the default value or Neo hierarchy information.
    """
    hierarchy = _get_hierarchy(timeseries)
    if hierarchy is not None:
        return hierarchy.get("comments", None)
    if timeseries.comments in (None, "", "no comments"):
        return None
    return timeseries.comments


def _annotations_equal(a, b):
    """Compare two annotation values, which may be NumPy arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    try:
        return bool(a == b)
    except ValueError:  # e.g. lists containing arrays
        return False


def _container_fields(container, exclude=()):
    """
    Return a dict containing those fields of an NWB container (e.g. an IntracellularElectrode
    or a Device) that can be stored as Neo annotations.
    """
    fields = {}
    for field_name in container.__class__.__nwbfields__:
        if field_name in exclude:
            continue
        value = getattr(container, field_name, None)
        if value is None:
            continue
        value = _to_annotation_value(value)
        if value is NOT_AN_ANNOTATION:
            logger.debug(f"Unable to store field '{field_name}' of '{container.name}' as an annotation")
        else:
            fields[field_name] = value
    return fields


def _ragged_rows(vector_index, rows):
    """
    Return the values of the given rows of a ragged column of a DynamicTable,
    as a list of arrays, given the VectorIndex for the column.

    We read the index and the data directly, since indexing a VectorIndex with an array of
    row numbers does not give the expected result in all versions of hdmf.
    """
    ends = np.asarray(vector_index.data[:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    data = vector_index.target.data
    if len(data) == 0:
        return [np.array([], dtype=str) for row in rows]
    data = np.asarray(data[:])
    if data.dtype.kind in "OS":
        data = data.astype(str)
    return [data[starts[row] : ends[row]] for row in rows]


def _init_proxy(proxy, name=None, description=None, annotations=None, array_annotations=None, n_channels=None):
    """
    Initialise the attributes of an NWB proxy object that would normally be set by
    `BaseNeo.__init__()` and `BaseProxy.__init__()`, i.e. name, description, annotations,
    array annotations and the links to parent containers.

    The NWB proxies do not call `BaseProxy.__init__()`, since it expects a RawIO.
    """
    annotations = annotations or {}
    _check_annotations(annotations)
    proxy.annotations = annotations
    proxy.name = name
    proxy.description = description
    proxy.file_origin = None
    for parent in proxy._parent_containers:
        setattr(proxy, parent, None)
    if n_channels is not None:
        proxy.array_annotations = ArrayDict(n_channels)
        if array_annotations:
            proxy.array_annotations.update(array_annotations)
    else:
        proxy.array_annotations = {}


def get_class(module, name):
    """
    Given a module path and a class name, return the class object
    """
    import pynwb

    module_path = module.split(".")
    if len(module_path) != 2:
        raise ValueError(
            f"`module_path` must be 2, not {module_path}"
        )  # todo: handle the general case where this isn't 2
    return getattr(getattr(pynwb, module_path[1]), name)


def statistics(block):  # todo: move this to be a property of Block
    """
    Return simple statistics about a Neo Block.
    """
    stats = {
        "SpikeTrain": {"count": 0},
        "AnalogSignal": {"count": 0},
        "IrregularlySampledSignal": {"count": 0},
        "Epoch": {"count": 0},
        "Event": {"count": 0},
    }
    for segment in block.segments:
        stats["SpikeTrain"]["count"] += len(segment.spiketrains)
        stats["AnalogSignal"]["count"] += len(segment.analogsignals)
        stats["IrregularlySampledSignal"]["count"] += len(segment.irregularlysampledsignals)
        stats["Epoch"]["count"] += len(segment.epochs)
        stats["Event"]["count"] += len(segment.events)
    return stats


def get_units_conversion(signal, timeseries_class):
    """
    Given a quantity array and a TimeSeries subclass, return
    the conversion factor and the expected units
    """
    # it would be nice if the expected units was an attribute of the PyNWB class
    if "CurrentClamp" in timeseries_class.__name__:
        expected_units = pq.volt
    elif "VoltageClamp" in timeseries_class.__name__:
        expected_units = pq.ampere
    else:
        # todo: warn that we don't handle this subclass yet
        expected_units = signal.units
    return float((signal.units / expected_units).simplified.magnitude), expected_units


def time_in_seconds(t):
    return float(t.rescale("second"))


def _decompose_unit(unit):
    """
    Given a quantities unit object, return a base unit name and a conversion factor.

    Example:

    >>> _decompose_unit(pq.mV)
    ('volt', 0.001)
    """
    if not isinstance(unit, pq.quantity.Quantity):
        raise TypeError(f"`unit` must be of type pq.quantity.Quantity and not type {type(unit)}")
    if unit.magnitude != 1:
        raise ValueError(f"The magnitude of the `unit` must be 1 not {unit.magnitude}")
    conversion = 1.0

    def _decompose(unit):
        dim = unit.dimensionality
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt-metre
        uq, n = list(dim.items())[0]
        if n != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt^2
        uq_def = uq.definition
        return float(uq_def.magnitude), uq_def

    conv, unit2 = _decompose(unit)
    while conv != 1:
        conversion *= conv
        unit = unit2
        conv, unit2 = _decompose(unit)
    return list(unit.dimensionality.keys())[0].name, conversion


def _recompose_unit(base_unit_name, conversion):
    """
    Given a base unit name and a conversion factor, return a quantities unit object

    Example:

    >>> _recompose_unit("ampere", 1e-9)
    UnitCurrent('nanoampere', 0.001 * uA, 'nA')

    """
    unit_name = None
    for cf in prefix_map:
        # conversion may have a different float precision to the keys in
        # prefix_map, so we can't just use `prefix_map[conversion]`
        if abs(conversion - cf) / cf < 1e-6:
            unit_name = prefix_map[cf] + base_unit_name
    if unit_name is None:
        raise ValueError(f"Can't handle this conversion factor: {conversion}")

    if unit_name[-1] == "s":  # strip trailing 's', e.g. "volts" --> "volt"
        unit_name = unit_name[:-1]
    try:
        return getattr(pq, unit_name)
    except AttributeError:
        logger.warning(f"Can't handle unit '{unit_name}'. Returning dimensionless")
        return pq.dimensionless


class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file, and "writing" a .nwb file from Neo
    """

    supported_objects = [
        Block,
        Segment,
        AnalogSignal,
        IrregularlySampledSignal,
        SpikeTrain,
        Epoch,
        Event,
        ImageSequence,
    ]
    readable_objects = supported_objects
    writeable_objects = supported_objects

    has_header = False
    support_lazy = True

    name = "NeoNWB IO"
    description = "This IO reads/writes experimental data from/to an .nwb dataset"
    extensions = ["nwb"]
    mode = "file"

    is_readable = True
    is_writable = True
    is_streameable = False

    def __init__(self, filename, mode="r", **annotations):
        """
        Arguments:
            filename : the filename
        """
        import pynwb

        BaseIO.__init__(self, filename=filename)
        self.filename = filename
        self.blocks_written = 0
        self.nwb_file_mode = mode
        self._blocks = {}
        self.annotations = annotations
        self._io_nwb = None

    def read_all_blocks(self, lazy=False, **kwargs):
        """
        Load all blocks in the file.
        """
        import pynwb

        if self.nwb_file_mode not in ("r",):
            raise ValueError("`mode` at init needs to be set to 'r' to read files")
        self._io_nwb = pynwb.NWBHDF5IO(
            self.filename, mode=self.nwb_file_mode, load_namespaces=True
        )  # Open a file with NWBHDF5IO
        try:
            self._file = self._io_nwb.read()
        except ValueError:
            self.logger.error("Unable to read this version of NWB file. Please convert to a later NWB format.")
            raise

        self.global_block_metadata = {}
        for annotation_name in GLOBAL_ANNOTATIONS:
            value = getattr(self._file, annotation_name, None)
            if value is not None:
                if annotation_name in POSSIBLE_JSON_FIELDS:
                    value = try_json_field(value)
                value = _to_annotation_value(value)
                if value is NOT_AN_ANNOTATION:
                    self.logger.debug(f"Unable to store NWB file attribute '{annotation_name}' as an annotation")
                    continue
                self.global_block_metadata[annotation_name] = value
        if "session_description" in self.global_block_metadata:
            self.global_block_metadata["description"] = self.global_block_metadata["session_description"]
        self.global_block_metadata["file_origin"] = self.filename
        if "session_start_time" in self.global_block_metadata:
            self.global_block_metadata["rec_datetime"] = self.global_block_metadata["session_start_time"]
        if "file_create_date" in self.global_block_metadata:
            self.global_block_metadata["file_datetime"] = self.global_block_metadata["rec_datetime"]

        self._blocks = {}
        self._read_acquisition_group(lazy=lazy)
        self._read_stimulus_group(lazy)
        self._read_units(lazy=lazy)
        self._read_epochs_group(lazy)

        if not self._blocks:
            # No data objects could be read, but we still return the file metadata
            self._blocks["default"] = Block(name="default", **self.global_block_metadata)
        return list(self._blocks.values())

    def read_block(self, lazy=False, block_index=0, **kargs):
        """
        Load the first block in the file.
        """
        return self.read_all_blocks(lazy=lazy)[block_index]

    def _get_segment(self, block_name, segment_name):
        # If we've already created a Block with the given name return it,
        #   otherwise create it now and store it in self._blocks.
        # If we've already created a Segment in the given block, return it,
        #   otherwise create it now and return it.
        if block_name in self._blocks:
            block = self._blocks[block_name]
        else:
            block = Block(name=block_name, **self.global_block_metadata)
            self._blocks[block_name] = block
        segment = None
        for seg in block.segments:
            if segment_name == seg.name:
                segment = seg
                break
        if segment is None:
            segment = Segment(name=segment_name)
            block.segments.append(segment)
        return segment

    def _read_epochs_group(self, lazy):
        if self._file.epochs is not None:
            try:
                # NWB files created by Neo store the segment, block and epoch names as extra
                # columns
                segment_names = self._file.epochs.segment[:]
                block_names = self._file.epochs.block[:]
                epoch_names = self._file.epochs._name[:]
            except AttributeError:
                epoch_names = None

            if epoch_names is not None:
                epoch_names = np.asarray(epoch_names)
                segment_names = np.asarray(segment_names)
                block_names = np.asarray(block_names)
                unique_epoch_names = np.unique(epoch_names)
                for epoch_name in unique_epoch_names:
                    (index,) = np.where((epoch_names == epoch_name))
                    epoch = EpochProxy(self._file.epochs, epoch_name, index)
                    if not lazy:
                        epoch = epoch.load()
                    segment_name = np.unique(segment_names[index])
                    block_name = np.unique(block_names[index])
                    if segment_name.size != block_name.size == 1:
                        raise ValueError("the `segment_name` and the `block_name` should be the same")
                    segment = self._get_segment(block_name[0], segment_name[0])
                    segment.epochs.append(epoch)
            else:
                epoch = EpochProxy(self._file.epochs)
                if not lazy:
                    epoch = epoch.load()
                segment = self._get_segment("default", "default")
                segment.epochs.append(epoch)

    def _read_timeseries_group(self, group_name, lazy):
        import pynwb

        group = getattr(self._file, group_name)
        for timeseries in group.values():
            if not isinstance(timeseries, pynwb.TimeSeries):
                # todo: handle containers such as LFP, Position, BehavioralEvents,
                #       and neurodata types from extensions
                self.logger.info(
                    f"Skipping '{timeseries.name}' in '{group_name}': "
                    f"reading {timeseries.__class__.__name__} objects is not yet supported"
                )
                continue
            if isinstance(timeseries, pynwb.image.ImageSeries) or len(timeseries.data.shape) > 2:
                # todo: read ImageSeries as ImageSequence
                self.logger.info(
                    f"Skipping '{timeseries.name}' in '{group_name}': "
                    f"reading {timeseries.__class__.__name__} objects is not yet supported"
                )
                continue
            # NWB files created by Neo store the segment and block names in the comments field.
            # For NWB files created with other applications, we put everything in a single
            # segment in a single block
            # todo: investigate whether there is a reliable way to create multiple segments,
            #       e.g. using Trial information
            hierarchy = _get_hierarchy(timeseries)
            if hierarchy is None:
                block_name = "default"
                segment_name = "default"
            else:
                block_name = hierarchy["block"]
                segment_name = hierarchy["segment"]
            segment = self._get_segment(block_name, segment_name)
            if isinstance(timeseries, pynwb.misc.AnnotationSeries):
                event = EventProxy(timeseries, group_name)
                if not lazy:
                    event = event.load()
                segment.events.append(event)
            elif timeseries.rate:  # AnalogSignal
                signal = AnalogSignalProxy(timeseries, group_name)
                if not lazy:
                    signal = signal.load()
                segment.analogsignals.append(signal)
            else:  # IrregularlySampledSignal
                signal = AnalogSignalProxy(timeseries, group_name)
                if not lazy:
                    signal = signal.load()
                segment.irregularlysampledsignals.append(signal)

    def _read_units(self, lazy):
        if self._file.units is not None and "spike_times" not in self._file.units.colnames:
            # e.g. units tables in some ophys files, which hold only cell metadata
            self.logger.info("Skipping the units table, since it does not contain spike times")
        elif self._file.units:
            # Used for units without observation intervals: read the flattened spike times
            # of all units once, rather than once per unit.
            all_spike_times = self._file.units["spike_times"].target.data[:]
            if len(all_spike_times) > 0:
                default_t_start = min(0.0, float(np.min(all_spike_times)))
                default_t_stop = float(np.max(all_spike_times))
            else:
                default_t_start = default_t_stop = 0.0
            for id in range(len(self._file.units)):
                try:
                    # NWB files created by Neo store the segment and block names as extra columns
                    segment_name = self._file.units.segment[id]
                    block_name = self._file.units.block[id]
                except AttributeError:
                    # For NWB files created with other applications, we put everything in a single
                    # segment in a single block
                    segment_name = "default"
                    block_name = "default"
                segment = self._get_segment(block_name, segment_name)
                spiketrain = SpikeTrainProxy(
                    self._file.units, id, default_t_start=default_t_start, default_t_stop=default_t_stop
                )
                if not lazy:
                    spiketrain = spiketrain.load()
                segment.spiketrains.append(spiketrain)

    def _read_acquisition_group(self, lazy):
        self._read_timeseries_group("acquisition", lazy)

    def _read_stimulus_group(self, lazy):
        self._read_timeseries_group("stimulus", lazy)

    def _build_global_annotations(self, blocks):
        annotations = {}
        for annotation_name in GLOBAL_ANNOTATIONS:
            if annotation_name in self.annotations:
                annotations[annotation_name] = self.annotations[annotation_name]
            else:
                # values may be unhashable (e.g. arrays of keywords), so we can't use a set
                values = []
                for block in blocks:
                    if annotation_name in block.annotations:
                        value = block.annotations[annotation_name]
                        if annotation_name in POSSIBLE_JSON_FIELDS and isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        if not any(_annotations_equal(value, other) for other in values):
                            values.append(value)
                if len(values) > 1:
                    raise NotImplementedError(f"We don't yet support multiple values for {annotation_name}")
                if values:
                    annotations[annotation_name] = values[0]

        if "identifier" not in annotations:
            annotations["identifier"] = str(self.filename)
        if "session_description" not in annotations:
            annotations["session_description"] = blocks[0].description or str(self.filename)
            # need to use str() here because self.filename may be a pathlib path object
            # todo: concatenate descriptions of multiple blocks if different
        if annotations.get("session_start_time", None) is None:
            if "rec_datetime" in annotations:
                annotations["session_start_time"] = annotations["rec_datetime"]
            else:
                raise Exception("Writing to NWB requires an annotation 'session_start_time'")
        return annotations

    def write_all_blocks(self, blocks, validate=True, **kwargs):
        """
        Write list of blocks to the file
        """
        import pynwb

        if len(blocks) == 0:
            raise ValueError("No blocks to write")
        # names given to TimeSeries in the NWB file, which must be unique
        self._nwb_names = set()
        self._dropped_fields = set()
        global_annotations = self._build_global_annotations(blocks)
        self._nwbfile = pynwb.NWBFile(**global_annotations)

        if sum(statistics(block)["SpikeTrain"]["count"] for block in blocks) > 0:
            self._nwbfile.add_unit_column("_name", "the name attribute of the SpikeTrain")
            self._nwbfile.add_unit_column("_description", "the description attribute of the SpikeTrain")
            self._nwbfile.add_unit_column("segment", "the name of the Neo Segment to which the SpikeTrain belongs")
            self._nwbfile.add_unit_column("block", "the name of the Neo Block to which the SpikeTrain belongs")

        if sum(statistics(block)["Epoch"]["count"] for block in blocks) > 0:
            self._nwbfile.add_epoch_column("_name", "the name attribute of the Epoch")
            # nwbfile.add_epoch_column('_description', 'the description attribute of the Epoch')
            self._nwbfile.add_epoch_column("segment", "the name of the Neo Segment to which the Epoch belongs")
            self._nwbfile.add_epoch_column("block", "the name of the Neo Block to which the Epoch belongs")

        for i, block in enumerate(blocks):
            self._write_block(block)

        if self.nwb_file_mode not in ("w",):
            raise ValueError("mode must be 'w' in order to write files")  # possibly expand to 'a'ppend later
        if self.nwb_file_mode == "w" and os.path.exists(self.filename):
            os.remove(self.filename)
        io_nwb = pynwb.NWBHDF5IO(self.filename, mode=self.nwb_file_mode)
        io_nwb.write(self._nwbfile)
        io_nwb.close()

        if validate:
            self.validate_file()

    def validate_file(self):
        import pynwb

        with pynwb.NWBHDF5IO(self.filename, "r") as io_validate:
            errors = pynwb.validate(io_validate, namespace="core")
            if errors:
                raise Exception(f"Errors found when validating {self.filename}")

    def write_block(self, block, **kwargs):
        """
        Write a single Block to the file
            :param block: Block to be written
        """
        return self.write_all_blocks([block], **kwargs)

    def _write_block(self, block):
        """
        Write a Block to the file
            :param block: Block to be written
        """
        electrodes = self._write_electrodes(self._nwbfile, block)
        if not block.name:
            block.name = f"block{self.blocks_written}"
        for i, segment in enumerate(block.segments):
            if segment.block is not block:
                raise TypeError(f"segment.block must be block it is {segment.block}")
            if not segment.name:
                segment.name = f"{block.name}  segment{i}"
            self._write_segment(self._nwbfile, segment, electrodes)
        self.blocks_written += 1

    def _write_electrodes(self, nwbfile, block):
        # this handles only icephys_electrode for now
        electrodes = {}
        devices = {}
        for segment in block.segments:
            for signal in chain(segment.analogsignals, segment.irregularlysampledsignals):
                if "nwb_electrode" in signal.annotations:
                    elec_meta = signal.annotations["nwb_electrode"].copy()
                    if elec_meta["name"] not in electrodes:
                        # todo: check for consistency if the name is already there
                        if elec_meta["device"]["name"] in devices:
                            device = devices[elec_meta["device"]["name"]]
                        else:
                            device = self._nwbfile.create_device(**elec_meta["device"])
                            devices[elec_meta["device"]["name"]] = device
                        elec_meta.pop("device")
                        electrodes[elec_meta["name"]] = self._nwbfile.create_icephys_electrode(
                            device=device, **elec_meta
                        )
        return electrodes

    def _write_segment(self, nwbfile, segment, electrodes):
        # maybe use NWB trials to store Segment metadata?
        # Note that we do not modify the names of the Neo objects. Where necessary, a different,
        # unique name is used in the NWB file, and the original Neo name is stored in the
        # hierarchy information in the TimeSeries comments field.
        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            if signal.segment is not segment:
                raise TypeError(f"signal.segment must be segment and is {signal.segment}")
            nwb_name = self._unique_nwb_name(signal.name, f"{segment.name} analogsignal{i}")
            self._write_signal(self._nwbfile, signal, electrodes, nwb_name)

        for i, train in enumerate(segment.spiketrains):
            if train.segment is not segment:
                raise TypeError(f"train.segment must be segment and is {train.segment}")
            # an empty string in the _name column represents a SpikeTrain without a name
            self._write_spiketrain(self._nwbfile, train, train.name or "")

        for i, event in enumerate(segment.events):
            if event.segment is not segment:
                raise TypeError(f"event.segment must be segment and is {event.segment}")
            nwb_name = self._unique_nwb_name(event.name, f"{segment.name} event{i}")
            self._write_event(self._nwbfile, event, nwb_name)

        for i, epoch in enumerate(segment.epochs):
            # the _name column is used to group rows of the epochs table into Epochs when reading,
            # so Epochs without a name need a unique generated name
            self._write_epoch(self._nwbfile, epoch, epoch.name or f"{segment.name} epoch{i}")

    def _unique_nwb_name(self, name, default):
        """
        Return a name for a TimeSeries that is unique within the NWB file being written.

        Uses `name` if it is not None and has not already been used,
        otherwise `default`, followed by a numerical suffix if necessary.
        """
        candidate = name or default
        if candidate in self._nwb_names:
            base = f"{default} {name}" if name else default
            candidate = base
            n = 1
            while candidate in self._nwb_names:
                candidate = f"{base} ({n})"
                n += 1
        self._nwb_names.add(candidate)
        return candidate

    @staticmethod
    def _accepted_args(timeseries_class):
        """Return the names of the arguments accepted by the constructor of an NWB class."""
        from hdmf.utils import get_docval

        return {arg["name"] for arg in get_docval(timeseries_class.__init__)}

    @staticmethod
    def _missing_required_args(timeseries_class, provided):
        """
        Return the names of the required arguments of the constructor of an NWB class
        that are not in `provided`.
        """
        from hdmf.utils import get_docval

        return [
            arg["name"]
            for arg in get_docval(timeseries_class.__init__)
            if "default" not in arg and arg["name"] not in provided
        ]

    def _filter_nwb_kwargs(self, timeseries_class, kwargs):
        """
        Remove arguments not accepted by the constructor of `timeseries_class`.

        Values taken from `nwb:` annotations may include fields that cannot be set
        when constructing the target class (e.g. fields that have fixed values for IZeroClampSeries),
        or that belong to a different class.
        """
        accepted = self._accepted_args(timeseries_class)
        dropped = sorted(set(kwargs) - accepted)
        for field_name in dropped:
            kwargs.pop(field_name)
            key = (timeseries_class.__name__, field_name)
            if key not in self._dropped_fields:
                self._dropped_fields.add(key)
                self.logger.warning(
                    f"Annotation 'nwb:{field_name}' cannot be written to a {timeseries_class.__name__}, ignoring it"
                )
        return kwargs

    def _hierarchy_json(self, obj, segment, nwb_name):
        """
        Return the JSON string that Neo stores in the `comments` field of a TimeSeries,
        recording the names of the Block and Segment the object belongs to,
        the Neo name of the object if this differs from the name used in the NWB file,
        and any NWB comments.
        """
        hierarchy = {"block": segment.block.name, "segment": segment.name}
        if obj.name != nwb_name:
            hierarchy["name"] = obj.name
        if "nwb:comments" in obj.annotations:
            hierarchy["comments"] = obj.annotations["nwb:comments"]
        return json.dumps(hierarchy)

    def _write_signal(self, nwbfile, signal, electrodes, nwb_name):
        import pynwb

        comments = self._hierarchy_json(signal, signal.segment, nwb_name)
        timeseries_class = pynwb.TimeSeries  # default
        if "nwb_neurodata_type" in signal.annotations:
            try:
                timeseries_class = get_class(*signal.annotations["nwb_neurodata_type"])
            except (ValueError, AttributeError, ImportError) as err:
                self.logger.warning(
                    f"Unable to find the NWB class {signal.annotations['nwb_neurodata_type']} ({err}), "
                    f"writing signal '{nwb_name}' as a TimeSeries"
                )
        additional_metadata = {
            name[4:]: value
            for name, value in signal.annotations.items()
            if name.startswith("nwb:") and name[4:] not in WRITE_HANDLED_FIELDS
        }
        if "nwb_electrode" in signal.annotations:
            electrode_name = signal.annotations["nwb_electrode"]["name"]
            additional_metadata["electrode"] = electrodes[electrode_name]
        missing = self._missing_required_args(
            timeseries_class, set(additional_metadata) | {"name", "data", "unit", "rate", "starting_time", "timestamps"}
        )
        if missing:
            # todo: write links to other NWB objects, e.g. the electrodes of an ElectricalSeries
            self.logger.warning(
                f"Unable to write signal '{nwb_name}' as a {timeseries_class.__name__}, "
                f"since the following information is missing: {', '.join(missing)}. Writing it as a TimeSeries."
            )
            timeseries_class = pynwb.TimeSeries
        if timeseries_class != pynwb.TimeSeries:
            conversion, units = get_units_conversion(signal, timeseries_class)
            additional_metadata["conversion"] = conversion
        else:
            units = signal.units
        description = signal.description
        if description is None and isinstance(signal.annotations.get("notes", None), dict):
            description = json.dumps(signal.annotations["notes"])
        if description is not None:
            additional_metadata["description"] = description
        additional_metadata = self._filter_nwb_kwargs(timeseries_class, additional_metadata)
        if hasattr(signal, "proxy_for") and signal.proxy_for in [AnalogSignal, IrregularlySampledSignal]:
            signal = signal.load()
        if issubclass(timeseries_class, pynwb.icephys.PatchClampSeries):
            if signal.shape[1] != 1:
                raise ValueError(
                    "To store patch clamp data in NWB, please ensure that each AnalogSignal"
                    f"contains only one channel. The current signal has {signal.shape[1]} channels."
                )
            # see https://github.com/NeurodataWithoutBorders/pynwb/issues/1300
            data = signal.ravel()  # convert to 1D
        else:
            data = signal
        kwargs = dict(name=nwb_name, data=data, unit=units.dimensionality.string, comments=comments)
        if isinstance(signal, AnalogSignal):
            kwargs["starting_time"] = time_in_seconds(signal.t_start)
            kwargs["rate"] = float(signal.sampling_rate.rescale("Hz"))
            # todo: try to add array_annotations via "control" attribute
        elif isinstance(signal, IrregularlySampledSignal):
            kwargs["timestamps"] = signal.times.rescale("second").magnitude
        else:
            raise TypeError(
                f"signal has type { signal.__class__.__name__}, should be AnalogSignal or IrregularlySampledSignal"
            )
        if "unit" not in self._accepted_args(timeseries_class):
            # some classes have fixed units, e.g. ElectricalSeries (volts)
            kwargs.pop("unit")
        tS = timeseries_class(**kwargs, **additional_metadata)
        nwb_group = signal.annotations.get("nwb_group", "acquisition")
        add_method_map = {"acquisition": self._nwbfile.add_acquisition, "stimulus": self._nwbfile.add_stimulus}
        if nwb_group in add_method_map:
            add_time_series = add_method_map[nwb_group]
        else:
            raise NotImplementedError(f"NWB group '{nwb_group}' not yet supported")
        add_time_series(tS)
        return tS

    def _write_spiketrain(self, nwbfile, spiketrain, name):
        segment = spiketrain.segment
        if hasattr(spiketrain, "proxy_for") and spiketrain.proxy_for is SpikeTrain:
            spiketrain = spiketrain.load()
        if "nwb:obs_intervals" in spiketrain.annotations:
            obs_intervals = np.asarray(spiketrain.annotations["nwb:obs_intervals"], dtype=float)
        else:
            obs_intervals = [[float(spiketrain.t_start.rescale("s")), float(spiketrain.t_stop.rescale("s"))]]
        self._nwbfile.add_unit(
            spike_times=spiketrain.rescale("s").magnitude,
            obs_intervals=obs_intervals,
            _name=name,
            # an empty string represents a SpikeTrain without a description
            _description=spiketrain.description or "",
            segment=segment.name,
            block=segment.block.name,
        )
        # todo: handle annotations (using add_unit_column()?)
        # todo: handle Neo Units
        # todo: handle spike waveforms, if any (see SpikeEventSeries)
        return self._nwbfile.units

    def _write_event(self, nwbfile, event, nwb_name):
        import pynwb

        segment = event.segment
        comments = self._hierarchy_json(event, segment, nwb_name)
        if hasattr(event, "proxy_for") and event.proxy_for == Event:
            event = event.load()
        description = event.description
        if description is None and isinstance(event.annotations.get("notes", None), dict):
            description = json.dumps(event.annotations["notes"])
        tS_evt = pynwb.misc.AnnotationSeries(
            name=nwb_name,
            data=event.labels,
            timestamps=event.times.rescale("second").magnitude,
            description=description or "",
            comments=comments,
        )
        self._nwbfile.add_acquisition(tS_evt)
        return tS_evt

    def _write_epoch(self, nwbfile, epoch, name):
        segment = epoch.segment
        if hasattr(epoch, "proxy_for") and epoch.proxy_for == Epoch:
            epoch = epoch.load()
        labels = epoch.labels
        if len(labels) != len(epoch):
            # Neo allows Epochs without labels
            labels = [""] * len(epoch)
        for t_start, duration, label in zip(
            epoch.rescale("s").magnitude, epoch.durations.rescale("s").magnitude, labels
        ):
            self._nwbfile.add_epoch(
                start_time=t_start,
                stop_time=t_start + duration,
                tags=[label] if label else [],
                _name=name,
                segment=segment.name,
                block=segment.block.name,
            )
        return self._nwbfile.epochs

    def close(self):
        if self._io_nwb:
            self._io_nwb.close()


class AnalogSignalProxy(BaseAnalogSignalProxy):
    common_metadata_fields = (
        # fields that are the same for all TimeSeries subclasses
        "comments",
        "description",
        "unit",
        "starting_time",
        "timestamps",
        "rate",
        "data",
        "starting_time_unit",
        "timestamps_unit",
        "electrode",
        "stream_id",
        # scaling fields, which are applied to the data when loading
        "conversion",
        "offset",
        "channel_conversion",
    )

    def __init__(self, timeseries, nwb_group):
        self._timeseries = timeseries
        self.shape = self._timeseries.data.shape
        if len(self.shape) == 1:
            self.shape = (self.shape[0], 1)
        self._set_scaling(timeseries)
        if timeseries.starting_time is not None:
            self.t_start = timeseries.starting_time * pq.s
        else:
            self.t_start = timeseries.timestamps[0] * pq.s
        if timeseries.rate:
            self.sampling_rate = timeseries.rate * pq.Hz
        else:
            self.sampling_rate = None
            # this proxy class is used for both regularly and irregularly sampled signals
            self.proxy_for = IrregularlySampledSignal

        annotations = {"nwb_group": nwb_group}
        description = timeseries.description
        notes = _parse_json_dict(description)
        if notes is not None:
            annotations["notes"] = notes
            description = None
        comments = _foreign_comments(timeseries)
        if comments is not None:
            annotations["nwb:comments"] = comments
        metadata_fields = [
            field_name
            for field_name in timeseries.__nwbfields__
            if field_name not in self.__class__.common_metadata_fields
        ]
        for field_name in metadata_fields:
            value = getattr(timeseries, field_name)
            if value is not None:
                value = _to_annotation_value(value)
                if value is NOT_AN_ANNOTATION:
                    logger.debug(f"Unable to store field '{field_name}' of '{timeseries.name}' as an annotation")
                else:
                    annotations[f"nwb:{field_name}"] = value
        annotations["nwb_neurodata_type"] = (timeseries.__class__.__module__, timeseries.__class__.__name__)
        if getattr(timeseries, "electrode", None) is not None:
            # todo: once the Group class is available, we could add electrode metadata
            #       to a Group containing all signals that share that electrode
            #       This would reduce the amount of redundancy (repeated metadata in every signal)
            electrode = timeseries.electrode
            electrode_metadata = _container_fields(electrode, exclude=("device",))
            electrode_metadata["name"] = electrode.name
            electrode_metadata["device"] = _container_fields(electrode.device)
            annotations["nwb_electrode"] = electrode_metadata

        name = timeseries.name
        hierarchy = _get_hierarchy(timeseries)
        if hierarchy is not None and "name" in hierarchy:
            name = hierarchy["name"]
        _init_proxy(self, name=name, description=description, annotations=annotations, n_channels=self.shape[-1])

    def _set_scaling(self, timeseries):
        """
        Determine the units of the signal, and the gain and offset to be applied to the data
        when loading, from the `unit`, `conversion`, `offset` and (for ElectricalSeries)
        `channel_conversion` fields of the TimeSeries.

        In NWB, the value in `unit` is data * channel_conversion * conversion + offset.
        Where possible, the conversion factor is folded into the units (e.g. a conversion
        of 1e-3 with unit "volts" gives units of mV), so that no scaling of the data is needed.
        """
        conversion = timeseries.conversion if timeseries.conversion else 1.0
        offset = getattr(timeseries, "offset", None) or 0.0
        try:
            self.units = _recompose_unit(timeseries.unit, conversion)
            gain = 1.0
            unit_conversion = conversion
        except ValueError:
            # conversion factor is not a power of ten, so we apply it to the data
            self.units = _recompose_unit(timeseries.unit, 1.0)
            gain = conversion
            unit_conversion = 1.0
        channel_conversion = getattr(timeseries, "channel_conversion", None)
        if channel_conversion is not None:
            channel_conversion = np.asarray(channel_conversion[:], dtype=float)
            if channel_conversion.shape == (self.shape[-1],):
                gain = gain * channel_conversion
            else:
                logger.warning(
                    f"Ignoring channel_conversion of '{timeseries.name}': it has {channel_conversion.size} values "
                    f"but the data have {self.shape[-1]} channels"
                )
        self._gain = gain
        self._offset = offset / unit_conversion

    def _scale(self, data):
        """Apply the gain and offset determined by `_set_scaling()` to the data."""
        if np.all(np.asarray(self._gain) == 1.0) and self._offset == 0.0:
            return data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data * self._gain + self._offset

    def load(self, time_slice=None, strict_slicing=True):
        """
        Load AnalogSignalProxy args:
            :param time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :param strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        i_start, i_stop, sig_t_start = None, None, self.t_start
        if time_slice:
            if self.sampling_rate is None:
                i_start, i_stop = np.searchsorted(self._timeseries.timestamps, time_slice)
            else:
                i_start, i_stop, sig_t_start = self._time_slice_indices(time_slice, strict_slicing=strict_slicing)
        signal = self._scale(self._timeseries.data[i_start:i_stop])
        if self.sampling_rate is None:
            return IrregularlySampledSignal(
                self._timeseries.timestamps[i_start:i_stop] * pq.s,
                signal,
                units=self.units,
                t_start=sig_t_start,
                sampling_rate=self.sampling_rate,
                name=self.name,
                description=self.description,
                array_annotations=None,
                **self.annotations,
            )  # todo: timeseries.control / control_description

        else:
            return AnalogSignal(
                signal,
                units=self.units,
                t_start=sig_t_start,
                sampling_rate=self.sampling_rate,
                name=self.name,
                description=self.description,
                array_annotations=None,
                **self.annotations,
            )  # todo: timeseries.control / control_description


class EventProxy(BaseEventProxy):

    def __init__(self, timeseries, nwb_group):
        self._timeseries = timeseries
        self.shape = self._timeseries.data.shape
        annotations = {"nwb_group": nwb_group}
        description = timeseries.description
        notes = _parse_json_dict(description)
        if notes is not None:
            annotations["notes"] = notes
            description = None
        comments = _foreign_comments(timeseries)
        if comments is not None:
            annotations["nwb:comments"] = comments
        name = timeseries.name
        hierarchy = _get_hierarchy(timeseries)
        if hierarchy is not None and "name" in hierarchy:
            name = hierarchy["name"]
        _init_proxy(self, name=name, description=description, annotations=annotations, n_channels=self.shape[-1])

    def load(self, time_slice=None, strict_slicing=True):
        """
        Load EventProxy args:
            :param time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :param strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        if time_slice:
            raise NotImplementedError("todo")
        else:
            times = self._timeseries.timestamps[:]
            labels = self._timeseries.data[:]
        return Event(times * pq.s, labels=labels, name=self.name, description=self.description, **self.annotations)


class EpochProxy(BaseEpochProxy):

    def __init__(self, time_intervals, epoch_name=None, index=None):
        """
        :param time_intervals: An epochs table,
            which is a specific TimeIntervals table that stores info about long periods
        :param epoch_name: (str)
            Name of the epoch object
        :param index: (np.array, slice)
            Array of integer indices or of bool values, or slice object, selecting the rows of
            time_intervals to be used. In the case of an array of bool values,
            it has to have the same shape as `time_intervals`.
        """
        self._time_intervals = time_intervals
        if index is not None:
            self._index = index
            if isinstance(index, slice):
                self.shape = (len(range(*index.indices(len(time_intervals)))),)
            elif np.asarray(index).dtype == bool:
                self.shape = (int(np.sum(index)),)
            else:
                self.shape = (len(index),)
        else:
            self._index = slice(None)
            self.shape = (len(time_intervals),)
        _init_proxy(self, name=epoch_name, n_channels=self.shape[-1])

    def load(self, time_slice=None, strict_slicing=True):
        """
        Load EpochProxy args:
            :param time_slice: None or tuple of the time slice expressed with quantities.
                            None is all of the intervals.
            :param strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        if time_slice:
            raise NotImplementedError("todo")
        else:
            start_times = np.asarray(self._time_intervals.start_time[self._index])
            stop_times = np.asarray(self._time_intervals.stop_time[self._index])
            durations = stop_times - start_times
            if "tags" in self._time_intervals.colnames:
                # each row may have zero, one or several tags; Neo has a single label per epoch
                rows = np.arange(len(self._time_intervals))[self._index]
                tags = _ragged_rows(self._time_intervals["tags"], rows)
                labels = np.array([", ".join(str(tag) for tag in row_tags) for row_tags in tags], dtype=str)
            else:
                labels = None

        return Epoch(times=start_times * pq.s, durations=durations * pq.s, labels=labels, name=self.name)


class SpikeTrainProxy(BaseSpikeTrainProxy):

    def __init__(self, units_table, id, default_t_start=0.0, default_t_stop=0.0):
        """
        :param units_table: A Units table
        (see https://pynwb.readthedocs.io/en/stable/pynwb.misc.html#pynwb.misc.Units)
        :param id: the cell/unit ID (integer)
        :param default_t_start: start time (in seconds) to use if the unit has no observation intervals
        :param default_t_stop: stop time (in seconds) to use if the unit has no observation intervals
        """
        self._units_table = units_table
        self.id = id
        self.units = pq.s
        annotations = {"nwb_group": "acquisition"}
        if "obs_intervals" in units_table.colnames:
            obs_intervals = np.asarray(units_table.get_unit_obs_intervals(id), dtype=float)
        else:
            obs_intervals = np.empty((0, 2))
        if len(obs_intervals) == 0:
            t_start, t_stop = default_t_start, default_t_stop
        else:
            t_start, t_stop = obs_intervals[:, 0].min(), obs_intervals[:, 1].max()
            if len(obs_intervals) > 1:
                annotations["nwb:obs_intervals"] = obs_intervals
        self.t_start = t_start * pq.s
        self.t_stop = t_stop * pq.s
        if "_name" in units_table.colnames:
            # NWB files created by Neo store the name as an extra column
            name = units_table["_name"][id] or None
        else:
            name = None
        if "_description" in units_table.colnames:
            # NWB files created by recent versions of Neo also store the description
            description = units_table["_description"][id] or None
        else:
            description = None
        self.shape = None  # no way to get this without reading the data
        _init_proxy(self, name=name, description=description, annotations=annotations)

    def load(self, time_slice=None, strict_slicing=True):
        """
        Load SpikeTrainProxy args:
            :param time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire spike train.
            :param strict_slicing: True by default.
                Control if an error is raised or not when one of the time_slice members
                (t_start or t_stop) is outside the real time range of the segment.
        """
        interval = None
        t_start, t_stop = self.t_start, self.t_stop
        if time_slice:
            interval = tuple(float(t.rescale(pq.s)) for t in time_slice)
            t_start = max(t_start, time_slice[0])
            t_stop = min(t_stop, time_slice[1])
        spike_times = np.asarray(self._units_table.get_unit_spike_times(self.id, in_interval=interval))
        if spike_times.size > 0:
            # some files contain spikes outside the observation intervals
            if spike_times.min() < float(t_start.rescale(pq.s)) or spike_times.max() > float(t_stop.rescale(pq.s)):
                logger.debug(f"Spike times of unit {self.id} extend beyond its observation intervals")
                t_start = min(t_start, spike_times.min() * pq.s)
                t_stop = max(t_stop, spike_times.max() * pq.s)
        return SpikeTrain(
            spike_times * self.units,
            t_stop,
            units=self.units,
            # sampling_rate=array(1.) * Hz,
            t_start=t_start,
            # waveforms=None,
            # left_sweep=None,
            name=self.name,
            # file_origin=None,
            description=self.description,
            # array_annotations=None,
            **self.annotations,
        )
