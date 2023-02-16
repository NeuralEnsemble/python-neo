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
from collections import defaultdict
from itertools import chain
from json.decoder import JSONDecodeError

import numpy as np
import quantities as pq
from neo.core import (Segment, SpikeTrain, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, Block, ImageSequence)
from neo.io.baseio import BaseIO
from neo.io.proxyobjects import (
    AnalogSignalProxy as BaseAnalogSignalProxy,
    EventProxy as BaseEventProxy,
    EpochProxy as BaseEpochProxy,
    SpikeTrainProxy as BaseSpikeTrainProxy
)


logger = logging.getLogger("Neo")

GLOBAL_ANNOTATIONS = (
    "session_start_time", "identifier", "timestamps_reference_time", "experimenter",
    "experiment_description", "session_id", "institution", "keywords", "notes",
    "pharmacology", "protocol", "related_publications", "slices", "source_script",
    "source_script_file_name", "data_collection", "surgery", "virus", "stimulus_notes",
    "lab", "session_description", "rec_datetime",
)

POSSIBLE_JSON_FIELDS = (
    "source_script", "description"
)

prefix_map = {
    1e9: 'giga',
    1e6: 'mega',
    1e3: 'kilo',
    1: '',
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano',
    1e-12: 'pico'
}


def try_json_field(content):
    """
    Try to interpret a string as JSON data.

    If successful, return the JSON data (dict or list)
    If unsuccessful, return the original string
    """
    try:
        return json.loads(content)
    except JSONDecodeError:
        return content


def get_class(module, name):
    """
    Given a module path and a class name, return the class object
    """
    import pynwb

    module_path = module.split(".")
    assert len(module_path) == 2  # todo: handle the general case where this isn't 2
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
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
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
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event, ImageSequence]
    readable_objects = supported_objects
    writeable_objects = supported_objects

    has_header = False
    support_lazy = True

    name = 'NeoNWB IO'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    is_readable = True
    is_writable = True
    is_streameable = False

    def __init__(self, filename, mode='r', **annotations):
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

        assert self.nwb_file_mode in ('r',)
        self._io_nwb = pynwb.NWBHDF5IO(self.filename, mode=self.nwb_file_mode,
                             load_namespaces=True)  # Open a file with NWBHDF5IO
        try:
            self._file = self._io_nwb.read()
        except ValueError:
            print("Error: Unable to read this version of NWB file.")
            print("Please convert to a later NWB format.")
            raise

        self.global_block_metadata = {}
        for annotation_name in GLOBAL_ANNOTATIONS:
            value = getattr(self._file, annotation_name, None)
            if value is not None:
                if annotation_name in POSSIBLE_JSON_FIELDS:
                    value = try_json_field(value)
                self.global_block_metadata[annotation_name] = value
        if "session_description" in self.global_block_metadata:
            self.global_block_metadata["description"] = self.global_block_metadata[
                "session_description"]
        self.global_block_metadata["file_origin"] = self.filename
        if "session_start_time" in self.global_block_metadata:
            self.global_block_metadata["rec_datetime"] = self.global_block_metadata[
                "session_start_time"]
        if "file_create_date" in self.global_block_metadata:
            self.global_block_metadata["file_datetime"] = self.global_block_metadata[
                "rec_datetime"]

        self._blocks = {}
        self._read_acquisition_group(lazy=lazy)
        self._read_stimulus_group(lazy)
        self._read_units(lazy=lazy)
        self._read_epochs_group(lazy)

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
            segment.block = block
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
                unique_epoch_names = np.unique(epoch_names)
                for epoch_name in unique_epoch_names:
                    index, = np.where((epoch_names == epoch_name))
                    epoch = EpochProxy(self._file.epochs, epoch_name, index)
                    if not lazy:
                        epoch = epoch.load()
                    segment_name = np.unique(segment_names[index])
                    block_name = np.unique(block_names[index])
                    assert segment_name.size == block_name.size == 1
                    segment = self._get_segment(block_name[0], segment_name[0])
                    segment.epochs.append(epoch)
                    epoch.segment = segment
            else:
                epoch = EpochProxy(self._file.epochs)
                if not lazy:
                    epoch = epoch.load()
                segment = self._get_segment("default", "default")
                segment.epochs.append(epoch)
                epoch.segment = segment

    def _read_timeseries_group(self, group_name, lazy):
        import pynwb

        group = getattr(self._file, group_name)
        for timeseries in group.values():
            try:
                # NWB files created by Neo store the segment and block names in the comments field
                hierarchy = json.loads(timeseries.comments)
            except JSONDecodeError:
                # For NWB files created with other applications, we put everything in a single
                # segment in a single block
                # todo: investigate whether there is a reliable way to create multiple segments,
                #       e.g. using Trial information
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
                event.segment = segment
            elif timeseries.rate:  # AnalogSignal
                signal = AnalogSignalProxy(timeseries, group_name)
                if not lazy:
                    signal = signal.load()
                segment.analogsignals.append(signal)
                signal.segment = segment
            else:  # IrregularlySampledSignal
                signal = AnalogSignalProxy(timeseries, group_name)
                if not lazy:
                    signal = signal.load()
                segment.irregularlysampledsignals.append(signal)
                signal.segment = segment

    def _read_units(self, lazy):
        if self._file.units:
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
                spiketrain = SpikeTrainProxy(self._file.units, id)
                if not lazy:
                    spiketrain = spiketrain.load()
                segment.spiketrains.append(spiketrain)
                spiketrain.segment = segment

    def _read_acquisition_group(self, lazy):
        self._read_timeseries_group("acquisition", lazy)

    def _read_stimulus_group(self, lazy):
        self._read_timeseries_group("stimulus", lazy)

    def _build_global_annotations(self, blocks):
        annotations = defaultdict(set)
        for annotation_name in GLOBAL_ANNOTATIONS:
            if annotation_name in self.annotations:
                annotations[annotation_name] = self.annotations[annotation_name]
            else:
                for block in blocks:
                    if annotation_name in block.annotations:
                        try:
                            annotations[annotation_name].add(block.annotations[annotation_name])
                        except TypeError:
                            if annotation_name in POSSIBLE_JSON_FIELDS:
                                encoded = json.dumps(block.annotations[annotation_name])
                                annotations[annotation_name].add(encoded)
                            else:
                                raise
                if annotation_name in annotations:
                    if len(annotations[annotation_name]) > 1:
                        raise NotImplementedError(
                            "We don't yet support multiple values for {}".format(annotation_name))
                    # take single value from set
                    annotations[annotation_name], = annotations[annotation_name]

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

        global_annotations = self._build_global_annotations(blocks)
        self._nwbfile = pynwb.NWBFile(**global_annotations)

        if sum(statistics(block)["SpikeTrain"]["count"] for block in blocks) > 0:
            self._nwbfile.add_unit_column('_name', 'the name attribute of the SpikeTrain')
            # nwbfile.add_unit_column('_description',
            # 'the description attribute of the SpikeTrain')
            self._nwbfile.add_unit_column(
                'segment', 'the name of the Neo Segment to which the SpikeTrain belongs')
            self._nwbfile.add_unit_column(
                'block', 'the name of the Neo Block to which the SpikeTrain belongs')

        if sum(statistics(block)["Epoch"]["count"] for block in blocks) > 0:
            self._nwbfile.add_epoch_column('_name', 'the name attribute of the Epoch')
            # nwbfile.add_epoch_column('_description', 'the description attribute of the Epoch')
            self._nwbfile.add_epoch_column(
                'segment', 'the name of the Neo Segment to which the Epoch belongs')
            self._nwbfile.add_epoch_column('block',
                                     'the name of the Neo Block to which the Epoch belongs')

        for i, block in enumerate(blocks):
            self._write_block(block)

        assert self.nwb_file_mode in ('w',)  # possibly expand to 'a'ppend later
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
            block.name = "block%d" % self.blocks_written
        for i, segment in enumerate(block.segments):
            assert segment.block is block
            if not segment.name:
                segment.name = "%s : segment%d" % (block.name, i)
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
        for i, signal in enumerate(
                chain(segment.analogsignals, segment.irregularlysampledsignals)):
            assert signal.segment is segment
            if hasattr(signal, 'name'):
                signal.name = "%s %s %i" % (segment.name, signal.name, i)
                logging.warning("Warning signal name exists. New name: %s" % (signal.name))
            else:
                signal.name = "%s : analogsignal%s %i" % (segment.name, signal.name, i)
            self._write_signal(self._nwbfile, signal, electrodes)

        for i, train in enumerate(segment.spiketrains):
            assert train.segment is segment
            if not train.name:
                train.name = "%s : spiketrain%d" % (segment.name, i)
            self._write_spiketrain(self._nwbfile, train)

        for i, event in enumerate(segment.events):
            assert event.segment is segment
            if hasattr(event, 'name'):
                event.name = "%s  %s %i" % (segment.name, event.name, i)
                logging.warning("Warning event name exists. New name: %s" % (event.name))
            else:
                event.name = "%s : event%s %d" % (segment.name, event.name, i)
            self._write_event(self._nwbfile, event)

        for i, epoch in enumerate(segment.epochs):
            if not epoch.name:
                epoch.name = "%s : epoch%d" % (segment.name, i)
            self._write_epoch(self._nwbfile, epoch)

    def _write_signal(self, nwbfile, signal, electrodes):
        import pynwb

        hierarchy = {'block': signal.segment.block.name, 'segment': signal.segment.name}
        if "nwb_neurodata_type" in signal.annotations:
            timeseries_class = get_class(*signal.annotations["nwb_neurodata_type"])
        else:
            timeseries_class = pynwb.TimeSeries  # default
        additional_metadata = {name[4:]: value
                               for name, value in signal.annotations.items()
                               if name.startswith("nwb:")}
        if "nwb_electrode" in signal.annotations:
            electrode_name = signal.annotations["nwb_electrode"]["name"]
            additional_metadata["electrode"] = electrodes[electrode_name]
        if timeseries_class != pynwb.TimeSeries:
            conversion, units = get_units_conversion(signal, timeseries_class)
            additional_metadata["conversion"] = conversion
        else:
            units = signal.units
        if hasattr(signal, 'proxy_for') and signal.proxy_for in [AnalogSignal,
                                                                 IrregularlySampledSignal]:
            signal = signal.load()
        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            tS = timeseries_class(
                name=signal.name,
                starting_time=time_in_seconds(signal.t_start),
                data=signal,
                unit=units.dimensionality.string,
                rate=float(sampling_rate),
                comments=json.dumps(hierarchy),
                **additional_metadata)
            # todo: try to add array_annotations via "control" attribute
        elif isinstance(signal, IrregularlySampledSignal):
            tS = timeseries_class(
                name=signal.name,
                data=signal,
                unit=units.dimensionality.string,
                timestamps=signal.times.rescale('second').magnitude,
                comments=json.dumps(hierarchy),
                **additional_metadata)
        else:
            raise TypeError(
                "signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(
                    signal.__class__.__name__))
        nwb_group = signal.annotations.get("nwb_group", "acquisition")
        add_method_map = {
            "acquisition": self._nwbfile.add_acquisition,
            "stimulus": self._nwbfile.add_stimulus
        }
        if nwb_group in add_method_map:
            add_time_series = add_method_map[nwb_group]
        else:
            raise NotImplementedError("NWB group '{}' not yet supported".format(nwb_group))
        add_time_series(tS)
        return tS

    def _write_spiketrain(self, nwbfile, spiketrain):
        segment = spiketrain.segment
        if hasattr(spiketrain, 'proxy_for') and spiketrain.proxy_for is SpikeTrain:
            spiketrain = spiketrain.load()
        self._nwbfile.add_unit(
                         spike_times=spiketrain.rescale('s').magnitude,
                         obs_intervals=[[float(spiketrain.t_start.rescale('s')),
                                         float(spiketrain.t_stop.rescale('s'))]],
                         _name=spiketrain.name,
                         # _description=spiketrain.description,
                         segment=segment.name,
                         block=segment.block.name)
        # todo: handle annotations (using add_unit_column()?)
        # todo: handle Neo Units
        # todo: handle spike waveforms, if any (see SpikeEventSeries)
        return self._nwbfile.units

    def _write_event(self, nwbfile, event):
        import pynwb

        segment = event.segment
        if hasattr(event, 'proxy_for') and event.proxy_for == Event:
            event = event.load()
        hierarchy = {'block': segment.block.name, 'segment': segment.name}
        tS_evt = pynwb.misc.AnnotationSeries(
            name=event.name,
            data=event.labels,
            timestamps=event.times.rescale('second').magnitude,
            description=event.description or "",
            comments=json.dumps(hierarchy))
        self._nwbfile.add_acquisition(tS_evt)
        return tS_evt

    def _write_epoch(self, nwbfile, epoch):
        segment = epoch.segment
        if hasattr(epoch, 'proxy_for') and epoch.proxy_for == Epoch:
            epoch = epoch.load()
        for t_start, duration, label in zip(epoch.rescale('s').magnitude,
                                            epoch.durations.rescale('s').magnitude,
                                            epoch.labels):
            self._nwbfile.add_epoch(t_start, t_start + duration, [label], [],
                              _name=epoch.name,
                              segment=segment.name,
                              block=segment.block.name)
        return self._nwbfile.epochs

    def close(self):
        if self._io_nwb:
            self._io_nwb.close()


class AnalogSignalProxy(BaseAnalogSignalProxy):
    common_metadata_fields = (
        # fields that are the same for all TimeSeries subclasses
        "comments", "description", "unit", "starting_time", "timestamps", "rate",
        "data", "starting_time_unit", "timestamps_unit", "electrode",
        "stream_id",
    )

    def __init__(self, timeseries, nwb_group):
        self._timeseries = timeseries
        self.units = timeseries.unit
        if timeseries.conversion:
            self.units = _recompose_unit(timeseries.unit, timeseries.conversion)
        if timeseries.starting_time is not None:
            self.t_start = timeseries.starting_time * pq.s
        else:
            self.t_start = timeseries.timestamps[0] * pq.s
        if timeseries.rate:
            self.sampling_rate = timeseries.rate * pq.Hz
        else:
            self.sampling_rate = None
        self.name = timeseries.name
        self.annotations = {"nwb_group": nwb_group}
        self.description = try_json_field(timeseries.description)
        if isinstance(self.description, dict):
            self.annotations["notes"] = self.description
            if "name" in self.annotations:
                self.annotations.pop("name")
            self.description = None
        self.shape = self._timeseries.data.shape
        if len(self.shape) == 1:
            self.shape = (self.shape[0], 1)
        metadata_fields = list(timeseries.__nwbfields__)
        for field_name in self.__class__.common_metadata_fields:  # already handled
            try:
                metadata_fields.remove(field_name)
            except ValueError:
                pass
        for field_name in metadata_fields:
            value = getattr(timeseries, field_name)
            if value is not None:
                self.annotations[f"nwb:{field_name}"] = value
        self.annotations["nwb_neurodata_type"] = (
            timeseries.__class__.__module__,
            timeseries.__class__.__name__
        )
        if hasattr(timeseries, "electrode"):
            # todo: once the Group class is available, we could add electrode metadata
            #       to a Group containing all signals that share that electrode
            #       This would reduce the amount of redundancy (repeated metadata in every signal)
            electrode_metadata = {"device": {}}
            metadata_fields = list(timeseries.electrode.__class__.__nwbfields__) + ["name"]
            metadata_fields.remove("device")  # needs special handling
            for field_name in metadata_fields:
                value = getattr(timeseries.electrode, field_name)
                if value is not None:
                    electrode_metadata[field_name] = value
            for field_name in timeseries.electrode.device.__class__.__nwbfields__:
                value = getattr(timeseries.electrode.device, field_name)
                if value is not None:
                    electrode_metadata["device"][field_name] = value
            self.annotations["nwb_electrode"] = electrode_metadata

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
                i_start, i_stop, sig_t_start = self._time_slice_indices(
                    time_slice, strict_slicing=strict_slicing)
        signal = self._timeseries.data[i_start: i_stop]
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
                **self.annotations)  # todo: timeseries.control / control_description

        else:
            return AnalogSignal(
                signal,
                units=self.units,
                t_start=sig_t_start,
                sampling_rate=self.sampling_rate,
                name=self.name,
                description=self.description,
                array_annotations=None,
                **self.annotations)  # todo: timeseries.control / control_description


class EventProxy(BaseEventProxy):

    def __init__(self, timeseries, nwb_group):
        self._timeseries = timeseries
        self.name = timeseries.name
        self.annotations = {"nwb_group": nwb_group}
        self.description = try_json_field(timeseries.description)
        if isinstance(self.description, dict):
            self.annotations.update(self.description)
            self.description = None
        self.shape = self._timeseries.data.shape

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
        return Event(times * pq.s,
                     labels=labels,
                     name=self.name,
                     description=self.description,
                     **self.annotations)


class EpochProxy(BaseEpochProxy):

    def __init__(self, time_intervals, epoch_name=None, index=None):
        """
            :param time_intervals: An epochs table,
                which is a specific TimeIntervals table that stores info about long periods
            :param epoch_name: (str)
                Name of the epoch object
            :param index: (np.array, slice)
                Slice object or array of bool values masking time_intervals to be used. In case of
                an array it has to have the same shape as `time_intervals`.
        """
        self._time_intervals = time_intervals
        if index is not None:
            self._index = index
            self.shape = (index.sum(),)
        else:
            self._index = slice(None)
            self.shape = (len(time_intervals),)
        self.name = epoch_name

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
            start_times = self._time_intervals.start_time[self._index]
            stop_times = self._time_intervals.stop_time[self._index]
            durations = stop_times - start_times
            labels = self._time_intervals.tags[self._index]

        return Epoch(times=start_times * pq.s,
                     durations=durations * pq.s,
                     labels=labels,
                     name=self.name)


class SpikeTrainProxy(BaseSpikeTrainProxy):

    def __init__(self, units_table, id):
        """
            :param units_table: A Units table
            (see https://pynwb.readthedocs.io/en/stable/pynwb.misc.html#pynwb.misc.Units)
            :param id: the cell/unit ID (integer)
        """
        self._units_table = units_table
        self.id = id
        self.units = pq.s
        obs_intervals = units_table.get_unit_obs_intervals(id)
        if len(obs_intervals) == 0:
            t_start, t_stop = None, None
        elif len(obs_intervals) == 1:
            t_start, t_stop = obs_intervals[0]
        else:
            raise NotImplementedError("Can't yet handle multiple observation intervals")
        self.t_start = t_start * pq.s
        self.t_stop = t_stop * pq.s
        self.annotations = {"nwb_group": "acquisition"}
        try:
            # NWB files created by Neo store the name as an extra column
            self.name = units_table._name[id]
        except AttributeError:
            self.name = None
        self.shape = None  # no way to get this without reading the data

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
        if time_slice:
            interval = (float(t) for t in time_slice)  # convert from quantities
        spike_times = self._units_table.get_unit_spike_times(self.id, in_interval=interval)
        return SpikeTrain(
            spike_times * self.units,
            self.t_stop,
            units=self.units,
            # sampling_rate=array(1.) * Hz,
            t_start=self.t_start,
            # waveforms=None,
            # left_sweep=None,
            name=self.name,
            # file_origin=None,
            # description=None,
            # array_annotations=None,
            **self.annotations)
