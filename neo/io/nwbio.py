"""
NWBIO
========

IO class for reading data from a Neurodata Without Borders (NWB) dataset

Documentation : https://neurodatawithoutborders.github.io
Depends on: h5py, nwb, dateutil
Supported: Read, Write
Specification - https://github.com/NeurodataWithoutBorders/specification
Python APIs - (1) https://github.com/AllenInstitute/nwb-api/tree/master/ainwb
	          (2) https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/core/nwb_data_set.py
              (3) https://github.com/NeurodataWithoutBorders/api-python
Sample datasets from CRCNS - https://crcns.org/NWB
Sample datasets from Allen Institute - http://alleninstitute.github.io/AllenSDK/cell_types.html#neurodata-without-borders
"""

from __future__ import absolute_import
from __future__ import division
from itertools import chain
import shutil
import tempfile
from datetime import datetime
from os.path import join
import json
from json.decoder import JSONDecodeError
from collections import defaultdict
import dateutil.parser
import numpy as np
import random

import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block, ImageSequence)
from collections import OrderedDict

# Standard Python imports
from tempfile import NamedTemporaryFile
import os
import glob

# PyNWB imports
try:
    import pynwb
    from pynwb import *
    from pynwb import NWBFile, TimeSeries, get_manager
    from pynwb.base import ProcessingModule
    from pynwb.ecephys import ElectricalSeries, Device, EventDetection
    from pynwb.behavior import SpatialSeries
    from pynwb.misc import AnnotationSeries
    from pynwb import image
    from pynwb.image import ImageSeries
    from pynwb.spec import NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBNamespace, NWBNamespaceBuilder
    from pynwb.device import Device
    from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence # For calcium imaging data
    have_pynwb = True
except ImportError:
    have_pynwb = False

# hdmf imports
try:
    from hdmf.spec import LinkSpec, GroupSpec, DatasetSpec, SpecNamespace,\
                           NamespaceBuilder, AttributeSpec, DtypeSpec, RefSpec
    from hdmf import *
    have_hdmf = True
except ImportError:
    have_hdmf = False


GLOBAL_ANNOTATIONS = (
    "session_start_time", "identifier", "timestamps_reference_time", "experimenter",
    "experiment_description", "session_id", "institution", "keywords", "notes",
    "pharmacology", "protocol", "related_publications", "slices", "source_script",
    "source_script_file_name", "data_collection", "surgery", "virus", "stimulus_notes",
    "lab", "session_description"
)
POSSIBLE_JSON_FIELDS = (
    "source_script", "description"
)


def try_json_field(content):
    try:
        return json.loads(content)
    except JSONDecodeError:
        return content


class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file, and "writing" a .nwb file from Neo
    """
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event, ImageSequence]
    readable_objects  = supported_objects
    writeable_objects = supported_objects

    has_header = False

    name = 'NeoNWB IO'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    is_readable = True
    is_writable = True
    is_streameable = False

    def __init__(self, filename, mode='r'):
        """
        Arguments:
            filename : the filename
        """
        if not have_pynwb:
            raise Exception("Please install the pynwb package to use NWBIO")
        if not have_hdmf:
            raise Exception("Please install the hdmf package to use NWBIO")
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
        self.blocks_written = 0

    def read_all_blocks(self, lazy=False, **kwargs):
        """

        """
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        self._file = io.read()

        self.global_block_metadata = {}
        for annotation_name in GLOBAL_ANNOTATIONS:
            value = getattr(self._file, annotation_name, None)
            if value is not None:
                if annotation_name in POSSIBLE_JSON_FIELDS:
                    value = try_json_field(value)
                self.global_block_metadata[annotation_name] = value
        if "session_description" in self.global_block_metadata:
            self.global_block_metadata["description"] = self.global_block_metadata["session_description"]
        self.global_block_metadata["file_origin"] = self.filename
        if "session_start_time" in self.global_block_metadata:
            self.global_block_metadata["rec_datetime"] = self.global_block_metadata["session_start_time"]
        if "file_create_date" in self.global_block_metadata:
            self.global_block_metadata["file_datetime"] = self.global_block_metadata["file_create_date"]

        self._blocks = {}
        self._handle_acquisition_group(lazy=lazy)
        self._handle_stimulus_group(lazy)
        self._handle_units(lazy=lazy)
        self._handle_epochs_group(lazy)

        # self._handle_calcium_imaging_data(_file, block)
        return list(self._blocks.values())

    def read_block(self, lazy=False, **kargs):
        """
        Load the first block in the file.
        """
        return self.read_all_blocks(lazy=lazy)[0]

    def write_all_blocks(self, blocks, **kwargs):
        """
        Write list of blocks to the file
        """
        # todo: allow metadata in NWBFile constructor to be taken from kwargs
        start_time = datetime.now()
        annotations = defaultdict(set)
        for annotation_name in GLOBAL_ANNOTATIONS:
            if annotation_name in kwargs:
                annotations[annotation_name] = kwargs[annotation_name]
            else:
                for block in blocks:
                    if annotation_name in block.annotations:
                        annotations[annotation_name].add(block.annotations[annotation_name])
                if annotation_name in annotations:
                    if len(annotations[annotation_name]) > 1:
                        raise NotImplementedError("We don't yet support multiple values for {}".format(annotation_name))
                    annotations[annotation_name], = annotations[annotation_name]  # take single value from set
        if "identifier" not in annotations:
            annotations["identifier"] = self.filename
        if "session_description" not in annotations:
            annotations["session_description"] = blocks[0].description or self.filename
            # todo: concatenate descriptions of multiple blocks if different
        if "session_start_time" not in annotations:
            annotations["session_start_time"] = datetime.now()
        # todo: handle subject
        # todo: store additional Neo annotations somewhere in NWB file
        nwbfile = NWBFile(**annotations)

        io_nwb = pynwb.NWBHDF5IO(self.filename, manager=get_manager(), mode='w')

        nwbfile.add_unit_column('_name', 'the name attribute of the SpikeTrain')
        #nwbfile.add_unit_column('_description', 'the description attribute of the SpikeTrain')
        nwbfile.add_unit_column('segment', 'the name of the Neo Segment to which the SpikeTrain belongs')
        nwbfile.add_unit_column('block', 'the name of the Neo Block to which the SpikeTrain belongs')

        nwbfile.add_epoch_column('_name', 'the name attribute of the Epoch')
        #nwbfile.add_unit_column('_description', 'the description attribute of the SpikeTrain')
        nwbfile.add_epoch_column('segment', 'the name of the Neo Segment to which the Epoch belongs')
        nwbfile.add_epoch_column('block', 'the name of the Neo Block to which the Epoch belongs')

        for i, block in enumerate(blocks):
            self.write_block(nwbfile, block)
            #self.write_calcium_imaging_data(nwbfile, block, i)
        io_nwb.write(nwbfile)
        io_nwb.close()

    def write_block(self, nwbfile, block, **kwargs):
        """
        Write a Block to the file
            :param block: Block to be written
        """
        if not block.name:
            block.name = "block%d" % self.blocks_written
        for i, segment in enumerate(block.segments):
            assert segment.block is block
            if not segment.name:
                segment.name = "%s : segment%d" % (block.name, i)
            self._write_segment(nwbfile, segment)
        self.blocks_written += 1

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

    def _handle_epochs_group(self, lazy):
        if self._file.epochs is not None:
            start_times = self._file.epochs.start_time[:]
            stop_times = self._file.epochs.stop_time[:]
            durations = stop_times - start_times
            labels = self._file.epochs.tags[:]
            try:
                # NWB files created by Neo store the segment, block and epoch names as extra columns
                segment_names = self._file.epochs.segment[:]
                block_names = self._file.epochs.block[:]
                epoch_names = self._file.epochs._name[:]
            except AttributeError:
                epoch_names = None

            if epoch_names is not None:
                unique_epoch_names = np.unique(epoch_names)
                for epoch_name in unique_epoch_names:
                    index = (epoch_names == epoch_name)
                    epoch = Epoch(times=start_times[index] * pq.s,
                                durations=durations[index] * pq.s,
                                labels=labels[index],
                                name=epoch_name)
                                # todo: handle annotations, array_annotations
                    segment_name = np.unique(segment_names[index])
                    block_name = np.unique(block_names[index])
                    assert segment_name.size == block_name.size == 1
                    segment = self._get_segment(block_name[0], segment_name[0])
                    segment.epochs.append(epoch)
                    epoch.segment = segment
            else:
                epoch = Epoch(times=start_times * pq.s,
                            durations=durations * pq.s,
                            labels=labels)
                segment = self._get_segment("default", "default")
                segment.epochs.append(epoch)
                epoch.segment = segment

    def _handle_timeseries_group(self, group_name, lazy):
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
            annotations = {"nwb_group" : group_name}
            description = try_json_field(timeseries.description)
            if isinstance(description, dict):
                annotations.update(description)
                description = None
            if isinstance(timeseries, AnnotationSeries):
                event = Event(timeseries.timestamps[:] * pq.s,
                              labels=timeseries.data[:],
                              name=timeseries.name,
                              description=description,
                              **annotations)
                segment.events.append(event)
                event.segment = segment
            elif timeseries.rate:
                signal = AnalogSignal(
                            timeseries.data[:],
                            units=timeseries.unit,
                            t_start=timeseries.starting_time * pq.s,  # use timeseries.starting_time_units
                            sampling_rate=timeseries.rate * pq.Hz,
                            name=timeseries.name,
                            file_origin=self._file.session_description,
                            description=description,
                            array_annotations=None,
                            **annotations)  # todo: timeseries.control / control_description
                segment.analogsignals.append(signal)
                signal.segment = segment
            else:
                signal = IrregularlySampledSignal(
                            timeseries.timestamps[:] * pq.s,
                            timeseries.data[:],
                            units=timeseries.unit,
                            name=timeseries.name,
                            file_origin=self._file.session_description,
                            description=description,
                            array_annotations=None,
                            **annotations)  # todo: timeseries.control / control_description
                segment.irregularlysampledsignals.append(signal)
                signal.segment = segment

    def _handle_units(self, lazy):
        if self._file.units:
            for id in self._file.units.id[:]:
                spike_times = self._file.units.get_unit_spike_times(id)
                t_start, t_stop = self._file.units.get_unit_obs_intervals(id)[0]
                try:
                    # NWB files created by Neo store the segment and block names as extra columns
                    name = self._file.units._name[id]
                    segment_name = self._file.units.segment[id]
                    block_name = self._file.units.block[id]
                except AttributeError:
                    # For NWB files created with other applications, we put everything in a single
                    # segment in a single block
                    name = None
                    segment_name = "default"
                    block_name = "default"
                segment = self._get_segment(block_name, segment_name)
                spiketrain = SpikeTrain(
                                spike_times,
                                t_stop * pq.s,
                                units='s',
                                #sampling_rate=array(1.) * Hz,
                                t_start=t_start * pq.s,
                                #waveforms=None,
                                #left_sweep=None,
                                name=name,
                                #file_origin=None,
                                #description=None,
                                #array_annotations=None,
                                #**annotations
                                nwb_group="acquisition"
                                )
                segment.spiketrains.append(spiketrain)
                spiketrain.segment = segment

    def _handle_acquisition_group(self, lazy):
        self._handle_timeseries_group("acquisition", lazy)

    def _handle_stimulus_group(self, lazy):
        self._handle_timeseries_group("stimulus", lazy)

    def _handle_calcium_imaging_data(self):
        """
        Function to read calcium imaging data.
        """
        pass

    def write_calcium_imaging_data(self, nwbfile, block, i):
        """
        Function to write calcium imaging data. This involves three main steps:
        - Acquiring two-photon images
        - Image segmentation
        - Fluorescence and dF/F response

        Adding metadata about acquisition
        """
        name_imaging_device = "imaging_device %s %d" % (block.name, i)
        device = Device(name_imaging_device)

        nwbfile.add_device(device)

        # To define the manifold
        l = []
        for frame in range(50):
            l.append([])
            for y in range(100):
                l[frame].append([])
                for x in range(100):
                    l[frame][y].append(random.randint(0, 50))

        # OpticalChannel
        name_optical_channel = "optical_channel %s %d" %(block.name, i)
        optical_channel = OpticalChannel(
                                         name = name_optical_channel,
                                         description = 'description',
                                         emission_lambda = 500.) # Emission wavelength for channel, in nm

        name_imaging_plane = "imaging_plane %s %d " %(block.name, i)

        imaging_plane = nwbfile.create_imaging_plane(
                                               name_imaging_plane, # name
                                               optical_channel, # optical_channel
                                               'a very interesting part of the brain', # description
                                               device, # device
                                               600., # excitation_lambda
                                               300., # imaging_rate
                                               'GFP', # indicator
                                               'my favorite brain location', # location
                                               l[frame][y].append(random.randint(0, 50)), # manifold
                                               1.0, # conversion
                                               'manifold unit', # unit
                                               'A frame to refer to' # reference_frame
                                              )

        """
        Adding two-photon image data
        """
        name_twophotonseries = "two_photon_series %s %d" %(block.name, i)
        image_series = TwoPhotonSeries(
                                       name=name_twophotonseries,
                                       dimension=[2],
                                       external_file=['images.tiff'],
                                       imaging_plane=imaging_plane,
                                       starting_frame=[0],
                                       format='tiff',
                                       starting_time=0.0,
                                       rate=1.0
                                      )

        nwbfile.add_acquisition(image_series)

        """
        Storing image segmentation output
        """
        name_processing_module = "processing_module %s %d" %(block.name, i)
        mod = nwbfile.create_processing_module(
                                               name_processing_module, # Example : 'ophys'
                                               'contains optical physiology processed data'
                                              )

        img_seg = ImageSegmentation()
        mod.add(img_seg)

        name_plane_segmentation = "plane_segmentation %s %d" %(block.name, i)
        ps = img_seg.create_plane_segmentation(
                                               description = 'output from segmenting my favorite imaging plane',
                                               imaging_plane = imaging_plane, # link to OpticalChannel
                                               name = name_plane_segmentation,
                                               reference_images = image_series # link to TwoPhotonSeries
                                              )

        """
        Add the resulting ROIs
        """
        w, h = 3, 3
        pix_mask1 = [(0, 0, 1.1), (1, 1, 1.2), (2, 2, 1.3)]
        img_mask1 = [[0.0 for x in range(w)] for y in range(h)]
        img_mask1[0][0] = 1.1
        img_mask1[1][1] = 1.2
        img_mask1[2][2] = 1.3
        ps.add_roi(pixel_mask=pix_mask1, image_mask=img_mask1)

        pix_mask2 = [(0, 0, 2.1), (1, 1, 2.2)]
        img_mask2 = [[0.0 for x in range(w)] for y in range(h)]
        img_mask2[0][0] = 2.1
        img_mask2[1][1] = 2.2
        ps.add_roi(pixel_mask=pix_mask2, image_mask=img_mask2)

        """
        Storing fluorescence measurements
        """
        # Create a data interface
        fl = Fluorescence()
        mod.add(fl)

        # Reference to the ROIs
        rt_region = ps.create_roi_table_region(
                                                'the first of two ROIs',
                                                region=[0]
                                              )

        # RoiResponseSeries
        data = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rrs = fl.create_roi_response_series(
                                            'my_rrs',
                                            data,
                                            rt_region,
                                            unit='lumens',
                                            timestamps=timestamps
                                           )

            # if ImageSequence:
            #     imagesequence_name = ("ImageSequence %s %s %d" % (block.name, segment.name, i))
            #     sampling_rate = signal.sampling_rate.rescale("Hz")
            #     image = pynwb.image.ImageSeries(
            #                                   name=imagesequence_name,
            #                                   data=[[[column for column in range(2)]for row in range(3)] for frame in range(4)],
            #                                   unit=None,
            #                                   format=None,
            #                                   external_file=None,
            #                                   starting_frame=None,
            #                                   bits_per_pixel=None,
            #                                   dimension=None,
            #                                   resolution=-1.0,
            #                                   conversion=float(1*pq.micrometer),
            #                                   timestamps=None,
            #                                   starting_time=None,
            #                                   rate=float(sampling_rate),
            #                                   comments='no comments',
            #                                   description='no description',
            #                                   control=None,
            #                                   control_description=None
            #                                 )

    def _write_segment(self, nwbfile, segment):
        # maybe use NWB trials to store Segment metadata?

        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            assert signal.segment is segment
            if not signal.name:
                signal.name = "%s : analogsignal%d" % (segment.name, i)
            self._write_signal(nwbfile, signal)

        for i, train in enumerate(segment.spiketrains):
            assert train.segment is segment
            if not train.name:
                train.name = "%s : spiketrain%d" % (segment.name, i)
            self._write_spiketrain(nwbfile, train)

        for i, event in enumerate(segment.events):
            assert event.segment is segment
            if not event.name:
                event.name = "%s : event%d" % (segment.name, i)
            self._write_event(nwbfile, event)

        for i, epoch in enumerate(segment.epochs):
            if not epoch.name:
                epoch.name = "%s : epoch%d" % (segment.name, i)
            self._write_epoch(nwbfile, epoch)

    def _write_signal(self, nwbfile, signal):
        hierarchy = {'block': signal.segment.block.name, 'segment': signal.segment.name}
        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            tS = TimeSeries(name=signal.name,
                            starting_time=time_in_seconds(signal.t_start),
                            data=signal,
                            unit=signal.units.dimensionality.string,
                            rate=float(sampling_rate),
                            comments=json.dumps(hierarchy))
                            # todo: try to add array_annotations via "control" attribute
        elif isinstance(signal, IrregularlySampledSignal):
            tS = TimeSeries(name=signal.name,
                            data=signal,
                            unit=signal.units.dimensionality.string,
                            timestamps=signal.times.rescale('second').magnitude,
                            comments=json.dumps(hierarchy))
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))
        nwbfile.add_acquisition(tS)
        return tS

    def _write_spiketrain(self, nwbfile, spiketrain):
        nwbfile.add_unit(spike_times=spiketrain.rescale('s').magnitude,
                         obs_intervals=[[float(spiketrain.t_start.rescale('s')),
                                         float(spiketrain.t_stop.rescale('s'))]],
                         _name=spiketrain.name,
                         #_description=spiketrain.description,
                         segment=spiketrain.segment.name,
                         block=spiketrain.segment.block.name)
        # todo: handle annotations (using add_unit_column()?)
        # todo: handle Neo Units
        # todo: handle spike waveforms, if any (see SpikeEventSeries)
        return nwbfile.units

    def _write_event(self, nwbfile, event):
        hierarchy = {'block': event.segment.block.name, 'segment': event.segment.name}
        tS_evt = AnnotationSeries(
                        name=event.name,
                        data=event.labels,
                        timestamps=event.times.rescale('second').magnitude,
                        description=event.description or "",
                        comments=json.dumps(hierarchy))
        nwbfile.add_acquisition(tS_evt)
        return tS_evt

    def _write_epoch(self, nwbfile, epoch):
        for t_start, duration, label in zip(epoch.rescale('s').magnitude,
                                            epoch.durations.rescale('s').magnitude,
                                            epoch.labels):
            nwbfile.add_epoch(t_start, t_start + duration, [label], [],
                              _name=epoch.name,
                              segment=epoch.segment.name,
                              block=epoch.segment.block.name)
        return nwbfile.epochs


def time_in_seconds(t):
    return float(t.rescale("second"))


def _decompose_unit(unit):
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
    conversion = 1.0
    def _decompose(unit):
        dim = unit.dimensionality
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt-metre
        uq, n = dim.items()[0]
        if n != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt^2
        uq_def = uq.definition
        return float(uq_def.magnitude), uq_def
    conv, unit2 = _decompose(unit)
    while conv != 1:
        conversion *= conv
        unit = unit2
        conv, unit2 = _decompose(unit)
    return conversion, unit.dimensionality.keys()[0].name


prefix_map = {
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano'
}