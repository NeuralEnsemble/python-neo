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
from scipy.io import loadmat

# PyNWB imports
import pynwb
from pynwb import *
from pynwb import NWBFile,TimeSeries, get_manager
from pynwb.base import ProcessingModule
from pynwb.ecephys import ElectricalSeries, Device, EventDetection
from pynwb.behavior import SpatialSeries
from pynwb import image
from pynwb.image import ImageSeries
from pynwb.spec import NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBNamespace, NWBNamespaceBuilder
from pynwb.device import Device
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation, Fluorescence # For calcium imaging data

# hdmf imports
from hdmf.spec import LinkSpec, GroupSpec, DatasetSpec, SpecNamespace,\
                       NamespaceBuilder, AttributeSpec, DtypeSpec, RefSpec
from hdmf import *


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

    def __init__(self, filename, mode):
        """
        Arguments:
            filename : the filename
        """
        if not pynwb:
            raise Exception("Please install the pynwb package to use NWBIO")
        if not hdmf:
            raise Exception("Please install the hdmf package to use NWBIO")        
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
   
    def read_all_blocks(self, lazy=False, **kwargs):        
        """
        Loads all blocks in the file that are attached to the root.
        Here, we assume that a neo block is a sub-part of a branch, into a NWB file;
        """
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        self._file = io.read()

        blocks = []
        for node in (self._file.acquisition, self._file.units, self._file.epochs):
            blocks.append(self._read_block(self._file, node, blocks))
        return blocks

    def read_block(self, lazy=False, **kargs):
        """
        Load the first block in the file.
        """
        return self.read_all_blocks(lazy=lazy)[0]

    def _read_block(self, _file, node, blocks, lazy=False, cascade=True, **kwargs):
        """
        Main method to load a block
        """
        self._lazy = lazy

        file_access_dates = _file.file_create_date
        identifier = _file.identifier
        if identifier == '_neo':
            identifier = None
        description = _file.session_description
        if description == "no description":
            description = None

        block = Block(name=identifier,
                      description=description,
                      file_origin=self.filename,
                      file_datetime=file_access_dates,
                      rec_datetime=_file.session_start_time,
                      file_access_dates=file_access_dates,
                      file_read_log='')
        if cascade:
            self._handle_general_group(block)
            self._handle_epochs_group(_file, block)
            self._handle_acquisition_group(lazy, _file, block)
            self._handle_stimulus_group(lazy, _file, block)
            self._handle_processing_group(_file, block)
            self._handle_analysis_group(block)
            self._handle_calcium_imaging_data(_file, block) # Calcium imaging data
        self._lazy = False
        return block

    def write_all_blocks(self, blocks, **kwargs):
        """
        Write list of blocks to the file
        """
        start_time = datetime.now()
        nwbfile = NWBFile(self.filename,
                               session_start_time=start_time,
                               identifier='',
                               file_create_date=None,
                               timestamps_reference_time=None,
                               experimenter=None,
                               experiment_description=None,
                               session_id=None,
                               institution=None,
                               keywords=None,
                               notes=None,
                               pharmacology=None,
                               protocol=None,
                               related_publications=None,
                               slices=None,
                               source_script=None,
                               source_script_file_name=None,
                               data_collection=None,
                               surgery=None,
                               virus=None,
                               stimulus_notes=None,
                               lab=None,
                               acquisition=None,
                               stimulus=None,
                               stimulus_template=None,
                               epochs=None,
                               epoch_tags=set(),
                               trials=None,
                               invalid_times=None,
                               units=None,
                               electrodes=None,
                               electrode_groups=None,
                               ic_electrodes=None,
                               sweep_table=None,
                               imaging_planes=None,
                               ogen_sites=None,
                               devices=None,
                               subject=None
                               )
        io_nwb = pynwb.NWBHDF5IO(self.filename, manager=get_manager(), mode='w')

        if Block in self.writeable_objects:
            for i, block in enumerate(blocks):
                block_name = block.name or  "blocks%d" % i
                self.write_block(nwbfile, block)
                self.write_calcium_imaging_data(nwbfile, block, i)
            io_nwb.write(nwbfile)
            return list(block.segments)
        io_nwb.close()

    def write_block(self, nwbfile, block, **kwargs):
        """
        Write a Block to the file
            :param block: Block to be written
        """
        self._write_block_children(nwbfile, block)

    def _write_block_children(self, nwbfile, block=None, **kwargs):
        for i, segment in enumerate(block.segments):
            self._write_segment(nwbfile, block, segment)
            segment_name = segment.name
#            print("segment_name = ", segment_name)
            tS_seg = TimeSeries(
                        name=segment_name,
                        data=[segment],
                        timestamps=[1],
                        description="",
                        )

    def _handle_general_group(self, block):
        pass

    def _handle_epochs_group(self, _file, block):
        """
        Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        """
        epochs = _file.epochs
        timeseries=[]
        segment = Segment(name=self.name)
        segment.epochs.append(Epoch)

        for obj in timeseries:
            obj.segment = segment
            if isinstance(obj, AnalogSignal):
                segment.analogsignals.append(obj)
                segment.epochs.append(obj)
            elif isinstance(obj, IrregularlySampledSignal):
                segment.irregularlysampledsignals.append(obj)
            elif isinstance(obj, Event):
                segment.events.append(obj)
            elif isinstance(obj, Epoch):
                segment.epochs.append(obj)
        segment.block = block
        block.segments.append(segment)

    def _handle_acquisition_group(self, lazy, _file, block):
        acq = _file.acquisition

    def _handle_stimulus_group(self, lazy, _file, block):
        sti = _file.stimulus
        for name in sti:
            segment_name_sti = _file.epochs
            desc_sti = _file.get_stimulus(name).unit
            segment_sti = segment_name_sti
            if lazy==True:
                times = np.array(())
                lazy_shape = _file.get_stimulus(name).data.shape
            else:
                current_shape = _file.get_stimulus(name).data.shape[0]
                times = np.zeros(current_shape)
                for j in range(0, current_shape):
                    times[j]=1./_file.get_stimulus(name).rate*j+_file.get_acquisition(name).starting_time # times = 1./frequency [Hz] + t_start [s]
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)

    def _handle_processing_group(self, _file, block):
        segment = Segment(name=self.name)

    def _handle_analysis_group(self, block):
        pass

    def _handle_calcium_imaging_data(self, _file, block):
        """
        Function to read calcium imaging data.
        """
#        print("*** def _handle_calcium_imaging_data ***")
        pass


#############################
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
#        print("imaging_plane = ", imaging_plane)

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
#        print("image_series = ", image_series)

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
#        print("ps = ", ps)

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
#        print("rrs = ", rrs)
#############################


    def _write_segment(self, nwbfile, block, segment):
        block_name = block.name or  "blocks %d" % i
        segment_name = segment.name

        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)): # analogsignals
            self._write_signal(nwbfile, block, signal, i, segment)
            analogsignal_name = signal.name or ("analogsignal %s %s %d" % (block_name, segment_name, i))
#            print("analogsignal_name = ", analogsignal_name)
            tS_signal = TimeSeries(
                                    name=analogsignal_name,
                                    data=signal,
                                    timestamps=[1],
                                    description="",
                                   )

#############################
            if ImageSequence:
                imagesequence_name = ("ImageSequence %s %s %d" % (block_name, segment_name, i))
#                print("imagesequence_name = ", imagesequence_name)
                sampling_rate = signal.sampling_rate.rescale("Hz")
                image = pynwb.image.ImageSeries(
                                              name=imagesequence_name,
                                              data=[[[column for column in range(2)]for row in range(3)] for frame in range(4)],
                                              unit=None,
                                              format=None,
                                              external_file=None,
                                              starting_frame=None,
                                              bits_per_pixel=None,
                                              dimension=None,
                                              resolution=-1.0,
                                              conversion=float(1*pq.micrometer),
                                              timestamps=None,
                                              starting_time=None,
                                              rate=float(sampling_rate),
                                              comments='no comments',
                                              description='no description',
                                              control=None,
                                              control_description=None
                                            )
#                print("image 2 = ", image)
#############################

        for i, train in enumerate(segment.spiketrains):
            self._write_spiketrains(nwbfile, train, i, segment)
            spiketrains_name = train.name or ("spiketrains %s %s %d" % (block_name, segment_name, i))
            ts_name = "{0}".format(spiketrains_name)
            tS_train = TimeSeries(
                        name=spiketrains_name,
                        data=train,
                        timestamps=[1],
                        description="",
                        )
        for i, event in enumerate(segment.events):
            self._write_event(nwbfile, event, nwb_epoch, i)
        for i, neo_epoch in enumerate(segment.epochs):
            self._write_neo_epoch(nwbfile, neo_epoch, i, segment)
            epochs_name = neo_epoch.name or ("neo epochs %s %s %d" % (block_name, segment_name, i))
            ts_name = "{0}".format(epochs_name)
            tS_epc = TimeSeries(
                        name=epochs_name,
                        data=neo_epoch,
                        timestamps=neo_epoch.times.rescale('second').magnitude,
                        description=neo_epoch.description or "",
                        )

        nwbfile.add_acquisition(tS_signal) # For analogsignals
        nwbfile.add_acquisition(tS_train) # For spiketrains
        nwbfile.add_acquisition(tS_epc) # For Neo segment (Neo epoch)
        nwbfile.add_acquisition(image) # for ImageSequence

    def _write_signal(self, nwbfile, block, signal, i, segment): # analogsignals
        block_name = block.name  or  "blocks %d" % i
        segment_name = segment.name
        signal_name = signal.name or ("signal %s %s %d" % (block_name, segment_name, i))
        ts_name = "{0}".format(signal_name)
        conversion = _decompose_unit(signal.units)
        attributes = {"conversion": conversion,
                      "resolution": float('nan')}

        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            signal.sampling_rate = sampling_rate
            # All signals should go in /acquisition
            tS = TimeSeries(name=ts_name, starting_time=time_in_seconds(signal.t_start), data=segment.analogsignals, rate=float(sampling_rate))
            #ts = nwbfile.add_acquisition(tS)
            return [segment.analogsignals]

#########################   # ImageSequence
        elif isinstance(signal, ImageSequence): # ImageSequence
            imagesequence_name = "ImageSequence %d" % i
            sampling_rate = signal.sampling_rate.rescale("Hz")
            signal.sampling_rate = sampling_rate
            # All signals should go in /acquisition
           
            image = pynwb.image.ImageSeries(
                  name=imagesequence_name,
                  data=[[[column for column in range(2)]for row in range(3)] for frame in range(4)],
                  unit=None,
                  format=None, 
                  external_file=None, 
                  starting_frame=None, 
                  bits_per_pixel=None, 
                  dimension=None,
                  resolution=-1.0,
                  conversion=float(1*pq.micrometer),
                  timestamps=None,
                  starting_time=None, 
                  rate=float(sampling_rate), #sampling_rate
                  comments='no comments', 
                  description='no description', 
                  control=None, 
                  control_description=None
              )
#            print("image = ", image)
######################### 


        elif isinstance(signal, IrregularlySampledSignal):
            tS = TimeSeries(name=ts_name, starting_time=time_in_seconds(signal.t_start), data=signal, timestamps=signal.times.rescale('second').magnitude)
            return [segment.irregularlysampledsignals]
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))
        ####ts = nwbfile.add_acquisition(tS)

    def _write_spiketrains(self, nwbfile, spiketrains, i, segment): # spiketrains
        spiketrain = segment.spiketrains
        for i, train in enumerate(segment.spiketrains): # spiketrains
            spiketrains_name = train.name or "spiketrains %d" % i
            ts_name = "{0}".format(spiketrains_name)
            tS_train = TimeSeries(
                        name=spiketrains_name,
                        data=train,
                        timestamps=[1],
                        description="",
                        )
            return [segment.spiketrains]

    def _write_event(self, nwbfile, event, nwb_epoch, i):
        event_name = event.name or "event{0}".format(i)
        ts_name = "{0}".format(event_name)
        tS = TimeSeries(
                        name=ts_name,
                        data=event, 
                        timestamps=event.times.rescale('second').magnitude,
                        description=event.description or "",
                        )

    def _write_neo_epoch(self, nwbfile, neo_epoch, i, segment): # epochs
        for i, epoch in enumerate(segment.epochs): # epochs
            epochs_name = epoch.name or "epochs %d" % i
            ts_name = "{0}".format(epochs_name)
            tS_epc = TimeSeries(
                        name=epochs_name,
                        data=epoch,
                        timestamps=[1],
                        description="",
                        )
            return [segment.epochs]


def time_in_seconds(t):
    return float(t.rescale("second"))

def _decompose_unit(unit):
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
    conversion = 1.0
    def _decompose(unit):
        dim = unit.dimensionality
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")
        uq, n = dim.items()[0]
        if n != 1:
            raise NotImplementedError("Compound units not yet supported")
        uq_def = uq.definition
        return float(uq_def.magnitude), uq_def

prefix_map = {
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano'
}