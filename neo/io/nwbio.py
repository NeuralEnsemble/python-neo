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

import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block)

# neo imports
from collections import OrderedDict

# Standard Python imports
from tempfile import NamedTemporaryFile
import os
import glob
from scipy.io import loadmat

# PyNWB imports
import pynwb
from pynwb import *
# Creating and writing NWB files
from pynwb import NWBFile,TimeSeries, get_manager
from pynwb.base import ProcessingModule
# Creating TimeSeries
from pynwb.ecephys import ElectricalSeries, Device, EventDetection
from pynwb.behavior import SpatialSeries
from pynwb.image import ImageSeries
from pynwb.core import set_parents
# For Neurodata Type Specifications
from pynwb.spec import NWBAttributeSpec # Attribute Specifications
from pynwb.spec import NWBDatasetSpec # Dataset Specifications
from pynwb.spec import NWBGroupSpec
from pynwb.spec import NWBNamespace


class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file.
    """

    name = 'NWB'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    def __init__(self, filename):
        """
        Arguments:
            filename : the filename
        """
        print("*** __init__ ***")
        BaseIO.__init__(self, filename=filename)
        #BaseIO.__init__(self)
        #print("filename = ", filename)
        self.filename = filename
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        self._file = io.read() # Define the file as a NWBFile object

    def read_block(self, lazy=False, cascade=True, **kwargs):
        print("*** read_block ***")
        self._lazy = lazy
        #print("lazy = ", lazy)
        file_access_dates = self._file.file_create_date
        #print("file_access_dates = ", file_access_dates)
        identifier = self._file.identifier # or experimenter ?
        #print("identifier = ", identifier)
        if identifier == '_neo':  # this is an automatically generated name used if block.name is None
            identifier = None
        description = self._file.session_description # or experiment_description ?
        #print("description = ", description)
        if description == "no description":
            description = None
        block = Block(name=identifier, 
                      description=description,
                      file_origin=self.filename,
                      file_datetime=file_access_dates,
                      rec_datetime=self._file.session_start_time,
                      #nwb_version=self._file.get('nwb_version').value,
                      file_access_dates=file_access_dates,
                      file_read_log='')
        print("block = ", block)
        if cascade:
            self._handle_general_group(block)
            self._handle_epochs_group(lazy, block) #self._handle_epochs_group(block)
            self._handle_acquisition_group(lazy, block) # self._handle_acquisition_group(block)
            self._handle_stimulus_group(lazy, block) # self._handle_stimulus_group(block)
            self._handle_processing_group(block)
            self._handle_analysis_group(block)
        self._lazy = False
        return block

    def _handle_general_group(self, block):
        print("*** def _handle_general_group ***")
        #block.annotations['file_read_log'] += ("general group not handled\n")

    def _handle_epochs_group(self, lazy, block):
        print("*** def _handle_epochs_group ***")
        self._lazy = lazy
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
###        epochs = self._file.epochs
        epochs = self._file.acquisition
        #print("epochs = ", epochs)

        # todo: handle epochs.attrs.get('tags')
        ##for name, epoch in epochs.items():
#        for name, epoch in epochs:
        for key in epochs:    
            #print("key = ", key)
            #print("epochs = ", epochs)
             # todo: handle epoch.attrs.get('links')
            timeseries = []
            #print("timeseries = ", timeseries)
            current_shape = self._file.get_acquisition(key).data.shape[0] # sample number
            #print("current_shape = ", current_shape)
            times = np.zeros(current_shape)
           #for key, value in epoch.items():
            for j in range(0, current_shape):
                times[j]=1./self._file.get_acquisition(key).rate*j+self._file.get_acquisition(key).starting_time
                if times[j] == self._file.get_acquisition(key).starting_time:
                    t_start = times[j] * pq.second
                    #print("t_start = ", t_start)
                elif times[j]==times[-1]:
                    t_stop = times[j] * pq.second
                    #print("t_stop = ", t_stop)
                else:
                    # todo: handle value['count']
                    # todo: handle value['idx_start']
                    #timeseries.append(self._handle_timeseries(key, value.get('timeseries')))
                    timeseries.append(self._handle_timeseries(self._lazy, key, times[j]))
                    #print(timeseries)
                    #print("timeSeries 1 bis = ", timeseries)
                #print(timeseries)
                #print("timeseries 2nd bis = ", timeseries)
#            segment = Segment(name=name)
                segment = Segment(name=j)
                #print("segment = ", segment)
            for obj in timeseries:
                #print("obj = ", obj)
                #print("timeseries = ", timeseries)
#            for obj in self._file.acquisition:
###                print("obj in segment = ", obj)
                obj.segment = segment
#                segment==obj.segment
                #print("segment in loop = ", segment)
#                print("obj.segment = ", obj.segment)
                if isinstance(obj, AnalogSignal):
                    print("*** isinstance(obj, AnalogSignal) ***")
                    segment.analogsignals.append(obj)
                    #print("obj=AnalogSignal")
                elif isinstance(obj, IrregularlySampledSignal):
                    print("*** isinstance(obj, IrregularlySampledSignal) ***")
                    segment.irregularlysampledsignals.append(obj)
                    #print("obj=IrregularlySampledSignal")
                elif isinstance(obj, Event):
                    print("*** isinstance(obj, Event) ***")
                    segment.events.append(obj)
                    #print("obj=Event")
                elif isinstance(obj, Epoch):
                    print("*** isinstance(obj, Epoch) ***")
                    segment.epochs.append(obj)
                    #print("obj=Epoch")
            segment.block = block
            #print("segment = ", segment)
            #print("block = ", block)
#            block.segments.append(segment)
            return obj, segment


    def _handle_timeseries(self, lazy, name, timeseries):
        print("*** def _handle_timeseries ***")
        # todo: check timeseries.attrs.get('schema_id')
        # todo: handle timeseries.attrs.get('source')
#         subtype = timeseries.attrs['ancestry'][-1]
        for i in self._file.acquisition:
            data_group = self._file.get_acquisition(i).data
            #print("data_group = ", data_group)
            dtype = data_group.dtype
            #print("dtype = ", dtype)
            #if self._lazy:
            if lazy==True:
                data = np.array((), dtype=dtype)
                #print("data if lazy = ", data)
                lazy_shape = data_group.shape  # inefficient to load the data to get the shape
                #print("lazy_shape = ", lazy_shape)
            else:
                data = data_group
            if dtype.type is np.string_:
                if self._lazy:
                    times = np.array(())
                else:
                    times = self._file.get_acquisition(i).timestamps
                    #print("times in timestamps = ", times)
                duration = 1/self._file.get_acquisition(i).rate
                #print("************************ duration = ", duration)
                if durations:
                    # Epoch
                    if self._lazy:
                        durations = np.array(())
                    obj = Epoch(times=times,
                                durations=durations,
                                labels=data,
                                units='second')
                    #print("obj if duration = ", obj)
                else:
                    # Event
                    obj = Event(times=times,
                                labels=data,
                                units='second')
                    #print("obj Event = ", obj)
            else:
                #units = get_units(data_group)
                units = self._file.get_acquisition(i).unit
                #print("units = ", units)
            current_shape = self._file.get_acquisition(i).data.shape[0] # number of samples
            #print("current_shape = ", current_shape)
            times = np.zeros(current_shape)
            for j in range(0, current_shape):
                times[j]=1./self._file.get_acquisition(i).rate*j+self._file.get_acquisition(i).starting_time
                if times[j] == self._file.get_acquisition(i).starting_time:
                    # AnalogSignal
                    sampling_metadata = times[j]
                    #print("sampling_metadata = ", sampling_metadata)        
                    t_start = sampling_metadata * pq.s
                    #print("t_start = ", t_start)
                    sampling_rate = self._file.get_acquisition(i).rate * pq.Hz
                    #print("sampling_rate = ", sampling_rate)
                    #assert sampling_metadata.attrs.get('unit') == 'Seconds'
###                    assert sampling_metadata.unit == 'Seconds'                    
#                 # todo: handle data.attrs['resolution']
                    obj = AnalogSignal(data,
                                       units=units,
                                       sampling_rate=sampling_rate,
                                       t_start=t_start,
                                       name=name)
                    #print("obj = ", obj)
#                elif 'timestamps' in timeseries:
                elif self._file.get_acquisition(i).timestamps:
                    # IrregularlySampledSignal
                    if self._lazy:
                        time_data = np.array(())
                    else:
                        time_data = self._file.get_acquisition(i).timestamps
###                        assert time_data.attrs.get('unit') == 'Seconds'
#                 obj = IrregularlySampledSignal(time_data.value,
#                                                data,
#                                                units=units,
#                                                time_units=pq.second)
#             else:
#                 raise Exception("Timeseries group does not contain sufficient time information")
#         if self._lazy:
#             obj.lazy_shape = lazy_shape
            return obj


    def _handle_acquisition_group(self, lazy, block):
        print("*** def _handle_acquisition_group ***")
        acq = self._file.acquisition
        #print("acq = ", acq)
#         images = acq.get('images')
#         if images and len(images) > 0:
#             block.annotations['file_read_log'] += ("file contained {0} images; these are not currently handled by Neo\n".format(len(images)))


        # todo: check for signals that are not contained within an NWB Epoch,
        #       and create an anonymous Segment to contain them

        ###segment_acq = dict((segment.name, segment) for segment in block.segments)
        ###print("segment_acq = ", segment_acq)
        for name in acq:
            #print("name = ", name) # Sample number 'index_'
#             if name == 'unit_list':
#                 pass  # todo
#             else:
#            segment_name = name
            # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
            segment_name = self._file.epochs
            #print("segment_name = ", segment_name)
            desc = self._file.get_acquisition(name).unit
            #print("desc = ", desc)
###            segment = segment_acq[segment_name]
            segment = segment_name
#            print("segment = ", segment)
            #if self._lazy:
            if lazy==True:
                times = np.array(())
                #print("times = ", times)
                #lazy_shape = group['times'].shape
                lazy_shape = self._file.get_acquisition(name).data.shape
                #print("lazy_shape = ", lazy_shape)
            else:
                current_shape = self._file.get_acquisition(name).data.shape[0] # sample number
                #print("current_shape = ", current_shape)
                times = np.zeros(current_shape)
                for j in range(0, current_shape): # For testing !
                    times[j]=1./self._file.get_acquisition(name).rate*j+self._file.get_acquisition(name).starting_time # temps = 1./frequency [Hz] + t_start [s]
                #print("times[j] = ", times)
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)  # todo: this is a custom Neo value, general NWB files will not have this - use segment.t_stop instead in that case?
                #print("spiketrain in _handle_acquisition_group = ", spiketrain)
            #if self._lazy:
###            if lazy==True:
###                spiketrain.lazy_shape = lazy_shape
            if segment is not None:
                spiketrain.segment = segment
                #print("segment = ", segment)
                segment.spiketrains.append(spiketrain)
        return spiketrain

    def _handle_stimulus_group(self, lazy, block):
        print("*** def _handle_stimulus_group ***")
        #block.annotations['file_read_log'] += ("stimulus group not handled\n")
        # The same as acquisition for stimulus for spiketrain...

        sti = self._file.stimulus
        #print("sti = ", sti)
#         images = sti.get('images')
#         if images and len(images) > 0:
#             block.annotations['file_read_log'] += ("file contained {0} images; these are not currently handled by Neo\n".format(len(images)))

###        segment_sti = dict((segment.name, segment) for segment in block.segments)
###        print("segment_sti = ", segment_sti)
        for name in sti:
            #print("name = ", name) # Sample number 'index_'
#             if name == 'unit_list':
#                 pass  # todo
#             else:
#            segment_name = name
            # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
            segment_name_sti = self._file.epochs
            #print("segment_name_sti = ", segment_name_sti)
            desc_sti = self._file.get_stimulus(name).unit
            #print("desc_sti = ", desc_sti)
###            segment = segment_acq[segment_name]
            segment_sti = segment_name_sti
            #print("segment_sti = ", segment_sti)
            #if self._lazy:
            if lazy==True:
                times = np.array(())
                #print("times = ", times)
                #lazy_shape = group['times'].shape
                lazy_shape = self._file.get_stimulus(name).data.shape
                #print("lazy_shape = ", lazy_shape)
            else:
                current_shape = self._file.get_stimulus(name).data.shape[0] # sample number
                #print("current_shape = ", current_shape)
                times = np.zeros(current_shape)
                for j in range(0, current_shape): # For testing !
                    times[j]=1./self._file.get_stimulus(name).rate*j+self._file.get_stimulus(name).starting_time # temps = 1./frequency [Hz] + t_start [s]
                #print("times[j] = ", times)
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)  # todo: this is a custom Neo value, general NWB files will not have this - use segment.t_stop instead in that case?
                #print("spiketrain = ", spiketrain)
            #if self._lazy:
###            if lazy==True:
###                spiketrain.lazy_shape = lazy_shape
            if segment_sti is not None:
                spiketrain.segment_sti = segment_sti
                #print("segment_sti = ", segment_sti)
                segment_sti.spiketrains.append(spiketrain)


    def _handle_processing_group(self, block):
        print("*** def _handle_processing_group ***")
        # todo: handle other modules than Units
##        units_group = self._file.get('processing/Units/UnitTimes')
        #segment_map = dict((segment.name, segment) for segment in block.segments)
        #print("segment_map = ", segment_map)
#         for name, group in units_group.items():
#             if name == 'unit_list':
#                 pass  # todo
#             else:
#                 segment_name = group['source'].value
#                 #desc = group['unit_description'].value  # use this to store Neo Unit id?
#                 segment = segment_map[segment_name]
#                 if self._lazy:
#                     times = np.array(())
#                     lazy_shape = group['times'].shape
#                 else:
#                     times = group['times'].value
#                 spiketrain = SpikeTrain(times, units=pq.second,
#                                         t_stop=group['t_stop'].value*pq.second)  # todo: this is a custom Neo value, general NWB files will not have this - use segment.t_stop instead in that case?
#                 if self._lazy:
#                     spiketrain.lazy_shape = lazy_shape
#                 spiketrain.segment = segment
#                 segment.spiketrains.append(spiketrain)

    def _handle_analysis_group(self, block):
        print("*** def _handle_analysis_group ***")
        #block.annotations['file_read_log'] += ("analysis group not handled\n")




# def time_in_seconds(t):
#     print("*** def time_in_seconds ***")
#     return float(t.rescale("second"))


# def _decompose_unit(unit):
#     print("*** def _decompose_unit ***")
#     """unit should be a Quantity object with unit magnitude
#     Returns (conversion, base_unit_str)
#     Example:
#         >>> _decompose_unit(pq.nA)
#         (1e-09, 'ampere')
#     """
#     assert isinstance(unit, pq.quantity.Quantity)
#     assert unit.magnitude == 1
#     conversion = 1.0
#     def _decompose(unit):
#         dim = unit.dimensionality
#         if len(dim) != 1:
#             raise NotImplementedError("Compound units not yet supported")  # e.g. volt-metre
#         uq, n = dim.items()[0]
#         if n != 1:
#             raise NotImplementedError("Compound units not yet supported")  # e.g. volt^2
#         uq_def = uq.definition
#         return float(uq_def.magnitude), uq_def
#     conv, unit2 = _decompose(unit)
#     while conv != 1:
#         conversion *= conv
#         unit = unit2
#         conv, unit2 = _decompose(unit)
#     return conversion, unit.dimensionality.keys()[0].name


prefix_map = {
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano'
}

def get_units(data_group):
    print("*** def get_units ***")
    #print("data_group = ", data_group)
#     conversion = data_group.attrs.get('conversion')
    #base_units = data_group.attrs.get('unit')
    base_units = data_group.units
    #print("base_units = ", base_units)
#     return prefix_map[conversion] + base_units


