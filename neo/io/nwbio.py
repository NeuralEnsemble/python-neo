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

    is_readable = True
    is_writable = True
    is_streameable = False
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event]
    readable_objects  = supported_objects
    writeable_objects = supported_objects

    has_header = False

    name = 'NWB'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    def __init__(self, filename):
        """
        Arguments:
            filename : the filename
        """
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        self._file = io.read() # Define the file as a NWBFile object

    def read_block(self, lazy=False, cascade=True, **kwargs):
        self._lazy = lazy
        file_access_dates = self._file.file_create_date
        identifier = self._file.identifier # or experimenter ?
        if identifier == '_neo':  # this is an automatically generated name used if block.name is None
            identifier = None
        description = self._file.session_description # or experiment_description ?
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
        if cascade:
            self._handle_general_group(block)
            self._handle_epochs_group(lazy, block)
            self._handle_acquisition_group(lazy, block)
            self._handle_stimulus_group(lazy, block)
            self._handle_processing_group(block)
            self._handle_analysis_group(block)
        self._lazy = False
        return block



    def write_block(self, block, **kwargs):
        start_time = datetime.now()
        self._file = NWBFile(self.filename,
                             session_start_time=start_time,
                             identifier=self._file.name,
                             )
        for segment in block.segments:
            self._write_segment(segment)
        self._file.close()

        if block.file_origin is None:
            block.file_origin = self.filename

        self._file = h5py.File(self.filename, "r+")
        nwb_create_date = self._file['file_create_date'].value
        if block.file_datetime:
            del self._file['file_create_date']
            self._file['file_create_date'] = np.array([block.file_datetime.isoformat(), nwb_create_date])
        else:
            block.file_datetime = parse_datetime(nwb_create_date[0])
        self._file.close()


    def _handle_general_group(self, block):
        print("*** def _handle_general_group ***")
        #block.annotations['file_read_log'] += ("general group not handled\n")

    def _handle_epochs_group(self, lazy, block):
        self._lazy = lazy
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
###        epochs = self._file.epochs
        epochs = self._file.acquisition

        for key in epochs:    
            timeseries = []
            current_shape = self._file.get_acquisition(key).data.shape[0] # sample number
            times = np.zeros(current_shape)
            for j in range(0, current_shape):
                times[j]=1./self._file.get_acquisition(key).rate*j+self._file.get_acquisition(key).starting_time
                if times[j] == self._file.get_acquisition(key).starting_time:
                    t_start = times[j] * pq.second
                elif times[j]==times[-1]:
                    t_stop = times[j] * pq.second
                else:
                    timeseries.append(self._handle_timeseries(self._lazy, key, times[j]))
                segment = Segment(name=j)
            for obj in timeseries:
#                print("obj = ", obj)
                obj.segment = segment
                if isinstance(obj, AnalogSignal):
                    #print("AnalogSignal")
                    segment.analogsignals.append(obj)
                elif isinstance(obj, IrregularlySampledSignal):
                    #print("IrregularlySampledSignal")
                    segment.irregularlysampledsignals.append(obj)
                elif isinstance(obj, Event):
                    #print("Event")
                    segment.events.append(obj)
                elif isinstance(obj, Epoch):
                    #print("Epoch")
                    segment.epochs.append(obj)
            segment.block = block
            segment.times=times
#            block.segments.append(segment)
#            print("segment = ", segment)
###            print("segment.analogsignals = ", segment.analogsignals)
            return segment, obj, times



    def _handle_timeseries(self, lazy, name, timeseries):
        for i in self._file.acquisition:
            data_group = self._file.get_acquisition(i).data*self._file.get_acquisition(i).conversion
            dtype = data_group.dtype
            if lazy==True:
                data = np.array((), dtype=dtype)
                lazy_shape = data_group.shape
            else:
                data = data_group
            if dtype.type is np.string_:
                if self._lazy:
                    times = np.array(())
                else:
                    times = self._file.get_acquisition(i).timestamps
                duration = 1/self._file.get_acquisition(i).rate
                if durations:
                    # Epoch
                    if self._lazy:
                        durations = np.array(())
                    obj = Epoch(times=times,
                                durations=durations,
                                labels=data,
                                units='second')
                else:
                    # Event
                    obj = Event(times=times,
                                labels=data,
                                units='second')
            else:
                units = self._file.get_acquisition(i).unit
            current_shape = self._file.get_acquisition(i).data.shape[0] # number of samples
            times = np.zeros(current_shape)
            for j in range(0, current_shape):
                times[j]=1./self._file.get_acquisition(i).rate*j+self._file.get_acquisition(i).starting_time
                if times[j] == self._file.get_acquisition(i).starting_time:
                    # AnalogSignal
                    sampling_metadata = times[j]
                    t_start = sampling_metadata * pq.s
                    sampling_rate = self._file.get_acquisition(i).rate * pq.Hz
                    #assert sampling_metadata.attrs.get('unit') == 'Seconds'
###                    assert sampling_metadata.unit == 'Seconds'
                    obj = AnalogSignal(data,
                                       units=units,
                                       sampling_rate=sampling_rate,
                                       t_start=t_start,
                                       name=name)
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
            return obj


    def _handle_acquisition_group(self, lazy, block):
        acq = self._file.acquisition

        # todo: check for signals that are not contained within an NWB Epoch,
        #       and create an anonymous Segment to contain them

        ###segment_acq = dict((segment.name, segment) for segment in block.segments)
        for name in acq:
            # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
            segment_name = self._file.epochs
            desc = self._file.get_acquisition(name).unit
            segment = segment_name
            if lazy==True:
                times = np.array(())
                lazy_shape = self._file.get_acquisition(name).data.shape
            else:
                current_shape = self._file.get_acquisition(name).data.shape[0] # sample number
                times = np.zeros(current_shape)
                for j in range(0, current_shape): # For testing !
                    times[j]=1./self._file.get_acquisition(name).rate*j+self._file.get_acquisition(name).starting_time # times = 1./frequency [Hz] + t_start [s]
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)  # todo: this is a custom Neo value, general NWB files will not have this - use segment.t_stop instead in that case?
            if segment is not None:
                spiketrain.segment = segment
                segment.spiketrains.append(spiketrain)
        return spiketrain

    def _handle_stimulus_group(self, lazy, block):
        #block.annotations['file_read_log'] += ("stimulus group not handled\n")
        # The same as acquisition for stimulus for spiketrain...

        sti = self._file.stimulus
###        segment_sti = dict((segment.name, segment) for segment in block.segments)

        for name in sti:
#             if name == 'unit_list':
#                 pass  # todo
#             else:
#            segment_name = name
            # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
            segment_name_sti = self._file.epochs
            desc_sti = self._file.get_stimulus(name).unit
###            segment = segment_acq[segment_name]
            segment_sti = segment_name_sti
            if lazy==True:
                times = np.array(())
                lazy_shape = self._file.get_stimulus(name).data.shape
            else:
                current_shape = self._file.get_stimulus(name).data.shape[0] # sample number
                times = np.zeros(current_shape)
                for j in range(0, current_shape): # For testing !
                    times[j]=1./self._file.get_stimulus(name).rate*j+self._file.get_acquisition(name).starting_time # times = 1./frequency [Hz] + t_start [s]
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)  # todo: this is a custom Neo value, general NWB files will not have this - use segment.t_stop instead in that case?
            if segment_sti is not None:
                spiketrain.segment_sti = segment_sti
                segment_sti.spiketrains.append(spiketrain)


    def _handle_processing_group(self, block):
        print("*** def _handle_processing_group ***")


    def _handle_analysis_group(self, block):
        print("*** def _handle_analysis_group ***")
        #block.annotations['file_read_log'] += ("analysis group not handled\n")


    def _write_segment(self, segment):
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.

        nwb_epoch = self._file.add_epoch(
                                         self._file, 
                                         self._file.epochs, #segment.name
                                         #start_time=time_in_seconds(segment.t_start),
                                         start_time=self._handle_epochs_group(True, Block)[2][0],
###                                         start_time=time_in_seconds(self._handle_epochs_group(True, Block)[2][0]),
                                         #stop_time=time_in_seconds(segment.t_stop),
                                         stop_time=self._handle_epochs_group(True, Block)[2][-1],
                                         )

        for i, signal in enumerate(chain(self._handle_epochs_group(True, Block)[0].analogsignals, self._handle_epochs_group(True, Block)[0].irregularlysampledsignals)): # segment.analogsignals, segment.irregularlysampledsignals
            self._write_signal(signal, nwb_epoch, i)
        self._write_spiketrains(self._handle_acquisition_group, self._handle_epochs_group(True, Block)[0]) #(segment.spiketrains, segment)
        for i, event in enumerate(self._handle_epochs_group(True, Block)[0].events): # segment.event
            self._write_event(event, nwb_epoch, i)
        for i, neo_epoch in enumerate(self._handle_epochs_group(True, Block)[0].epochs): # segment.epochs
            self._write_neo_epoch(neo_epoch, nwb_epoch, i)


    def _write_signal(self, signal, epoch, i):
        print("   ")
        print("*** def _write_signal ***")
        signal_name = signal.name or "signal{0}".format(i)
        ts_name = "{0}_{1}".format(signal.segment.name, signal_name)

        #ts = self._file.make_group("<MultiChannelTimeSeries>", ts_name, path="/acquisition/timeseries") ###
##        conversion, base_unit = _decompose_unit(signal.units)
#        attributes = {"conversion": conversion,
#                      "unit": base_unit,
#                      "resolution": float('nan')}

        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            signal.sampling_rate = sampling_rate
#            ts.set_dataset("starting_time", time_in_seconds(signal.t_start),
#                           attrs={"rate": float(sampling_rate)})
#        elif isinstance(signal, IrregularlySampledSignal):
#            ts.set_dataset("timestamps", signal.times.rescale('second').magnitude)
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))
#        ts.set_dataset("data", signal.magnitude,
#                       dtype=np.float64,  #signal.dtype,
#                       attrs=attributes)
#        ts.set_dataset("num_samples", signal.shape[0])   # this is supposed to be created automatically, but is not
#        #ts.set_dataset("num_channels", signal.shape[1])
#        ts.set_attr("source", signal.name or "unknown")
#        ts.set_attr("description", signal.description or "")

        self._file.add_epoch(
                            epoch,
                            signal_name,                            
                            start_time = time_in_seconds(signal.segment.t_start),
                            stop_time = time_in_seconds(signal.segment.t_stop),
#                           ts
                            )



    def _write_spiketrains(self, spiketrains, segment):
        print("*** def _write_spiketrains ***")

    def _write_event(self, event, nwb_epoch, i):
        print("*** def _write_event ***")
        event_name = event.name or "event{0}".format(i)
        ts_name = "{0}_{1}".format(event.segment.name, event_name)

#        ts = self._file.make_group("<AnnotationSeries>", ts_name, path="/acquisition/timeseries")
#        ts.set_dataset("timestamps", event.times.rescale('second').magnitude)
#        ts.set_dataset("data", event.labels)
#        ts.set_dataset("num_samples", event.size)   # this is supposed to be created automatically, but is not
#        ts.set_attr("source", event.name or "unknown")
#        ts.set_attr("description", event.description or "")

        self._file.add_epoch_ts(
                               nwb_epoch,
                               time_in_seconds(event.segment.t_start),
                               time_in_seconds(event.segment.t_stop),
                               event_name,
#                               ts
                                )


    def _write_neo_epoch(self, neo_epoch, nwb_epoch, i):
        print("*** def _write_neo_epoch ***")



def time_in_seconds(t):
    print("*** def time_in_seconds ***")
    return float(t.rescale("second"))
    print("float(t.rescale(second)) = ",float(t.rescale("second")))


def _decompose_unit(unit):
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
    conversion = 1.0
    def _decompose(unit):
        dim = unit.dimensionality
        print("dim = ", dim)
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")

        print("list(dim.keys())[0] = ", list(dim.keys())[0])
        print("list(dim.values())[0] = ", list(dim.values())[0])
        print("list(dim.values())[1] = ", list(dim.values())[:])
        print("dim.unicode = ", dim.unicode)


###        uq, n = dim.items()[0]
#
#        print("unit.dimensionality.items()[0] = ", unit.dimensionality.items()[0])
#        print("unit.dimensionality.items()[0] = ", unit.dimensionality)
#        print("unit.dimensionality[0] = ", unit.dimensionality[0])

#        if n != 1:
#            raise NotImplementedError("Compound units not yet supported")
#        uq_def = uq.definition
#        return float(uq_def.magnitude), uq_def
#    conv, unit2 = _decompose(unit)
#    while conv != 1:
#        conversion *= conv
#        unit = unit2
#        conv, unit2 = _decompose(unit)
#    return conversion, unit.dimensionality.keys()[0].name


prefix_map = {
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano'
}
