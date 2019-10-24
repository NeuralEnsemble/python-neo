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
from pynwb import NWBFile,TimeSeries, get_manager
from pynwb.base import ProcessingModule
from pynwb.ecephys import ElectricalSeries, Device, EventDetection
from pynwb.behavior import SpatialSeries
from pynwb.image import ImageSeries
#from pynwb.core import set_parents
from pynwb.spec import NWBAttributeSpec # Attribute Specifications
from pynwb.spec import NWBDatasetSpec # Dataset Specifications
from pynwb.spec import NWBGroupSpec
from pynwb.spec import NWBNamespace
from pynwb.spec import NWBNamespaceBuilder
from hdmf.spec import LinkSpec, GroupSpec, DatasetSpec, SpecNamespace,\
                       NamespaceBuilder, AttributeSpec, DtypeSpec, RefSpec
from hdmf import *

# allensdk package
#import allensdk
#from allensdk import *
#from pynwb import load_namespaces
#from allensdk.brain_observatory.nwb.metadata import load_LabMetaData_extension
#from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema
#load_LabMetaData_extension(OphysBehaviorMetaDataSchema, 'AIBS_ophys_behavior')
#load_LabMetaData_extension(OphysBehaviorTaskParametersSchema, 'AIBS_ophys_behavior')


neo_extension = {"fs": {"neo": {
    "info": {
        "name": "Neo TimeSeries extension",
        "version": "0.9.0",
        "date": "2019",
        "authors": "Elodie Legouée, Andrew Davison",
        "contacts": "elodie.legouee@unic.cnrs-gif.fr, andrew.davison@unic.cnrs-gif.fr",
        "description": ("Extension defining a new TimeSeries type, named 'MultiChannelTimeSeries'")
    },

    "schema": {
        "<MultiChannelTimeSeries>/": {
            "description": "Similar to ElectricalSeries, but without the restriction to volts",
            "merge": ["core:<TimeSeries>/"],
            "attributes": {
                "ancestry": {
                    "data_type": "text",
                    "dimensions": ["2"],
                    "value": ["TimeSeries", "MultiChannelTimeSeries"],
                    "const": True},
                "help": {
                    "data_type": "text",
                    "value": "A multi-channel time series",
                    "const": True}},
            "data": {
                "description": ("Multiple measurements are recorded at each point of time."),
                "dimensions": ["num_times", "num_channels"],
                "data_type": "float32"},
        },

        "<AnnotatedIntervalSeries>/": {
            "description": "Represents a series of annotated time intervals",
            "merge": ["core:<AnnotationSeries>/"],
            "attributes": {
                "ancestry": {
                    "data_type": "text",
                    "dimensions": ["3"],
                    "value": ["TimeSeries", "AnnotationSeries", "AnnotatedIntervalSeries"],
                    "const": True},
                "help": {
                    "data_type": "text",
                    "value": "A series of annotated time intervals",
                    "const": True}},
            "durations": {
                "description": ("Durations for intervals whose start times are stored in timestamps."),
                "data_type": "float64!",
                "dimensions": ["num_times"],
                "attributes": {
                    "unit": {
                        "description": ("The string \"Seconds\""),
                        "data_type": "text", "value": "Seconds"}}
            },
        }
    }
}}}

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

    def __init__(self, filename, mode):
        """
        Arguments:
            filename : the filename
        """
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
#        if mode=='r':
#            self.read_block()
#        else:
#            self.write_block()
##        if mode=='w':
##            self.write_block(self.block)

    def read_block(self, lazy=False, cascade=True, **kwargs):
#        print("*** read_block ***")
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        _file = io.read()
        self._lazy = lazy

        file_access_dates = _file.file_create_date
        identifier = _file.identifier
        if identifier == '_neo': # this is an automatically generated name used if block.name is None
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
            self._handle_processing_group(block)
            self._handle_analysis_group(block)
        self._lazy = False
        return block

    def write_block(self, block, **kwargs):
#        print("*** ----------- write_block ------------ ***")

        start_time = datetime.now()
        self._file = NWBFile(self.filename,            
                               session_start_time=start_time,
#                               identifier=block.name or "_neo",
                               identifier='test',
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
                               time_intervals=None,
                               units=None,
                               modules=None,
                               electrodes=None,
                               electrode_groups=None,
                               ic_electrodes=None,
                               sweep_table=None,
                               imaging_planes=None,
                               ogen_sites=None,
                               devices=None,
                               #subject=None
                               )
        io_nwb = pynwb.NWBHDF5IO(self.filename, manager=get_manager(), mode='w')
#        print("io_nwb = ", io_nwb)

        file_access_dates = self._file.file_create_date
        identifier = self._file.identifier
        if identifier == '_neo':  # this is an automatically generated name used if block.name is None
            identifier = None
        description = self._file.session_description
        if description == "no description":
            description = None

#        print("block.segments = ", block.segments)
        for segment in block.segments:
            print("segment = ", segment)
            self._write_segment(self._file, segment)            

        print("END loop block.segment")
        io_nwb.write(self._file)
        print("io_nwb.write")
        io_nwb.close()
        print("io_nwb.close")

    def _handle_general_group(self, block):
        pass

    def _handle_epochs_group(self, _file, block):
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        epochs = _file.acquisition
        for key in epochs:
            timeseries = []
            current_shape = _file.get_acquisition(key).data.shape[0] # or 1 if multielectrode ?
            times = np.zeros(current_shape)

            for j in range(0, current_shape):# to do w/ ecephys data (e.g. multielectrode: how is it organised?)
                times[j]=1./_file.get_acquisition(key).rate*j+_file.get_acquisition(key).starting_time
                if times[j] == _file.get_acquisition(key).starting_time:
                    t_start = times[j] * pq.second
                elif times[j]==times[-1]:
                    t_stop = times[j] * pq.second
                else:
                    timeseries.append(self._handle_timeseries(_file, key, times[j]))                    
            segment = Segment(name=j)
            for obj in timeseries:
                obj.segment = segment
                if isinstance(obj, AnalogSignal):
                    segment.analogsignals.append(obj)
                elif isinstance(obj, IrregularlySampledSignal):
                    segment.irregularlysampledsignals.append(obj)
                elif isinstance(obj, Event):
                    segment.events.append(obj)
                elif isinstance(obj, Epoch):
                    segment.epochs.append(obj)
            segment.block = block
            block.segments.append(segment)

            segment.times=times
            return segment, obj, times


    def _handle_timeseries(self, _file, name, timeseries):
        for i in _file.acquisition:
            data_group = _file.get_acquisition(i).data*_file.get_acquisition(i).conversion
            dtype = data_group.dtype
            data = data_group

            if dtype.type is np.string_:
                if self._lazy:
                    times = np.array(())
                else:
                    times = _file.get_acquisition(i).timestamps
                duration = 1/_file.get_acquisition(i).rate
                if durations:
                    # Epoch
                    if self._lazy:
                        durations = np.array(())
                    obj = Epoch(times=times,
                                durations=durations,
                                labels=data_group,
                                units='second')
                else:
                    # Event
                    obj = Event(times=times,
                                labels=data_group,
                                units='second')
            else:
                units = _file.get_acquisition(i).unit

                current_shape = _file.get_acquisition(i).data.shape[0] # number of samples
                times = np.zeros(current_shape)
                for j in range(0, current_shape):
                    times[j]=1./_file.get_acquisition(i).rate*j+_file.get_acquisition(i).starting_time
                    if times[j] == _file.get_acquisition(i).starting_time:
                        # AnalogSignal
                        sampling_metadata = times[j]
                        t_start = sampling_metadata * pq.s
                        sampling_rate = _file.get_acquisition(i).rate * pq.Hz
                        obj = AnalogSignal( 
                                           data_group,
                                           units=units,
                                           sampling_rate=sampling_rate,
                                           t_start=t_start,
                                           name=name)
                    elif _file.get_acquisition(i).timestamps:
                        if self._lazy:
                            time_data = np.array(())
                        else:
                            time_data = _file.get_acquisition(i).timestamps
                        obj = IrregularlySampledSignal(
                                                    data_group,
                                                    units=units,
                                                    time_units=pq.second)
            return obj

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
                current_shape = _file.get_stimulus(name).data.shape[0] # sample number
                times = np.zeros(current_shape)
                for j in range(0, current_shape):
                    times[j]=1./_file.get_stimulus(name).rate*j+_file.get_acquisition(name).starting_time # times = 1./frequency [Hz] + t_start [s]
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)

    def _handle_processing_group(self, block):
        pass

    def _handle_analysis_group(self, block):
        pass

    def _write_segment(self, _file, segment):
        start_time = segment.t_start
        stop_time = segment.t_stop

        nwb_epoch = self._file.add_epoch(       
                                        self._file,
                                        segment.name,
                                        start_time=float(start_time),
                                        stop_time=float(stop_time),
                                        )
        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            #print("signal = ", signal)
            print("i = ", i)
            self._write_signal(signal, nwb_epoch, i, segment)
        self._write_spiketrains(segment.spiketrains, segment)
        for i, event in enumerate(segment.events):
#            print("event = ", event)
            self._write_event(event, nwb_epoch, i)
        for i, neo_epoch in enumerate(segment.epochs):
#            print("neo_epoch = ", neo_epoch)
            self._write_neo_epoch(neo_epoch, nwb_epoch, i)


    def _write_signal(self, signal, epoch, i, segment):
        # i=index
        print("-------------------------------- segment.ind = ", segment.index)
        print("*** def _write_signal ***")
        print("segment.name = ", segment.name) # index

#        print("i = ", i)

        signal_name = signal.name or "signal{0}".format(i)
        ts_name = "{0}".format(signal_name)

        # Create a builder for the namespace
        ns_builder_signal = NWBNamespaceBuilder('Extension to neo signal', "neo_signal")
#        print("ns_builder_signal = ", ns_builder_signal)
        ns_builder_signal.include_type('TimeSeries', namespace='core')

        # Group Specifications
        # Create extensions
        ts_signal = NWBGroupSpec('A custom TimeSeries interface for signal',
#                           attributes=[NWBAttributeSpec('timeseries', '', 'int')],
                           #datasets=[],
                           #groups=[],
                           groups=[NWBGroupSpec('An included TimeSeries instance for signal', neurodata_type_inc='TimeSeries')],
                           neurodata_type_inc='TimeSeries',
                           neurodata_type_def='MultiChannelTimeSeries'
                          )
        print("ts_signal = ", ts_signal)
        print("   ")

        # Add the extension
        ext_source_signal = 'nwb_neo_extension_signal.specs.yaml'
        ns_builder_signal.add_spec(ext_source_signal,
                            ts_signal
                            )
#        print("ns_builder_signal = ", ns_builder_signal)

        # Save the namespace and extensions
        ns_path_signal = "nwb_neo_extension_signal.namespace.yaml"
        ns_builder_signal.export(ns_path_signal)

        # Incorporating extensions
        load_namespaces(ns_path_signal)

#        NWBSignalSeries = get_class('MultiChannelTimeSeries', 'neo_signal') # Classe abstraite !
        # TimeSeries
        NWBSignalSeries = get_class('TimeSeries', 'neo_signal') # class pynwb.base.TimeSeries
        # NWB File
        #NWBSignalSeries = get_class('NWBFile', namespace='core') # class pynwb.base.TimeSeries
#        print("NWBSignalSeries = ", NWBSignalSeries)

        # NWB File
#        self._file

###        pynwb.file = NWB File
#        ts = NWBSignalSeries(
#                            identifier='',
#                            session_description='session_description',
#                            session_start_time=datetime(2019, 10,22)
#                                )

#        # TimeSeries
#        ts = NWBSignalSeries(
#                                name='',
#                                data=np.arange(10),
#                                resolution=3.0,
#                                rate=10.0,
#                                unit='unit of data',
#                            )

    
#        MultiChannelTimeSeries = pynwb.core.NWBDataInterface(name='test_multi')
#        print("MultiChannelTimeSeries = ", MultiChannelTimeSeries)




        #ts = NWBSignalSeries('MultiChannelTimeSeries', time_series=self._file ,rate=1.0)
        ts = NWBSignalSeries(
###        ts = TimeSeries(
                        'MultiChannelTimeSeries123_index_%d_%s' % (i, segment.name), #index
                        #'MultiChannelTimeSeries123_%d_%s' % (ind, segment.name), #index
#                        'MultiChannelTimeSeries123_%s' % (segment.name), #index
                        #'TimeSeries', # name of the class
                        [ts_signal], 
                        rate=1.0
                       )
        ##ts = NWBSignalSeries('MultiChannelTimeSeries', time_series=MultiChannelTimeSeries ,rate=1.0) 
        print("   ")
        print("ts = ", ts)
        print("   ")
#        print("self._file = ", self._file)
        print("self._file.acquisition = ", self._file.acquisition)
#        print("self._file.epochs = ", self._file.epochs)
     #   self._file.add_acquisition(ts)
#        print("ok")

        ###test_ac = self._file.add_acquisition(ts)
#        test_ac = self._file.get_acquisition('MultiChannelTimeSeries')
#        print("test_ac = ", test_ac)


        """
        # create a builder for the namespace
        ns_builder = NWBNamespaceBuilder("Extension for use in my laboratory", "mylab")
        """


        conversion = _decompose_unit(signal.units)
        attributes = {"conversion": conversion,
                      "resolution": float('nan')}

        if isinstance(signal, AnalogSignal):
            print("isinstance(signal, AnalogSignal)")
            test_ac = self._file.add_acquisition(ts)
            print("test_ac = ", test_ac)

            sampling_rate = signal.sampling_rate.rescale("Hz")
            signal.sampling_rate = sampling_rate
            ts_signal.add_dataset(
                            doc='',
                            neurodata_type_def='MultiChannelTimeSeries',
                          )
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))
        print("END def _write_signal")

    def _write_spiketrains(self, spiketrains, segment):
        print("*** def _write_spiketrains ***")
        """
        mod = NWBGroupSpec('A custom TimeSeries interface',
                            attributes=[],
                            datasets=[],
                            groups=[],
                            neurodata_type_inc='TimeSeries',
                            neurodata_type_def='Module')

        ext_source = 'nwb_neo_extension.specs.yaml'
        mod.add_dataset(
                        doc='',
                        neurodata_type_def='Module',
                      )
        """

#    def _write_event(self, _file, event, nwb_epoch):
    def _write_event(self, event, nwb_epoch, i):
        print("*** def _write_event ***")

        """
        event_name = event.name or "event{0}".format(i)
#        print("event_name = ", event_name)
        ts_name = "{0}".format(event_name)
#        print("ts_name = ", ts_name)

        ts = NWBGroupSpec('A custom TimeSeries interface',
                           attributes=[],
                           datasets=[],
                           groups=[],
                           neurodata_type_inc='TimeSeries',
                           neurodata_type_def='AnnotationSeries')
#        print("ts = ", ts)

        ext_source = 'nwb_neo_extension.specs.yaml'
        ts.add_dataset(
                        doc='',
                        neurodata_type_def='AnnotationSeries',
                      )

        self._file.add_epoch(            
                               time_in_seconds(event.times[0]),
                               time_in_seconds(event.times[1]),
                                )
        """


    def _write_neo_epoch(self, neo_epoch, nwb_epoch, i):
        print("*** def _write_neo_epoch ***")
        neo_epoch_name = neo_epoch.name or "intervalseries{0}".format(i)
#        print("neo_epoch_name = ", neo_epoch_name)
#        ts_name = "{0}_{1}".format(neo_epoch.segment.name, neo_epoch_name)
#        print("ts_name = ", ts_name)

#        ts.set_dataset("timestamps", neo_epoch.times.rescale('second').magnitude)
#        ts.set_dataset("durations", neo_epoch.durations.rescale('second').magnitude)
#        ts.set_dataset("data", neo_epoch.labels)
#        ts.set_attr("source", neo_epoch.name or "unknown")
#        ts.set_attr("description", neo_epoch.description or "")

#        print("   ")
###        neo_AnnotatedIntervalSeries = neo_extension["fs"]["neo"]["schema"]["<AnnotatedIntervalSeries>/"]
###        print("neo_AnnotatedIntervalSeries = ", neo_AnnotatedIntervalSeries)


        
        # Create a builder for the namespace
        ns_builder_neo_epoch = NWBNamespaceBuilder('Extension to neo epoch', "neo_epoch")
#        ns_builder = NWBNamespaceBuilder('Extension to neo epoch', "neo_AnnotatedIntervalSeries")
#        print("ns_builder = ", ns_builder)
        ns_builder_neo_epoch.include_type('TimeSeries', namespace='core')
#        ns_builder.include_type('neo_AnnotatedIntervalSeries', namespace='core')        

        # Group Specifications
        # Create extensions
        ts_neo_epoch = NWBGroupSpec('A custom TimeSeries interface',
#                           attributes=[NWBAttributeSpec('timeseries', '', 'int')],
                           #datasets=[],
                           #groups=[],
                           groups=[NWBGroupSpec('An included TimeSeries instance', neurodata_type_inc='TimeSeries')],
                           neurodata_type_inc='TimeSeries',
                           neurodata_type_def='AnnotatedIntervalSeries'
                          )
#        print("ts = ", ts)
#        print("   ")


        # Add the extension
        ext_source_neo_epoch = 'nwb_neo_extension.specs.yaml'
        ns_builder_neo_epoch.add_spec(ext_source_neo_epoch,

                            ts_neo_epoch
                            )

        # Include an existing namespace
#        ns_builder_neo_epoch.include_namespace('collab_ts')

        # Save the namespace and extensions
        ns_path_neo_epoch = "nwb_neo_extension.namespace.yaml"
#        print("   ")
        ns_builder_neo_epoch.export(ns_path_neo_epoch)
#        ns_builder.export("AnnotatedIntervalSeries")

        load_namespaces(ns_path_neo_epoch)

#        AutoNeoEpochSeries = get_class('AnnotatedIntervalSeries', 'neo_epoch')
        AutoNeoEpochSeries = get_class('TimeSeries', 'neo_epoch')
#        print("AutoNeoEpochSeries = ", AutoNeoEpochSeries)

        print("END def _write_neo_epoch")
        


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