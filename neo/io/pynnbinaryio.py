# -*- coding: utf-8 -*-
"""
Class for reading/writing pyNN output files saved in numpy binary format.

pyNN is available at http://neuralensemble.org/trac/PyNN

Supported : Read/Write

@author : pyger

"""

from baseio import BaseIO
#from neo.core import *
from ..core import *
import numpy

class PyNNBinaryIO(BaseIO):
    
    is_readable        = True
    is_writable        = False  
    has_header         = True
    name               = "pyNN Numpy Binary File"
    supported_objects  = [Block, Segment]
    readable_objects   = [Segment, SpikeTrain, SpikeTrainList, AnalogSignal, AnalogSignalList]
    writeable_objects  = []
    
    read_params        = { Segment : [] }
    write_params       = None    
    
    has_header         = True
    is_streameable     = False
    extensions         = ['pynn']
    
    def __init__(self , filename=None , **kargs ) :
        BaseIO.__init__(self)
        self.filename = filename
        self.fileobj  = open(self.filename, 'r')        

    def _check_metadata(self, metadata):
        if 'dt' in metadata:
            if metadata['dt'] is None and 'dt' in metadata:
                metadata['dt'] = metadata['dt']
        if not ('id_list' in metadata) or (metadata['id_list'] is None):
            if ('first_id' in metadata) and ('last_id' in metadata):
                metadata['id_list'] = range(int(metadata['first_id']), int(metadata['last_id'])+1)
            else:
                raise Exception("id_list can not be infered while reading %s" %self.filename)
        elif isinstance(metadata['id_list'], int): # allows to just specify the number of neurons
            metadata['id_list'] = range(metadata['id_list'])
        elif not isinstance(metadata['id_list'], list):
            raise Exception("id_list should be an int or a list !")
        return metadata

    def get_data(self):
        data = numpy.load(self.fileobj)['data']
        self.fileobj.seek(0)
        return data

    def read(self , **kargs):
        """
        Read the file.
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment(**kargs)
    
    def read_segment(self, **kargs):
        seg       = Segment()        
        metadata  = self.read_header()
        if metadata['variable'] == 'spikes':
            spk_list            = self.read_spiketrainlist(**kargs)            
            seg._spiketrains   += spk_list.spiketrains.values() 
        else:
            ag_list             = self.read_analogsignallist(**kargs)
            seg._analogsignals += ag_list
        return seg
    
    def read_spiketrain(self, id, **kargs):
        metadata    = self.read_header()
        metadata.update(kargs)
        if metadata[name] != 'spikes':
            raise Exception("The pyNN file contain analog signals !")
        signals     = self.get_data()
        idx         = numpy.where(spikes[:,0] == id)[0]
        signal      = signals[idx, 1]
        res         = SpikeTrain(spike_times=spike_times)
        return res
        
    def read_analogsignal(self, id, **kargs):
        metadata    = self.read_header()
        metadata.update(kargs)
        if metadata['variable'] == 'spikes':
            raise Exception("The pyNN file contain spikes signals !")
        signals     = self.get_data()
        idx         = numpy.where(signals[:, 1] == id)[0]
        signal      = signals[idx, 0]
        res         = AnalogSignal(signal=signal, dt=metadata['dt'])
        return res
        
    def read_spiketrainlist(self, **kargs):
        metadata = self.read_header()
        metadata.update(kargs)
        if metadata['variable'] != 'spikes':
            raise Exception("The pyNN file contain analog signals !")
        metadata = self._check_metadata(metadata)
        spikes   = self.get_data()
        N        = len(spikes)        
        if N > 0:
            idx          = numpy.argsort(spikes[:,1])
            spikes       = spikes[idx]
            break_points = numpy.where(numpy.diff(spikes[:, 1]) > 0)[0] + 1
            break_points = numpy.concatenate(([0], break_points, [N]))
            res          = []
            for idx in xrange(len(break_points)-1):
                id  = spikes[break_points[idx], 1]
                if id in metadata['id_list']:
                    nrn  = Neuron(id = id)
                    data = spikes[break_points[idx]:break_points[idx+1], 0]
                    res += [SpikeTrain(spike_times=1e-3*data, neuron=nrn, **kargs)]
        result = SpikeTrainList(spiketrains=res, **kargs)
        result.complete(metadata['id_list'])      
        return result

    def read_analogsignallist(self, **kargs):
        metadata = self.read_header()
        metadata.update(kargs)
        if metadata['variable'] == 'spikes':
            raise Exception("The pyNN file contain spikes signals !")
        metadata = self._check_metadata(metadata)
        signals  = self.get_data()
        res      = []
        for id in metadata['id_list']:
            idx    = numpy.where(signals[:, 1] == id)[0]
            signal = signals[idx, 0]
            if len(signal) > 0:
                res += [AnalogSignal(signal=signal, dt=metadata['dt'])]
        return res

    def read_header(self):
        metadata = {}
        for name,value in numpy.load(self.fileobj)['metadata']:
            if name == 'variable':
                metadata[name] = value
            else:
                metadata[name] = eval(value)
        self.fileobj.seek(0)
        return metadata    