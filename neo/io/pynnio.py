# -*- coding: utf-8 -*-
"""
Class for reading/writing pyNN output files saved in text format.

pyNN is available at http://neuralensemble.org/trac/PyNN

Supported : Read/Write

@author : pyger

"""

from baseio import BaseIO
#from neo.core import *
from ..core import *

try:
    import TableIO
    HAVE_TABLEIO = True
except ImportError:
    HAVE_TABLEIO = False
    
DEFAULT_BUFFER_SIZE = 100000

def _savetxt(filename, data, format, delimiter):
    """
    Due to the lack of savetxt in older versions of numpy
    we provide a cut-down version of that function.
    """
    f = open(filename, 'w')
    for row in data:
        f.write(delimiter.join([format%val for val in row]) + '\n')
    f.close()


class PyNNIO(BaseIO):
    
    is_readable        = True
    is_writable        = False  
    has_header         = True
    name               = "pyNN Text File"
    supported_objects  = [Segment, SpikeTrain, SpikeTrainList, AnalogSignal, AnalogSignalList]
    readable_objects   = [Segment, SpikeTrain, SpikeTrainList, AnalogSignal, AnalogSignalList]
    writeable_objects  = [SpikeTrainList, AnalogSignalList]
    
    read_params        = { Segment : [] }
    write_params       = None    
    
    has_header         = True
    is_streameable     = False
    extensions         = ['pynn']
    
    filemode = True
    
            
    def __init__(self , filename=None , **kargs ) :
        BaseIO.__init__(self)
        self.filename = filename
    
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

    def get_data(self, sepchar = "\t", skipchar = "#"):
        if HAVE_TABLEIO:
            data = numpy.fliplr(TableIO.readTableAsArray(self.filename, skipchar))
        else:
            contents = self.fileobj.readlines()
            self.fileobj.close()
            for i in xrange(idx, len(contents)):
                line = contents[i].strip().split(sepchar)
                id   = [float(line[-1])]
                id  += map(float, line[0:-1])
                data.append(id)
            logging.debug("Loaded %d lines of data from %s" % (len(data), self))
            data = numpy.array(data, numpy.float32)
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
        if metadata['variable'] != 'spikes':
            raise Exception("The pyNN file contain analog signals !")
        signals     = self.get_data()
        idx         = numpy.where(spikes[:,1] == id)[0]
        signal      = signals[idx, 0]
        res         = SpikeTrain(spike_times=spike_times, **kargs)
        return res
        
    def read_analogsignal(self, id, **kargs):
        metadata    = self.read_header()
        metadata.update(kargs)
        if metadata['variable'] == 'spikes':
            raise Exception("The pyNN file contain spikes signals !")
        signals     = self.get_data()
        signal      = numpy.transpose(data[data[:,0] == id, 1:])[0]
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
            idx          = numpy.argsort(spikes[:,0])
            spikes       = spikes[idx]
            break_points = numpy.where(numpy.diff(spikes[:, 0]) > 0)[0] + 1
            break_points = numpy.concatenate(([0], break_points))
            break_points = numpy.concatenate((break_points, [N]))
            res          = []
            for idx in xrange(len(break_points)-1):
                id  = spikes[break_points[idx], 0]
                if id in metadata['id_list']:
                    nrn  = Neuron(id = int(id))
                    data = spikes[break_points[idx]:break_points[idx+1], 1]
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
        data     = self.get_data()
        res      = []
        for id in metadata['id_list']:
            signal = numpy.transpose(data[data[:,0] == id, 1:])[0]
            if len(signal) > 0:
                res += [AnalogSignal(signal=signal, dt=metadata['dt'])]
        return res

    def read_header(self):
        metadata = {}
        self.fileobj  = open(self.filename, 'r', DEFAULT_BUFFER_SIZE)        
        cmd = ''
        for line in self.fileobj.readlines():
            if line[0] == '#':
                if line[1:].strip().find('variable') == -1:
                    cmd += line[1:].strip() + ';'
                else:
                    tmp = line[1:].strip().split(" = ")
            else:
                break
        exec cmd in None, metadata
        metadata[tmp[0]] = tmp[1]
        return metadata
    
    #def write_spiketrainlist(self, **kargs):
            
    #def write_analogsignallist(self, **kargs):    