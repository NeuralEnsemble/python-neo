import os, sys, numpy
sys.path.append(os.path.abspath('..'))
from neo.io import *
DEFAULT_BUFFER_SIZE = 100000

try:
    import TableIO
    HAVE_TABLEIO = True
except ImportError:
    HAVE_TABLEIO = False
    

class TextFile(BaseFile):
    
    is_readable     = True
    is_writable     = True  
    has_header      = True
    nfiles          = 1
    name            = "pyNN Text File"
    supported_types = [SpikeTrainList, AnalogSignalList]
    
    def __init__(self , filename=None , **kargs ) :
        BaseFile.__init__(self, filename, **kargs)
        self.fileobj = None
        #if self.metadata.has_key('dt'):
            #self.object = ["AnalogSignalList"]
        #else:
            #self.object = ["SpikeTrainList"]
    
    def _get_metadata(self, object):
        """
        Fill the metadata from those of a NeuroTools object before writing the object
        """
        metadata = {}
        if len(object.id_list() > 0):
            metadata['first_id'] = numpy.min(object.id_list())
            metadata['last_id']  = numpy.max(object.id_list())
        if hasattr(object, 'dt'):
            metadata['dt']       = object.dt
        return metadata
    
    def _check_metadata(self, metadata):
        """
        Establish a control/completion/correction of the metadata to create an object by 
        using comparison and data extracted from the metadata.
        """
        if 'dt' in metadata:
            if metadata['dt'] is None and 'dt' in self.metadata:
                metadata['dt'] = self.metadata['dt']
        if not ('id_list' in metadata) or (metadata['id_list'] is None):
            if ('first_id' in self.metadata) and ('last_id' in self.metadata):
                metadata['id_list'] = range(int(self.metadata['first_id']), int(self.metadata['last_id'])+1)
            else:
                raise Exception("id_list can not be infered while reading %s" %self.filename)
        elif isinstance(metadata['id_list'], int): # allows to just specify the number of neurons
            metadata['id_list'] = range(metadata['id_list'])
        elif not isinstance(metadata['id_list'], list):
            raise Exception("id_list should be an int or a list !")
        return metadata

    def get_data(self, sepchar = "\t", skipchar = "#"):
        """
        Load data from a text file and returns an array of the data
        """
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
    
    def write(self, object):
        # can we write to the file more than once? In this case, should use seek, tell
        # to always put the header information at the top?
        # write header
        self.write_header(self.__get_metadata(object))
        self.close()
        self.fileobj = open(self.filename, 'w', DEFAULT_BUFFER_SIZE)
        numpy.savetxt(fileobj, object.raw_data(), fmt = '%g', delimiter='\t')
        self.fileobj.close()

    def read(self, **kargs ):
        """
        bulk read the file at the highest level possible
        """
        if self.object == "AnalogSignal":
            return self.read_analogsignalslist(kargs)
        else:
            return self.read_spiketrainlists(kargs)
        
    def write(self, **kargs):
        """
        bulk write the file at the highest level possible
        """
        if self.object == "AnalogSignal":
            return self.write_analogsignalslist(kargs)
        else:
            return self.write_spiketrainlists(kargs)
    
    def read_spiketrainlists(self, **kargs):
        """
        Read SpikeTrainList objects from a file
        
        Examples:

        """
        self.read_header()        
        p      = self._check_metadata(kargs)
        spikes = self.get_data()
        N      = len(spikes)        
        if N > 0:
            idx          = numpy.argsort(spikes[:,0])
            spikes       = spikes[idx]
            break_points = numpy.where(numpy.diff(spikes[:, 0]) > 0)[0] + 1
            break_points = numpy.concatenate(([0], break_points))
            break_points = numpy.concatenate((break_points, [N]))
            res          = []
            for idx in xrange(len(break_points)-1):
                id  = spikes[break_points[idx], 0]
                if id in p['id_list']:
                    nrn  = Neuron(id = id)
                    data = spikes[break_points[idx]:break_points[idx+1], 1]
                    res += [SpikeTrain(spike_times=data, neuron=nrn)]
        if kargs.has_key('t_start') & kargs.has_key('t_stop'):
            result = SpikeTrainList(spiketrains=res, t_start = kargs['t_start'], t_stop = kargs['t_stop'])
        else:
            result = SpikeTrainList(spiketrains=res)
        result.complete(p['id_list'])        
        return result

    def read_header(self):
        self.metadata = {}
        self.fileobj  = open(self.filename, 'r', DEFAULT_BUFFER_SIZE)        
        cmd = ''
        for line in self.fileobj.readlines():
            if line[0] == '#':
                cmd += line[1:].strip() + ';'
            else:
                break
        #self.fileobj.seek(0)
        exec cmd in None, self.metadata
     
    def write_spiketrainlists(self, **kargs):
        """
        Write SpikeTrainList objects from a file
        
        Examples:

        """
        return _abstract_method(self)
        
    def write_header(self, **kargs):
        """
        Write metadata/header from a file
        
        Examples:

        """
        header_lines = ["# %s = %s" % item for item in metadata.items()]
        fileobj.write("\n".join(header_lines) + '\n')