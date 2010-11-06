# -*- coding: utf-8 -*-
"""

Classe for reading/writing SpikeTrain in a text file.
It is the simple case where different spiketrain are written line by line.

Supported : Read/Write



@author : sgarcia

"""


from baseio import BaseIO
#from neo.core import *
from ..core import *
import numpy
from numpy import *




from baseio import BaseIO

from numpy import *




class AsciiSpikeIO(BaseIO):
    """
    Classe for reading/writing SpikeTrain in a text file.
    

    **Example**
    
    #read a file
    io = AsciiSpikeIO(filename = 'myfile.txt')
    seg = io.read() # read the entire file
    seg.get_spiketrains() # return all spiketrain
    
    # write a file
    io = AsciiSpikeIO(filename = 'myfile.txt')
    seg = Segment()
    io.write(seg)

    
    
    """
    
    is_readable        = True
    is_writable        = True
    
    supported_objects  = [Segment , SpikeTrain]
    readable_objects   = [Segment]
    writeable_objects  = [Segment]    
    
    has_header         = False
    is_streameable     = False
    
    read_params        = {
                            Segment : [
                                        ('delimiter' , {'value' :  '\t', 'possible' : ['\t' , ' ' , ',' , ';'] }) ,
                                        ('t_start' , { 'value' : 0., } ),
                                        ]
                            }
    write_params       = {
                            Segment : [
                                        ('delimiter' , {'value' :  '\t', 'possible' : ['\t' , ' ' , ',' , ';'] }) ,
                                        ]
                            }

    name               = None
    extensions          = [ 'txt' ]
    
    mode = 'file'
    


    def __init__(self , filename = None) :
        """
        This class read/write SpikeTrains in a text file.
        Each row is a spiketrain.
        
        **Arguments**
        
        filename : the filename to read/write
        
        """
        BaseIO.__init__(self)
        self.filename = filename

    def read(self , **kargs):
        """
        Read the file.
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment( **kargs)
    
    def read_segment(self, 
                            delimiter = '\t',
                            t_start = 0.,
                            ):
        """
        **Arguments**
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'
            t_start : time start of all spiketrain 0 by default
        """
        seg = Segment()
        
        f = open(self.filename, 'Ur')
        i = 0
        for line in f :
            
            all = line[:-1].split(delimiter)
            
            if all[-1] == '': all = all[:-1]
            if all[0] == '': all = all[1:]
            spike_times = array(all).astype('f')
            
            spiketr = SpikeTrain(spike_times = spike_times,
                                    t_start = t_start)
            spiketr.channel = i
            seg._spiketrains.append(spiketr)
            i+=1
        f.close()
        
        return seg
        
    def write(self , *args , **kargs):
        """
        Write SpikeTrain of a Segment in a txt file.
        See write_segment for detail.
        """
        self.write_segment(*args , **kargs)

    def write_segment(self, segment,
                                delimiter = '\t',
                                ):
        """
        Write SpikeTrain of a Segment in a txt file.
        Each row is a spiketrain.
        
         **Arguments**
            segment : the segment to write. Only analog signals will be written.
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'            
            
            information of t_start is lost
            
        """
        
        f = open(self.filename, 'w')
        
        for s,spiketr in enumerate(segment.get_spiketrains()) :
            for ts in spiketr.spike_times :
                f.write('%f%s'% (ts , delimiter) )
            f.write('\n')
        f.close()
        

