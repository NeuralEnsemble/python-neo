# -*- coding: utf-8 -*-

from baseio import BaseIO
from neo.core import *
import numpy
from numpy import *

"""
asciisignalio
==================

Classe for reading/writing SpikeTrain in a text file.
It is the simple case where different spiketrain are written line by line.


Classes
-------

AsciiSignalIO          - Classe for reading/writing spiketrain in a text file.

@author : sgarcia

"""



from baseio import BaseIO

from numpy import *




class AsciiSpikeIO(BaseIO):
    """
    Classe for reading/writing SpikeTrain in a text file.

    Cover many case when part of a file can be view as a CVS format.
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = True
    
    supported_objects            = [Segment , SpikeTrain]
    readable_objects    = [Segment]
    writeable_objects    = [Segment]    
    
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
    


    
    def __init__(self ) :
        """
        
        **Arguments**
        
        """
        
        BaseIO.__init__(self)


    def read(self , **kargs):
        """
        Read the file.
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment( **kargs)
    
    def read_segment(self, 
                            filename = '',
                            delimiter = '\t',
                            t_start = 0.,
                            ):
        """
        **Arguments**
            filename : filename
            TODO
            
        """
        seg = Segment()
        
        f = open(filename, 'Ur')
        for line in f :
            all = line[:-1].split(delimiter)
            
            if all[-1] == '': all = all[:-1]
            if all[0] == '': all = all[1:]
            spike_times = array(all).astype('f')
            
            spiketr = SpikeTrain(spike_times = spike_times,
                                    t_start = t_start)
            seg._spiketrains.append(spiketr)
        f.close()
        
        return seg
        
    


    def write(self , *args , **kargs):
        """
        Write segment in a raw file.
        See write_segment for detail.
        """
        self.write_segment(*args , **kargs)

    def write_segment(self, segment,
                                filename = '',
                                delimiter = '\t',
                                
                                skiprows =0,
                                
                                timecolumn = None,
                                
                                ):
        """
        
         **Arguments**
            segment : the segment to write. Only analog signals will be written.
            TODO
        """
        
        f = open(filename, 'w')
        
        for s,spiketr in enumerate(segment.get_spiketrains()) :
            for ts in spiketr.spike_times :
                f.write('%f%s'% (ts , delimiter) )
            f.write('\n')
        f.close()
        

