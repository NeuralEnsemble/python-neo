# encoding: utf-8
"""
Classe for reading/writing SpikeTrains in a text file.
It is the simple case where different spiketrains are written line by line.

Supported : Read/Write

Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship
import numpy as np
import quantities as pq


import csv
import os
from numpy import newaxis


# file no longer exists in Python3
try:
    file
except NameError:
    import io
    file = io.BufferedReader


class AsciiSpikeTrainIO(BaseIO):
    """

    Classe for reading/writing SpikeTrain in a text file.
    Each Spiketrain is a line.
    
    Usage:
        >>> from neo import io
        >>> r = io.AsciiSpikeTrainIO( filename = 'File_ascii_spiketrain_1.txt')
        >>> seg = r.read_segment(lazy = False, cascade = True,)
        >>> print seg.spiketrains

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

    def read_segment(self, 
                            lazy = False,
                            cascade = True,
                            delimiter = '\t',
                            t_start = 0.*pq.s,
                            unit = pq.s,
                            ):
        """
        Arguments:
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'
            t_start : time start of all spiketrain 0 by default
            unit : unit of spike times, can be a str or directly a Quantities
        """
        unit = pq.Quantity(1, unit)

        seg = Segment(file_origin = os.path.basename(self.filename))
        if not cascade:
            return seg
        
        f = file(self.filename, 'Ur')
        for i,line in enumerate(f) :
            if lazy:
                spike_times = [ ]
                t_stop = t_start
            else:
                all = line[:-1].split(delimiter)
                if all[-1] == '': all = all[:-1]
                if all[0] == '': all = all[1:]
                spike_times = np.array(all).astype('f')
                t_stop = spike_times.max()*unit
            
            sptr = SpikeTrain(spike_times*unit, t_start=t_start, t_stop=t_stop)
            sptr.annotate(channel_index = i)
            seg.spiketrains.append(sptr)
        f.close()
        
        create_many_to_one_relationship(seg)
        return seg

    def write_segment(self, segment,
                                delimiter = '\t',
                                ):
        """
        Write SpikeTrain of a Segment in a txt file.
        Each row is a spiketrain.
        
         Arguments:
            segment : the segment to write. Only analog signals will be written.
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'            
            
            information of t_start is lost
            
        """
        
        f = file(self.filename, 'w')        
        for s,sptr in enumerate(segment.spiketrains) :
            for ts in sptr :
                f.write('%f%s'% (ts , delimiter) )
            f.write('\n')
        f.close()
        



