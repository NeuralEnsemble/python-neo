# encoding: utf-8
"""
Class for reading/writing analog signals in a text file.
Each columns represents a AnalogSignal. All AnalogSignal have the same sampling rate.
Covers many case when part of a file can be viewed as a CSV format.

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


class AsciiSignalIO(BaseIO):
    """

    Class for reading signal in generic ascii format.
    Columns respresents signal. They share all the same sampling rate.
    The sampling rate is externally known or the first columns could hold the time
    vector.
    
    Usage:
        >>> from neo import io
        >>> r = io.AsciiSignalIO(filename='File_asciisignal_2.txt')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print seg.analogsignals
        [<AnalogSignal(array([ 39.0625    ,   0.        ,   0.        , ..., -26.85546875 ...

    """
    
    is_readable        = True
    is_writable        = True
    
    supported_objects  = [ Segment , AnalogSignal]
    readable_objects   = [ Segment]
    writeable_objects  = [Segment]

    has_header         = False
    is_streameable     = False

    read_params        = {
                            Segment : [
                                        ('delimiter' , {'value' :  '\t', 'possible' : ['\t' , ' ' , ',' , ';'] }) ,
                                        ('usecols' , { 'value' : None , 'type' : int } ),
                                        ('skiprows' , { 'value' :0 } ),
                                        ('timecolumn' , { 'value' : None, 'type' : int } ) ,
                                        ('unit' , { 'value' : 'V', } ),
                                        ('sampling_rate' , { 'value' : 1000., } ),
                                        ('t_start' , { 'value' : 0., } ),
                                        ('method' , { 'value' : 'homemade', 'possible' : ['genfromtxt' , 'csv' , 'homemade' ] }) ,
                                        ]
                            }
    write_params       = {
                            Segment : [
                                        ('delimiter' , {'value' :  '\t', 'possible' : ['\t' , ' ' , ',' , ';'] }) ,
                                        ('writetimecolumn' , { 'value' : True,  } ) ,
                                        ]
                            }
    
    name               = None
    extensions          = [ 'txt' , 'asc', ]
    
    mode = 'file'

    def __init__(self , filename = None) :
        """
        This class read/write AnalogSignal in a text file.
        Each signal is a column.
        One of the column can be the time vector
        
        Arguments:
            filename : the filename to read/write
        """
        BaseIO.__init__(self)
        self.filename = filename
    
    def read_segment(self, 
                                        lazy = False,
                                        cascade = True,
                                        delimiter = '\t',
                                        usecols = None,
                                        skiprows =0,
                                        
                                        timecolumn = None,
                                        sampling_rate = 1.*pq.Hz,
                                        t_start = 0.*pq.s,
                                        
                                        unit = pq.V,
                                        
                                        method = 'genfromtxt',
                                        
                                        ):
        """
        Arguments:
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'
            usecols : if None take all columns otherwise a list for selected columns
            skiprows : skip n first lines in case they contains header informations
            timecolumn :  None or a valid int that point the time vector
            samplerate : the samplerate of signals if timecolumn is not None this is not take in account
            t_start : time of the first sample
            unit : unit of AnalogSignal can be a str or directly a Quantities
            
            method :  'genfromtxt' or 'csv' or 'homemade'
                        in case of bugs you can try one of this methods
                        
                        'genfromtxt' use numpy.genfromtxt
                        'csv' use cvs module
                        'homemade' use a intuitive more robust but slow method
            
        """
        seg = Segment(file_origin = os.path.basename(self.filename))
        if not cascade:
            return seg
        
        if type(sampling_rate) == float or type(sampling_rate)==int:
            # if not quantitities Hz by default
            sampling_rate = sampling_rate*pq.Hz
        
        if type(t_start) == float or type(t_start)==int:
            # if not quantitities s by default
            t_start = t_start*pq.s
        
        unit = pq.Quantity(1, unit)
        
            
        
        #loadtxt
        if method == 'genfromtxt' :
            sig = np.genfromtxt(self.filename, 
                                        delimiter = delimiter,
                                        usecols = usecols ,
                                        skiprows = skiprows,
                                        dtype = 'f')
            if len(sig.shape) ==1:
                sig = sig[:,newaxis]
        elif method == 'csv' :
            tab = [l for l in  csv.reader( file(self.filename,'rU') , delimiter = delimiter ) ]
            tab = tab[skiprows:]
            sig = np.array( tab , dtype = 'f')
        elif method == 'homemade' :
            fid = open(self.filename,'rU')
            for l in range(skiprows):
                fid.readline()
            tab = [ ]
            for line in fid.readlines():
                line = line.replace('\r','')
                line = line.replace('\n','')
                l = line.split(delimiter)
                while '' in l :
                    l.remove('')
                tab.append(l)
            sig = np.array( tab , dtype = 'f')
        
        if timecolumn is not None:
            sampling_rate = 1./np.mean(np.diff(sig[:,timecolumn])) * pq.Hz
            t_start = sig[0,timecolumn] * pq.s
        
        
        
        for i in range(sig.shape[1]) :
            if timecolumn == i : continue
            if usecols is not None and i not in usecols: continue
            
            if lazy:
                signal = [ ]*unit
            else:
                signal = sig[:,i]*unit
            
            anaSig = AnalogSignal( signal , sampling_rate = sampling_rate ,t_start =t_start, name = 'Column %d'%i)
            if lazy:
                anaSig.lazy_shape = sig.shape
            anaSig.annotate( channel_index = i )
            seg.analogsignals.append( anaSig )
        
        create_many_to_one_relationship(seg)
        return seg

    def write_segment(self, segment,
                                delimiter = '\t',
                                
                                skiprows =0,
                                writetimecolumn = True,
                                
                                ):
        """
        Write a segment and AnalogSignal in a text file.
        
         **Arguments**
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'
            writetimecolumn :  True or Flase write time vector as first column
        """
        l = [ ]
        if writetimecolumn is not None:
            l.append(segment._analogsignals[0].times[:,newaxis])
        for anaSig in segment.analogsignals:
            l.append(anaSig.magnitude[:,newaxis])
        sigs = np.concatenate(l, axis=1)
        #print sigs.shape
        np.savetxt(self.filename , sigs , delimiter = delimiter)


