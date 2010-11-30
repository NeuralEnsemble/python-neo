# -*- coding: utf-8 -*-
"""
Class for reading/writing analog signals in a text file.
Covers many case when part of a file can be viewed as a CSV format.

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
import csv




class AsciiSignalIO(BaseIO):
    """
    Class for reading/writing data in a text file.
    Covers many cases when part of a file can be viewed as a CSV format.
    
    **Example**
    
    # read a file
    io = AsciiSignalIO(filename = 'myfile.txt')
    seg = io.read() # read the entire file
    seg.get_analogsignals() # return all AnalogSignals
    
    # write a file
    io = AsciiSignalIO(filename = 'myfile.txt')
    seg = Segment()
    io.write(seg)
    
    """
    
    is_readable        = True
    is_writable        = True
    
    supported_objects  = [Segment , AnalogSignal]
    readable_objects   = [Segment]
    writeable_objects  = [Segment]

    has_header         = False
    is_streameable     = False

    read_params        = {
                            Segment : [
                                        ('delimiter' , {'value' :  '\t', 'possible' : ['\t' , ' ' , ',' , ';'] }) ,
                                        ('usecols' , { 'value' : None , 'type' : int } ),
                                        ('skiprows' , { 'value' :0 } ),
                                        ('timecolumn' , { 'value' : None, 'type' : int } ) ,
                                        ('samplerate' , { 'value' : 1000., } ),
                                        ('t_start' , { 'value' : 0., } ),
                                        ('method' , { 'value' : 'homemade', 'possible' : ['genfromtxt' , 'csv' , 'homemade' ] }) ,
                                        ]
                            }
    write_params       = {
                            Segment : [
                                        ('delimiter' , {'value' :  '\t', 'possible' : ['\t' , ' ' , ',' , ';'] }) ,
                                        ('timecolumn' , { 'value' : None, 'type' : int } ) ,
                                        ]
                            }
    
    name               = None
    extensions          = [ 'txt' ]
    
    mode = 'file'



    
    def __init__(self , filename = None) :
        """
        This class read/write AnalogSignal in a text file.
        Each signal is a column.
        One of the column can be the time vector
        
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
                                        usecols = None,
                                        skiprows =0,
                                        
                                        timecolumn = None,
                                        samplerate = 1000.,
                                        t_start = 0.,
                                        
                                        method = 'genfromtxt',
                                        
                                        ):
        """
        **Arguments**
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'
            usecols : if None take all columns otherwise a list for selected columns
            skiprows : skip n first lines in case they contains header informations
            timecolumn :  None or a valid int that point the time vector
            samplerate : the samplerate of signals if timecolumn is not None this is not take in account
            t_start : time of the first sample
            
            method :  'genfromtxt' or 'csv' or 'homemade'
                        in case of bugs you can try one of this methods
                        
                        'genfromtxt' use numpy.genfromtxt
                        'csv' use cvs module
                        'homemade' use a intuitive more robust but slow method
            
        """
        #loadtxt
        if method == 'genfromtxt' :
            sig = genfromtxt(self.filename, 
                             delimiter = delimiter,
                            usecols = usecols ,
                            skiprows = skiprows,
                            dtype = 'f4')
            if len(sig.shape) ==1:
                sig = sig[:,newaxis]
        elif method == 'csv' :
            tab = [l for l in  csv.reader( open(self.filename,'rU') , delimiter = delimiter ) ]
            tab = tab[skiprows:]
            sig = array( tab , dtype = 'f4')
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
            sig = array( tab , dtype = 'f4')
        
        if timecolumn is not None:
            samplerate = 1./mean(diff(sig[:,timecolumn]))
            t_start = sig[0,timecolumn]
        
        seg = Segment()
        for i in xrange(sig.shape[1]) :
            if usecols is not None :
                if timecolumn == usecols[i] :
                    # time comlumn not a signal
                    continue
            else :
                if timecolumn == i :
                    continue
            #print 'lkjjlkj', len(sig[:,i])
            analogSig = AnalogSignal( signal = sig[:,i] ,
                                                    sampling_rate = samplerate,
                                                    t_start = t_start)
            analogSig.channel = i
            seg._analogsignals.append( analogSig )
        
        return seg


    def write(self , *args , **kargs):
        """
        Write segment in a raw file.
        See write_segment for detail.
        """
        self.write_segment(*args , **kargs)

    def write_segment(self, segment,
                                delimiter = '\t',
                                
                                skiprows =0,
                                
                                timecolumn = None,
                                
                                ):
        """
        Write a segment and AnalogSignal in a text file.
        
         **Arguments**
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'
            skiprows : skip n first lines in case they contains header informations
            timecolumn :  None or a valid int that point the time vector
        """
        
        
        sigs = None
        for analogSig in segment.get_analogsignals():
            if sigs is None :
                sigs = analogSig.signal[:,newaxis]
            else :
                sigs = concatenate ((sigs, analogSig.signal[:,newaxis]) , axis = 1 )
        
        if timecolumn is not None:
            t = segment.get_analogsignals()[0].t()
            print sigs.shape , t.shape
            sigs = concatenate ((sigs, t[:,newaxis]*nan) , axis = 1 )
            sigs[:,timecolumn+1:] = sigs[:,timecolumn:-1].copy()
            sigs[:,timecolumn] = t
        
        savetxt(self.filename , sigs , delimiter = delimiter)

