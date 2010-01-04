# -*- coding: utf-8 -*-

from baseio import BaseIO
from neo.core import *
import numpy
from numpy import *

"""
asciiio
==================

Classe for reading/writing data in a text file.
Cover many case when part of a file can be view as a CVS format.


Classes
-------

AsciiIO          - Classe for reading/writing data in a text file.
                    Cover many case.

@author : sgarcia

"""



from baseio import BaseIO

from numpy import *

import csv




class AsciiIO(BaseIO):
    """
    Class for reading/writing data in a text file.
    Cover many case when part of a file can be view as a CVS format.
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = True
    is_object_readable = False
    is_object_writable = False
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
    level              = None
    nfiles             = 0
    name               = None
    objects            = []
    supported_types    = [ Segment ]
    
    def __init__(self ) :
        """
        
        **Arguments**
        
        filename is optional if the file exist read() is call at __init__
        
        
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
                                        usecols = None,
                                        skiprows =0,
                                        
                                        timecolumn = None,
                                        samplerate = 1000.,
                                        t_start = 0.,
                                        
                                        method = 'genfromtxt',
                                        
                                        ):
        """
        **Arguments**
            filename : filename
            TODO
            
        """
        #loadtxt
        if method == 'genfromtxt' :
            sig = genfromtxt(filename, 
                             delimiter = delimiter,
                            usecols = usecols ,
                            skiprows = skiprows,
                            dtype = 'f4')
            if len(sig.shape) ==1:
                sig = sig[:,newaxis]
        elif method == 'csv' :
            tab = [l for l in  csv.reader( open(filename,'rU') , delimiter = delimiter ) ]
            tab = tab[skiprows:]
            sig = array( tab , dtype = 'f4')
        elif method == 'homemade' :
            fid = open(filename,'rU')
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
                                                    freq = samplerate,
                                                    t_start = t_start)
            seg._analogsignals.append( analogSig )
        
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
        
        
        sigs = None
        for analogSig in segment.get_analogsignals():
            if sigs is None :
                sigs = analogSig.signal[:,newaxis]
            else :
                sigs = concatenate ((sigs, analogSig.signal[:,newaxis]) , axis = 0 )
        
        if timecolumn is not None:
            t = segment.get_analogsignals()[0].t()
            sigs = concatenate ((sigs, t[:,newaxis]*nan) , axis = 1 )
            sigs[:,timecolumn+1:] = sigs[:,timecolumn:-1].copy()
            sigs[:,timecolumn] = t
        
        savetxt(filename , sigs , delimiter = delimiter)

