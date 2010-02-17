# -*- coding: utf-8 -*-
"""
Classe for reading/writing data in a raw binary compact file.
All channel are interleaved.

You must known externaly to the file : the sample rate, the dtype of file (int16, float32, float64), the number of channel, ...

This neutral format is quite widly used in old acquisition system.


Supported : Read/Write

@author : sgarcia

"""





from baseio import BaseIO
from neo.core import *
import numpy
from numpy import *

class RawIO(BaseIO):
    """
    Classe for reading/writing data in a raw binary compact file.
   
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = True

    supported_objects  = [Segment , AnalogSignal]
    readable_objects    = [ Segment]
    writeable_objects   = [Segment]
    
    has_header         = False
    is_streameable     = False
    read_params        = { Segment : [
                                        ('samplerate' , { 'value' : 1000. } ) ,
                                        ('nbchannel' , { 'value' : 16 } ),
                                        ('bytesoffset' , { 'value' : 0 } ),
                                        ('t_start' , { 'value' : 0. } ),
                                        ('dtype' , { 'value' : 'f4' , 'possible' : ['f4' , 'i2' , 'i4' , 'f8' ] } ),
                                        ('rangemin' , { 'value' : -10 } ),
                                        ('rangemax' , { 'value' : 10 } ),
                                    ]
                        }
    write_params       = { Segment : [
                                        ('bytesoffset' , { 'value' : 0 } ),
                                        ('dtype' , { 'value' : 'f4' , 'possible' : ['f4' , 'i2' , 'i4' , 'f8' ] } ),
                                        ('rangemin' , { 'value' : -10 } ),
                                        ('rangemax' , { 'value' : 10 } ),
                                    ]
                        }
                        
    name               = None
    extensions          = [ 'raw' ]
    
    def __init__(self , filename = None) :
        """
        This class read a binary file.
        
        **Arguments**
        
            filename : the filename to read
        
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
                                        samplerate = 1000.,
                                        nbchannel = 1,
                                        bytesoffset = 0,
                                        t_start = 0.,
                                        dtype = 'f4',
                                        rangemin = -10,
                                        rangemax = 10,
                                    ):
        """
        **Arguments**
            samplerate :  sample rate
            nbchannel : number of channel
            bytesoffset : nb of bytes offset at the start of file
            t_start : time of the first sample sample of each channel
            dtype : dtype of the data
            rangemin , rangemax : if the dtype is integer, range can give in volt the min and the max of the range
        """
        
        dtype = numpy.dtype(dtype)
        
        fid = open(self.filename , 'rb')
        buf = fid.read()
        fid.close()
        sig = numpy.fromstring(buf[bytesoffset:], dtype = dtype )
        
        if sig.size % nbchannel != 0 :
            sig = sig[:- sig.size%nbchannel]
        sig = sig.reshape((sig.size/nbchannel,nbchannel))
        
        if dtype.kind == 'i' :
            sig = sig.astype('f')
            sig /= 2**(8*dtype.itemsize-1)
            #~ print numpy.max(sig)
            sig *= ( rangemax-rangemin )
            #~ print numpy.max(sig)
        
        seg = Segment()
        
        for i in xrange(nbchannel) :
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
                                dtype = 'f4',
                                rangemin = -10,
                                rangemax = 10,
                                bytesoffset = 0,
                            ):
        """
        
         **Arguments**
            segment : the segment to write. Only analog signals will be written.
            
            dtype : dtype of the data
            rangemin , rangemax : if the dtype is integer, range can give in volt the min and the max of the range

        """
        
        dtype = numpy.dtype(dtype)
        
        sigs = None
        for analogSig in segment.get_analogsignals():
            if sigs is None :
                sigs = analogSig.signal[:,newaxis]
            else :
                sigs = concatenate ((sigs, analogSig.signal[:,newaxis]) , axis = 1 )
        
        if dtype.kind == 'i' :
            sigs /= (rangemax - rangemin)
            sigs *= 2**(8*dtype.itemsize-1)
            sigs = sigs.astype(dtype)
        else:
            sigs = sigs.astype(dtype)
        fid = open(self.filename , 'wb')
        fid.write( sigs.tostring() )
        fid.close()



