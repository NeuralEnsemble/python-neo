# -*- coding: utf-8 -*-
"""
elanio
==================

Classe for reading/writing data from Elan.
Elan is software for studying time frequency map of EEG data
Elan is develloped in France,lyon, inserm U821


Classes
-------

ElanIO          - Classe for reading/writing data in elan file.
                    Elan file is separated in 3 files :
                        .eeg          raw data file
                        .eeg.ent      hearder file
                        .eeg.pos      event file

@author : sgarcia

"""







from baseio import BaseIO
from neo.core import *
from numpy import *
import re
import datetime


class ElanIO(BaseIO):
    """
    Classe for reading/writing data from Elan.
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = True
    is_object_readable = False
    is_object_writable = False
    has_header         = True
    is_streameable     = False
    read_params        = {}
    write_params       = {}
    level              = None
    nfiles             = 0
    name               = None
    objects            = []
    supported_types    = []
    
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
        
        ## Read header file
        
        seg = Segment()
        
        f = open(filename+'.ent')
        #version
        version = f.readline()
        if version[:2] != 'V2' :
            raise('read only V2 .eeg.ent files')
            return
        
        #info
        seg.info1 = f.readline()[:-1]
        seg.info2 = f.readline()[:-1]
        
        #date1
        l = f.readline()
        r1 = re.findall('(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)',l)
        r2 = re.findall('(\d+):(\d+):(\d+)',l)
        r3 = re.findall('(\d+)-(\d+)-(\d+)',l)
        if len(r1) != 0 :
            DD , MM, YY, hh ,mm ,ss = r1[0]
            date1 = datetime.datetime(int(YY) , int(MM) , int(DD) , int(hh) , int(mm) , int(ss) )
        elif len(r2) != 0 :
            hh ,mm ,ss = r2[0]
            date1 = datetime.datetime(2000 , 1 , 1 , int(hh) , int(mm) , int(ss) )
        elif len(r3) != 0:
            DD , MM, YY= r3[0]
            date1 = datetime.datetime(int(YY) , int(MM) , int(DD) , 0,0,0 )
        else :
            date1 = datetime.now()
        seg.date1 = date1
        
        #date2
        l = f.readline()
        r1 = re.findall('(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)',l)
        r2 = re.findall('(\d+):(\d+):(\d+)',l)
        r3 = re.findall('(\d+)-(\d+)-(\d+)',l)
        if len(r1) != 0 :
            DD , MM, YY, hh ,mm ,ss = r1[0]
            date2 = datetime.datetime(int(YY) , int(MM) , int(DD) , int(hh) , int(mm) , int(ss) )
        elif len(r2) != 0 :
            hh ,mm ,ss = r2[0]
            date2 = datetime.datetime(2000 , 1 , 1 , int(hh) , int(mm) , int(ss) )
        elif len(r3) != 0:
            DD , MM, YY= r3[0]
            date2 = datetime.datetime(int(YY) , int(MM) , int(DD) , 0,0,0 )
        else :
            date2 = datetime.now()
        seg.date2 = date2
        
        
        l = f.readline()
        l = f.readline()
        l = f.readline()
        
        # frequencie sample
        l = f.readline()
        freq = 1./float(l)
#        print 'freq',freq
        
        # nb channel
        l = f.readline()
        nbchannel = int(l)
#        print 'nbchannel', nbchannel
        
        #channel label
        labels = [ ]
        for c in range(nbchannel) :
            labels.append(f.readline()[:-1])
#        print labels
        
        # channel type
        types = [ ]
        for c in range(nbchannel) :
            types.append(f.readline()[:-1])
#        print types
        
        # channel unit
        units = [ ]
        for c in range(nbchannel) :
            units.append(f.readline()[:-1])
#        print units
        
        #range
        min_physic = []
        for c in range(nbchannel) :
            min_physic.append( float(f.readline()) )
        max_physic = []
        for c in range(nbchannel) :
            max_physic.append( float(f.readline()) )
        min_logic = []
        for c in range(nbchannel) :
            min_logic.append( float(f.readline()) )
        max_logic = []
        for c in range(nbchannel) :
            max_logic.append( float(f.readline()) )
#        print min_physic,max_physic , min_logic, max_logic
        
        #info filter
        info_filter = []
        for c in range(nbchannel) :
            info_filter.append(f.readline()[:-1])
#        print info_filter
        
        f.close()
        
        #raw data
        n = int(round(log(max_logic[0]-min_logic[0])/log(2))/8)
#        print n
        data = fromfile(filename,dtype = 'i'+str(n) )
        data = data.byteswap().reshape( (data.size/nbchannel ,nbchannel) ).astype('f4')
        list_sig = [ ]
        for c in range(nbchannel) :
            sig = (data[:,c]-min_logic[c])/(max_logic[c]-min_logic[c])*\
                                (max_physic[c]-min_physic[c])+min_physic[c]
            analogSig = AnalogSignal(signal = sig,
                                freq = freq,
                                t_start=0)
            seg._analogsignals.append( analogSig )
        
        # triggers
        f = open(filename+'.pos')
        for l in f.readlines() :
            r = re.findall(' *(\d+) *(\d+) *(\d+) *',l)
            ev = Event( time = float(r[0][0])/freq )
            ev.label = str(r[0][1])
            ev.reject = str(r[0][2])
            seg._events.append( ev )
            
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
        
        pass

