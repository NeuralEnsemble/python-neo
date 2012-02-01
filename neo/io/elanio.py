# encoding: utf-8
"""
Class for reading/writing data from Elan.

Elan is software for studying time-frequency maps of EEG data.

Elan is developed in Lyon, France, at INSERM U821

An Elan dataset is separated into 3 files :
 - .eeg          raw data file
 - .eeg.ent      hearder file
 - .eeg.pos      event file


Depend on: 

Supported : Read and Write

Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship
import numpy as np
from numpy import dtype, zeros, fromstring, empty, log, fromfile
import quantities as pq

import os
import datetime
import re

class VersionError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ElanIO(BaseIO):
    """
    Classe for reading/writing data from Elan.
    
    Usage:
        >>> from neo import io
        >>> r = io.ElanIO( filename = 'File_elan_1.eeg')
        >>> seg = r.read_segment(lazy = False, cascade = True,)
        >>> print seg.analogsignals   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<AnalogSignal(array([ 89.21203613,  88.83666992,  87.21008301, ...,  64.56298828,
            67.94128418,  68.44177246], dtype=float32) * pA, [0.0 s, 101.5808 s], sampling rate: 10000.0 Hz)>]
        >>> print seg.spiketrains     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        []
        >>> print seg.eventarrays     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        []
        
    
    
    """
    
    is_readable        = True
    is_writable        = False

    supported_objects  = [Segment, AnalogSignal, EventArray]
    readable_objects   = [Segment]
    writeable_objects  = [ ]

    has_header         = False
    is_streameable     = False
    
    read_params        = { Segment : [ ] }
    write_params       = { Segment : [ ] }

    name               = None
    extensions         = ['eeg']
    
    mode = 'file'
    
    
    def __init__(self , filename = None) :
        """
        This class read/write a elan based file.
        
        **Arguments**
            filename : the filename to read or write
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read_segment(self, lazy = False, cascade = True):
        
        ## Read header file
        
        f = open(self.filename+'.ent' , 'rU')
        #version
        version = f.readline()
        if version[:2] != 'V2' and version[:2] != 'V3':
            # raise('read only V2 .eeg.ent files')
            raise VersionError('Read only V2 or V3 .eeg.ent files. %s given' %
                               version[:2]) 
            return
        
        #info
        info1 = f.readline()[:-1]
        info2 = f.readline()[:-1]
        
        # strange 2 line for datetime
        #line1
        l = f.readline()
        r1 = re.findall('(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)',l)
        r2 = re.findall('(\d+):(\d+):(\d+)',l)
        r3 = re.findall('(\d+)-(\d+)-(\d+)',l)
        YY, MM, DD, hh, mm, ss = (None, )*6
        if len(r1) != 0 :
            DD , MM, YY, hh ,mm ,ss = r1[0]
        elif len(r2) != 0 :
            hh ,mm ,ss = r2[0]
        elif len(r3) != 0:
            DD , MM, YY= r3[0]
        
        #line2
        l = f.readline()
        r1 = re.findall('(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)',l)
        r2 = re.findall('(\d+):(\d+):(\d+)',l)
        r3 = re.findall('(\d+)-(\d+)-(\d+)',l)
        if len(r1) != 0 :
            DD , MM, YY, hh ,mm ,ss = r1[0]
        elif len(r2) != 0 :
            hh ,mm ,ss = r2[0]
        elif len(r3) != 0:
            DD , MM, YY= r3[0]
        try:
            fulldatetime = datetime.datetime(int(YY) , int(MM) , int(DD) , int(hh) , int(mm) , int(ss) )
        except:
            fulldatetime = None
        
        
        seg = Segment(  file_origin = os.path.basename(self.filename),
                                    elan_version = version, 
                                    info1 = info1,
                                    info2 = info2,
                                    rec_datetime = fulldatetime,
                                    )
        
        if not cascade : return seg
        
        
        l = f.readline()
        l = f.readline()
        l = f.readline()
        
        # sampling rate sample
        l = f.readline()
        sampling_rate = 1./float(l) * pq.Hz
        
        # nb channel
        l = f.readline()
        nbchannel = int(l)-2
        
        #channel label
        labels = [ ]
        for c in range(nbchannel+2) :
            labels.append(f.readline()[:-1])
        
        # channel type
        types = [ ]
        for c in range(nbchannel+2) :
            types.append(f.readline()[:-1])
        
        # channel unit
        units = [ ]
        for c in range(nbchannel+2) :
            units.append(f.readline()[:-1])
        #print units
        
        #range
        min_physic = []
        for c in range(nbchannel+2) :
            min_physic.append( float(f.readline()) )
        max_physic = []
        for c in range(nbchannel+2) :
            max_physic.append( float(f.readline()) )
        min_logic = []
        for c in range(nbchannel+2) :
            min_logic.append( float(f.readline()) )
        max_logic = []
        for c in range(nbchannel+2) :
            max_logic.append( float(f.readline()) )
        
        #info filter
        info_filter = []
        for c in range(nbchannel+2) :
            info_filter.append(f.readline()[:-1])
        
        f.close()
        
        #raw data
        n = int(round(log(max_logic[0]-min_logic[0])/log(2))/8)
        data = fromfile(self.filename,dtype = 'i'+str(n) )
        data = data.byteswap().reshape( (data.size/(nbchannel+2) ,nbchannel+2) ).astype('f4')
        for c in range(nbchannel) :
            if lazy:
                sig = [ ]
            else:
                sig = (data[:,c]-min_logic[c])/(max_logic[c]-min_logic[c])*\
                                    (max_physic[c]-min_physic[c])+min_physic[c]
            
            try:
                unit = pq.Quantity(1, units[c] )
            except:
                unit = pq.Quantity(1, '' )
            
            
            anaSig = AnalogSignal( sig * unit,
                                                    sampling_rate = sampling_rate,
                                                    t_start=0.*pq.s,
                                                    name = labels[c],
                                                    )
            if lazy:
                anaSig.lazy_shape = data.shape[0]
            anaSig.annotate(channel_index = c)
            anaSig.annotate(channel_name= labels[c])
            seg.analogsignals.append( anaSig )
        
        # triggers
        f = open(self.filename+'.pos')
        times =[ ]
        labels = [ ]
        reject_codes = [ ]
        for l in f.readlines() :
            r = re.findall(' *(\d+) *(\d+) *(\d+) *',l)
            times.append( float(r[0][0])/sampling_rate.magnitude )
            labels.append(str(r[0][1]) )
            reject_codes.append( str(r[0][2]) )
        if lazy:
            times = [ ]*pq.S
            labels = np.array([ ], dtype = 'S')
            reject_codes = [ ]
        else:
            times =  np.array(times) * pq.s
            labels  = np.array(labels)
            reject_codes = np.array(reject_codes) 
        ea = EventArray( times = times,
                                    labels  = labels,
                                    reject_codes = reject_codes,
                                    )
        if lazy:
            ea.lazy_shape = len(times)
        seg.eventarrays.append(ea)
    
        
        f.close()
        
        create_many_to_one_relationship(seg)
        return seg
        

    #~ def write_segment(self, segment, ):
        #~ """
        
         #~ Arguments:
            #~ segment : the segment to write. Only analog signals and events will be written.
        #~ """
        #~ assert self.filename.endswith('.eeg')
        #~ fid_ent = open(self.filename+'.ent' ,'wt')
        #~ fid_eeg = open(self.filename ,'wt')
        #~ fid_pos = open(self.filename+'.pos' ,'wt')
        
        #~ seg = segment
        #~ sampling_rate = seg._analogsignals[0].sampling_rate
        #~ N = len(seg._analogsignals)
        
        #~ #
        #~ # header file
        #~ #
        #~ fid_ent.write('V2\n')
        #~ fid_ent.write('OpenElectrophyImport\n')
        #~ fid_ent.write('ELAN\n')
        #~ t =  datetime.datetime.now()
        #~ fid_ent.write(t.strftime('%d-%m-%Y %H:%M:%S')+'\n')
        #~ fid_ent.write(t.strftime('%d-%m-%Y %H:%M:%S')+'\n')
        #~ fid_ent.write('-1\n')
        #~ fid_ent.write('reserved\n')
        #~ fid_ent.write('-1\n')
        #~ fid_ent.write('%g\n' %  (1./sampling_rate))
        
        #~ fid_ent.write( '%d\n' % (N+2) )
        
        #~ # channel label
        #~ for i, anaSig in enumerate(seg.analogsignals) :
            #~ try :
                #~ fid_ent.write('%s.%d\n' % (anaSig.label, i+1 ))
            #~ except :
                #~ fid_ent.write('%s.%d\n' % ('nolabel', i+1 ))
        #~ fid_ent.write('Num1\n')
        #~ fid_ent.write('Num2\n')
        
        #~ #channel type
        #~ for i, anaSig in enumerate(seg.analogsignals) :
            #~ fid_ent.write('Electrode\n')
        #~ fid_ent.write( 'dateur echantillon\n')
        #~ fid_ent.write( 'type evenement et byte info\n')
        
        #~ #units
        #~ for i, anaSig in enumerate(seg._analogsignals) :
            #~ unit_txt = str(anaSig.units).split(' ')[1]
            #~ fid_ent.write('%s\n' % unit_txt)
        #~ fid_ent.write('sans\n')
        #~ fid_ent.write('sans\n')
    
        #~ #range and data
        #~ list_range = []
        #~ data = np.zeros( (seg._analogsignals[0].size , N+2)  , 'i2')
        #~ for i, anaSig in enumerate(seg._analogsignals) :
            #~ # in elan file unit is supposed to be in microV to have a big range
            #~ # so auto translate
            #~ if anaSig.units == pq.V or anaSig.units == pq.mV:
                #~ s = anaSig.rescale('uV').magnitude
            #~ elif anaSig.units == pq.uV:
                #~ s = anaSig.magnitude
            #~ else:
                #~ # automatic range in arbitrry unit
                #~ s = anaSig.magnitude
                #~ s*= 10**(int(np.log10(abs(s).max()))+1)
            
            #~ list_range.append( int(abs(s).max()) +1 )
            
            #~ s2 = s*65535/(2*list_range[i])
            #~ data[:,i] = s2.astype('i2')
            
        #~ for r in list_range :
            #~ fid_ent.write('-%.0f\n'% r)
        #~ fid_ent.write('-1\n')
        #~ fid_ent.write('-1\n')
        #~ for r in list_range :
            #~ fid_ent.write('%.0f\n'% r)
        #~ fid_ent.write('+1\n')
        #~ fid_ent.write('+1\n')
        
        #~ for i in range(N+2) :
            #~ fid_ent.write('-32768\n')
        #~ for i in range(N+2) :
            #~ fid_ent.write('+32767\n')
        
        #~ #info filter
        #~ for i in range(N+2) :
            #~ fid_ent.write('passe-haut ? Hz passe-bas ? Hz\n')
        #~ fid_ent.write('sans\n')
        #~ fid_ent.write('sans\n')
        
        #~ for i in range(N+2) :
            #~ fid_ent.write('1\n')
            
        #~ for i in range(N+2) :
            #~ fid_ent.write('reserved\n')
    
        #~ # raw file .eeg
        #~ if len(seg._eventarrays) == 1:
            #~ ea = seg._eventarrays[0]
            #~ trigs = (ea.times*sampling_rate).magnitude
            #~ trigs = trigs.astype('i')
            #~ trigs2 = trigs[ (trigs>0) & (trigs<data.shape[0]) ]
            #~ data[trigs2,-1] = 1
        #~ fid_eeg.write(data.byteswap().tostring())
        
        
        #~ # pos file  eeg.pos
        #~ if len(seg._eventarrays) == 1:
            #~ ea = seg._eventarray[0]
            #~ if 'reject_codes' in ea.annotations and len(ea.reject_codes) == len(ea.times):
                #~ rcs = ea.reject_codes
            #~ else:
                #~ rcs = np.array(  [ '' ]*ea.times.size)
            #~ if len(ea.labels) == len(ea.times):
                #~ labels = ea.labels
            #~ else:
                #~ labels = np.array(  [ '' ]*ea.times.size)
            
            #~ for t, label, rc in zip(ea.times, labels, rcs):
                #~ fid_pos.write('%d    %s    %s\n' % (trigs[i] , ev.label,0))
        
        #~ fid_ent.close()
        #~ fid_eeg.close()
        #~ fid_pos.close()
