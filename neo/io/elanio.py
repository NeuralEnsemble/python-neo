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
        #read a file
        io = ElanIO(filename = 'myfile.eeg')
        seg = io.read() # read the entire file
        
        seg is a Segment that contains AnalogSignals and Events
        
        # write a file
        io = ElanIO(filename = 'myfile.eeg')
        seg = Segment()
        io.write(seg)    
    
    """
    
    is_readable        = True
    is_writable        = True

    supported_objects            = [ Segment , AnalogSignal , Event]
    readable_objects    = [Segment]
    writeable_objects    = [Segment]  

    has_header         = False
    is_streameable     = False
    
    read_params        = { Segment : [] }
    write_params       = { Segment : [] }

    name               = None
    extensions          = [ 'eeg' ]
    
    
    def __init__(self , filename = None) :
        """
        This class read/write a elan based file.
        
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
    
    def read_segment(self, ):
        """
        **Arguments**
            no arguments
        """
        
        ## Read header file
        
        seg = Segment()
        
        f = open(self.filename+'.ent' , 'rU')
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
        nbchannel = int(l)-2
#        print 'nbchannel', nbchannel
        
        #channel label
        labels = [ ]
        for c in range(nbchannel+2) :
            labels.append(f.readline()[:-1])
#        print labels
        
        # channel type
        types = [ ]
        for c in range(nbchannel+2) :
            types.append(f.readline()[:-1])
#        print types
        
        # channel unit
        units = [ ]
        for c in range(nbchannel+2) :
            units.append(f.readline()[:-1])
#        print units
        
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
#        print min_physic,max_physic , min_logic, max_logic
        
        #info filter
        info_filter = []
        for c in range(nbchannel+2) :
            info_filter.append(f.readline()[:-1])
#        print info_filter
        
        f.close()
        
        #raw data
        n = int(round(log(max_logic[0]-min_logic[0])/log(2))/8)
#        print n
        data = fromfile(self.filename,dtype = 'i'+str(n) )
        data = data.byteswap().reshape( (data.size/(nbchannel+2) ,nbchannel+2) ).astype('f4')
        for c in range(nbchannel) :
            sig = (data[:,c]-min_logic[c])/(max_logic[c]-min_logic[c])*\
                                (max_physic[c]-min_physic[c])+min_physic[c]
            analogSig = AnalogSignal(signal = sig,
                                freq = freq,
                                t_start=0)
            seg._analogsignals.append( analogSig )
        
        # triggers
        f = open(self.filename+'.pos')
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
        Write segment in 3 files.
        See write_segment for detail.
        """
        self.write_segment(*args , **kargs)

    def write_segment(self, segment, ):
        """
        
         **Arguments**
            segment : the segment to write. Only analog signals and events will be written.
        """
        assert self.filename.endswith('.eeg')
        fid_ent = open(self.filename+'.ent' ,'wt')
        fid_eeg = open(self.filename ,'wt')
        fid_pos = open(self.filename+'.pos' ,'wt')
        
        seg = segment
        freq = seg.get_analogsignals()[0].freq
        N = len(seg.get_analogsignals())
        
        #
        # header file
        #
        fid_ent.write('V2\n')
        fid_ent.write('OpenElectrophyImport\n')
        fid_ent.write('ELAN\n')
        t =  datetime.datetime.now()
        fid_ent.write(t.strftime('%d-%m-%Y %H:%M:%S')+'\n')
        fid_ent.write(t.strftime('%d-%m-%Y %H:%M:%S')+'\n')
        fid_ent.write('-1\n')
        fid_ent.write('reserved\n')
        fid_ent.write('-1\n')
        fid_ent.write('%g\n' %  (1./freq))
        
        fid_ent.write( '%d\n' % (N+2) )
        
        # channel label
        for i, anaSig in enumerate(seg.get_analogsignals()) :
            if hasattr(anaSig , 'label') :
                fid_ent.write('%s.%d\n' % (anaSig.label, i+1 ))
            else :
                fid_ent.write('%s.%d\n' % ('nolabel', i+1 ))
        fid_ent.write('Num1\n')
        fid_ent.write('Num2\n')
        
        #channel type
        for i, anaSig in enumerate(seg.get_analogsignals()) :
            fid_ent.write('Electrode\n')
        fid_ent.write( 'dateur echantillon\n')
        fid_ent.write( 'type evenement et byte info\n')
        
        #units
        for i, anaSig in enumerate(seg.get_analogsignals()) :
            if hasattr(anaSig , 'unit') :
                fid_ent.write('%s\n' % anaSig.unit)
            else :
                fid_ent.write('microV\n')
        fid_ent.write('sans\n')
        fid_ent.write('sans\n')
    
        #range
        list_range = []
        for i, anaSig in enumerate(seg.get_analogsignals()) :
            # in elan file unit is supposed to be in microV to have a big range
            # so auto translate
            if hasattr(anaSig , 'unit') :
                if anaSig.unit =='V' :
                    anaSig.signal *= 1e6
                elif anaSig.unit =='mV' :
                    anaSig.signal *= 1e3
                elif anaSig.unit =='microV' :
                    anaSig.signal *= 1
                elif anaSig.unit =='µV' :
                    anaSig.signal *= 1
                else :
                    # automatic range in arbitrry unit
                    if abs(anaSig.signal).max() < 1.:
                        anaSig.signal *= 1e6*10**(int(log10(abs(anaSig.signal).max()))+1)
            else :
                # automatic range in arbitrry unit
                if abs(anaSig.signal).max() < 1.:
                    anaSig.signal *= 1e6*10**(int(log10(abs(anaSig.signal).max()))+1)
            list_range.append( int(abs(anaSig.signal).max()) +1 )
        for r in list_range :
            fid_ent.write('-%.0f\n'% r)
        fid_ent.write('-1\n')
        fid_ent.write('-1\n')
        for r in list_range :
            fid_ent.write('%.0f\n'% r)
        fid_ent.write('+1\n')
        fid_ent.write('+1\n')
        
        for i in range(N+2) :
            fid_ent.write('-32768\n')
        for i in range(N+2) :
            fid_ent.write('+32767\n')
        
        #info filter
        for i in range(N+2) :
            fid_ent.write('passe-haut ? Hz passe-bas ? Hz\n')
        fid_ent.write('sans\n')
        fid_ent.write('sans\n')
        
        for i in range(N+2) :
            fid_ent.write('1\n')
            
        for i in range(N+2) :
            fid_ent.write('reserved\n')
    
        #
        # raw file
        #
        data = zeros( (seg.get_analogsignals()[0].signal.size , N+2)  , 'i2')
        for i, anaSig in enumerate(seg.get_analogsignals()) :
            sig2 = anaSig.signal*65535/(2*list_range[i])
            data[:,i] = sig2.astype('i2')
        
        trigs = array([ ev.time for ev in seg.get_events() ])
        trigs *= freq
        trigs = trigs.astype('i')
        trigs2 = trigs[ (trigs>0) & (trigs<data.shape[0]) ]
        data[trigs2,-1] = 1
        
        fid_eeg.write(data.byteswap().tostring())
        
        #
        # pos file
        #
        for i, ev in enumerate( seg.get_events() ) :
            if hasattr(ev , 'label') and ev.label is not None:
                fid_pos.write('%d    %d    %d\n' % (trigs[i] , ev.label,0))
                label = ev.label
            else :
                fid_pos.write('%d    %d    %d\n' % (trigs[i] , 0,0))
            
        
        fid_ent.close()
        fid_eeg.close()
        fid_pos.close()
