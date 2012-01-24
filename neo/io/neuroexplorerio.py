# encoding: utf-8
"""

Class for reading data from NeuroExplorer (.nex)

Documentation for dev :
http://www.neuroexplorer.com/code.html

Depend on: scipy


Supported : Read

Author: sgarcia,luc estebanez

"""


from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship

import numpy as np
import quantities as pq

import struct
import datetime
import os



class NeuroExplorerIO(BaseIO):
    """
    Class for reading nex file.
    
    Usage:
        >>> from neo import io
        >>> r = io.NeuroExplorerIO(filename='File_neuroexplorer_1.nex')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print seg.analogsignals   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<AnalogSignal(array([ 39.0625    ,   0.        ,   0.        , ..., -26.85546875, ...
        >>> print seg.spiketrains     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<SpikeTrain(array([  2.29499992e-02,   6.79249987e-02,   1.13399997e-01, ...       
        >>> print seg.eventarrays     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<EventArray: @21.1967754364 s, @21.2993755341 s, @21.350725174 s, @21.5048999786 s, ...
        >>> print seg.epocharrays     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<neo.core.epocharray.EpochArray object at 0x10561ba90>, <neo.core.epocharray.EpochArray object at 0x10561bad0>]
    
    """
    
    is_readable        = True
    is_writable        = False
    
    supported_objects  = [Segment , AnalogSignal, SpikeTrain, EventArray, EpochArray]
    readable_objects    = [ Segment]
    writeable_objects   = []

    has_header         = False
    is_streameable     = False
    
    # This is for GUI stuf : a definition for parameters when reading.
    read_params        = {
    
                        Segment :  [ ]
                        }
    write_params       = None
    
    name               = 'NeuroExplorer'
    extensions          = [ 'nex' ]
    
    mode = 'file'
    
    
    def __init__(self , filename = None) :
        """
        This class read a nex file.
        
        Arguments:
        
            filename : the filename to read you can pu what ever it do not read anythings
        
        """
        BaseIO.__init__(self)
        self.filename = filename
    
    def read_segment(self, 
                                        lazy = False,
                                        cascade = True,
                                        ):
        
        
        fid = open(self.filename, 'rb')
        globalHeader = HeaderReader(fid , GlobalHeader ).read_f(offset = 0)
        #~ print globalHeader
        #~ print 'version' , globalHeader['version']
        seg = Segment()
        seg.file_origin = os.path.basename(self.filename)
        seg.annotate(neuroexplorer_version = globalHeader['version'])
        seg.annotate(comment = globalHeader['comment'])
        
        if not cascade :
            return seg
        
        offset = 544
        for i in range(globalHeader['nvar']):
            entityHeader = HeaderReader(fid , EntityHeader ).read_f(offset = offset+i*208)
            entityHeader['name'] = entityHeader['name'].replace('\x00','')
            
            #print 'i',i, entityHeader['type']
            
            if entityHeader['type'] == 0:
                # neuron
                if lazy:
                    spike_times = [ ]*pq.s
                else:
                    spike_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    spike_times = spike_times.astype('f8')/globalHeader['freq']*pq.s
                sptr = SpikeTrain( times= spike_times, 
                                                    t_start = globalHeader['tbeg']/globalHeader['freq']*pq.s,
                                                    t_stop = globalHeader['tend']/globalHeader['freq']*pq.s,
                                                    name = entityHeader['name'],
                                                    )
                if lazy:
                    sptr.lazy_shape = entityHeader['n']
                sptr.annotate(channel_index = entityHeader['WireNumber'])
                seg.spiketrains.append(sptr)
            
            if entityHeader['type'] == 1:
                # event 
                if lazy:
                    event_times = [ ]*pq.s
                else:
                    event_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    event_times = event_times.astype('f8')/globalHeader['freq'] * pq.s
                labels = np.array(['']*event_times.size, dtype = 'S')
                evar = EventArray(times = event_times, labels=labels, channel_name = entityHeader['name'] )
                if lazy:
                    evar.lazy_shape = entityHeader['n']
                seg.eventarrays.append(evar)
            
            if entityHeader['type'] == 2:
                # interval
                if lazy:
                    start_times = [ ]*pq.s
                    stop_times = [ ]*pq.s
                else:
                    start_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    start_times = start_times.astype('f8')/globalHeader['freq']*pq.s       
                    stop_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset']+entityHeader['n']*4,
                                                    )
                    stop_times = stop_times.astype('f')/globalHeader['freq']*pq.s
                epar = EpochArray(times = start_times,
                                  durations =  stop_times - start_times,
                                  labels = np.array(['']*start_times.size, dtype = 'S'),
                                  channel_name = entityHeader['name'])
                if lazy:
                    epar.lazy_shape = entityHeader['n']
                seg.epocharrays.append(epar)
            
            if entityHeader['type'] == 3:
                # spiketrain and wavefoms
                if lazy:
                    spike_times = [ ]*pq.s
                    waveforms = None
                else:
                    
                    spike_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    spike_times = spike_times.astype('f8')/globalHeader['freq'] * pq.s
                    
                    waveforms = np.memmap(self.filename , np.dtype('i2') ,'r' ,
                                                shape = (entityHeader['n'] ,  1,entityHeader['NPointsWave']),
                                                offset = entityHeader['offset']+entityHeader['n'] *4,
                                                )
                    waveforms = (waveforms.astype('f')* entityHeader['ADtoMV'] +  entityHeader['MVOffset'])*pq.mV
                
                sptr = SpikeTrain(      times = spike_times,
                                                t_start = globalHeader['tbeg']/globalHeader['freq']*pq.s,
                                                t_stop = globalHeader['tend']/globalHeader['freq']*pq.s,

                                                name = entityHeader['name'],
                                                waveforms = waveforms,
                                                sampling_rate = entityHeader['WFrequency']*pq.Hz,
                                                left_sweep = 0*pq.ms,
                                                )
                if lazy:
                    sptr.lazy_shape = entityHeader['n'] 
                sptr.annotate(channel_index = entityHeader['WireNumber'])
                seg.spiketrains.append(sptr)
            
            if entityHeader['type'] == 4:
                # popvectors
                pass

            if entityHeader['type'] == 5:
                # analog
                
                    
                timestamps= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                        shape = (entityHeader['n'] ),
                                                        offset = entityHeader['offset'],
                                                        )
                timestamps = timestamps.astype('f8')/globalHeader['freq']
                fragmentStarts = np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                        shape = (entityHeader['n'] ),
                                                        offset = entityHeader['offset'],
                                                        )
                fragmentStarts = fragmentStarts.astype('f8')/globalHeader['freq']
                t_start =  timestamps[0] - fragmentStarts[0]/float(entityHeader['WFrequency'])
                del timestamps, fragmentStarts
                
                if lazy :
                    signal = [ ]*pq.mV
                else:
                    signal = np.memmap(self.filename , np.dtype('i2') ,'r' ,
                                                            shape = (entityHeader['NPointsWave'] ),
                                                            offset = entityHeader['offset'],
                                                            )
                    signal = signal.astype('f')
                    signal *= entityHeader['ADtoMV']
                    signal += entityHeader['MVOffset']
                    signal = signal*pq.mV
                
                anaSig = AnalogSignal(signal = signal , t_start =t_start*pq.s , sampling_rate  = entityHeader['WFrequency']*pq.Hz, name = entityHeader['name'])
                if lazy:
                    anaSig.lazy_shape = entityHeader['NPointsWave'] 
                anaSig.annotate(channel_index = entityHeader['WireNumber'])
                seg.analogsignals.append( anaSig )
                
            if entityHeader['type'] == 6:
                # markers  : TO TEST
                if lazy:
                    times = [ ]*pq.s
                    labels = np.array([ ], dtype = 'S')
                    markertype = None
                else:
                    times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    times = times.astype('f8')/globalHeader['freq'] * pq.s
                    fid.seek(entityHeader['offset'] + entityHeader['n']*4)
                    markertype = fid.read(64).replace('\x00','')
                    labels = np.memmap(self.filename, np.dtype('S' + str(entityHeader['MarkerLength'])) ,'r',
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'] + entityHeader['n']*4 + 64
                                                    )
                ea = EventArray( times = times,
                                            labels = labels.view(np.ndarray),
                                            name = entityHeader['name'],
                                            channel_index = entityHeader['WireNumber'],
                                            marker_type = markertype
                                            )
                if lazy:
                    ea.lazy_shape = entityHeader['n']
                seg.eventarrays.append(ea)
        
        
        create_many_to_one_relationship(seg)
        return seg



GlobalHeader = [
    ('signature' , '4s'),
    ('version','i'),
    ('comment','256s'),
    ('freq','d'),
    ('tbeg','i'),
    ('tend','i'),
    ('nvar','i'),
    ]

EntityHeader = [
    ('type' , 'i'),
    ('varVersion','i'),
    ('name','64s'),
    ('offset','i'),
    ('n','i'),
    ('WireNumber','i'),
    ('UnitNumber','i'),
    ('Gain','i'),
    ('Filter','i'),
    ('XPos','d'),
    ('YPos','d'),
    ('WFrequency','d'),
    ('ADtoMV','d'),
    ('NPointsWave','i'),
    ('NMarkers','i'),
    ('MarkerLength','i'),
    ('MVOffset','d'),
    ('dummy','60s'),
    ]


MarkerHeader = [
    ('type' , 'i'),
    ('varVersion','i'),
    ('name','64s'),
    ('offset','i'),
    ('n','i'),
    ('WireNumber','i'),
    ('UnitNumber','i'),
    ('Gain','i'),
    ('Filter','i'),
    ]




class HeaderReader():
    def __init__(self,fid ,description ):
        self.fid = fid
        self.description = description
    def read_f(self, offset =0):
        self.fid.seek(offset)
        d = { }
        for key, format in self.description :
            val = struct.unpack(format , self.fid.read(struct.calcsize(format)))
            if len(val) == 1:
                val = val[0]
            else :
                val = list(val)
            d[key] = val
        return d



