# encoding: utf-8
"""

Class for reading data from NeuroExplorer (.nex)

Documentation for dev :
http://www.neuroexplorer.com/code.html

Depend on: scipy


Supported : Read

Author: sgarcia,luc estebanez

"""


from baseio import BaseIO
from ..core import *
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
        >>> r = io.NeuroExplorerIO( filename = 'File_neuroexplorer_1.nex ')
        >>> seg = r.read_segment(lazy = False, cascade = True,)
        >>> print seg._analogsignals
        >>> print seg._spiketrains
        >>> print seg._eventarrays
        >>> print seg._epocharrays

    
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

    def read(self , **kargs):
        """
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment( **kargs)



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
        seg._annotations['neuroexplorer_version'] = globalHeader['version']
        seg._annotations['comment'] = globalHeader['comment']
        
        if not cascade :
            return seg
        
        offset = 544
        for i in range(globalHeader['nvar']):
            entityHeader = HeaderReader(fid , EntityHeader ).read_f(offset = offset+i*208)
            entityHeader['name'] = entityHeader['name'].replace('\x00','')
            
            print 'i',i, entityHeader['type']
            
            if entityHeader['type'] == 0:
                # neuron
                if lazy:
                    spike_times = [ ]*pq.s
                else:
                    spike_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    spike_times = spike_times.astype('f')/globalHeader['freq']*pq.s
                sptr = SpikeTrain( times= spike_times, 
                                                    t_start = globalHeader['tbeg']/globalHeader['freq']*pq.s,
                                                    t_stop = globalHeader['tend']/globalHeader['freq']*pq.s,
                                                    name = entityHeader['name'],
                                                    )
                sptr._annotations['channel_index'] = entityHeader['WireNumber']
                seg._spiketrains.append(sptr)
            
            if entityHeader['type'] == 1:
                # event 
                if lazy:
                    event_times = [ ]*pq.s
                else:
                    event_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    event_times = event_times.astype('f')/globalHeader['freq'] * pq.s
                evar = EventArray(times = event_times, channel_name = entityHeader['name'] )
                seg._eventarrays.append(evar)
            
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
                    start_times = start_times.astype('f')/globalHeader['freq']*pq.s       
                    stop_times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset']+entityHeader['n']*4,
                                                    )
                    stop_times = stop_times.astype('f')/globalHeader['freq']*pq.s
                epar = EpochArray( times = start_times, durations =  stop_times - start_times, channel_name = entityHeader['name'])
                seg._epocharrays.append(epar)
            
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
                    spike_times = spike_times.astype('f')/globalHeader['freq'] * pq.s
                    
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
                                                sampling_rate = entityHeader['WFrequency'],
                                                left_sweep = 0*pq.ms,
                                                )
                sptr._annotations['channel_index'] = entityHeader['WireNumber']
                seg._spiketrains.append(sptr)
            
            if entityHeader['type'] == 4:
                # popvectors
                pass

            if entityHeader['type'] == 5:
                # analog
                
                    
                timestamps= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                        shape = (entityHeader['n'] ),
                                                        offset = entityHeader['offset'],
                                                        )
                timestamps = timestamps.astype('f')/globalHeader['freq']
                fragmentStarts = np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                        shape = (entityHeader['n'] ),
                                                        offset = entityHeader['offset'],
                                                        )
                fragmentStarts = fragmentStarts.astype('f')/globalHeader['freq']
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
                
                anaSig = AnalogSignal(signal = signal , t_start =t_start , sampling_rate  = entityHeader['WFrequency'], name = entityHeader['name'])
                anaSig._annotations['channel_index'] = entityHeader['WireNumber']
                seg._analogsignals.append( anaSig )
                
            if entityHeader['type'] == 6:
                # markers  : TO TEST
                if lazy:
                    time = None
                    labels = None
                    markertype = None
                else:
                    times= np.memmap(self.filename , np.dtype('i4') ,'r' ,
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'],
                                                    )
                    times = times.astype('f')/globalHeader['freq'] * pq.s
                    fid.seek(entityHeader['offset'] + entityHeader['n']*4)
                    markertype = fid.read(64).replace('\x00','')
                    labels = np.memmap(self.filename, np.dtype('S' + entityHeader['MarkerLength']) ,'r',
                                                    shape = (entityHeader['n'] ),
                                                    offset = entityHeader['offset'] + entityHeader['n']*4 + 64
                                                    )
                ea = EventArray( times = times,
                                            labels = labels,
                                            name = entityHeader['name'],
                                            channel_index = entityHeader['WireNumber'],
                                            marker_type = markertype
                                            )
                seg._eventarrays.append(ea)
                
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



