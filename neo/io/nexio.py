# -*- coding: utf-8 -*-
"""

Class for reading data from NeuroExplorer (.nex)


Supported : Read

@author :  luc estebanez, sgarcia


"""






from baseio import BaseIO
#from neo.core import *
from neo.core import *

from numpy import *
import numpy as np
from scipy.io.numpyio import fread
import struct


class NexIO(BaseIO):
    """
    Class for reading/writing data in a fake file.
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = False
    
    supported_objects  = [Segment , AnalogSignal, SpikeTrain, Event, Epoch]
    readable_objects    = [ Segment]
    writeable_objects   = []

    has_header         = False
    is_streameable     = False
    
    # This is for GUI stuf : a definition for parameters when reading.
    read_params        = {
    
                        Segment :  [
                                        ('load_spike_waveform' , { 'value' : False } ) ,
                                        ]
                        }
    write_params       = None
    
    name               = 'NeuroExplorer'
    extensions          = [ 'nex' ]
    
    mode = 'file'
    
    
    def __init__(self , filename = None) :
        """
        This class read a abf file.
        
        **Arguments**
        
            filename : the filename to read you can pu what ever it do not read anythings
        
        """
        BaseIO.__init__(self)
        self.filename = filename

    def read(self , **kargs):
        """
        Read a fake file.
        Return a neo.Segment
        See read_block for detail.
        """
        return self.read_segment( **kargs)



    def read_segment(self, load_spike_waveform = False, analogtype='float32'):
        """
        """
        self.analogtype = analogtype
 

        fid = open(self.filename, 'rb')
        globalHeader = HeaderReader(fid , GlobalHeader ).read_f(offset = 0)
        
        seg = Segment()
        offset = 544
        for i in range(globalHeader['nvar']):
            entityHeader = HeaderReader(fid , EntityHeader ).read_f(offset = offset+i*208)
            entityHeader['name'] = entityHeader['name'].replace('\x00','')
            
            if entityHeader['type'] == 0:
                # neuron
                spike_times= memmap(self.filename , dtype('i4') ,'r' ,
                                                shape = (entityHeader['n'] ),
                                                offset = entityHeader['offset'],
                                                )
                spike_times = spike_times.astype('f')/globalHeader['freq']
                spikeTr = SpikeTrain( spike_times= spike_times, 
                                                    t_start = globalHeader['tbeg']/globalHeader['freq'],
                                                    t_stop = globalHeader['tend']/globalHeader['freq'],
                                                    )
                spikeTr.name = entityHeader['name']
                spikeTr.channel = entityHeader['WireNumber']
                seg._spiketrains.append(spikeTr)

            if entityHeader['type'] == 1:
                # event
                event_times= memmap(self.filename , dtype('i4') ,'r' ,
                                                shape = (entityHeader['n'] ),
                                                offset = entityHeader['offset'],
                                                )
                event_times = event_times.astype('f')/globalHeader['freq']
                for t in event_times:
                    ev = Event( time = t)
                    ev.type = entityHeader['name']
                    seg._events.append(ev)
                    


            if entityHeader['type'] == 2:
                # interval
                start_times= memmap(self.filename , dtype('i4') ,'r' ,
                                                shape = (entityHeader['n'] ),
                                                offset = entityHeader['offset'],
                                                )
                start_times = start_times.astype('f')/globalHeader['freq']                
                stop_times= memmap(self.filename , dtype('i4') ,'r' ,
                                                shape = (entityHeader['n'] ),
                                                offset = entityHeader['offset']+entityHeader['n']*4,
                                                )
                stop_times = stop_times.astype('f')/globalHeader['freq']
                for t_start, t_stop in zip(start_times, stop_times):
                    ep = Epoch(time = t_start , duration = t_stop-t_start)
                    seg._epochs.append(ep)



            if entityHeader['type'] == 3:
                # waves
                spike_times= memmap(self.filename , dtype('i4') ,'r' ,
                                                shape = (entityHeader['n'] ),
                                                offset = entityHeader['offset'],
                                                )
                spike_times = spike_times.astype('f')/globalHeader['freq']
                if load_spike_waveform :
                    waveforms = memmap(self.filename , dtype('i2') ,'r' ,
                                                shape = (entityHeader['n'] ,  1,entityHeader['NPointsWave']),
                                                offset = entityHeader['offset']+entityHeader['n'] *4,
                                                )
                    waveforms = waveforms.astype('f')* entityHeader['ADtoMV'] +  entityHeader['MVOffset']
                else:
                    waveforms = None
                spikeTr = SpikeTrain(  
                                                name = entityHeader['name'],
                                                channel = entityHeader['WireNumber'],
                                                sampling_rate = entityHeader['WFrequency'],
                                                spike_times = spike_times,
                                                waveforms = waveforms,
                                                t_start = globalHeader['tbeg']/globalHeader['freq'],
                                                t_stop = globalHeader['tend']/globalHeader['freq'],
                                                )
                seg._spiketrains.append(spikeTr)
            
            if entityHeader['type'] == 4:
                # popvectors
                pass

            if entityHeader['type'] == 5:
                # analog
                timestamps= memmap(self.filename , dtype('i4') ,'r' ,
                                                        shape = (entityHeader['n'] ),
                                                        offset = entityHeader['offset'],
                                                        )
                timestamps = timestamps.astype('f')/globalHeader['freq']
                fragmentStarts = memmap(self.filename , dtype('i4') ,'r' ,
                                                        shape = (entityHeader['n'] ),
                                                        offset = entityHeader['offset'],
                                                        )
                fragmentStarts = fragmentStarts.astype('f')/globalHeader['freq']
                t_start =  timestamps[0] - fragmentStarts[0]/float(entityHeader['WFrequency'])
                signal = memmap(self.filename , dtype('i2') ,'r' ,
                                                        shape = (entityHeader['NPointsWave'] ),
                                                        offset = entityHeader['offset'],
                                                        )
                signal = signal.astype(analogtype)
                signal *= entityHeader['ADtoMV']
                signal += entityHeader['MVOffset']
                # now, delete the timestamps vector : takes load of space !
                timestamps = 0
                #signal = np.fromstring(fid.read(entityHeader['NPointsWave']*2), np.int16)*entityHeader['ADtoMV'] + entityHeader['offset']
                #signal = fread(fid, entityHeader['NPointsWave'], 'i')

                anaSig = AnalogSignal(signal = signal , t_start =t_start , sampling_rate  = entityHeader['WFrequency'])
                anaSig.channel = entityHeader['WireNumber']
                seg._analogsignals.append( anaSig )
                
            if entityHeader['type'] == 6:
                # markers  : TO TEST
                times= memmap(self.filename , dtype('i4') ,'r' ,
                                                shape = (entityHeader['n'] ),
                                                offset = entityHeader['offset'],
                                                )
                times = times.astype('f')/globalHeader['freq']
                for j in xrange(entityHeader['NMarkers']):
                    fid.seek(entityHeader['offset'] + entityHeader['n']*4)
                    markertype = fid.read(64).replace('\x00','')
                    for k in xrange(times.size):
                        ev = Event( time = times[k] )
                        ev.type = markertype
                        ev.label = fid.read(entityHeader['MarkerLength']).replace('\x00','')
                        seg._events.append(ev)
                        
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



