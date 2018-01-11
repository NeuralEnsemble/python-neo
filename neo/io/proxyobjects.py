# -*- coding: utf-8 -*-
"""
Here a list of proxy object that can be used when lazy=True.

This idea is to be able to postpone that real in memory loading 
for objects that contains big data (AnalogSIgnal, SpikeTrain, Event, Epoch).

The implementation rely on neo.rawio, so it will available only for neo.io that
ineherits neo.rawio.


"""

from neo.core.baseneo import BaseNeo


class BaseProxy(BaseNeo):
    def __init__(self, rawio, **kargs):
        self._rawio = rawio
        BaseNeo.__init__(self, **kargs)


class AnalogSignalProxy(BaseProxy):
    _single_parent_objects = ('Segment', 'ChannelIndex')
    _necessary_attrs = (('sampling_rate', pq.Quantity, 0),
                                    ('t_start', pq.Quantity, 0))
    _recommended_attrs = BaseNeo._recommended_attrs
    
    def __init__(self, rawio=None, shape=None, units=None, dtype=None,
                t_start=None, samlpling_rate=None,  **kargs):
        BaseProxy.__init__(self, rawio=rawio, **kargs)
        
        self.units = units
        self.shape = shape
        self.dtype = dtype
        self.t_start = t_start
        self.sampling_rate = sampling_rate
        
        self._block_index = block_index
        self._seg_index = seg_index

    @property
    def duration(self):
        '''Signal duration'''
        return self.shape[0] / self.sampling_rate

    @property
    def t_stop(self):
        '''Time when signal ends'''
        return self.t_start + self.duration
    
    def load(self, time_slice=None, channel_indexes=None, magnitude_mode='raw'):
        sr = self.get_signal_sampling_rate(channel_indexes) * pq.Hz
        if time_slice is None:
            i_start, i_stop = None, None
            sig_t_start = self.t_start
        else:
            t_start, t_stop = time_slice
            if t_start is None:
                i_start = None
                sig_t_start = self.t_start
            else:
                t_start = ensure_second(t_start)
                assert t_start<=t_start<=self.t_stop, 't_start is outside'
                i_start = int((t_start-self.t_start).magnitude * sr.magnitude)
                #this needed to get the real t_start of the first sample
                #because do not necessary match what is demanded
                sig_t_start = self.t_start + i_start/sr 
                
            if t_stop is None:
                i_stop = None
            else:
                t_stop = ensure_second(t_stop)
                assert t_start<=t_stop<=self.t_stop, 't_stop is outside'
                i_stop = int((t_stop-self.t_start).magnitude * sr.magnitude)
        
        raw_signal = self._rawio.get_analogsignal_chunk(block_index=self._block_index, seg_index=self._seg_index,
                    i_start=i_start, i_stop=i_stop, channel_indexes=channel_indexes)
        
        if magnitude_mode=='raw':
            #~ sig = raw_signal
            #~ units = CompundUnits
            raise(NotImplementedError)
            
        elif magnitude_mode=='reascaled':
            sig = self.rescale_signal_raw_to_float(raw_signal,  dtype='float32',
                                                                                channel_indexes=channel_indexes)
            units = 'todo'
        
        anasig = AnalogSignal(sig, units=units, sampling_rate=self.sampling_rate,
                    name=self.name, )

#~ units=None, dtype=None, copy=True,
                #~ t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                #~ name=None, file_origin=None, description=None,
                #~ **annotations):




        


class SpikeTrainProxy(BaseProxy):
    _single_parent_objects = ('Segment', 'Unit')
    _quantity_attr = 'times'
    _necessary_attrs = (('t_start', pq.Quantity, 0),
                                    ('t_stop', pq.Quantity, 0))
    _recommended_attrs = ()    
    
    def __init__(self, rawio=None, **kargs):
        BaseProxy.__init__(self, rawio=rawio, **kargs)
        







class EventProxy(BaseProxy):
    _single_parent_objects = ('Segment',)
    _necessary_attrs = ()
    
    def __init__(self, rawio=None, **kargs):
        BaseProxy.__init__(self, rawio=rawio, **kargs)
        
    def load(self):
        pass


class EpochProxy(BaseProxy):
    _single_parent_objects = ('Segment',)
    _necessary_attrs = ()
    
    def __init__(self, rawio=None, **kargs):
        BaseProxy.__init__(self, rawio=rawio, **kargs)
        
    def load(self):
        pass



def ensure_second(v):
    if isinstance(v, float):
        return v*pq.s
    elif isinstance(v, pq.Quantity):
        return v.rescale('s')
    elif isinstance(v, int):
        return float(v)*pq.s
