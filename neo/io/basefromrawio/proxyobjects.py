# -*- coding: utf-8 -*-
"""
Here a list of proxy object that can be used when lazy=True.

This idea is to be able to postpone that real in memory loading 
for objects that contains big data (AnalogSIgnal, SpikeTrain, Event, Epoch).

The implementation rely on neo.rawio, so it will available only for neo.io that
ineherits neo.rawio.


"""

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo


from neo.core import (AnalogSignal, 
                      Epoch, Event, SpikeTrain)


class BaseProxy(BaseNeo):
    def __init__(self, **kargs):
        #this for py27 str vs py3 str in neo attributes ompatibility
        kargs = check_annotations(kargs)
        BaseNeo.__init__(self, **kargs)

class AnalogSignalProxy(BaseProxy):
    '''
    This object mimic AnalogSignal except that it does not
    have the signals array itself. All attributes and annotations are here.
    
    The goal is to postpone the loading of data into memory
    when reading a file with the new lazy load system based
    on neo.rawio.
    
    This object must not be constructed directly but is given
    neo.io when lazy=True.
    
    The AnalogSignalProxy is able to load:
      * only a slice of time
      * only a subset of channels
      * have an internal raw magnitude identic to the file (int16) with
        a pq.CompoundUnit().
    
    Usage:
    >>> proxy_anasig = AnalogSignalProxy(rawio=self.reader, channel_indexes=None,
                        block_index=0, seg_index=0,)
    >>> anasig = proxy_anasig.load()
    >>> slice_of_anasig = proxy_anasig.load(time_slice=(1.*pq.s, 2.*pq.s))
    >>> some_channel_of_anasig = proxy_anasig.load(channel_indexes=[0,5,10])
    
    '''
    _single_parent_objects = ('Segment', 'ChannelIndex')
    _necessary_attrs = (('sampling_rate', pq.Quantity, 0),
                                    ('t_start', pq.Quantity, 0))
    _recommended_attrs = BaseNeo._recommended_attrs
    
    def __init__(self, rawio=None, global_channel_indexes=None, block_index=0, seg_index=0):
        
        self._rawio = rawio
        self._block_index = block_index
        self._seg_index = seg_index
        if global_channel_indexes is None:
            global_channel_indexes = slice(None)
        total_nb_chan = self._rawio.header['signal_channels'].size
        self._global_channel_indexes = np.arange(total_nb_chan)[global_channel_indexes]
        self._nb_chan = self._global_channel_indexes.size
        
        sig_chans = self._rawio.header['signal_channels'][self._global_channel_indexes]
        
        assert np.unique(sig_chans['units']).size==1, 'Channel do not have same units'
        assert np.unique(sig_chans['dtype']).size==1, 'Channel do not have same dtype'
        assert np.unique(sig_chans['sampling_rate']).size==1, 'Channel do not have same sampling_rate'
        
        self.units = ensure_signal_units(sig_chans['units'][0])
        self.dtype = sig_chans['dtype'][0]
        self.sampling_rate = sig_chans['sampling_rate'][0] * pq.Hz
        sigs_size = self._rawio.get_signal_size(block_index=block_index, seg_index=seg_index, 
                                                            channel_indexes=self._global_channel_indexes)
        self.shape = (sigs_size, self._nb_chan)
        self.t_start = self._rawio.get_signal_t_start(block_index, seg_index, self._global_channel_indexes) * pq.s
        
        #magnitude_mode='raw' is supported only if all offset=0
        #and all gain are the same
        support_raw_magnitude = np.all(sig_chans['gain']==sig_chans['gain'][0]) and \
                                                    np.all(sig_chans['offset']==0.)
        if support_raw_magnitude:
            self._raw_units = pq.CompoundUnit('{}*{}'.format(sig_chans['gain'][0], sig_chans['units'][0]))
        else:
            self._raw_units = None

        #both necessary attr and annotations
        kargs = {}
        kargs['name'] = self._make_name(None)
        
        BaseProxy.__init__(self, **kargs)
    
    def _make_name(self, channel_indexes):
        sig_chans = self._rawio.header['signal_channels'][self._global_channel_indexes]
        if channel_indexes is not None:
            sig_chans = sig_chans[channel_indexes]
        name = 'Channel bundle ({}) '.format(','.join(sig_chans['name']))
        return name
    
    @property
    def duration(self):
        '''Signal duration'''
        return self.shape[0] / self.sampling_rate

    @property
    def t_stop(self):
        '''Time when signal ends'''
        return self.t_start + self.duration
    
    def load(self, time_slice=None, channel_indexes=None, magnitude_mode='rescaled'):
        '''
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :channel_indexes: None or list. Channels to load. None is all channels
                    Be carefull that channel_indexes represent the local channel index inside
                    the AnalogSignal and not the global_channel_indexes like in rawio.
            :magnitude_mode: 'rescaled' or 'raw'.
                For instance if the internal dtype is int16:
                    * **rescaled** give [1.,2.,3.]*pq.uV and the dtype is float32
                    * **raw** give [10, 20, 30]*pq.CompoundUnit('0.1*uV')
                The CompoundUnit with magnitude_mode='raw' is usefull to
                postpone the scaling when needed and having an internal dtype=int16
                but it less intuitive when you don't know so well quantities.
        '''
        
        if channel_indexes is None:
            channel_indexes = slice(None)
        
        sr = self.sampling_rate
        
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
                #the i_start is ncessary ceil
                i_start = int(np.ceil((t_start-self.t_start).magnitude * sr.magnitude))
                #this needed to get the real t_start of the first sample
                #because do not necessary match what is demanded
                sig_t_start = self.t_start + i_start/sr 
                
            if t_stop is None:
                i_stop = None
            else:
                t_stop = ensure_second(t_stop)
                assert t_start<=t_stop<=self.t_stop, 't_stop is outside'
                i_stop = int((t_stop-self.t_start).magnitude * sr.magnitude)
        
        raw_signal = self._rawio.get_analogsignal_chunk(block_index=self._block_index,
                    seg_index=self._seg_index, i_start=i_start, i_stop=i_stop,
                    channel_indexes=self._global_channel_indexes[channel_indexes])
        
        #if slice in channel so the name change
        #and also array_annotations
        #TODO later: implement array_annotations slice here
        if raw_signal.shape[1]!=self._nb_chan:
            name = self._make_name(channel_indexes)
        else:
            name = self.name
        
        if magnitude_mode=='raw':
            assert self._raw_units is not None,\
                    'raw magnitude is not support gain are not the same for all channel'
            sig = raw_signal
            units = self._raw_units
            
        elif magnitude_mode=='rescaled':
            sig = self._rawio.rescale_signal_raw_to_float(raw_signal,  dtype='float32',
                                            channel_indexes=self._global_channel_indexes[channel_indexes])
            units = self.units
        
        anasig = AnalogSignal(sig, units=units, copy=False, t_start=sig_t_start,
                    sampling_rate=self.sampling_rate, name=name,
                    file_origin=self.file_origin, description=self.description,
                    **self.annotations)
        
        return anasig


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



unit_convert = {'Volts': 'V',  'volts': 'V','Volt': 'V', 'volt': 'V', ' Volt' : 'V','microV': 'V'}
def ensure_signal_units(units):
    #test units
    units = units.replace(' ', '')
    if units in unit_convert:
        units = unit_convert[units]
    try:
        units = pq.Quantity(1, units)
    except:
        logging.warning('Units "{}" not understand use dimentionless instead'.format(units))
        units = ''
    return units

def check_annotations(annotations):
    #force type to str for some keys
    # imposed for tests
    for k in ('name', 'description', 'file_origin'):
        if k in annotations:
            annotations[k] = str(annotations[k])
    return annotations

def ensure_second(v):
    if isinstance(v, float):
        return v*pq.s
    elif isinstance(v, pq.Quantity):
        return v.rescale('s')
    elif isinstance(v, int):
        return float(v)*pq.s


