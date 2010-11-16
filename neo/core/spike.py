# -*- coding: utf-8 -*-

from baseneo import BaseNeo

class Spike(BaseNeo):
    definition = """An :class:`Spike` is an evenement at time t with a waveform.
    Note that the use of :class:`Spike` is optional. :class:`SpikeTrain` is normally
    suffisent to manage spike_times and waveforms. :class:`Spike` is an alternative and more 
    flexible way to manage spikes.
    """
    
     
    __doc__ = """
    Object to represent a spike

    **Definition**"""+definition+"""

    with arguments:
    
    ``time`` The spike time
    
    ``waveform`` An 2D array of the waveform 
                        dim 0 = trodnes (1 = normal, 2 = stereotrod, 4 = tetrod)
                        dim 1 = form for each one
    
    ``sampling_rate`` The waveform sampling rate
    
    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):
        
        for attr in [  'time' , 'waveform', 'sampling_rate', 'right_sweep', 'left_sweep' ]:
            if attr in karg:
                setattr(self, attr, karg[attr])
            else:
                setattr(self, attr, None)

