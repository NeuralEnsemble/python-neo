# -*- coding: utf-8 -*-

class Segment(object):
    
    definition = """A :class:`Segment` is a heterogeneous container for discrete
    or continous data sharing a common clock (time basis) but not necessary
    the same sampling rate, t_start and t_stop. In short, a :class:`Segment` is a 
    recording may contain AnalogSignal, SpikeTrain, Event or Epoch that share 
    the same logical clock.
    """
    
    
    

    
    __doc__ = """
    Heterogeneous container of several data sharing a common time base

    **Definition**
    """+definition+"""

    **Usage**


    **Example**

    >> seg = Segment()

    **Methods**

    get_analogsignals()

    get_spiketrains()

    get_events()

    get_epochs()

    """
    
    
    def __init__(self, *arg, **karg):
        self._analogsignals   = []
        self._spiketrains     = []
        self._epochs          = []
        self._events          = []
        self._recordingpoints = []
        self._neurons         = []
        
    def get_analogsignals(self):
        """
        Return a list of :class:`AnalogSignal`.
        """
        return self._analogsignals
        
    def get_spiketrains(self):
        """
        Return a list of :class:`SpikeTrain`.
        """
        return self._spiketrains
        
    def get_events(self):
        """
        Return a list of :class:`Event`.
        """
        return self._events

    def get_epochs(self):
        """
        Return a list of :class:`Epoch`.
        """
        return self._epochs

