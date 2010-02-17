# -*- coding: utf-8 -*-

class Segment(object):
    """
    Heterogeneous container of several data sharing a common time base

    **Definition**
    A :class:`Segment` is a container gathering discrete or continous data acquired during the same time lapse.
    In short, a :class:`Segment` may contain :class:`AnalogSignal`, :class:`SpikeTrain`, :class:`Event` and :class:`Epoch` 
    that share the same time base.

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
        Return  a list of :calss:`AnalogSignal`.
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
        self._epochs

