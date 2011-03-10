from neo.core.baseneo import BaseNeo

import numpy as np
import quantities as pq

class EventArray(BaseNeo):
    """
    Subset of event. Introduced for performance reason.
    An EventArray is prefered to a list of Event.
    

    Usage:
    
    
    Necessary Attributes/properties:
        times (quantitie array 1D):
        labels (numpy.array 1D dtype='S') ):
    
    Recommanded Attributes/properties:
        
    
    
    Container of:
        None
        
    """
    def __init__(self, times = np.array([ ]) * pq.s ,
                    durations = np.array([ ]) * pq.s ,
                    labels = np.array([ ] , dtype = 'S'),
                    **kargs):
        BaseNeo.__init__(self, **kargs)
        self.times = times
        self.labels = labels
