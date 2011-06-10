from neo.core.baseneo import BaseNeo

import numpy as np
import quantities as pq

class IrregularlySampledSignal(BaseNeo):
    """
    Object to manage signal when the sampling is not regular.
    In short this object manage both the signal values and its times vector.
    

    Usage:
    
    
    Necessary Attributes/properties:
        times (quantitie 1D):
        samples (quantities 1D): 
    
    Recommanded Attributes/properties:
        channel_name : 
        channel_index :
    
    
    Container of:
        None
        
    """
    def __init__(self, times = np.array([ ]) * pq.s ,
                    samples = np.array([ ]) * pq.dimensionless ,
                    **kargs):
        BaseNeo.__init__(self, **kargs)
        self.times = times
        self.samples = samples
