from neo.core.baseneo import BaseNeo

import quantities as pq


class Event(BaseNeo):
    """
    Object to represent ponctual time event.
    Useful for managing trigger, stimulus, ...
    
    

    Usage:
    
    
    Necessary Attributes/properties:
        time (quantitie):
        label (str): 
    
    Recommanded Attributes/properties:
        name:
        description:
        file_origin:    
        
    Container of:
        None
        
    """
    def __init__(self, time = 0*pq.s , duration = 0*pq.s , label = '',
                    **kargs):
        BaseNeo.__init__(self, **kargs)
        
        self.time = time
        self.label = label
