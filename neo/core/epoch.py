from neo.core.baseneo import BaseNeo
import quantities as pq

class Epoch(BaseNeo):
    """
    Similar as Event ( to represent ponctual time) but with a duration.
    Useful for describing a period, the state of a subject, ...
    

    Usage:
    
    
    Necessary Attributes/properties:
        time (quantitie):
        duration (quantitie):
        label (str): 
    
    Recommanded Attributes/properties:
        
    
    
    Container of:
        None
        
    """
    def __init__(self, time = 0*pq.s ,
                    duration = 0*pq.s ,
                    label = '',
                    **kargs):
        BaseNeo.__init__(self, **kargs)
        self.time = time
        self.duration = duration
        self.label = label
