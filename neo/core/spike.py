from neo.core.baseneo import BaseNeo


class Spike(BaseNeo):
    """
    Object to represent one spike emit by a Unit and represented by
    its time occurence and optional waveform.
    
    
    
    Usage:
    
    
    Necessary Attributes/properties:
        time (quantitie):
    
    Recommanded Attributes/properties:
        waveform (quantitie 2D (channel_index X time) = 
        sampling_rate = 
        left_sweep = 
    
    Properties:
        right_sweep
        duration
    
    """
    def __init__(self, time,
                    waveform=None,
                    sampling_rate= None,
                    left_sweep = None,
                    **kargs):
        BaseNeo.__init__(self , **kargs)
        self.time = time
        self.waveform = waveform
        self.left_sweep = left_sweep

    @property
    def duration(self):
        try:
            return self.waveform.shape[1]/self.sampling_rate
        except:
            return None

    @property
    def right_sweep(self):
        try:
            return self.left_sweep + self.duration()
        except:
            return None


