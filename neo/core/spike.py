from neo.core.baseneo import BaseNeo


class Spike(BaseNeo):
    """
    time quantified float
    waveforms is a 2D quantified array with dimensions: left_sweep, trodeness
   
    """
    def __init__(self, time, waveforms=None, sampling_rate=[])
        BaseNeo.__init__(self)
        self.time = time
        self.waveforms = waveforms
        self.sampling_rate = sampling_rate
        return self
