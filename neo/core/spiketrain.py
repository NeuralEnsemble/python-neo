from neo.core.baseneo import BaseNeo


class SpikeTrain(BaseNeo):
    """
    times is a 1D quantified array
    waveforms is a 3D quantified array with dimensions: element, trodeness, pts
    """
    def __init__(self, times, waveforms=None)
        BaseNeo.__init__(self)
        self.times = times
        self.waveforms = waveforms
        return self
