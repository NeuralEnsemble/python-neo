from neo.core.baseneo import BaseNeo
import quantities as pq


class Spike(BaseNeo):
    """
    Object to represent one spike emitted by a :class:`Unit` and represented by
    its time occurence and optional waveform.

    *Usage*:
        TODO

    *Required attributes/properties*:
        :time: (quantity)

    *Recommended attributes/properties*:
        :waveform: (quantity 2D (channel_index X time))
        :sampling_rate:
        :left_sweep:
        :name:
        :description:
        :file_origin:

    *Properties*:
        :right_sweep:
        :duration:

    """
    def __init__(self, time=0 * pq.s, waveform=None, sampling_rate=None,
                 left_sweep=None, name=None, description=None,
                 file_origin=None, **annotations):
        """Initialize a new Spike."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

        self.time = time

        self.waveform = waveform
        self.left_sweep = left_sweep
        self.sampling_rate = sampling_rate

        self.segment = None
        self.unit = None

    @property
    def duration(self):
        if self.waveform is None or self.sampling_rate is None:
            return None
        return self.waveform.shape[1] / self.sampling_rate

    @property
    def right_sweep(self):
        dur = self.duration
        if dur is None or self.left_sweep is None:
            return None
        return self.left_sweep + dur

    @property
    def sampling_period(self):
        if self.sampling_rate is None:
            return None
        return 1.0 / self.sampling_rate

    @sampling_period.setter
    def sampling_period(self, period):
        if period is None:
            self.sampling_rate = None
        else:
            self.sampling_rate =  1.0 / period

