"""
Class for reading/writing data in a raw binary interleaved compact file.
Sampling rate, units, number of channel and dtype must be externally known.
This generic format is quite widely used in old acquisition systems and is quite universal
for sharing data.

Supported : Read/Write

Author: sgarcia

"""

import numpy as np

from neo.core import Segment, AnalogSignal, NeoReadWriteError

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.rawbinarysignalrawio import RawBinarySignalRawIO


class RawBinarySignalIO(RawBinarySignalRawIO, BaseFromRaw):
    """
    Class for reading/writing data in a raw binary interleaved compact file.

    **Important release note**

    Since the version neo 0.6.0 and the neo.rawio API,
    argmuents of the IO (dtype, nb_channel, sampling_rate) must be
    given at the __init__ and not at read_segment() because there is
    no read_segment() in neo.rawio classes.

    So now the usage is:
        >>>>r = io.RawBinarySignalIO(filename='file.raw', dtype='int16',
                                    nb_channel=16, sampling_rate=10000.)


    """

    _prefered_signal_group_mode = "group-by-same-units"

    is_readable = True
    is_writable = True

    supported_objects = [Segment, AnalogSignal]
    readable_objects = [Segment]
    writeable_objects = [Segment]

    def __init__(
        self,
        filename,
        dtype="int16",
        sampling_rate=10000.0,
        nb_channel=2,
        signal_gain=1.0,
        signal_offset=0.0,
        bytesoffset=0,
    ):
        RawBinarySignalRawIO.__init__(
            self,
            filename=filename,
            dtype=dtype,
            sampling_rate=sampling_rate,
            nb_channel=nb_channel,
            signal_gain=signal_gain,
            signal_offset=signal_offset,
            bytesoffset=bytesoffset,
        )
        BaseFromRaw.__init__(self, filename)

    def write_segment(self, segment):
        """

        **Arguments**
            segment : the segment to write. Only analog signals will be written.

        Support only 2 cases:
          * segment.analogsignals have one 2D AnalogSignal
          * segment.analogsignals have several 1D AnalogSignal with
            same length/sampling_rate/dtype

        """

        if self.bytesoffset:
            raise NotImplementedError("bytesoffset values other than 0 " + "not supported")

        anasigs = segment.analogsignals
        if len(anasigs) == 0:
            raise NeoReadWriteError("No AnalogSignal to write")

        anasig0 = anasigs[0]
        if len(anasigs) == 1 and anasig0.ndim == 2:
            numpy_sigs = anasig0.magnitude
        else:

            if anasig0.ndim != 1 and (anasig0.ndim == 2 and anasig0.shape[1] != 1):
                raise NeoReadWriteError("Incorrect analogsignal shaping for write")
            # all AnaologSignal from Segment must have the same length/sampling_rate/dtype
            for anasig in anasigs[1:]:
                if anasig.shape != anasig0.shape:
                    raise ValueError("The shape of one of the analog signals does not match")
                if anasig.sampling_rate != anasig0.sampling_rate:
                    raise ValueError("The sampling_rate of one of the analog signals does not match")
                if anasig.dtype != anasig0.dtype:
                    raise ValueError("The dtype of one the analog signals does not match")

            numpy_sigs = np.empty((anasig0.size, len(anasigs)))
            for i, anasig in enumerate(anasigs):
                numpy_sigs[:, i] = anasig.magnitude.flatten()

        numpy_sigs -= self.signal_offset
        numpy_sigs /= self.signal_gain
        numpy_sigs = numpy_sigs.astype(self.dtype)

        with open(self.filename, "wb") as f:
            f.write(numpy_sigs.tobytes())
