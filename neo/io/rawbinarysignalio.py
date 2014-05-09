# -*- coding: utf-8 -*-
"""
Class for reading/writing data in a raw binary interleaved compact file.
Sampling rate, units, number of channel and dtype must be externally known.
This generic format is quite widely used in old acquisition systems and is quite universal
for sharing data.

Supported : Read/Write

Author: sgarcia

"""

import os

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, AnalogSignal


class RawBinarySignalIO(BaseIO):
    """
    Class for reading/writing data in a raw binary interleaved compact file.

    Usage:
        >>> from neo import io
        >>> r = io.RawBinarySignalIO( filename = 'File_ascii_signal_2.txt')
        >>> seg = r.read_segment(lazy = False, cascade = True,)
        >>> print seg.analogsignals  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ...

    """
    is_readable        = True
    is_writable        = True

    supported_objects  = [Segment , AnalogSignal]
    readable_objects    = [ Segment]
    writeable_objects   = [Segment]

    has_header         = False
    is_streameable     = False
    read_params        = { Segment : [
                                        ('sampling_rate' , { 'value' : 1000. } ) ,
                                        ('nbchannel' , { 'value' : 16 } ),
                                        ('bytesoffset' , { 'value' : 0 } ),
                                        ('t_start' , { 'value' : 0. } ),
                                        ('dtype' , { 'value' : 'float32' , 'possible' : ['float32' , 'float64',
                                                                                                'int16' , 'uint16', 'int32' , 'uint32',  ] } ),
                                        ('rangemin' , { 'value' : -10 } ),
                                        ('rangemax' , { 'value' : 10 } ),
                                    ]
                        }
    write_params       = { Segment : [
                                        ('bytesoffset' , { 'value' : 0 } ),
                                        ('dtype' , { 'value' : 'float32' , 'possible' :  ['float32' , 'float64',
                                                                                                    'int16' , 'uint16', 'int32' , 'uint32',  ] } ),
                                        ('rangemin' , { 'value' : -10 } ),
                                        ('rangemax' , { 'value' : 10 } ),
                                    ]
                        }

    name               = None
    extensions          = [ 'raw' ]


    mode = 'file'
    def __init__(self , filename = None) :
        """
        This class read a binary file.

        **Arguments**

            filename : the filename to read

        """
        BaseIO.__init__(self)
        self.filename = filename

    def read_segment(self,
                                        cascade = True,
                                        lazy = False,

                                        sampling_rate = 1.*pq.Hz,
                                        t_start = 0.*pq.s,
                                        unit = pq.V,

                                        nbchannel = 1,
                                        bytesoffset = 0,

                                        dtype = 'f4',
                                        rangemin = -10,
                                        rangemax = 10,
                                    ):
        """
        Reading signal in a raw binary interleaved compact file.

        Arguments:
            sampling_rate :  sample rate
            t_start : time of the first sample sample of each channel
            unit: unit of AnalogSignal can be a str or directly a Quantities
            nbchannel : number of channel
            bytesoffset : nb of bytes offset at the start of file

            dtype : dtype of the data
            rangemin , rangemax : if the dtype is integer, range can give in volt the min and the max of the range
        """
        seg = Segment(file_origin = os.path.basename(self.filename))
        if not cascade:
            return seg

        dtype = np.dtype(dtype)

        if type(sampling_rate) == float or type(sampling_rate)==int:
            # if not quantitities Hz by default
            sampling_rate = sampling_rate*pq.Hz

        if type(t_start) == float or type(t_start)==int:
            # if not quantitities s by default
            t_start = t_start*pq.s

        unit = pq.Quantity(1, unit)

        if not lazy:
            sig = np.memmap(self.filename, dtype = dtype, mode = 'r', offset = bytesoffset)
            if sig.size % nbchannel != 0 :
                sig = sig[:- sig.size%nbchannel]
            sig = sig.reshape((sig.size/nbchannel,nbchannel))
            if dtype.kind == 'i' :
                sig = sig.astype('f')
                sig /= 2**(8*dtype.itemsize)
                sig *= ( rangemax-rangemin )
                sig += ( rangemax+rangemin )/2.
            elif dtype.kind == 'u' :
                sig = sig.astype('f')
                sig /= 2**(8*dtype.itemsize)
                sig *= ( rangemax-rangemin )
                sig += rangemin
            sig_with_units =  pq.Quantity(sig, units=unit, copy = False)
        
        for i in range(nbchannel) :
            if lazy:
                signal = [ ]*unit
            else:
                signal = sig_with_units[:,i]

            anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                  t_start=t_start, channel_index=i, copy = False)
            
            if lazy:
                # TODO
                anaSig.lazy_shape = None
            seg.analogsignals.append(anaSig)

        seg.create_many_to_one_relationship()
        return seg

    def write_segment(self, segment, dtype='f4', rangemin=-10,
                      rangemax=10, bytesoffset=0):
        """

         **Arguments**
            segment : the segment to write. Only analog signals will be written.

            dtype : dtype of the data
            rangemin , rangemax : if the dtype is integer, range can give in volt the min and the max of the range

        """
        if bytesoffset:
            raise NotImplementedError('bytesoffset values other than 0 ' +
                                      'not supported')

        dtype = np.dtype(dtype)

        # all AnaologSignal from Segment must have the same length
        for anasig in segment.analogsignals[1:]:
            assert anasig.size == segment.analogsignals[0].size

        sigs = np.empty((segment.analogsignals[0].size, len(segment.analogsignals)))
        for i, anasig in enumerate(segment.analogsignals):
            sigs[:, i] = anasig.magnitude

        if dtype.kind == 'i':
            sigs -= ( rangemax+rangemin )/2.
            sigs /= (rangemax - rangemin)
            sigs *= 2 ** (8 * dtype.itemsize )
        elif dtype.kind == 'u' :
            sigs -= rangemin
            sigs /= (rangemax - rangemin)
            sigs *= 2 ** (8 * dtype.itemsize)
        sigs = sigs.astype(dtype)
            
        f = open(self.filename, 'wb')
        f.write(sigs.tostring())
        f.close()


