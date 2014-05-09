# -*- coding: utf-8 -*-
'''
This module defines :class:`Spike`, a single spike with an optional waveform.

:class:`Spike` derives from :class:`BaseNeo`, from :module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import quantities as pq

from neo.core.baseneo import BaseNeo


class Spike(BaseNeo):
    '''
    A single spike.

    Object to represent one spike emitted by a :class:`Unit` and represented by
    its time occurence and optional waveform.

    *Usage*::
        >>> from quantities import s
        >>> spk = Spike(3*s)
        >>> spk.time
        array(3.0) * s

    *Required attributes/properties*:
        :time: (quantity) The time of the spike.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :waveforms: (quantity array 2D (channel_index, time))
            The waveform of the spike.
        :sampling_rate: (quantity scalar) Number of samples per unit time
            for the waveform.
        :left_sweep: (quantity scalar) Time from the beginning
            of the waveform to the trigger time of the spike.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`sampling_rate`)
        :duration: (quantity scalar) Duration of :attr:`waveform`, read-only.
            (:attr:`waveform`.shape[1] * :attr:`sampling_period`)
        :right_sweep: (quantity scalar) Time from the trigger time of the spike
            to the end of the :attr:`waveform`, read-only.
            (:attr:`left_sweep` + :attr:`duration`)

    '''

    _single_parent_objects = ('Segment', 'Unit')
    _necessary_attrs = (('time', pq.Quantity, 0),)
    _recommended_attrs = ((('waveform', pq.Quantity, 2),
                           ('left_sweep', pq.Quantity, 0),
                           ('sampling_rate', pq.Quantity, 0)) +
                          BaseNeo._recommended_attrs)

    def __init__(self, time=0 * pq.s, waveform=None, sampling_rate=None,
                 left_sweep=None, name=None, description=None,
                 file_origin=None, **annotations):
        '''
        Initialize a new :class:`Spike` instance.
        '''
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
        '''
        Returns the duration of the :attr:`waveform`.

        (:attr:`waveform`.shape[1] * :attr:`sampling_period`)
        '''
        if self.waveform is None or self.sampling_rate is None:
            return None
        return self.waveform.shape[1] / self.sampling_rate

    @property
    def right_sweep(self):
        '''
        Time from the trigger time of the spike to the end of the
        :attr:`waveform`.

        (:attr:`left_sweep` + :attr:`duration`)
        '''
        dur = self.duration
        if dur is None or self.left_sweep is None:
            return None
        return self.left_sweep + dur

    @property
    def sampling_period(self):
        '''
        Interval between two samples.

        (1/:attr:`sampling_rate`)
        '''
        if self.sampling_rate is None:
            return None
        return 1.0 / self.sampling_rate

    @sampling_period.setter
    def sampling_period(self, period):
        '''
        Setter for :attr:`sampling_period`.
        '''
        if period is None:
            self.sampling_rate = None
        else:
            self.sampling_rate = 1.0 / period

    def merge(self, other):
        '''
        Merging is not supported in :class:`Spike`.
        '''
        raise NotImplementedError('Cannot merge Spike objects')
