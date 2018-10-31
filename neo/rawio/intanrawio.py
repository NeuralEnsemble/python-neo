# -*- coding: utf-8 -*-
"""

Support for intan tech rhd file.

See http://intantech.com/files/Intan_RHD2000_data_file_formats.pdf

Author: Samuel Garcia

"""
from __future__ import print_function, division, absolute_import
# from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np
from collections import OrderedDict


class IntanRawIO(BaseRawIO):
    """

    """
    extensions = ['rdh']
    rawmode = 'one-dir'
    