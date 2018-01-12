# -*- coding: utf-8 -*-
"""
Tests proxyobject mechanisms with ExampleRawIO
"""


import unittest

import numpy as np
import quantities as pq
from neo.rawio.examplerawio import ExampleRawIO
from neo.io.basefromrawio.proxyobjects import AnalogSignalProxy

from neo.core import (AnalogSignal, 
                      Epoch, Event, SpikeTrain)


from neo.test.tools import (
                            assert_arrays_almost_equal,
                            assert_neo_object_is_compliant,
                            #~ assert_same_sub_schema,
                            #~ assert_objects_equivalent,
                            assert_same_attributes,
                            #~ assert_same_sub_schema,
                            )



class BaseProxyTest(unittest.TestCase):
    def setUp(self):
        self.reader = ExampleRawIO()
        self.reader.parse_header()

class TestAnalogSignalProxy(BaseProxyTest):
    
    def test_AnalogSignalProxy(self):
        proxy_anasig = AnalogSignalProxy(rawio=self.reader, channel_indexes=None,
                        block_index=0, seg_index=0,)
        
        assert proxy_anasig.sampling_rate==10*pq.kHz
        assert proxy_anasig.t_start==0*pq.s
        assert proxy_anasig.t_stop==10*pq.s
        assert proxy_anasig.duration==10*pq.s
        
        #full load
        full_anasig = proxy_anasig.load(time_slice=None)
        assert isinstance(full_anasig, AnalogSignal)
        assert_same_attributes(proxy_anasig, full_anasig)
        
        
        #slice time
        anasig = proxy_anasig.load(time_slice=(2.*pq.s, 5*pq.s))
        assert anasig.t_start == 2.*pq.s
        assert anasig.duration == 3.*pq.s
        assert anasig.shape == (30000, 16)

        #ceil next sample when slicing
        anasig = proxy_anasig.load(time_slice=(1.99999*pq.s, 5.000001*pq.s))
        assert anasig.t_start == 2.*pq.s
        assert anasig.duration == 3.*pq.s
        assert anasig.shape == (30000, 16)
        
        #select channels
        anasig = proxy_anasig.load(channel_indexes=[3,4,9])
        assert anasig.shape[1]==3
        
        #select channels and slice times
        anasig = proxy_anasig.load(time_slice=(2.*pq.s, 5*pq.s), channel_indexes=[3,4,9])
        assert anasig.shape==(30000, 3)
        
        #magnitude mode rescaled
        anasig_float = proxy_anasig.load(magnitude_mode='rescaled')
        assert anasig_float.dtype=='float32'
        assert anasig_float.units==pq.uV
        assert anasig_float.units==proxy_anasig.units

        #magnitude mode rescaled
        anasig_int = proxy_anasig.load(magnitude_mode='raw')
        assert anasig_int.dtype=='int16'
        assert anasig_int.units==pq.CompoundUnit('0.0152587890625*uV')
        
        assert_arrays_almost_equal(anasig_float, anasig_int.rescale('uV'), 1e-9)




if __name__ == "__main__":
    unittest.main()
