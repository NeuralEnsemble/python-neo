# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.tdtrawio import TdtRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestTdtRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = TdtRawIO
    entities_to_test = ['aep_05']
    
    files_to_download = [
        'aep_05/Block-1/aep_05_Block-1.Tbk',
        'aep_05/Block-1/aep_05_Block-1.Tdx',
        'aep_05/Block-1/aep_05_Block-1.tev',
        'aep_05/Block-1/aep_05_Block-1.tsq',

        #~ 'aep_05/Block-2/aep_05_Block-2.Tbk',
        #~ 'aep_05/Block-2/aep_05_Block-2.Tdx',
        #~ 'aep_05/Block-2/aep_05_Block-2.tev',
        #~ 'aep_05/Block-2/aep_05_Block-2.tsq',

        #~ 'aep_05/Block-3/aep_05_Block-3.Tbk',
        #~ 'aep_05/Block-3/aep_05_Block-3.Tdx',
        #~ 'aep_05/Block-3/aep_05_Block-3.tev',
        #~ 'aep_05/Block-3/aep_05_Block-3.tsq', 
    ]
    


    
if __name__ == "__main__":
    unittest.main()
