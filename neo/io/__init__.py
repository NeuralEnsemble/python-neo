from baseio import *

all_format = [ ]

try:
    from pynn import TextFile
    all_format += ['PyNN']
except ImportError:
    print "Error while loading pyNN IO module"

#try:
    #from spike2 import *
    #all_format += ['PyNN']
#except ImportError:
    #pass



#~ if sys.platform =='win32':
	#~ from neurshare import Neuroshare
	#~ all_IOclass += [ Neuroshare ]
