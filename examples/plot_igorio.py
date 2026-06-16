"""
IgorProIO Demo
=======================

"""

###########################################################
# Import our packages

from urllib.request import urlretrieve
import matplotlib.pyplot as plt
from neo.io import get_io

#############################################################
# Then download some data
# we can try out some data on the NeuralEnsemble ephy testing repo

url_repo = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/"
distantfile = url_repo + "igor/win-version2.ibw"
localfile = "win-version2.ibw"
urlretrieve(distantfile, localfile)


# ######################################################
# Once we have our data we can use `get_io` to find an
# io (Igor in this case). Then we read the analogsignals
# Finally we will make some nice plots
#
# Note: not all IOs have all types of read functionality
# see our documentation for a better understanding of the 
# Neo object hierarchy and the functionality of differnt IOs

reader = get_io(localfile)
signal = reader.read_analogsignal()
plt.plot(signal.times, signal)
plt.xlabel(signal.sampling_period.dimensionality)
plt.ylabel(signal.dimensionality)

plt.show()
