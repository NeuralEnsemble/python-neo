"""
IgorProIO Demo (BROKEN)
=======================

"""

###########################################################
# Import our packages
import os
from urllib.request import urlretrieve
import zipfile
import matplotlib.pyplot as plt
from neo.io import get_io

#############################################################
# Then download some data
# Downloaded from Human Brain Project Collaboratory
# Digital Reconstruction of Neocortical Microcircuitry (nmc-portal)
# http://microcircuits.epfl.ch/#/animal/8ecde7d1-b2d2-11e4-b949-6003088da632
# NOTE: this dataset is not found as the link is broken.

# datafile_url = "https://microcircuits.epfl.ch/data/released_data/B95.zip"
# filename_zip = "B95.zip"
# filename = "grouped_ephys/B95/B95_Ch0_IDRest_107.ibw"
# urlretrieve(datafile_url, filename_zip)

# zip_ref = zipfile.ZipFile(filename_zip)  # create zipfile object
# zip_ref.extract(path=".", member=filename)  # extract file to dir
# zip_ref.close()

# ######################################################
# # Once we have our data we can use `get_io` to find an
# # io (Igor in this case). Then we read the analogsignals
# # Finally we will make some nice plots
# reader = get_io(filename)
# signal = reader.read_analogsignal()
# plt.plot(signal.times, signal)
# plt.xlabel(signal.sampling_period.dimensionality)
# plt.ylabel(signal.dimensionality)

# plt.show()
