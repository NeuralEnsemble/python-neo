"""
ImageSequences
==============

"""

##########################################################
# Let's import some packages

from neo.core import ImageSequence
from neo.core import RectangularRegionOfInterest, CircularRegionOfInterest, PolygonRegionOfInterest
import matplotlib.pyplot as plt
import quantities as pq

import random


############################################################
# Now we need to generate some data
# We will just make a nice box and then we can attach this
# ImageSequence to a variety of ROIs
# our ImageSequence will be 50 frames of 100x100 pixel images

l = []
for frame in range(50):
    l.append([])
    for y in range(100):
        l[frame].append([])
        for x in range(100):
            l[frame][y].append(random.randint(0, 50))

#####################################################################
# we then make our image sequence and pull out our results from the
# image_seq

image_seq = ImageSequence(l, sampling_rate=500 * pq.Hz, spatial_scale="m", units="V")

result = image_seq.signal_from_region(
    CircularRegionOfInterest(image_seq, 50, 50, 25),
    CircularRegionOfInterest(image_seq, 10, 10, 5),
    PolygonRegionOfInterest(image_seq, (50, 25), (50, 45), (14, 65), (90, 80)),
)

###############################################################
# It is easy to plot our results using matplotlib

for i in range(len(result)):
    plt.figure()
    plt.plot(result[i].times, result[i])
    plt.xlabel("seconde")
    plt.ylabel("valeur")
plt.tight_layout()
plt.show()
