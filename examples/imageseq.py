from neo.core import ImageSequence
from neo.core import RectangularRegionOfInterest, CircularRegionOfInterest, PolygonRegionOfInterest
import matplotlib.pyplot as plt
import quantities as pq

import random

# generate data

l = []
for frame in range(50):
    l.append([])
    for y in range(100):
        l[frame].append([])
        for x in range(100):
            l[frame][y].append(random.randint(0, 50))

image_seq = ImageSequence(l, sampling_rate=500 * pq.Hz, spatial_scale='m', units='V')

result = image_seq.signal_from_region(CircularRegionOfInterest(50, 50, 25),
                                      CircularRegionOfInterest(10, 10, 5),
                                      PolygonRegionOfInterest((50, 25), (50, 45), (14, 65),
                                                              (90, 80)))

for i in range(len(result)):
    plt.figure()
    plt.plot(result[i].times, result[i])
    plt.xlabel("seconde")
    plt.ylabel("valeur")

plt.show()
