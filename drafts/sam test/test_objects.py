import sys

sys.path.append('../../')
import neo
from neo import *


bl = Block( toto = 'yep')
print bl.toto

for e in dir(bl):
    print e
print
for e in dir(Block.__class__):
    print e
print 
print Block.__class__.__name__

print Block.__name__
print type(Segment).__name__

#~ sptr = SpikeTrain([ ] , 0 , 5)

#~ neo.test

