#!/usr/bin/env python
########################
## Constant Sclar ops ##
########################

import tensorflow as t0

x=t0.constant(3)
y=t0.constant(7)

with t0.Session() as ss:
    print "Constant addition : %i" % ss.run(x+y)
    print "Constant multiplication : %i" % ss.run(x*y)
