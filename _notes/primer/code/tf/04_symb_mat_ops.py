#!/usr/bin/env python

import tensorflow as t0

x=t0.placeholder(t0.types.int16)
y=t0.placeholder(t0.types.int16)

mul_op = t0.mul(x,y)
add_op = t0.add(x,y)

with t0.Session() as ss:
	print "Symbolic Constant addition : %i" % ss.run(add_op,feed_dict={x:3,y:7})
	print "Symbolic Constant multiplication : %i" % ss.run(mul_op,feed_dict={x:3,y:7})
