#!/usr/bin/env python

import tensorflow as t0

x=t0.constant([[1.,2.]])
y=t0.constant([[3.],[4.]])

# define operation mul_op
mul_op = t0.matmul(x,y)
#add_op = t0.add(x,y)

with t0.Session() as ss:
	#print "Matrix addition : %i" % ss.run(add_op)
	print "Constant Matrix multiplication : %i" % ss.run(mul_op)
