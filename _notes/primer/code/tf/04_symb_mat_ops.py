#!/usr/bin/env python

import tensorflow as t0

x=t0.placeholder("float",[1,2])
y=t0.placeholder("float",[2,1])

mul_op = t0.matmul(x,y)

with t0.Session() as ss:
	a=t0.constant([[1.,2.]])
	b=t0.constant([[3.],[4.]])
	z = ss.run(mul_op(a,b))
	#print "Symbolic Constant multiplication : %i" % ss.run(z)
