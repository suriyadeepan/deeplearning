# Resilient Backpropagation

RProp is a weight update rule which just considers the sign of the partial derivative. If the partial derivative of error w.r.t. the weight is of the same sign as before, the update value of the weight is multiplied by a constant n+ where n+ > 0, while if the partial derivative is of the different sign, the update value is multiplied with a n- where n- < 0.

## Why does it work?

Read [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.4576)
