# Restricted Boltzmann Machines

RBMs are bipartite graphs of neurons : two fully connected layers of neurons with no interconnections within a layer. One is a visible layer and the other is hidden. An unrestricted bolztmann machine allows connections between hidden layer units. This restriction in RBM helps in training[[2](http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf)] it efficiently.

Deep Belief Networks can be formed by stacking RBMs and fine tuning the resulting deep network by using gradient descent and backpropagation.

## Energy Model

$E(v_{i},h_{j}) = -\sum_{i} a_{i}v_{i} - \sum_{j} b_{j}h{j} - \sum_{i}\sum_{j} v_{i}w_{i,j}h_{j}$

where <br />
$v_{i}$ : ith unit of visible layer<br />
$h_{j}$ : jth unit of hidden  layer<br />
$w_{i,j}$ : (i,j)th unit of weight matrix connecting v and h<br />
a : visible layer bias<br />
b : hidden  layer bias<br />


## References

1. [Wikipedia](http://www.wikiwand.com/en/Restricted_Boltzmann_machine)
2. [On Contrastive Divergence](http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf)