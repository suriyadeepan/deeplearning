# Restricted Boltzmann Machines

RBMs are bipartite graphs of neurons : two fully connected layers of neurons with no interconnections within a layer. One is a visible layer and the other is hidden. An unrestricted bolztmann machine allows connections between hidden layer units. This restriction in RBM helps in training[[2](http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf)] it efficiently.

Deep Belief Networks can be formed by stacking RBMs and fine tuning the resulting deep network by using gradient descent and backpropagation.


## Energy Based Models

Each configuration of nodes produce a particular value of energy(scalar). Learning the right configuration means minimizing the energy. Probability distribution of any energy based model is given by <br />

$p(x) = \frac{e^{-E(x)}}{Z}$

where the normalizing factor Z is called the partition function, is the summation of energy of all possible configuration of the model.

$Z = \sum_x e^{-E(x)}$ <br />

Learning of energy models can be achieved by minimizing the negative log likelihood.

Likelihood, $L(D,\theta) = \frac{1}{N} \sum_{x(i) \epsilon D} log(p(x^{(i)}))$ <br />
Loss, $l(D,\theta) = - L(D,\theta)$ <br />

where <br />
$\theta$ : paramters of model <br />
D : input dataset <br />
$x(i)$ : an example in D <br />


## Energy Based Models with Hidden units

Hidden units increase the expressive power of the model. 

$P(x) = \sum_{h} P(x,h) = \sum_{h} \frac{e^{-E(x,h)}}{Z}$

Free Energy, $F(x) = -log( \sum_{h} e^{-E(x,h)} )$

Now $P(x) = \frac{e^{-F(x)}}{Z}$



## Energy Model of RBM

$E(v_{i},h_{j}) = -\sum_{i} a_{i}v_{i} - \sum_{j} b_{j}h{j} - \sum_{i}\sum_{j} v_{i}w_{i,j}h_{j}$

where <br />
$v_{i}$ : ith unit of visible layer<br />
$h_{j}$ : jth unit of hidden  layer<br />
$w_{i,j}$ : (i,j)th unit of weight matrix connecting v and h<br />
a : visible layer bias<br />
b : hidden  layer bias<br />

In Matrix notation, <br />

$E(v,h) = -a^Tv -b^Th - v^TWh$


## References

1. [Wikipedia](http://www.wikiwand.com/en/Restricted_Boltzmann_machine)
2. [On Contrastive Divergence](http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf)