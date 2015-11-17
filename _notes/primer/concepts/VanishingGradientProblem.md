# Vanishing Gradient Problem

In any neural network, the error calculated at the last layer of the network is propagated backwards. As it propagates, the value of the gradients gradually diminish. This causes the the initial layers of the network to train slowly compared to the layers close to the output. This is known as the vanishing gradients problem. This problem affects especially the *deep* networks with multiple layers. 

Alternatively, derivatives of activation functions that can accommodate large values, lead to *Exploding Gradient Problem*. 

## Multi-level Hierarchy

Each layer of the network is pre-trained, one at a time through unsupervised learning, to represent a compressed form of the observations, which is fed to the next layer. Backpropagation is used for fine-tuning.

Read more [here](ftp://195.176.70.136/pub/juergen/chunker.pdf)

## Resilient Backpropagation

RProp is a weight update rule which just considers the sign of the partial derivative. If the partial derivative of error w.r.t. the weight is of the same sign as before, the update value of the weight is multiplied by a constant n+ where n+ > 0, while if the partial derivative is of the different sign, the update value is multiplied with a n- where n- < 0.

RProp is used in Long Short Term Memory. Read about it [here](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)

**Why does it work?** Read [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.4576)

## Faster Hardware

[How does faster hardware solve Vanishing Gradient problem?](http://arxiv.org/abs/1404.7828)


