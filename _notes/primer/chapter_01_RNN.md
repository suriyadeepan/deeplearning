# Recurrent Neural Networks

## Reference

1. [Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network)

## Models

### Bidirectional RNN

Input sequence is processed by two RNN's. One process it from left to right and the other processed it from right to left. This way each element has two contexts : the past and the future contexts. Consider a paragraph in which we are trying to label or calculate a probability score of a sentence. We can combine the score obtained from two contexts. 

## LSTM

## Hopfield Network

## Continuous-time RNN

* Use in [Evolutionary Robotics](http://users.sussex.ac.uk/~inmanh/csrp317.pdf)
* [Cooperation](http://groups.lis.illinois.edu/amag/langev/localcopy/pdf/quinn01evolvingCommunication.pdf)

## Recurrent Multilayer Perceptron

It consists of multiple sub-networks cascaded together. Each sub-network is a feed-forward network, while the last layer has feedback connections. Each sub-network is feed-forward connected. 

Read More [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.3527)

## Second Order RNN

The weights are of higher order Wijk : 3 dimensional connections between layers. LSTM is an example of second order RNN. This concept is used to implement Finite State Machines in RNN. Read more [here](https://clgiles.ist.psu.edu/pubs/NC1992-recurrent-NN.pdf)

## Multiple Timescales RNN (MTRNN)

Read [here](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2570613/pdf/pcbi.1000220.pdf]

## Neural Turing Machine 

A RNN connected to external memory elements, functions like a Turing Machine/ Von Neumann Machine. It is fully differentiable and can be trained through gradient descent. 