# Recurrent Neural Networks

## Reference

1. [Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network)
2. [Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
3. [WildML](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) 

## Abstract

RNNs combine the input vector with their current state vector, using a learned function, to produce a new state vector. 

> Training Neural networks is optimization over functions, training RNN is optimization over programs.

```python
class RNN:
  # ...
  def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y
```

Mathematically,

$h = h_{t-1}W_{hh} + W_{xh}x_t$

During each step, RNN combines the current hidden state with previous hidden state, by matrix addition. The current hidden state is the product of input x and weight matrix $W_{xh}$ and the previous hidden state term is the product of $h_{t-1}$ and $W_{hh}$, where $h_{t-1} = h_{t-2}W_{hh} + W_{xh}x_t$


## Models

### Bidirectional RNN

Input sequence is processed by two RNN's. One process it from left to right and the other processed it from right to left. This way each element has two contexts : the past and the future contexts. Consider a paragraph in which we are trying to label or calculate a probability score of a sentence. We can combine the score obtained from two contexts. 

### LSTM

### Hopfield Network

### Continuous-time RNN

* Use in [Evolutionary Robotics](http://users.sussex.ac.uk/~inmanh/csrp317.pdf)
* [Cooperation](http://groups.lis.illinois.edu/amag/langev/localcopy/pdf/quinn01evolvingCommunication.pdf)

### Recurrent Multilayer Perceptron

It consists of multiple sub-networks cascaded together. Each sub-network is a feed-forward network, while the last layer has feedback connections. Each sub-network is feed-forward connected. 

Read More [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.3527)

### Second Order RNN

The weights are of higher order Wijk : 3 dimensional connections between layers. LSTM is an example of second order RNN. This concept is used to implement Finite State Machines in RNN. Read more [here](https://clgiles.ist.psu.edu/pubs/NC1992-recurrent-NN.pdf)

### Multiple Timescales RNN (MTRNN)

Read [here](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2570613/pdf/pcbi.1000220.pdf]

### Neural Turing Machine 

A RNN connected to external memory elements, functions like a Turing Machine/ Von Neumann Machine. It is fully differentiable and can be trained through gradient descent. 


## Training

### Backpropagation Through Time (BPTT)

Assuming that a RNN consists of two sub-networks f and g. Where f is the hidden layer which gets connected to multiple instances of itself over time. BPTT unfolds f into k instances of f connected in series as a single deep feed-forward network, with the 0th instance of f connected to input and kth instance of f connected to g (presumably the output layer). The error at g is backpropagated like it would be in a normal feed-forward network. 

Read more [here](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) and [here](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf)

### Genetic Algorithms

Read [here](http://arimaa.com/arimaa/about/Thesis/)



## Notes

1. Recurrent Neural Networks are a special case of Recursive Neural Network with a particular structure : a linear chain.  
2. RNNs are (Turing-complete](http://binds.cs.umass.edu/papers/1995_Siegelmann_Science.pdf)
3. [Attention based Model for recognizing objects in images](http://arxiv.org/abs/1412.7755)