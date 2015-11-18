# Neural Turing Machine

A turing machine basically has a R/W head that reads and write to blocks of data memory based on the instructions provided by the program memory. A Neural Turing Machine (NTM) is a Neural Network which reads and writes to an external memory.

> A Neural Turing Machine is a differentiable computer which can be trained using gradient descent, to learn programs. 

The network can read and write to the memory by using a weight vector to control the amount of attention given to a place (content) in memory and is able to perform tasks of temporal nature. 

## Introduction

Recurrent nets are Turing complete[4](http://www.sciencedirect.com/science/article/pii/S0022000085710136) and can simulate any procedure. Turing machine is enriched by an infinite tape to write to and read from. Similarly NTM is enriched by using a large, addressable memory.

**Working Memory** is a short term memory in the brain, over which the central executes focusses its attention on to perform certain operation on data. 

> Recursive processing of variable-length structures continues to be regarded as a hallmark of human cognition.

RNN are said to possess dynamic state (i.e) the state of the system is not only dependent on the current input, they are also dependant on the prvious states. 

**Blurry Read/Write** operations in NTM, makes the controller interact with the memory elements, with a degree of focus. The degree of blurriness is controlled by a attention focus mechanism. The memory locations focussed depend on the *specialized* output (heads) emitted by the network. The heads (R/W) consists of a weight vector $w_{t}$ which specifies the degree of read and write on the memory elements of the *memory matrix*. Consider the memory as a long list of vectors of size M and the number of elements of list as N.  

### Reading 

N - number of memory locations <br />
M - vector size at each location <br />
$w_{t}$ - normalized weightings over N locations emitted by read head at time t <br />

$\sum_{i} w_{t} = 1$

Read vector $r_{t} \leftarrow \sum_{i} w_{t}(i)M_{t}(i)$

### Writing

Each write consists of two operations : *erase* followed by an *add*.

**Erase Operation** 
$\tilde{M}_{t}(i) \leftarrow M_{t-1}(i)[1 - w_{t}(i)e_{t}]$

**Add Operation**
$M_{t}(i) \leftarrow \tilde{M}_{t}(i) + w_{t}(i)a_{t}$

### Addressing Mechanisms

1. Content-based Addressing
2. Location-based Addressing

Content-based addressing mechanism produces a normalized weight vector $w_{i}(t)$ using a positive key strength $\beta_{t}$ and a simularity measure between key vector $k_{t}$ of length M, provided by the write/read head and each vector $M_{t}(i)$. 

$w_{t}^{c}(i) \leftarrow \frac{exp\lgroup\beta_{t}K[ k_{t},M_{t}(i) ]\rgroup}{\sum_j exp\lgroup\beta_{t}K[ k_{t},M_{t}(j) ]\rgroup}$ <br />
K : cosine

Location-based addressing is designed to facilitate simple iteration and jumps. Rotional shifts results in shift in attention to right or left depending on the value being positive or negative. A scalar value $g_{t}$ (Interpolation gate (0,1) emitted by the head) is used to blend the previous weighting and the current weighting produced by the content system, to yeild the current gated weighting.

$w_{t}^{g} \leftarrow g_{t}w_{t}^{c} + (1 - g_{t})w_{t-1}$









## Notes

1. Hidden Markov Model vs RNN : dynamic, distributed state

## References

1. [http://lepisma.svbtle.com/neural-turing-machines-for-dummies](http://lepisma.svbtle.com/neural-turing-machines-for-dummies)
2. [Implementation](https://blog.wtf.sg/2014/10/27/neural-turing-machines-a-first-look/)
3. [Overview](www.i-programmer.info/news/105-artificial-intelligence/7923-neural-turing-machines-learn-their-algorithms.html)
4. [Turing complete](http://www.sciencedirect.com/science/article/pii/S0022000085710136)