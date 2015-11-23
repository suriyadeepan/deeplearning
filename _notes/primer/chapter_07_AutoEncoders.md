# Autoencoders

Given inputs {$x^{1},x^{2},x^{3},...x^{i},...x^{m}$}, an autoencoder network sets the target value $y^{i} = x^{i}$, using backpropagation. It tries to learn the function $h_{W,b} \approx x$.

Consider a network that takes 10x10 = 100 pixel images as input. Here $x \epsilon R^{100}$ and $y \epsilon R^{100}$, let the hidden layer contain 50 nodes (i.e) the vector of hidden layer activation $a \epsilon R^{50}$. In this case, the hidden layer forms a compressed representation of the input, from which it reproduces an approximate form of the input, in the output layer. This network discovers the correlations between differnt features in the input. A low dimensional representation of the input is learnt by this network, similar to PCA. 

To summarize, propagating from input layer to hidden layer is the compression step, while propagating from hidden layer to output layer is the decompression step. 

### Sparse Activation

We can keep the number of hidden units to be larger than the number of input units, but put a sparsity constraint. Sparsity means, keeping most of the hidden unit activation values low, close to zero. This sparsity constraint term is added to the cost function used for backpropagation. 

Average activation value for hidden neuron j over the training set is given by,

$\hat{p}_{j} = \frac{1}{m} \sum \limits_{i=1}^{m} [ a_{j}x^{(i)}]$ <br />

where $a_{j}x^{(i)}$ is the activation of jth hidden neuron for ith training example.

![Activations over training examples](/home/jabroni/_/deeplearning/_notes/primer/svg/autoencoder1.png)

![Average Activation over training examples](/home/jabroni/_/deeplearning/_notes/primer/svg/autoencoder2.png)

Now that we have the average activation term, we can define the sparsity cost term, which basically is the difference between the average activation and the desired average activation value, which is typically set to something close 0, say 0.05.

The sparsity cost term is defined as,

$\sum\limits_{j=1}^{n} p log \frac{p}{\hat{p}_{j}} + (1-p)log \frac{1-p}{1-\hat{p}_j}$

where n is the number of hidden units. The sparsity penalty term is basically the KL-divergence between p and $\hat{p}$, written as

$\sum\limits_{j=1}^{n} KL(p||\hat{p}_{j})$

The overall cost function can now be written as

$J_{sparse} = \alpha J(W,b) + \beta \sum\limits_{j=1}^{n} KL(p||\hat{p}_{j})$






## Keyterms

1. **Overcomplete Representations** : If the size of the intermediate representation is larger than the input dimension.