# Autoencoders

Given inputs {$x^{1},x^{2},x^{3},...x^{i},...x^{m}$}, an autoencoder network sets the target value $y^{i} = x^{i}$, using backpropagation. It tries to learn the function $h_{W,b} \approx x$.

Consider a network that takes 10x10 = 100 pixel images as input. Here $x \epsilon R^{100}$ and $y \epsilon R^{100}$, let the hidden layer contain 50 nodes (i.e) the vector of hidden layer activation $a \epsilon R^{50}$. In this case, the hidden layer forms a compressed representation of the input, from which it reproduces an approximate form of the input, in the output layer. This network discovers the correlations between differnt features in the input. A low dimensional representation of the input is learnt by this network, similar to PCA. [[1](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)]

To summarize, propagating from input layer to hidden layer is the compression step, while propagating from hidden layer to output layer is the decompression step. [[2](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/autoencoders.html)]

### Sparse Activation

We can keep the number of hidden units to be larger than the number of input units, but put a sparsity constraint. Sparsity means, keeping most of the hidden unit activation values low, close to zero. This sparsity constraint term is added to the cost function used for backpropagation.

Average activation value for hidden neuron j over the training set is given by,

$\hat{p}_{j} = \frac{1}{m} \sum \limits_{i=1}^{m} [ a_{j}x^{(i)}]$ <br />

where $a_{j}x^{(i)}$ is the activation of jth hidden neuron for ith training example.[[3](https://chrisjmccormick.wordpress.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/)]

![Activations over training examples](/home/jabroni/_/deeplearning/_notes/primer/svg/autoencoder1.png)

![Average Activation over training examples](/home/jabroni/_/deeplearning/_notes/primer/svg/autoencoder2.png)

Now that we have the average activation term, we can define the sparsity cost term, which basically is the difference between the average activation and the desired average activation value, which is typically set to something close 0, say 0.05.

The sparsity cost term is defined as,

$\sum\limits_{j=1}^{n} p log \frac{p}{\hat{p}_{j}} + (1-p)log \frac{1-p}{1-\hat{p}_j}$

where n is the number of hidden units. The sparsity penalty term is basically the **KL-divergence** between p and $\hat{p}$, written as

$\sum\limits_{j=1}^{n} KL(p||\hat{p}_{j})$

The overall cost function can now be written as

$J_{sparse} = \alpha J(W,b) + \beta \sum\limits_{j=1}^{n} KL(p||\hat{p}_{j})$

## Understanding by Visualization

[[1](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)]Consider the case of 10x10 = 100 pixel input images. A hidden unit is given by

$a_{i} = f( \sum\limits_{j=1}^{100} W_{ij}x_{j} + b_{i} )$

Consider a hidden unit i. What input image,x will cause maximal activation of unit i? Given that the input image is normalized such that $||x||^{2} = \sum\limits_{i=1}^{100} x_{i}^{2} \leq 1$, a pixel j of such an image is given by,

$x_j = \frac{W_{ij}}{\sqrt{\sum\limits_{j=1}^{100} (W_{ij})^2} }$

From the above expression, it is clear that the normalized vectors of images that are parallel to the normalized vectors of hidden unit weights $W_{ij}$ cause them to activate. An image formed by the pixels j will activate unit i to the greatest extent. By visualizing images made of such pixels that maximally activate each of i hidden units, we can see the kind of high level features in the input images that are captured by the hidden layer. 

![Features captured by hidden units](/home/jabroni/_/deeplearning/_notes/primer/svg/autoencoder3.png)



## Keyterms

1. **Overcomplete Representations** : If the size of the intermediate representation is larger than the input dimension.
2. **KL-Divergence** : a measure of how different two distributions are.

## Notes

1. An autoencoder is a combination of encoder and decoder. Input to hidden layer pass encodes the input in a compressed form. Hidden layer to output layer pass decodes it to reproduce an approximation of the original input.

## References

1. [UFLDL : Stanford](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
2. [Shark](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/autoencoders.html)
3. [Chris McCormick](https://chrisjmccormick.wordpress.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/)
