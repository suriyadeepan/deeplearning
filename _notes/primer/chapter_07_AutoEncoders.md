# Autoencoders

Given inputs {$x^{1},x^{2},x^{3},...x^{i},...x^{m}$}, an autoencoder network sets the target value $y^{i} = x^{i}$, using backpropagation. It tries to learn the function $h_{W,b} \approx x$.

Consider a network that takes 10x10 = 100 pixel images as input. Here $x \epsilon R^{100}$ and $y \epsilon R^{100}$, let the hidden layer contain 50 nodes (i.e) the vector of hidden layer activation $a \epsilon R^{50}$. In this case, the hidden layer forms a compressed representation of the input, from which it reproduces an approximate form of the input, in the output layer. This network discovers the correlations between differnt features in the input. A low dimensional representation of the input is learnt by this network, similar to PCA. [[1](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)]

To summarize, propagating from input layer to hidden layer is the compression step, while propagating from hidden layer to output layer is the decompression step. [[2](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/autoencoders.html)]


### Motivation

[[4](http://deeplearning.net/tutorial/dA.html)]Without any constraints on the hidden layer ( number of hidden layer units is equal to input dimensions), an autoencoder just learns an identity function that maps input to itself, which is pretty useless. But when experimented with more hidden units than in the input, the hidden layer captures the features in the input and provides a useful representation of the input. *Useful* in the sense, this representation (encoding) can yield better classification error. 

For achieving a good reconstruction, the encoding layer requires small weights to bring the non-linear units into the linear regime. To minimize the reconstruction error, the decoding layer requires large weights. If we use regularization (weight decay) to keep the weights in check(small), the encodings produced capture the statistical regularities in the training set, rather than just serving as an identity function. 


This can also be achieved by putting a constraint on the number of hidden units or introducing a sparsity constraint in the cost function, which are discussed in detail below. 


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

### Understanding by Visualization

[[1](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)]Consider the case of 10x10 = 100 pixel input images. A hidden unit is given by

$a_{i} = f( \sum\limits_{j=1}^{100} W_{ij}x_{j} + b_{i} )$

Consider a hidden unit i. What input image,x will cause maximal activation of unit i? Given that the input image is normalized such that $||x||^{2} = \sum\limits_{i=1}^{100} x_{i}^{2} \leq 1$, a pixel j of such an image is given by,

$x_j = \frac{W_{ij}}{\sqrt{\sum\limits_{j=1}^{100} (W_{ij})^2} }$

From the above expression, it is clear that the normalized vectors of images that are parallel to the normalized vectors of hidden unit weights $W_{ij}$ cause them to activate. An image formed by the pixels j will activate unit i to the greatest extent. By visualizing images made of such pixels that maximally activate each of i hidden units, we can see the kind of high level features in the input images that are captured by the hidden layer. 

![Features captured by hidden units](/home/jabroni/_/deeplearning/_notes/primer/svg/autoencoder3.png)


### Reconstruction

The hidden layer, $y = a(Wx + b)$ is also known as the latent representation or **code**. The code is mapped back to reconstruction **z**, of the same shape as **x**. 

$z = a(W'y + b')$

*tied weights* : Optionally, the weight matrix $W'$ can be tied together with W, with a constraint $W' = W^T$. 

To measure the reconstruction error, we can used the squared error $||z-x||^2$. 


### Denoising Autoencoders

>  *reconstruct the input from a corrupted version of it*

Denoising autoencoder is a combination of an autoencoder that encodes and reconstructs an input and a stochastic corruptions process. The stochastic corruption process basically corrupts the input, by removing some components from the input randomly. So, the denoising autoencoder tries to predict the corrupted values from the uncorrupted values. Being able to do that is sufficient condition for capturing the joint probability distribution between a set of variables. Look at it this way. By understanding the probabilistic connections between the features in the input from the uncorrupted inputs, the network is able to fill in the blanks in the corrupted inputs, and thus reconstruct them. 

A separate stochastic corruption mechanism is added, which basically masks some entries of the input randomly and sets them to zero. 


## Keyterms

1. **Overcomplete Representations** : If the size of the intermediate representation is larger than the input dimension.
2. **KL-Divergence** : a measure of how different two distributions are.
3. **Tied Weights** : $W' = W^T$

## Notes

1. An autoencoder is a combination of encoder and decoder. Input to hidden layer pass encodes the input in a compressed form. Hidden layer to output layer pass decodes it to reproduce an approximation of the original input.

2. **PCA and autoencoders** : The hidden units capture the first k (number of hidden units) principle components of the input. This is a lossy compression. Reconstruction of the original input is done using these k principle components. Like PCA, autoencoders when used with linear hidden units, try to provide a projection of high dimensional data, onto a low dimensional space. But if non-linear hidden units are used, the hidden layer captures multi-modal aspects of the data (*needs further investigation*).

3. **Stacked Autoencoder** : A deep network can be formed by stacking autoencoders together. This is done by feeding the latent representation of an autoencoder in the layer below, as input to the autoencoder above. Layer by layer greedy pretraining is done. This deep network is used as a feedforward network for classification tasks, by finetuning the network using backpropagation similar to a normal feedforward MLP.

## References

1. [UFLDL : Stanford](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)
2. [Shark](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/autoencoders.html)
3. [Chris McCormick](https://chrisjmccormick.wordpress.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/)
4. [deeplearning.net](http://deeplearning.net/tutorial/dA.html)