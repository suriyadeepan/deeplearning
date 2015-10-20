# Richard Zemel's Machine Learning and Data Mining


## Introduction

* Prerequisites : Sheldon Ross's A First Course in Probability
* Textbooks
	- Christopher Bishop's *Pattern Recognition and Machine Learning*
	- *Introduction to Machine Learning* by Ethem Alpaydin
	- Machine Learning : A Probabilistic Perspective by Kevin Murphy
	- Information Theory, Inference and Learning Algorithms by David Mackay

## Regression

* latent variables
* Update rule : w <- w + (dJ/dw).lambda
* Stochastic (SGD) vs Batch updates
* Non-iterative Least Squares Regression?
* Weight Decay in ML is *analogous* to Ridge Regression in Statistics

## Linear Classification

* Decision boundary
* Hyper plane : w0 + w.x = 0 
* In 2D, hyperplane (line : w0 + w.x = 0) is orthogonal to w
* w : direction; w0 : location
* Zero-one loss L0-1
* Asymmetric binary loss
* Precision, Recall, F1 score

## Probabilistic Classification

* It is difficult to optimize the loss function of sign( w0 + wx )
* Use of logistic function (smoother function : why relevant?)
* 1/(1 + e^-z) => gradient based learning is now possible
* Liberal use of the term *posterior*
* Posterier p(C=0|x) = sigmoid(wx + w0)
* p(C=1|x) = 1 - p(C=0|x)
* Training examples are sampled *I.I.D* : Independent, Identically Distributed
* Binomial distribution 
* Learning model by maximizing conditional likelihood 
* Find w such that summationOf(p(t|x))_over_N_examples is maximum
* Loss function : negative log-likelihood of the function above : Minimization problem

## K-Nearest Neighbours

* Classification is intrinsically non-linear
* Linear Classification : the part that adapts is linear : z(x) = w.x + w0
* Instance-based learning : Learning => simply storing training data
* Test instances classified using *similar* training instances
* Each training example is a point in d-dimensional space
* Find 'k' examples closest to test instance x
* Voronoi diagram visualization?
* "remove examples that lie within the Voronoi region"?
* kd-tree? search tree?
* KNN is a non-parametric model?

## Decision Trees

* Decision boundary as composition of several simple boundaries
* View decision tree as recursive nested if-else conditions
* Nodes : test attributes(conditions) ; Branch : attribute value; Leaf : output
* Information theory for constructing Decision Tree
* Shannon Entropy : "how much you can compress your data" ( H(X) )
