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


## Probabilistic Classifiers


### Bayes Classifier

* Simple Explanation given [here](http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification)

* **Conditional Probability** : What is the probability of event E occurring given some other event D has already happened?
	- Let's say that there is some Outcome O. And some Evidence E. From the way these probabilities are defined: The Probability of having both the Outcome O and Evidence E is: (Probability of O occurring) multiplied by the (Prob of E given that O happened)
	- Evidence and Outcome

* **Naive Bayes** : we have to predict an outcome given multiple evidence. In that case, the math gets very complicated. To get around that complication, one approach is to 'uncouple' multiple pieces of evidence, and to treat each of piece of evidence as independent. This approach is why this is called naive Bayes.

> P(Outcome/Multiple Evidence) = P(Evidence1/Outcome) x P(Evidence2/outcome) x ... x P(EvidenceN/outcome) x P(Outcome) scaled by P(Multiple Evidence)

> P(outcome/evidence) = ( P(Likelihood of Evidence) x Prior prob of outcome ) / P(Evidence)
                    			
* Naive Bayes Classifier ultimately *reduces* to

P(Banana/evidence) = 1/z * Prob(Banana) x Prob(Evidence1/Banana).Prob(Evidence2/Banana)...

P(Orange/Evidence) = 1/z * Prob(Orange) x Prob(Evidence1/Orange).Prob(Evidence2/Orange)...

P(Other Fruit/Evidence) = 1/z * Prob(Other) x Prob(Evidence1/Other).Prob(Evidence2/Other)...

### MLE for Gaussian

* MLE is a parametric model, where you estimate parameters of a distribution, which maximizes the probability of the observation of data X

* Laplace smoothing?

## Neural Network

* Linear classifier : linear combination of input features
* Use of cross entropy error function for Binary classification, with sigmoid as activation function
* For multi-class classification, we use softmax activation 

* **Replicated Feature Approach** : use many different copies of the same feature detector, in slightly different positions ( Replicated pool of detectors )

* Adapt back-propagation for replicated weights 
	- if w1 = w2
	- 	gradient of w1 = (dE/dw1) + (dE/dw2)

* Incorporate knowledge of invariances (in the network) by creating more training examples ( by artificially applying invariances to the training set ). This works very well

