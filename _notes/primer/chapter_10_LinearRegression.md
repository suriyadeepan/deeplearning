*I asked myself several times. Why am I writing this? Its ridiculous. It is just linear regression. But everything starts with a "hello world" right? Plus, this article would serve as an intro to numpy and theano.*

# Linear Regression

In linear regression, we basically fit a model for given data-target pairs. The data $x \epsilon \Re^d$ is a vector of features. Target $y \epsilon \Re$ is a scalar value, which is the correct target value. Consider the example of housing, where the features are dimensions of the house, number of rooms,etc, and the target value is the price of the house. 

We have the data and target, we now need to find the model that will map the data to the target. We define this model with an objective function $y = h(\theta,x)$, where $\theta$ is the parameters of the models, which actually define the function. To estimate the best model is to figure out the best values of $\theta$ that makes the model the best fit for given data. 

Objective function,
$h(\theta,x) = \sum\limits_j \theta_jx_j$

In matrix form,
$h(\theta,x) = \theta^T.x$

$y = h(x)$ gives the predicted value. This predicted value should ideally be equal(or close) to the original target value. This difference between the predicted and the actual target value gives an idea about the performance of our parameters($\theta$). This is measured using the cost function $J(\theta)$, given by

$J(\theta) = \frac{1}{2} \sum \limits_i ( h_{\theta}(x^{(i)}) - y^{(i)} )^2$

where *i* is the index of the input data example $(x^{(i)},y^{(i)})$.

Now the goal is the minimize the cost function by adjusting the values of $\theta$. The $\theta$ that gives the lowest cost value, is the best fit for the given data. 


