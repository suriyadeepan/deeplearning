# Gradient Checking

Backpropagation is difficult to implement correctly and it often leads to incorrect gradient values. Gradient checking is a numerical technique to make sure the correctness of the implemented backpropagation algorithm. 

$\frac{dJ(\theta)}{d\theta} = \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$

Typically $\epsilon$ is set as $10^{-4}$.

This works well if $\theta \epsilon \mathbb{R}$. But we need a slightly different numerical algorithm if $\theta$ is a vector. The derivative of $J(\theta)$ w.r.t. element $\theta_i$ of $\theta$ can be written as

$\frac{dJ(\theta)}{d\theta_i} = \frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2\epsilon}$

where $\theta^{(i+)} = \theta + \epsilon \times \vec{e_i}$ <br />
$\theta^{(i-)} = \theta - \epsilon \times \vec{e_i}$ <br />
			
and $\vec{e_i}$ is a vector filled with zeros, except at position i it has a value of 1.



## Reference

1. [http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization](http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)