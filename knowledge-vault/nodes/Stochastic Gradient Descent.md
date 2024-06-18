> 2024-04-30

#sprout


---
**links**: 
**brain-dump**: 

---

**Stochastic Gradient Descent (SGD)** is an optimization algorithm commonly used in machine learning for minimizing the loss function. It's an extension of the gradient descent algorithm that deals with the problem of minimizing the objective function that has a gradient but can be noisy or computationally expensive.

![[Pasted image 20231011133640.png]]

### What is Gradient Descent?

Gradient Descent is an optimization algorithm used to minimize a function. It works by iteratively moving towards the minimum of the function based on the gradient.

### Stochastic Gradient Descent vs Batch Gradient Descent

In Batch Gradient Descent, the update to the model parameters is done after computing the gradient of the error with respect to the entire training set. This can be computationally expensive for large datasets. 

Stochastic Gradient Descent (SGD) solves this problem by updating the model parameters based on each data point, making it much faster and capable of handling large datasets.

---

## Mathematical Foundations

### Update Rule

The update rule for SGD in its simplest form is given by:

$$
\theta = \theta - \alpha \nabla_\theta J(\theta; x^{(i)}, y^{(i)})
$$

Where:
- $( \theta )$ are the model parameters
- $( \alpha )$ is the learning rate
- $( \nabla_\theta J(theta; x^{(i)}, y^{(i)}) )$ is the gradient of the objective function $( J ) at $( \theta )$, considering the $(i^{th})$ data point $( (x^{(i)}, y^{(i)}) )$

### Learning Rate

The learning rate \( \alpha \) controls the step size during the optimization and is a crucial hyper-parameter. Too large of a learning rate can lead to oscillation, while too small can make the optimization very slow.

---

## Practical Considerations

### Learning Rate Scheduling

Over time, it's often beneficial to decrease the learning rate. Several schedules like step decay, exponential decay, or using a learning rate finder can be effective. 

> See [[Learning Rate Scheduling]] for more 

### Momentum

SGD can be slow or can oscillate. The momentum term helps to overcome these issues by adding a fraction of the previous update to the current one:

$$
v = \beta v - \alpha \nabla_\theta J(\theta)
$$
$$
\theta = \theta + v
$$

Where $( \beta )$ is the momentum term.

### Variants

There are several variants of SGD like RMSprop, Adam, Adagrad that adapt the learning rates during training and combine the benefits of both SGD and momentum.

---

## Advantages and Disadvantages

### Advantages

1. **Efficiency**: Suitable for large-scale data.
2. **Ease of implementation**: The algorithm is straightforward to implement.

### Disadvantages

1. **Hyperparameter Sensitivity**: Requires careful selection of $( \alpha )$ and sometimes other hyper-parameters like momentum.
2. **Noise**: The updates can be noisy due to the stochastic nature, which sometimes is an advantage for non-convex functions but can also lead to suboptimal convergence.

---

## Summary

Stochastic Gradient Descent is a cornerstone optimization algorithm in machine learning. Its stochastic nature makes it suitable for large datasets and complex optimization problems, but care must be taken in hyperparameter tuning and handling its stochastic nature.

By understanding the intricacies of SGD, you are better equipped to tackle optimization challenges in machine learning.



