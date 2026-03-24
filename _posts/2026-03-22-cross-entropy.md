---
title: "cross-entropy: the simplest gradient in all of machine learning"
date: 2026-03-21 10:00:00 +0000
categories: [idlemachines, math, practice]
tags: [softmax, activation functions, math]
math: true
---

Cross-entropy is one of the most fundamental and widely used loss functions in all of machine learning, and the gradient is so beautifully simple. And basically nobody knows how to actually derive it on pen and paper. *(I'll put an exercise at the end if you just want to try it yourself first, but if not let's get into it).*

![Cross-entropy loss curve](/assets/img/cross_entropy_loss_curve.png)
*Loss spikes exponentially when you're wrong, but the gradient responds linearly — a counterintuitive feature that keeps learning stable even when losses blow up.*

## From probabilities to likelihoods
As a fair warning this is going to need a bit of maths, if that's not your thing, feel free to skim down to the code and it should make sense. But if you want to follow along, let's start with the basics.

We have a model, and for our sins, we say it outputs a set of probabilities
$$
p(x) = \mathbb{F}(x_i)
$$
and we know there is some true label $y_i$, which we'll say is a one-hot vector. 
We can define some loss as 
$$
\mathcal{L} = p(y)
$$
which is just the probability of the true label. This is a perfectly good loss function, and in general we like to take logs of probabilities, so we can rewrite this as
$$
\mathcal{L} = \log p(y)
$$
then flip the sign 
$$
\mathcal{L} = -\log p(y)
$$
Cross-entropy is negative log-likelihood in this setting — and that's exactly why it works. It's the negative log likelihood of the true label under the model's predicted distribution.
Or in other words, it's how unlikely the true label is according to the model's predictions. The lower the cross-entropy, the better the model's predictions align with the true labels.

## General form
Moving from a single label to a vector we get 
$$
\mathcal{L} = -\sum_{i} y_i \log p_i
$$
For one-hot labels, this picks out the log probability of the correct class. Here's the key insight: because $y_i = 0$ for all incorrect classes, the sum collapses to a single term corresponding to the true class. For soft labels, this becomes a weighted average of the log probabilities, where the weights are the true label distribution.

## Interpretation
What does the cross-entropy actually mean? A fairly common way of describing it is as how "surprised" the model is by the true label. If the model assigns a high probability to the true label, the cross-entropy will be low, indicating that the model is not surprised. Conversely, if the model assigns a low probability to the true label, the cross-entropy will be high, indicating that the model is very surprised.

This is fine, but to be honest I don't really like the framing. Personally I prefer to think of it as a measure of how well the model's predicted distribution matches the true label distribution. If the model's predictions are close to the true labels, the cross-entropy will be low. If the model's predictions are far from the true labels, the cross-entropy will be high.

This perspective is much more aligned to classical statistics like we used at CERN to understand the discrepancy between our data and our models. It's not about surprise, it's about how well the model's predicted distribution matches the true label distribution. Seeing cross-entropy in this light makes it much clearer why the loss function behaves as it does, surprise feels ephemeral, but the idea of matching distributions is much more concrete and intuitive. (in my opinion, but maybe you disagree, and I'd like to hear it).

## Connection to KL divergence

Cross-entropy has a useful decomposition:
$$
H(p, q) = H(p) + D_{KL}(p || q)
$$
where $H(p)$ is the entropy of the true distribution and $D_{KL}(p||q)$ is the KL divergence between the true and predicted distributions.

For classification with one-hot labels, $H(p) = 0$ (the true distribution has no entropy), so minimizing cross-entropy is exactly equivalent to minimizing KL divergence. More importantly, since $H(p)$ is constant during training, the gradients of cross-entropy and KL divergence are identical — this is why you'll see people use these terms interchangeably in classification contexts.

## Numerical stability
In practice, we almost never compute cross-entropy from probabilities directly. 

Instead, we work with logits and use the form:
$$
\mathcal{L} = -\sum_{i} y_i x_i + \log \sum_{j} e^{x_j}
$$
This avoids ever taking the log of very small numbers and uses the same stability tricks we'll see below (shifting by the max). 

This is why most frameworks implement softmax and cross-entropy as a single fused operation — it's both more stable and more efficient.

# Softmax meets cross-entropy
When we combine softmax with cross-entropy, we get a very nice simplification. The softmax function transforms the logits into probabilities, and the cross-entropy loss then measures how well these probabilities match the true labels. 

If we start with 
$$
p_i = \mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$
then we can plug this into the cross-entropy loss:
$$
\mathcal{L} = -\sum_{i} y_i \log \left( \frac{e^{x_i}}{\sum_{j} e^{x_j}} \right)
$$
This can be rewritten as:
$$
\mathcal{L} = -\sum_{i} y_i (x_i - \log \sum_{j} e^{x_j})
$$
which simplifies to:
$$
\mathcal{L} = -\sum_{i} y_i x_i + \log \sum_{j} e^{x_j}
$$
This is the cross-entropy loss expressed in terms of the logits. The first term is just weighting the logits by the true labels, and the second term is a normalization term that ensures the loss is properly scaled.

For the first term, when we have a one-hot label, this just picks out the logit corresponding to the true class. For soft labels, this is a weighted sum of the logits, where the weights are the true label distribution. This term increases the logit of the correct class, while the log-sum-exp term prevents all logits from growing arbitrarily. 


### log-sum-exp trick
This is one of the most important tricks to know off the top of your head when working with loss functions, they are full of logs and exponentials, and the log-sum-exp trick is an incredibly simple way to compute these in a numerically stable way. The idea is to factor out the maximum logit from the sum of exponentials, which prevents overflow and underflow issues.
$$\log \sum_{j} e^{x_j} = \log \sum_{j} e^{x_j - \max(x)} + \max(x)$$
This way, we are always working with numbers that are less than or equal to zero, which prevents overflow when we take the exponential.

## Deriving the gradient
Now for the fun part, let's derive the gradient of the cross-entropy loss with respect to the logits. Hopefully you're still with me, because this is one of the simplest and most elegant gradients in all of machine learning. 

First, let's write down the loss again:
$$\mathcal{L} = -\sum_{i} y_i x_i + \log \sum_{j} e^{x_j}$$
Now we can take the derivative with respect to $x_k$:
$$\frac{d\mathcal{L}}{dx_k} = -y_k + \frac{e^{x_k}}{\sum_{j} e^{x_j}}$$
note that the first term picks out just the $k$-th term from the sum, and the second term is the derivative of the log-sum-exp function. This can be rewritten as:

$$\frac{d\mathcal{L}}{dx_k} = \mathrm{softmax}(x_k) - y_k$$
*quick aside on where that comes from*
When we take the derivative of the log-sum-exp function, we get:
$$\frac{d}{dx_k} \log \sum_{j} e^{x_j} = \frac{1}{\sum_{j} e^{x_j}} \cdot \frac{d}{dx_k} \sum_{j} e^{x_j}$$
The derivative of the sum of exponentials is just the exponential of the $k$-th term, so we get:
$$\frac{d}{dx_k} \log \sum_{j} e^{x_j} = \frac{e^{x_k}}{\sum_{j} e^{x_j}}$$
which is exactly the softmax of $x_k$.

Then we can put this all together to get the final expression for the gradient:
$$\frac{d\mathcal{L}}{dx_k} = \mathrm{softmax}(x_k) - y_k$$
This is a beautifully simple expression. Suggestively if we rearrange it a bit we can write it as:
$$\frac{d\mathcal{L}}{dx_k} = p_k - y_k$$
where $p_k$ is the predicted probability for class $k$. This means the gradient is just the difference between the predicted probability and the true label. If the model is predicting the true label with high probability, the gradient will be small, and if the model is predicting the true label with low probability, the gradient will be large. This makes intuitive sense, and it's one of the absolutely canonical gradients in all of machine learning. 

Doing this on a whiteboard in front of people and knowing all these terms will cancel as you point out it's just the probabilities, it's such a good feeling. And I hope this gives you some confidence you could do this too. 


More seriously this is a really important gradient to understand, it makes it clear how the model is learning to adjust its predictions based on the true labels, and how simple the signal that feeds into a model can be. It's just the difference between what the model thinks is true and what is actually true, and this simple signal is what drives the learning process in so many models.

## Why this works so well

This pairing — softmax with cross-entropy — works unusually well, and that's not an accident.

First, the gradient is extremely well behaved. It doesn't involve exponentials or logs — it's just a difference between probabilities. That means it's naturally well-scaled and avoids the saturation issues you get with other combinations like sigmoid + MSE. With sigmoid + MSE, when the prediction is very wrong, the sigmoid saturates and the gradient becomes tiny, slowing down learning exactly when you need it most. Here, when $p_k \to 0$ for the true class, the gradient is $p_k - 1 \to -1$ — bounded, but still providing clear signal.

This is genuinely counterintuitive: the loss grows exponentially (that $-\log(p)$ term explodes as $p \to 0$), but the gradient responds *linearly*. When your model assigns probability 0.01 to the true class, the loss is 4.6, but the gradient is just -0.99. When it assigns 0.5, the loss is 0.69 and the gradient is -0.5. The gradient tracks the error, not the panic. This is the entire reason we can use large learning rates without everything exploding.

Second, while the gradient is bounded, it still gives a very strong signal when the model is wrong. The loss itself screams at you (going to infinity), making it easy to monitor training, while the gradient stays composed and pushes steadily in the right direction.

This combination — stable gradients, strong error signal, and a clean probabilistic interpretation — is why softmax + cross-entropy shows up almost everywhere.

There are a few other things worth noting. First, this is only possible because we combined the softmax and cross-entropy together. If we had a different activation function, or a different loss function, we wouldn't get this nice simplification. This is one of the reasons why softmax and cross-entropy are so commonly used together in classification problems. Second, this gradient is very interpretable, it directly tells us how the model's predictions differ from the true labels, which can be really helpful for debugging and understanding the learning process.

Walk through one example with me. Imagine we have logits $x = [2.0, 3.5, 1.5]$ for a 3-class classification problem, and the true label is class 1 (which corresponds to the second entry in the vector, so $y = [0, 1, 0]$). 

First, we compute the softmax to get probabilities:
$$p = \mathrm{softmax}(x) \approx [0.1, 0.7, 0.2]$$
The cross-entropy loss would be:
$$\mathcal{L} = -\sum_{i} y_i \log p_i = -\log 0.7 \approx 0.357$$
Now we can compute the gradient with respect to the logits:
$$\frac{d\mathcal{L}}{dx} = p - y = [0.1, 0.7, 0.2] - [0, 1, 0] = [0.1, -0.3, 0.2]$$
This means that the model will adjust its logits in the direction of increasing the probability of the true class (class 1) and decreasing the probabilities of the other classes. The magnitude of the adjustments is proportional to how far off the model's predictions are from the true labels, which is exactly what we want in a learning algorithm.


## Let's implement this
Let's implement this properly, using the stable logits form we derived. For a single example:

```python
import numpy as np

def softmax(x):
    """Numerically stable softmax."""
    exp_shifted = np.exp(x - np.max(x))
    return exp_shifted / np.sum(exp_shifted)

def cross_entropy_forward(logits, y):
    """Compute cross-entropy from logits directly (stable)."""
    # Using the form: L = -sum(y * logits) + log(sum(exp(logits)))
    max_logit = np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(logits - max_logit))) + max_logit
    return -np.sum(y * logits) + log_sum_exp

def cross_entropy_backward(logits, y):
    """Gradient is just softmax(logits) - y."""
    return softmax(logits) - y
```

For batched inputs with shape `(batch_size, num_classes)`, we need to handle the reduction:
```python
def cross_entropy_forward_batched(logits, y):
    """Batched cross-entropy. Returns mean loss over batch."""
    batch_size = logits.shape[0]
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - max_logits), axis=1)) + max_logits.squeeze()
    loss_per_sample = -np.sum(y * logits, axis=1) + log_sum_exp
    return np.mean(loss_per_sample)  # Mean over batch

def cross_entropy_backward_batched(logits, y):
    """Batched gradient."""
    batch_size = logits.shape[0]
    probs = softmax(logits)  # Apply along correct axis
    return (probs - y) / batch_size  # Scale by batch size for mean reduction
```
The key implementation details:

1. **Work with logits directly** — never compute probabilities just to take their log
2. **Use log-sum-exp trick** with max subtraction for numerical stability
3. **Batch reduction matters** — use mean (not sum) over batch so gradient scale doesn't depend on batch size
4. **The backward pass is identical** whether we use sum or mean reduction, we just need to scale by batch size

The simplicity of the backward pass is what makes this so efficient. No matter how complex the forward computation looks, the gradient is always just $p - y$.


*If you want to work through this properly — stable forward pass, backward pass, and gradient checks — I've put it up as an exercise here: [idlemachines.co.uk/questions/cross-entropy-forward-backward](https://idlemachines.co.uk/questions/cross-entropy-forward-backward)*