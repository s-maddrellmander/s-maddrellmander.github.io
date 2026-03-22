---
title: "sigmoid: don't overflow, it's embarrassing"
date: 2026-03-21 10:00:00 +0000
categories: [idlemachines, math, practice]
tags: [sigmoid, activation functions, math]
math: true
---

We all know the sigmoid function. It's almost the canonical activation function. The first one you learn in a deep learning 101 course. It's a simple function that takes any real-valued input and squashes it to a value between 0 and 1 as

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

and we often pretend this gives us a probability, or at least something pseudo-probabilistic.

So why are so many people implementing it wrong? 


It's easy to translate this into numpy
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
and we can plot it to see what it looks like:
![Sigmoid function plot](/assets/img/sigmoid-plot.png)

So far so simple. I'm sure lots of you can see where I'm going here, any time we see a function with an exponential we have to think what happens with extreme inputs? 

Neural networks have loads of parameters, and even when most of them are well behaved there are often some which grow and compound. And when they do exponentials start overflowing, and we get NaNs.

If we have a nice large precision, like float64 we can get away with murder here, even values as high as `x=-700` won't overflow, but if instead we try float 32, or float16 suddenly our range is limited to `x=-89` or `x=-11`. 
And as modern frameworks get more optimized for lower precision, this becomes a real concern.

Obviously in the real world this is largely abstracted away, but it's important to understand as it impacts the way weights grow and how gradients flow. 

How do we fix it? We start by seeing there are two cases. X>= 0 and x < 0. When x>=0 because the sign on the exponent is negative we ultiamtely only ever tend towards 0, and because 

$$
\frac{1}{1 + e^{-x}} \sim \frac{1}{1 + 0} = 1
$$
positive values are numerically safe. 

Negative values are more complex. 


$$
\frac{1}{1 + e^{-x}} \sim \frac{1}{1 + \infty} = \mathrm{NaN}
$$

Or in some cases actually even worse, these can surface as silent errors, and clip to 0. Sometimes this is a last resort on large and unstable training runs, but generally this is a sign of something going very wrong, and we want to fix it.

Fortunately here the fix is easy, we can use the fact that anything multiplied by 1 is itself, and rewrite the function as:
$$
\sigma(x) = \frac{1}{1 + e^{-x}} \cdot \frac{e^x}{e^x} = \frac{e^x}{e^x + 1}
$$

Then if we play the same game we can see that for x < 0, the exponent is negative, and we only ever tend towards 0, so we can be sure to avoid overflow.
We can then combine these two cases into a single function a little more formally like this:
$$
\sigma(x) = \begin{cases}
\frac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\
\frac{e^x}{e^x + 1} & \text{if } x < 0
\end{cases}
$$
and express this as a simple numpy function as well:
```python
def stable_sigmoid(x):
   return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), # x >= 0 case
                    np.exp(x) / (np.exp(x) + 1) # x < 0 case
                ) 
```
The `np.where` allows us to vectorise this function - in general if you can try to use where rather than conditionals in your code, it's usually more efficient and less error prone.


## The derivative of sigmoid
Again, the canonical derivative of sigmoid is one of those really nice expressions 
$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$
But it's important to be able to derive this yourself, and understand where it comes from.

We can start with the original definition of sigmoid:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
Then we can use the quotient rule to find the derivative:
$$\frac{d\sigma}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2}$$

Then we can start expressing things in terms of $\sigma(x)$, first pulling out one of the factors in the denominator:
$$\frac{d\sigma}{dx} = \frac{e^{-x}}{(1 + e^{-x})(1 + e^{-x})} = \sigma(x) \cdot \frac{e^{-x}}{1 + e^{-x}}$$

Then we can use a neat little trick, where we add and subtract 1 in the numerator, which allows us to express the second term in terms of $\sigma(x)$ as well:
$$\frac{d\sigma}{dx} = \sigma(x) \cdot \frac{1 + e^{-x} - 1}{1 + e^{-x}} = \sigma(x) \cdot \left(1 - \frac{1}{1 + e^{-x}}\right) = \sigma(x) \cdot (1 - \sigma(x))$$

This last trick feels a bit like cheating every time I do it, but it's a really common technique to manipulate expressions into a more convenient form.

Generally the days of needing to write this out on a whiteboard during an interview are largely over, but it's still really useful to understand where these expressions come from, and be able to derive them yourself.

Something I hope you notice here is that the derivative of the sigmoid function is always relying on the output of the sigmoid function, not the input, but only ever the output. This is a useful feature from an engineering perspective, it means when we compute the forward pass we can store the output directly as a cache and then use it in the backward pass without needing to recompute any exponentials, which is a nice little efficiency boost.

## The backwards pass 

When we are computing the backwards pass of a sigmoid layer in a neural network we obviously don't use the derivative in isolation. We are using the chain rule to compute the gradient with respect to the upstream gradients, probably from a loss function, but in theory anything that is further down the computational graph.
By the chain rule:
$$\frac{dL}{dx} = \frac{dL}{d\sigma} \cdot \frac{d\sigma}{dx}$$

Where $L$ is the loss function, and $\sigma$ is the output of the sigmoid function. 
Then we can substitute in our expression for the derivative of sigmoid:
$$\frac{dL}{dx} = \frac{dL}{d\sigma} \cdot\sigma(x)\cdot(1 - \sigma(x))$$

(We will save a detailed discussion of the chain rule in backpropagation for another time, I've got some thoughts about how to think about the Jacobian)

But the thing that's really worth thinking about at this stage is what does this expression tell us about the flow of gradients through a sigmoid layer? The activation function is an elementwise function, meaning each input is transformed independently. And so are the gradients. This means that the gradient with respect to each input is only dependent on the output of the sigmoid function for that input, and the upstream gradient for that input. From a practical point of view this means the whole backwards pass is reduced simply down to an elementwise multiplication, which is really efficient to compute.

We can put this all together in a nice simple readable form like this:
```python
import numpy as np

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), # x >= 0 case
                    np.exp(x) / (np.exp(x) + 1) # x < 0 case
                ) 

def sigmoid_backward(dL_dsigma, sigma_x):
    # Where sigma_x is the cached output of the forward pass
    return dL_dsigma * sigma_x * (1 - sigma_x)

# Sanity check
x = np.array([-1000, -10, 0, 10, 1000])
sigma_x = sigmoid(x)
dL_dsigma = np.array([1, 1, 1, 1, 1]) # Upstream gradient of 1 for simplicity
dL_dx = sigmoid_backward(dL_dsigma, sigma_x)
print(dL_dx) # Should be close to 0 for extreme values, and around 0.25 for x=0
```

The exteme values will return exactly 0.

## Why care? 
It's a fair question, and if you got this far I hope you've at least got an intuition as to why. But fundamentally the problem in a large part of modern machine learning isn't about the math, it's about the engineering. And understanding these kinds of numerical stability issues is really important to be able to build and train large models that don't just blow up in your face.

Most numerical stability issues get fixed by casting to float32. That's papering over the problem. Understanding why it breaks means you fix it once, correctly, rather than chasing NaNs through a training run at 3am. If we think about activation values as drawn from a distribution, all we've done is make it less likely we get values at the tail which case overflow, but the code still isn't overflow safe, if we think about what exponentals do, and how to protect against these issues, we can avoid the problem entirely.  


If you want to try and work through this yourself, have a look at some of the problems (including this one) over on [idlemachines.co.uk](https://idlemachines.co.uk) and see if you can work through the solutions yourself.
<!-- https://idlemachines.co.uk/questions/sigmoid-stable -->
<!-- Could use this direct link instead? -->