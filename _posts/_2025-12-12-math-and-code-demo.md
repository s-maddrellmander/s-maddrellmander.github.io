---
title: "Math and Code Demo"
date: 2025-12-12 12:30:00 +0000
categories: [Demo, Technical]
tags: [math, code, demo]
math: true
---

## Mathematics Rendering

The site supports beautiful LaTeX math rendering via MathJax.

### Inline Math

Consider the loss function $$\mathcal{L}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f_\theta(x), y)]$$ where $\theta$ represents our model parameters.

### Display Math

The gradient descent update rule:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

For a momentum-based optimizer:

$$
\begin{align}
m_{t+1} &= \beta m_t + (1-\beta) \nabla_\theta \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta m_{t+1}
\end{align}
$$

### More Complex Equations

The Bellman optimality equation for Q-learning:

$$
Q^*(s,a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\left[r(s,a) + \gamma \max_{a'} Q^*(s', a')\right]
$$

## Code Blocks

Syntax highlighting works beautifully with the warm paper theme.

### Python

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training loop
model = SimpleNet(784, 256, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
```

### Shell Commands

```bash
# Set up the environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib

# Run training
python train.py --config configs/default.yaml
```

### YAML Config

```yaml
model:
  type: transformer
  d_model: 512
  n_heads: 8
  n_layers: 6
  
training:
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 1000
  max_steps: 100000
  
data:
  train_path: data/train.jsonl
  val_path: data/val.jsonl
  max_seq_len: 512
```

## Tables

| Method | Accuracy | Training Time | GPU Memory |
|--------|----------|---------------|------------|
| FP32   | 94.2%    | 8.5h         | 24GB       |
| FP16   | 94.1%    | 4.2h         | 12GB       |
| FP8    | 93.8%    | 2.1h         | 8GB        |

## Blockquotes

> **Key Insight**: Lower precision training isn't just about speed—it fundamentally changes the optimization landscape. The quantization noise can act as implicit regularization.

## Lists

Key considerations for FP8 training:

1. **Loss Scaling**: Dynamic loss scaling becomes critical
2. **Gradient Accumulation**: More steps needed for stability
3. **Layer-wise Precision**: Not all layers need the same precision
4. **Hardware Support**: Requires modern GPUs (H100, Blackwell)

Technical benefits:

- 2-4x throughput improvement
- Reduced memory footprint
- Minimal accuracy degradation (when done right)
- Better energy efficiency

---

This demonstrates the site's capability for technical writing with math, code, and structured content.
