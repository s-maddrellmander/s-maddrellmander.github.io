---
title: "FP8/FP4 Training on Blackwell: Initial Experiments"
date: 2025-12-18 14:00:00 +0000
categories: [Projects, Low-Precision Training]
tags: [fp8, fp4, gpu, optimization, blackwell]
math: true
published: false
---

## Overview

Exploring ultra-low precision (FP8/FP4) training on NVIDIA's Blackwell architecture—infrastructure setup, pitfalls, and early results.

---

## Motivation

**Why FP8/FP4?**

- **Throughput**: 2-4x faster matrix operations vs. FP16
- **Memory**: Train larger models on same hardware
- **Energy**: Reduced power consumption
- **Theory**: Quantization noise as implicit regularization

**Challenges**:

- Numerical stability
- Gradient underflow
- Layer-wise precision requirements
- Hardware/software maturity

---

## Infrastructure

### Hardware Setup

- **GPU**: NVIDIA Blackwell (simulated via H100 for now)
- **Memory**: 80GB HBM3
- **Interconnect**: NVLink 4.0 for multi-GPU

### Software Stack

```bash
# Environment setup
conda create -n fp8_experiments python=3.11
conda activate fp8_experiments

# PyTorch with FP8 support
pip install torch==2.5.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformer-engine  # NVIDIA's FP8 library

# Monitoring
pip install wandb tensorboard nvidia-smi
```

### Code Infrastructure

```python
import torch
from transformer_engine.pytorch import Linear, LayerNormLinear
from transformer_engine.common import recipe

class FP8Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # FP8-aware layers
        self.layers = nn.ModuleList([
            LayerNormLinear(config.d_model, config.d_ff)
            for _ in range(config.n_layers)
        ])
        
    def forward(self, x, fp8_recipe):
        with recipe.DelayedScaling(
            fp8_format=recipe.Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max"
        ):
            for layer in self.layers:
                x = layer(x)
        return x
```

---

## Key Challenges

### 1. Loss Scaling

FP8 has limited dynamic range → gradients can underflow.

**Solution**: Adaptive loss scaling

$$
\text{loss}_{\text{scaled}} = \text{loss} \times 2^s, \quad s \in [0, 32]
$$

Update $s$ based on gradient magnitudes:

```python
class AdaptiveLossScaler:
    def __init__(self, init_scale=2**10, growth_rate=2.0):
        self.scale = init_scale
        self.growth_rate = growth_rate
        
    def update(self, grads_finite):
        if grads_finite:
            self.scale *= self.growth_rate  # Increase
        else:
            self.scale /= 2.0  # Decrease on overflow
```

### 2. Layer-wise Precision

Not all layers need FP8:

- **Embeddings**: Keep in FP16 (small, critical)
- **Attention**: FP8 works well
- **Layer norms**: FP16 for stability
- **Output logits**: FP16 to preserve precision

```python
# Hybrid precision config
precision_map = {
    "embed": torch.float16,
    "attn": torch.float8_e4m3fn,
    "mlp": torch.float8_e4m3fn,
    "norm": torch.float16,
    "output": torch.float16
}
```

### 3. Gradient Accumulation

Smaller precision → need more steps for stability:

```python
accumulation_steps = 8  # vs. 4 for FP16

for step, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Experimental Setup

### Baseline Model

- **Architecture**: Transformer (GPT-style)
- **Size**: 350M parameters
- **Dataset**: OpenWebText (10B tokens)
- **Hardware**: 8x H100 GPUs

### Configurations to Compare

| Config | Precision | Batch Size | Throughput | Memory |
|--------|-----------|------------|------------|--------|
| FP32   | Full      | 32         | 100%       | 80GB   |
| FP16   | Mixed     | 64         | 180%       | 48GB   |
| FP8    | Mixed     | 128        | 320%       | 32GB   |
| FP4    | Extreme   | 256        | ???        | 20GB   |

---

## Early Results

### Convergence Comparison

_Plots to come: training loss, validation perplexity, wall-clock time_

**Observations so far**:

- FP8 matches FP16 accuracy with proper scaling
- 2.5x throughput improvement on matrix ops
- Gradient noise observable but not catastrophic

### Pitfalls Encountered

1. **Overflow in attention scores**: Softmax inputs too large
   - **Fix**: Pre-scaling before attention
   
2. **Embedding collapse**: FP8 quantization too aggressive
   - **Fix**: Keep embeddings in FP16
   
3. **Batch norm instability**: Running statistics diverge
   - **Fix**: Use LayerNorm instead, or FP16 BN

---

## Metrics to Track

Beyond standard loss/accuracy:

- **Gradient SNR**: $\frac{\|\mathbb{E}[g]\|}{\text{std}(g)}$
- **Activation ranges**: Monitor for overflow/underflow
- **Weight update magnitudes**: Ensure meaningful updates
- **Memory bandwidth**: Profile actual utilization

---

## Next Steps

- [ ] Complete FP8 baseline runs
- [ ] Attempt FP4 experiments (higher risk)
- [ ] Measure real Blackwell performance (when available)
- [ ] Write custom CUDA kernels for edge cases
- [ ] Compare against Google's FP8 work (TPUv5)

---

## Resources

- [NVIDIA Transformer Engine Docs](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [FP8 Formats for Deep Learning](arxiv.org/abs/2209.05433)
- [Blackwell Architecture Whitepaper](nvidia.com/blackwell)

---

_Experiments ongoing. Results and code will be shared as they stabilize._
