#!/usr/bin/env python3
"""
Generate cross-entropy hero image: loss spike vs linear gradient.
Clean visual statement - minimal labels, maximum clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

# Styling to match softmax plot
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Generate data
p = np.linspace(0.01, 0.99, 500)
loss = -np.log(p)  # Exponential spike
gradient_mag = np.abs(p - 1)  # Linear response

# Create side-by-side figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#f8f9fa')

# ============ LEFT: LOSS CURVE (exponential spike) ============
ax1.set_facecolor('#f8f9fa')

# Plot loss with dramatic red gradient
ax1.fill_between(p, 0, loss, color='#c44e52', alpha=0.2)
ax1.plot(p, loss, linewidth=3, color='#8b0000', alpha=0.9)

# Add shadow effect
ax1.plot(p, loss, linewidth=5, color='#c44e52', alpha=0.3, zorder=1)

# Mark key points
key_points = [(0.05, -np.log(0.05)), (0.3, -np.log(0.3)), (0.7, -np.log(0.7)), (0.95, -np.log(0.95))]
for px, lx in key_points:
    ax1.scatter([px], [lx], s=200, c='#8b0000', edgecolors='white', 
               linewidths=3, zorder=5)

# Title only
ax1.set_title('Loss', fontsize=32, weight='bold', pad=20, 
             color='#8b0000', family='monospace')

# Styling - clean and minimal
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 5)
ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color='#999999')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax1.spines[spine].set_color('#cccccc')
    ax1.spines[spine].set_linewidth(1.5)
# Hide tick labels for cleaner look
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.tick_params(length=0)

# ============ RIGHT: GRADIENT (linear, bounded) ============
ax2.set_facecolor('#f8f9fa')

# Plot gradient with blue
ax2.fill_between(p, 0, gradient_mag, color='#4c72b0', alpha=0.2)
ax2.plot(p, gradient_mag, linewidth=3, color='#2c5aa0', alpha=0.9)

# Add shadow effect
ax2.plot(p, gradient_mag, linewidth=5, color='#4c72b0', alpha=0.3, zorder=1)

# Show it's bounded at 1
ax2.axhline(y=1, color='#2c5aa0', linestyle='--', linewidth=2, alpha=0.4)

# Mark same key points
for px, _ in key_points:
    gx = abs(px - 1)
    ax2.scatter([px], [gx], s=200, c='#2c5aa0', edgecolors='white', 
               linewidths=3, zorder=5)

# Title only
ax2.set_title('Gradient', fontsize=32, weight='bold', pad=20, 
             color='#2c5aa0', family='monospace')

# Styling - clean and minimal
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.15)
ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color='#999999')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax2.spines[spine].set_color('#cccccc')
    ax2.spines[spine].set_linewidth(1.5)
# Hide tick labels for cleaner look
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.tick_params(length=0)

plt.tight_layout()

# Save
output_path = 'assets/img/cross_entropy_loss_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
           facecolor='#f8f9fa', edgecolor='none')
print(f"✓ Saved cross-entropy loss vs gradient to {output_path}")

plt.close()
