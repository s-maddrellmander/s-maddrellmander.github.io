import numpy as np
import matplotlib.pyplot as plt

def stable_softmax(x, axis=-1):
    """Stable softmax implementation with axis support."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Figure 1: Single vector before and after softmax - HERO IMAGE
np.random.seed(42)
# Next token prediction: "the cat sat on a __"
tokens = ['mat', 'chair', 'table', 'rug', 'floor', 'bed', 'sofa']
logits = np.array([3.2, 1.8, 1.0, 2.5, 0.5, 0.8, 0.3])
probabilities = stable_softmax(logits)

# Create figure with dramatic layout
fig = plt.figure(figsize=(18, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[5, 1, 5], wspace=0.12)
ax_before = fig.add_subplot(gs[0])
ax_arrow = fig.add_subplot(gs[1])
ax_after = fig.add_subplot(gs[2])

x_pos = np.arange(len(logits))

# Create vivid color gradients
colors_before = plt.cm.Greens(0.45 + logits / logits.max() * 0.55)
colors_after = plt.cm.Purples(0.35 + probabilities / probabilities.max() * 0.65)

# Add shadows/depth by plotting shifted bars underneath
shadow_offset = 0.1
shadow_color = '#d0d0d0'

# Plot 1: Logits with shadow
ax_before.bar(x_pos, logits, color=shadow_color, edgecolor='none', width=0.8, 
              bottom=shadow_offset, alpha=0.3, zorder=1)
bars_before = ax_before.bar(x_pos, logits, color=colors_before, edgecolor='white', 
                            linewidth=2, width=0.8, zorder=2)
ax_before.set_ylim(0, max(logits) * 1.25)
for spine in ax_before.spines.values():
    spine.set_visible(False)
ax_before.set_xticks(x_pos)
ax_before.set_xticklabels(tokens, fontsize=17, fontfamily='monospace', fontweight='bold')
ax_before.set_yticks([])
ax_before.tick_params(axis='x', length=0, pad=12)
ax_before.set_facecolor('#f8f8f8')

# Add prompt text above
ax_before.text(0.5, 1.12, '"the cat sat on a ___"', transform=ax_before.transAxes,
               ha='center', fontsize=15, style='italic', color='#666666')
ax_before.text(0.5, 1.05, 'Logits', transform=ax_before.transAxes,
               ha='center', fontsize=20, fontweight='bold', color='#1e5a3a')

# Arrow in middle - make it bolder
ax_arrow.axis('off')
ax_arrow.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                  arrowprops=dict(arrowstyle='->', lw=5, color='#444444',
                                mutation_scale=25),
                  xycoords='axes fraction')
ax_arrow.text(0.5, 0.32, 'softmax', ha='center', va='top', 
              fontsize=15, style='italic', color='#444444',
              fontweight='bold', transform=ax_arrow.transAxes)

# Plot 2: Probabilities with shadow and winner highlight
ax_after.bar(x_pos, probabilities, color=shadow_color, edgecolor='none', width=0.8,
             bottom=shadow_offset * max(probabilities)/max(logits), alpha=0.3, zorder=1)
bars_after = ax_after.bar(x_pos, probabilities, color=colors_after, edgecolor='white',
                          linewidth=2, width=0.8, zorder=2)
ax_after.set_ylim(0, max(probabilities) * 1.25)
for spine in ax_after.spines.values():
    spine.set_visible(False)
ax_after.set_xticks(x_pos)
ax_after.set_xticklabels(tokens, fontsize=17, fontfamily='monospace', fontweight='bold')
ax_after.set_yticks([])
ax_after.tick_params(axis='x', length=0, pad=12)
ax_after.set_facecolor('#f8f8f8')

# Add title
ax_after.text(0.5, 1.05, 'Probabilities', transform=ax_after.transAxes,
              ha='center', fontsize=20, fontweight='bold', color='#5a1a66')

# Dramatic winner highlight
max_idx = probabilities.argmax()
ax_after.plot([max_idx], [probabilities[max_idx] * 1.15], 'v', 
              markersize=20, color='#FFD700', zorder=10, 
              markeredgecolor='#FFA500', markeredgewidth=2)
# Add percentage text
ax_after.text(max_idx, probabilities[max_idx] * 1.2, 
              f'{probabilities[max_idx]*100:.0f}%',
              ha='center', va='bottom', fontsize=13, fontweight='bold',
              color='#FFD700', bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', edgecolor='#FFD700', linewidth=2))

plt.savefig('assets/img/softmax_distribution_shift.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
print("Saved: assets/img/softmax_distribution_shift.png")
plt.close()


# Figure 2: 2D visualization - batch of vectors before and after softmax
np.random.seed(43)
n_samples = 8
n_classes = 6

# Generate logits with some structure
logits_2d = np.random.randn(n_samples, n_classes) * 0.8 + 1.5
for i in range(n_samples):
    dominant_class = np.random.randint(0, n_classes)
    logits_2d[i, dominant_class] += np.random.uniform(1.5, 2.5)

# Apply softmax
probabilities_2d = stable_softmax(logits_2d, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot 1: Heatmap of logits (green palette)
im1 = axes[0].imshow(logits_2d.T, aspect='auto', cmap='Greens', 
                     interpolation='nearest')
axes[0].spines['top'].set_visible(True)
axes[0].spines['right'].set_visible(True)
axes[0].spines['bottom'].set_visible(True)
axes[0].spines['left'].set_visible(True)
axes[0].tick_params(length=0)

# Plot 2: Heatmap of probabilities (purple palette)
im2 = axes[1].imshow(probabilities_2d.T, aspect='auto', cmap='Purples', 
                     interpolation='nearest')
axes[1].spines['top'].set_visible(True)
axes[1].spines['right'].set_visible(True)
axes[1].spines['bottom'].set_visible(True)
axes[1].spines['left'].set_visible(True)
axes[1].tick_params(length=0)

plt.tight_layout()
plt.savefig('assets/img/softmax_focus_2d.png', dpi=300, bbox_inches='tight')
print("Saved: assets/img/softmax_focus_2d.png")
plt.close()


# Figure 3: Show the effect of axis parameter with a simple example
# Figure 3: Show the effect of axis parameter with a simple example
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Create a simple 3x4 matrix
example_logits = np.array([
    [2.0, 1.0, 0.5, 0.1],
    [0.1, 3.0, 0.2, 0.3],
    [1.0, 0.5, 2.5, 0.8]
])

# Apply softmax along different axes
softmax_axis1 = stable_softmax(example_logits, axis=1)  # across columns (classes)
softmax_axis0 = stable_softmax(example_logits, axis=0)  # across rows (batch)

# Plot original (green)
im1 = axes[0].imshow(example_logits, cmap='Greens', aspect='auto')
for i in range(3):
    for j in range(4):
        axes[0].text(j, i, f'{example_logits[i, j]:.1f}', 
                    ha='center', va='center', color='black', fontsize=11, fontweight='bold')
axes[0].tick_params(length=0)

# Plot softmax along axis=1 (purple)
im2 = axes[1].imshow(softmax_axis1, cmap='Purples', aspect='auto')
for i in range(3):
    for j in range(4):
        axes[1].text(j, i, f'{softmax_axis1[i, j]:.2f}', 
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
axes[1].tick_params(length=0)

# Plot softmax along axis=0 (purple)
im3 = axes[2].imshow(softmax_axis0, cmap='Purples', aspect='auto')
for i in range(3):
    for j in range(4):
        axes[2].text(j, i, f'{softmax_axis0[i, j]:.2f}', 
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
axes[2].tick_params(length=0)

plt.tight_layout()
plt.savefig('assets/img/softmax_axis_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: assets/img/softmax_axis_comparison.png")
plt.close()

print("\nAll plots generated successfully!")
