import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def stable_softmax(x, temperature=1.0):
    """Stable softmax with temperature parameter."""
    x_temp = x / temperature
    x_shifted = x_temp - np.max(x_temp)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


# Setup data
tokens = ['mat', 'chair', 'table', 'rug', 'floor', 'bed', 'sofa']
logits = np.array([3.2, 1.8, 1.0, 2.5, 0.5, 0.8, 0.3])

# Temperature values to scan through
# High temp -> uniform, low temp -> one-hot
temperatures = np.concatenate([
    np.linspace(5.0, 0.1, 40),  # Cool down
    np.linspace(0.1, 0.1, 10),  # Hold at min
    np.linspace(0.1, 5.0, 40),  # Heat up
    np.linspace(5.0, 5.0, 10),  # Hold at max
])

fig = plt.figure(figsize=(18, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[5, 1, 5], wspace=0.12)
ax_before = fig.add_subplot(gs[0])
ax_arrow = fig.add_subplot(gs[1])
ax_after = fig.add_subplot(gs[2])

x_pos = np.arange(len(logits))

def init():
    """Initialize the plot."""
    ax_before.clear()
    ax_arrow.clear()
    ax_after.clear()
    
    # Setup left plot (logits - static)
    colors_before = plt.cm.Greens(0.45 + logits / logits.max() * 0.55)
    shadow_offset = 0.1
    shadow_color = '#d0d0d0'
    
    ax_before.bar(x_pos, logits, color=shadow_color, edgecolor='none', width=0.8, 
                  bottom=shadow_offset, alpha=0.3, zorder=1)
    ax_before.bar(x_pos, logits, color=colors_before, edgecolor='white', 
                  linewidth=2, width=0.8, zorder=2)
    ax_before.set_ylim(0, max(logits) * 1.25)
    for spine in ax_before.spines.values():
        spine.set_visible(False)
    ax_before.set_xticks(x_pos)
    ax_before.set_xticklabels(tokens, fontsize=17, fontfamily='monospace', fontweight='bold')
    ax_before.set_yticks([])
    ax_before.tick_params(axis='x', length=0, pad=12)
    ax_before.set_facecolor('#f8f8f8')
    ax_before.text(0.5, 1.12, '"the cat sat on a ___"', transform=ax_before.transAxes,
                   ha='center', fontsize=15, style='italic', color='#666666')
    ax_before.text(0.5, 1.05, 'Logits', transform=ax_before.transAxes,
                   ha='center', fontsize=20, fontweight='bold', color='#1e5a3a')
    
    # Setup arrow
    ax_arrow.axis('off')
    ax_arrow.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=5, color='#444444',
                                    mutation_scale=25),
                      xycoords='axes fraction')
    ax_arrow.text(0.5, 0.32, 'softmax', ha='center', va='top', 
                  fontsize=15, style='italic', color='#444444',
                  fontweight='bold', transform=ax_arrow.transAxes)
    
    return fig,

def animate(frame):
    """Update plot for each frame."""
    temperature = temperatures[frame]
    probabilities = stable_softmax(logits, temperature)
    
    # Clear right plot
    ax_after.clear()
    
    # Setup right plot (probabilities - animated)
    colors_after = plt.cm.Purples(0.35 + probabilities / probabilities.max() * 0.65)
    shadow_offset = 0.1
    shadow_color = '#d0d0d0'
    
    ax_after.bar(x_pos, probabilities, color=shadow_color, edgecolor='none', width=0.8,
                 bottom=shadow_offset * max(probabilities)/max(logits), alpha=0.3, zorder=1)
    ax_after.bar(x_pos, probabilities, color=colors_after, edgecolor='white',
                 linewidth=2, width=0.8, zorder=2)
    ax_after.set_ylim(0, 1.0)
    for spine in ax_after.spines.values():
        spine.set_visible(False)
    ax_after.set_xticks(x_pos)
    ax_after.set_xticklabels(tokens, fontsize=17, fontfamily='monospace', fontweight='bold')
    ax_after.set_yticks([])
    ax_after.tick_params(axis='x', length=0, pad=12)
    ax_after.set_facecolor('#f8f8f8')
    
    # Add title with temperature
    ax_after.text(0.5, 1.05, f'Probabilities (T={temperature:.2f})', 
                  transform=ax_after.transAxes,
                  ha='center', fontsize=20, fontweight='bold', color='#5a1a66')
    
    # Highlight winner
    max_idx = probabilities.argmax()
    ax_after.plot([max_idx], [probabilities[max_idx] * 1.15], 'v', 
                  markersize=20, color='#FFD700', zorder=10, 
                  markeredgecolor='#FFA500', markeredgewidth=2)
    
    return fig,

print("Creating animation...")
anim = FuncAnimation(fig, animate, init_func=init, frames=len(temperatures), 
                     interval=50, blit=True, repeat=True)

print("Saving GIF...")
writer = PillowWriter(fps=20)
anim.save('assets/img/softmax_temperature.gif', writer=writer, dpi=150)
print("Saved: assets/img/softmax_temperature.gif")
plt.close()
