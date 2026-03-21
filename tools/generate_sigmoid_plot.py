import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Define stable sigmoid
def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (np.exp(x) + 1))

# Define sigmoid derivative
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Generate x values
x = np.linspace(-10, 10, 1000)
y_sigmoid = sigmoid(x)
y_derivative = sigmoid_derivative(x)

# Create figure with custom colors
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with beautiful colors
ax.plot(x, y_sigmoid, linewidth=3, label='σ(x)', color='#2E86AB', alpha=0.9)
ax.plot(x, y_derivative, linewidth=3, label="σ'(x)", color='#A23B72', alpha=0.9)

# Add horizontal line at y=0.5 for sigmoid
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.3)

# Labels and title
ax.set_xlabel('x', fontsize=14, fontweight='bold')
ax.set_ylabel('y', fontsize=14, fontweight='bold')
ax.set_title('Sigmoid Function and Its Derivative', fontsize=16, fontweight='bold', pad=20)

# Legend
ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=12)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Set limits
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(-10, 10)

# Tight layout
plt.tight_layout()

# Save with high DPI
plt.savefig('assets/img/sigmoid-plot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Sigmoid plot saved to assets/img/sigmoid-plot.png")

plt.close()
