import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    'FRED LSTM',
    'FRED Transformer',
    'FRED CNN+Transformer (Event + RGB)',
    'DroneStalker-LSTM-0.3'
]

# Metrics (ADE / FDE / mIoU)
ade_scores = [82.73, 49.43, 47.49, 32.63]
fde_scores = [90.35, 69.07, 65.89, 49.02]
miou_scores = [0.044, 0.283, 0.279, 0.390]

# Metric names
metrics = ['ADE\n(pixels)', 'FDE\n(pixels)', 'mIoU']

# Create figure with 4 subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Comparison: 0.4s Frame Prediction - All Metrics per Model',
             fontsize=16, fontweight='bold', y=0.995)

# Flatten axes for easier iteration
axes = axes.flatten()

# Colors for metrics
metric_colors = ['#E74C3C', '#3498DB', '#2ECC71']

# Plot each model's metrics
for idx, (model, ade, fde, miou) in enumerate(zip(models, ade_scores, fde_scores, miou_scores)):
    ax = axes[idx]

    # Since metrics have different scales, we'll use normalized values for visualization
    # But show actual values in labels
    values = [ade, fde, miou * 100]  # Scale mIoU to be more visible

    # Create bars with different colors for each metric
    bars = ax.bar(range(3), values, color=metric_colors, edgecolor='black', linewidth=1.2, alpha=0.8)

    # Highlight if it's our model
    if idx == 3:
        ax.set_facecolor('#FFF9E6')
        ax.set_title(f'{model}', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='#FFD700', alpha=0.3))
    else:
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars with actual values
    actual_values = [ade, fde, miou]
    for bar, value, actual in zip(bars, values, actual_values):
        height = bar.get_height()
        if actual < 1:  # mIoU
            label = f'{actual:.3f}'
        else:
            label = f'{actual:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add annotations for better/worse indicators
    if idx == 3:
        # Add improvement indicators compared to best baseline
        best_ade = min(ade_scores[:3])
        best_fde = min(fde_scores[:3])
        best_miou = max(miou_scores[:3])

        improvement_ade = ((best_ade - ade) / best_ade) * 100
        improvement_fde = ((best_fde - fde) / best_fde) * 100
        improvement_miou = ((miou - best_miou) / best_miou) * 100

        # Add text box with improvements
        textstr = f'vs Best Baseline:\n'
        textstr += f'ADE: {improvement_ade:+.1f}%\n'
        textstr += f'FDE: {improvement_fde:+.1f}%\n'
        textstr += f'mIoU: {improvement_miou:+.1f}%'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('benchmarking/model_comparison_per_model.png', dpi=300, bbox_inches='tight')
print("Chart saved to benchmarking/model_comparison_per_model-2.png")
plt.show()
