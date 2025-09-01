import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["svg.fonttype"] = "none"
tp = 198  # True Positives
fp = 49   # False Positives
fn = 45   # False Negatives
tn = 205  # True Negatives
total = 497

# Data preparation for donut chart
data = [tp, fp, fn, tn]
labels = ['True Positives', 'False Positives', 'False Negatives', 'True Negatives']
sample_counts = [f'{tp} samples', f'{fp} samples', f'{fn} samples', f'{tn} samples']

# Colors as specified
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

# Create figure with polar projection
fig = plt.figure(figsize=(12, 8), dpi=80)
fig.patch.set_facecolor('#f7f9fc')
ax = plt.subplot(polar=True)
plt.axis('off')

# Constants for layout
upperLimit = max(data) * 1.2
lowerLimit = 0
labelPadding = max(data) * 0.1

# Compute scaling
max_value = max(data)
slope = (max_value - lowerLimit) / max_value
heights = slope * np.array(data) + lowerLimit

# Compute widths and angles
width = 2 * np.pi / len(data)
indexes = list(range(1, len(data) + 1))
angles = [element * width for element in indexes]

# Draw bars (donut segments)
bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=1.5, 
    edgecolor="white",  
    color=colors  
)

# Calculate percentages for each segment
percentages = [(value / total * 100) for value in data]

# Add labels with sample counts
for bar, angle, height, label, sample_count, percentage in zip(bars, angles, heights, labels, sample_counts, percentages):
    rotation = np.rad2deg(angle)
    alignment = "right" if np.pi / 2 <= angle < 3 * np.pi / 2 else "left"
    rotation = rotation + 180 if alignment == "right" else rotation

    # Add label with sample count
    ax.text(
        x=angle,
        y=lowerLimit + bar.get_height() + labelPadding, 
        s=f"{label}\n{sample_count}", 
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor",
        font="serif", fontsize=16, weight="bold",
        color="#333333", alpha=0.9
    )
    
    # Add percentage inside each segment
    ax.text(
        x=angle,
        y=lowerLimit + bar.get_height() / 2,
        s=f"{percentage:.1f}%",
        ha='center',
        va='center',
        rotation=rotation,
        font="serif", fontsize=16, weight="bold",
        color="black"
    )


# Add space
fig.subplots_adjust(top=0.82)

# Save as SVG
plt.savefig('confusion_matrix_donut_new.svg', format='svg', bbox_inches='tight', dpi=300)
print("Saved as confusion_matrix_donut_new.svg")

# Show the plot
plt.show()