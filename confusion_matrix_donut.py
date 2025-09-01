import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["svg.fonttype"] = "none"

# Confusion matrix values
tp = 380
fp = 117
fn = 96
tn = 407
total = 1000

# Labels and sizes for the donut chart
labels = ['True Positives\n380 samples', 'False Positives\n117 samples', 'False Negatives\n96 samples', 'True Negatives\n407 samples']
sizes = [tp, fp, fn, tn]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']  # Using specified color scheme

# Explosion for each slice
explode = (0.05, 0.05, 0.05, 0.05)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create pie chart (donut)
wedges, texts, autotexts = ax.pie(
    sizes, 
    colors=colors, 
    labels=labels, 
    autopct='%1.1f%%', 
    startangle=90, 
    pctdistance=0.85, 
    explode=explode,
    textprops={'fontsize': 16}
)

# Draw white circle in the center to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal') 

# Add total count in the center
ax.text(0, 0, f'Evalution Samples\n{total}', 
        ha='center', va='center', 
        fontsize=18, fontweight='bold')

# Improve text formatting
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(16)

plt.tight_layout()

# Save as SVG
plt.savefig('confusion_matrix_donut.svg', format='svg', bbox_inches='tight', dpi=300)
print("Saved as confusion_matrix_donut.svg")

# Show the plot
plt.show()