import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import Seaborn

# 1) Load the JSON data
with open("embeddings_custom_2000_test.json", "r") as f:
    data = json.load(f)

# 2) Extract the dictionary (e.g., "LaCLIP")
motion_data = data["MotionLaCLIP+"]

# 3) Prepare a figure
plt.figure(figsize=(10, 8))

# 4) Get label list
labels = list(motion_data.keys())
num_labels = len(labels)

# Use a Seaborn color palette with enough distinct colors
# e.g., "hls", "Set2", "tab10", etc.
colors = sns.color_palette("tab10", n_colors=num_labels)

for i, label in enumerate(labels):
    points_2d = motion_data[label]["embeddings_2d"]
    x_vals = [p[0] for p in points_2d]
    y_vals = [p[1] for p in points_2d]

    # Get the color for this label from the palette
    color = colors[i]

    # Plot first point as a star
    plt.scatter(
        x_vals[0],
        y_vals[0],
        color=color,
        marker="*",
        s=200,
        alpha=0.8,
        edgecolors="black",
        linewidths=1,
        label=label
    )

    # Plot remaining points as circles
    if len(x_vals) > 1:
        plt.scatter(
            x_vals[1:],
            y_vals[1:],
            color=color,
            marker="o",
            alpha=0.7,
            label=None
        )

# 5) Legend inside the plot
plt.title("2D Embeddings by Label")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("plot_new.png", dpi=300)
plt.show()
