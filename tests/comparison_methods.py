import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# -----------------------------
# Example data (edit these)
# -----------------------------
methods = ["AUTO", "Poscidyn (uncompiled)", "Poscidyn (compiled)", "Poscidyn (batched)"]
times = np.array([3.26, 6.93, 1.71, 0.32])

# -----------------------------
# Create soft green → red colormap
# -----------------------------
soft_cmap = LinearSegmentedColormap.from_list(
    "soft_green_red",
    ["#4C9A2A", "#F0F0F0", "#B22222"]  # muted green → light gray → muted red
)

norm = Normalize(vmin=times.min(), vmax=times.max())
colors = soft_cmap(norm(times))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 4))

bars = plt.barh(methods, times, color=colors)

# Add time labels
for bar, t in zip(bars, times):
    plt.text(
        bar.get_width() + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.2f} s",
        va='center'
    )

plt.xlabel("Completion Time (seconds)")
plt.title("Method Performance Comparison")
plt.xlim(0, times.max() * 1.15)

plt.tight_layout()
plt.show()
