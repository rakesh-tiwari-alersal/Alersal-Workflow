import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Plastic Spiral setup
# -----------------------------
rho = 1.324718
a = 1.0
b = (2 * np.log(rho)) / np.pi   # r(θ) = exp(b*θ) since a=1

def theta_of_L(L):
    """Inverse arc-length -> angle for logarithmic spiral r = exp(b*θ)."""
    return (1.0 / b) * np.log(1.0 + (b / a) / np.sqrt(1 + b**2) * L)

# Extend slightly beyond 528 so the 510 marker sits comfortably inside
theta_max = theta_of_L(528) * 1.01
theta = np.linspace(0, theta_max, 6000)
r = np.exp(b * theta)

# -----------------------------
# Long-cycle anchors (SPX separated at 493) + add AMZN & CAT
# -----------------------------
cycles = {
    "Bitcoin (237)": 237,
    "Uber (260)": 260,
    "AMZN (385)": 385,           
    "XLE (408)": 408,
    "Crude / Gold (510)": 510,
    "SPX (528)": 528,
    "CAT (541)": 541,        
}
colors = {
    "Bitcoin (237)": "tab:blue",
    "Uber (260)": "tab:green",
    "AMZN (385)": "tab:olive",
    "XLE (408)": "tab:purple",
    "Crude / Gold (510)": "tab:red",
    "SPX (528)": "tab:pink",
    "CAT (541)": "tab:orange",
}

# Compute marker positions and radial limit
marker_points = []
max_r = r.max()
for label, S in cycles.items():
    th = theta_of_L(S)
    rr = np.exp(b * th)
    marker_points.append((th, rr, label, colors[label]))
    max_r = max(max_r, rr)

# -----------------------------
# Guide circles (ONLY these four)
# -----------------------------
guide_levels = [220, 291, 385, 510]
guide_rs = [np.exp(b * theta_of_L(L)) for L in guide_levels]

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection="polar")

# Clean background
ax.grid(False)

# Spiral in blue
ax.plot(theta, r, linewidth=2, color="blue")

# Outer limit with a small buffer
ax.set_rlim(0, max_r * 1.03)

# Faint guide circles
for r_guide in guide_rs:
    ax.plot(theta, np.full_like(theta, r_guide),
            linestyle="--", linewidth=1, color="lightgrey", alpha=0.7)

# Markers (large dots)
for th, rr, label, color in marker_points:
    ax.plot(th, rr, "o", markersize=16, color=color, mec="black", mew=0.6, zorder=5)

# Legend outside
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[label],
           markeredgecolor="black", markersize=10, label=label)
    for label in cycles.keys()
]
ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.0, 1.0),
          borderaxespad=0.0, title="Assets with Plastic Cycles")

# Title with math notation and extra padding
# ax.set_title(r"Capital Lattice Expressed in Plastic Spiral  $(r=\rho^{2\theta/\pi})$",
#             va="bottom", pad=30)

# Hide tick labels for a clean look
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()
plt.show()
