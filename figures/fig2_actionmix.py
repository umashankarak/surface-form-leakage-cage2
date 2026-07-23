"""
Figure 2 — Action distribution by model (the play-style confound).

Grouped bars, greyscale-distinguishable via hatching + distinct grey levels so
the figure survives black-and-white printing. Makes the central case-study
finding immediate: the 7B model (which floors) monitors passively and rarely
removes, while 3B and 14B (which defend competently) remove aggressively.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.linewidth": 0.6,
})

# data from Table III (percentages)
actions = ["Monitor", "Remove", "Analyse", "Decoy", "Other"]
models = ["3B", "7B", "14B"]
data = {
    "3B":  [34, 46, 3,  0.5, 17],
    "7B":  [71, 2,  7,  16,  4],
    "14B": [44, 31, 15, 3,   7],
}

x = np.arange(len(actions))
width = 0.25

# greyscale-safe styling: distinct fills + hatches
styles = {
    "3B":  dict(color="#3f3f3f", hatch=None),
    "7B":  dict(color="#ffffff", hatch="////", edgecolor="#2b2b2b"),
    "14B": dict(color="#9a9a9a", hatch=None),
}

fig, ax = plt.subplots(figsize=(6.6, 3.2))
for i, m in enumerate(models):
    s = styles[m]
    ax.bar(x + (i - 1) * width, data[m], width,
           label=f"{m}", color=s["color"], hatch=s.get("hatch"),
           edgecolor=s.get("edgecolor", "#2b2b2b"), linewidth=0.8, zorder=3)

ax.set_ylabel("Share of actions (%)")
ax.set_xticks(x)
ax.set_xticklabels(actions)
ax.set_ylim(0, 80)
ax.yaxis.grid(True, linewidth=0.4, color="#cccccc", zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# value labels on top of each bar -- clearer than arrows, no overlap
for i, m in enumerate(models):
    for j, v in enumerate(data[m]):
        if v >= 1:
            ax.text(x[j] + (i - 1) * width, v + 1.2, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=6.8, color="#2b2b2b",
                    zorder=4)

ax.legend(title="Model", frameon=False, loc="upper right",
          bbox_to_anchor=(1.0, 1.02), fontsize=8.5, title_fontsize=8.5,
          handlelength=1.6, labelspacing=0.35)

plt.tight_layout(pad=0.4)
fig.savefig("./fig2_actionmix.pdf", bbox_inches="tight")
fig.savefig("./fig2_actionmix.png", dpi=300, bbox_inches="tight")
print("Figure 2 written: fig2_actionmix.pdf / .png")
