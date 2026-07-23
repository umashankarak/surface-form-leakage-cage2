"""
Figure 1 (v2) — CAGE-2 topology with a clean blocked-path barrier.

Fixes the confusing curved arrow: the permitted route is two straight arrows
(User->Enterprise->Operational); the blocked direct route is shown as a short
straight arrow from the User zone toward the Operational zone, interrupted by a
barrier symbol (a red no-entry marker) with the substring-gate label beneath it.
No arrow has to navigate around obstacles, so the asymmetry reads at a glance.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.linewidth": 0.6,
})

fig, ax = plt.subplots(figsize=(7.0, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.2)
ax.axis("off")

# ---- subnet zones -----------------------------------------------------------
zones = [
    (0.2, "User subnet", "#eef2f7"),
    (3.6, "Enterprise subnet", "#e4ecf3"),
    (7.0, "Operational subnet", "#dfe9f0"),
]
zone_w = 2.8
for x, label, color in zones:
    ax.add_patch(Rectangle((x, 1.15), zone_w, 3.55, facecolor=color,
                           edgecolor="#8a99a8", linewidth=0.7, zorder=0))
    ax.text(x + zone_w / 2, 4.44, label, ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#2b3a4a")

# ---- hosts ------------------------------------------------------------------
def host(x, y, label, w=1.75, h=0.54, bold=False, crown=False):
    fc = "#ffffff" if not crown else "#c9d6e2"
    lw = 1.5 if crown else 0.9
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=fc, edgecolor="#2b3a4a", linewidth=lw, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=8.2,
            fontweight="bold" if bold else "normal", zorder=4)
    return (x, y)

host(1.6, 3.55, "User0\n(entry)", bold=True)
host(1.6, 2.15, "User1–4")
host(5.0, 3.55, "Enterprise0–2")
host(5.0, 2.15, "Defender")
host(8.4, 3.55, "Op_Server0\n(crown jewel)", crown=True, bold=True)
host(8.4, 2.15, "Op_Host0–2")

# ---- permitted pivots (dark solid arrows, top row) --------------------------
def arrow(p1, p2, color="#2b3a4a", lw=1.4):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                        color=color, linewidth=lw,
                        shrinkA=6, shrinkB=6, zorder=2)
    ax.add_patch(a)

arrow((2.50, 3.55), (4.10, 3.55))   # User -> Enterprise
arrow((5.90, 3.55), (7.50, 3.55))   # Enterprise -> Operational

# ---- blocked direct path (bottom band): straight arrow + barrier ------------
yb = 1.55
# left stub: from User zone heading right
a1 = FancyArrowPatch((2.50, yb), (4.35, yb), arrowstyle="-",
                     color="#7a1f1f", linewidth=1.4,
                     linestyle=(0, (5, 3)), shrinkA=6, shrinkB=0, zorder=2)
ax.add_patch(a1)
# right stub: continuing to the Operational zone, with arrowhead
a2 = FancyArrowPatch((5.65, yb), (7.50, yb), arrowstyle="-|>",
                     mutation_scale=13, color="#7a1f1f", linewidth=1.4,
                     linestyle=(0, (5, 3)), shrinkA=0, shrinkB=6, zorder=2)
ax.add_patch(a2)

# barrier symbol (no-entry) at the midpoint
bx, by = 5.0, yb
ax.add_patch(Circle((bx, by), 0.26, facecolor="#f2dcdc",
                    edgecolor="#7a1f1f", linewidth=1.6, zorder=5))
ax.plot([bx - 0.15, bx + 0.15], [by, by], color="#7a1f1f",
        linewidth=1.8, solid_capstyle="round", zorder=6)

ax.text(3.55, yb + 0.30, "direct path", fontsize=7.8, color="#7a1f1f",
        ha="center", style="italic")

# gate label beneath the barrier
ax.add_patch(Rectangle((bx - 1.95, by - 0.92), 3.9, 0.44,
                       facecolor="#f7ecec", edgecolor="#7a1f1f",
                       linewidth=1.0, zorder=5))
ax.text(bx, by - 0.70, "access gate:  if 'Enterprise' in host",
        ha="center", va="center", fontsize=8, family="monospace",
        color="#7a1f1f", zorder=6)

# ---- legend -----------------------------------------------------------------
legend_elems = [
    Line2D([0], [0], color="#2b3a4a", lw=1.4, marker=">", markersize=5,
           label="permitted pivot"),
    Line2D([0], [0], color="#7a1f1f", lw=1.4, linestyle=(0, (5, 3)),
           label="blocked direct path (substring-gated)"),
]
ax.legend(handles=legend_elems, loc="lower center", ncol=2,
          bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8,
          handlelength=2.4, columnspacing=1.8)

plt.tight_layout(pad=0.4)
fig.savefig("./fig1_topology.pdf", bbox_inches="tight")
fig.savefig("./fig1_topology.png", dpi=300, bbox_inches="tight")
print("Figure 1 v2 written.")
