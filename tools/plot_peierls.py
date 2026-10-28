"""Plot core position along the glide direction (top) and core height y (bottom) vs applied stress
from one or more Peierls-ramp CSVs.

Usage: python plot_peierls.py out.png label1=data1.csv [label2=data2.csv ...]
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JUMP = 2.8553 * 6 ** 0.5 / 3
COLORS = ["#2a78d6", "#eb6834"]               # categorical slots 1-2 (reference palette)
fig, (ax, ay) = plt.subplots(2, 1, figsize=(6.4, 6.6), dpi=200, sharex=True, height_ratios=[3, 2])
for k, spec in enumerate(sys.argv[2:]):
    label, path = spec.split("=", 1)
    rows = list(csv.DictReader(open(path)))
    for kind, marker in [("coarse", "o"), ("fine", "s")]:
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        s = [float(x["sigma_GPa"]) for x in r]
        z = [float(x["core_z_cum_A"]) for x in r]
        y = [float(x["core_y_A"]) for x in r]
        if kind == "coarse":
            ax.plot(s, z, "-", color=COLORS[k], lw=2, zorder=2)
            ay.plot(s, y, "-", color=COLORS[k], lw=2, zorder=2)
        ay.scatter(s, y, s=36 if kind == "coarse" else 30, marker=marker, color=COLORS[k], edgecolor="white",
                   linewidth=1.2, zorder=3)
        ax.scatter(s, z, s=36 if kind == "coarse" else 30, marker=marker, color=COLORS[k], edgecolor="white",
                   linewidth=1.2, zorder=3, label=f"{label}, {kind} ({'50' if kind == 'coarse' else '10'} MPa steps)")
for n in (1, 2, 3):
    ax.axhline(n * JUMP, color="#c3c2b7", lw=0.8, ls="--", zorder=1)
ax.text(0.01, JUMP + 0.15, "one Peierls jump, a√6/3 = 2.331 Å", fontsize=7, color="#555", transform=ax.get_yaxis_transform())
ay.set_xlabel("applied glide stress σ_xy (GPa)")
ay.set_ylabel("core height y (Å)")
ay.set_title("tracked core height (energy centroid of atoms within 4 Å of the original glide plane;\n"
             "under-reads a move to another plane)", fontsize=8)
ax.set_ylabel("core displacement along z = [-1 -1 2] (Å)")
ax.set_title("Screw dislocation in Fe (Mendelev 2003 #2), χ = 0: core position vs stress\n"
             "(core displacement since the start of the ramp; a runaway step is stopped after 3 jumps)", fontsize=9)
for a in (ax, ay):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(sys.argv[1])
print("saved", sys.argv[1])
