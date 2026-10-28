"""figures/peierls_vs_chi.png (Session 13; Session 14: D5 events and D6 baseline).
Every measured bracket (error bar = bracket, point = midpoint) from data/audit_moves.csv (D5 audit of every step). Marker =
event at the first move: circle = glide jump on (-110), down-triangle = glide jump on (-1 0 1), open diamond = easy -> split
transition (split core on a vertex column; a known Mendelev-potential artifact). Colour: small-cell sweep, 2x cell, live AL.
chi = 0 baseline = B2 (sweep protocol, D6); Gate B' (uncorrected loading) shown as history. Curves: Schmid 1.315 / cos chi,
the Vitek-type fit to our (-110) glide jumps (data/vitek_fit.json), and the Chaussidon et al. 2006 fit (s0 = 1604 MPa,
a = 0.61, as read by the planning chat). Run (ml env): python scripts/plot_peierls_vs_chi.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fit_vitek import vitek_tau, CHAUSSIDON, ROOT
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
MK = {"(-110)": "o", "(0 1 -1)": "^", "(-1 0 1)": "v"}
rows = list(csv.DictReader(open(f"{ROOT}/data/audit_moves.csv")))
vf = json.load(open(f"{ROOT}/data/vitek_fit.json"))
fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=170)
g = np.linspace(-30, 30, 241)
tau0 = 1.315                                          # D6 baseline (B2)
ax.plot(g, tau0 / np.cos(np.radians(g)), color=MUTED, lw=1.3, ls="--", label="Schmid: 1.315 / cos χ (χ = 0 baseline, B2)")
ax.plot(g, vitek_tau(g, vf["s0_GPa"], vf["a"]), color=INK, lw=1.2, label=f"Vitek-type fit to our (−110) glide jumps (s0 = {vf['s0_GPa'] * 1000:.0f} MPa, a = {vf['a']:.2f})")
ax.plot(g, vitek_tau(g, **CHAUSSIDON), color=MUTED, lw=1.2, ls=":", label="Chaussidon et al. 2006 fit (s0 = 1604 MPa, a = 0.61)")
seen = set()
for r in rows:
    chi, lo, hi = float(r["chi_deg"]), float(r["tau_last_unmoved"]), float(r["tau_first_moved"])
    if "gateBp" in r["run"]:
        ax.errorbar(chi - 0.8, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt="x", ms=6, color=MUTED, capsize=2, label="χ = 0, Gate B' (uncorrected loading; history)")
        continue
    live, x2 = "live" in r["cell"], r["cell"] == "2x"
    col = ORANGE if live else (AQUA if x2 else BLUE)
    split = r["event"].startswith("easy")
    mk = "D" if split else MK.get(r["glide_plane"], "o")
    key = ("live AL" if live else "2× cell" if x2 else "χ sweep, small cell", "easy → split" if split else f"glide jump {r['glide_plane']}")
    lab = f"{key[0]}: {key[1]}" if key not in seen else None
    seen.add(key)
    ax.errorbar(chi + (0.8 if x2 else 0), (lo + hi) / 2, yerr=(hi - lo) / 2, fmt=mk, ms=7, color=col, mfc="white" if split else col, mec=col,
                mew=1.6, capsize=3, label=lab)
    if live:
        k = r["cell"].split()[-1].rstrip(")")
        off = {"1": ((8, 26), "left"), "3": ((-4, 12), "right")}.get(k, ((-6, 6), "right"))
        ax.annotate(f"AL {k}", (chi, (lo + hi) / 2), xytext=off[0], textcoords="offset points", ha=off[1], fontsize=7.5, color=INK)
ax.axvline(0, color=MUTED, lw=0.8)
for x, txt, ha in ((-29.5, "twinning side (χ < 0)", "left"), (29.5, "anti-twinning side (χ > 0)", "right")):
    ax.text(x, 0.02, txt, transform=ax.get_xaxis_transform(), ha=ha, color=MUTED, fontsize=8.5)
ax.set_xlabel("MRSS angle χ (degrees) from the (−110) plane", color=INK); ax.set_ylabel("Peierls stress τ_P (GPa)", color=INK)
ax.set_xticks(range(-30, 31, 6)); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Screw dislocation in Fe (Mendelev 2003 #2), 0 K: Peierls stress vs χ (146.87 Å cell; 2× at ±30)", fontsize=9.5, color=INK)
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/peierls_vs_chi.png"); print("saved figures/peierls_vs_chi.png")
