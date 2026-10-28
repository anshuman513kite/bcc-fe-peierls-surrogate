"""Session 12, Part B step 6: figures/surrogate_chi_v1.png.
Top: tau_P vs chi: the sweep points (v1), the live active-learning points (numbered by round), the pre-registered predictions
(+-2 spread bars, made before each run), the surrogate mean with its +-2 band (best LOO model refitted on all points), and
Schmid's law tau0 / cos(chi). Bottom: the same on the Schmid ratio r = tau_P cos(chi) / tau0 (Schmid = 1).
Twinning side chi < 0, anti-twinning side chi > 0 (Session 9). Run (ml env): python scripts/plot_surrogate.py
"""
import csv, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surrogate_common import ROOT, TAU0, load_train, make_model, TRAIN_V1, LIVE_RESULTS
torch.set_num_threads(1)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
name = "gp_tau_d4"                       # D4/D7: the live-AL model from round 3 on (Session 14), not re-selected
chi, tau = load_train()
v1 = [(float(r["chi_deg"]), float(r["tau_mid"])) for r in csv.DictReader(open(TRAIN_V1))]
live = list(csv.DictReader(open(LIVE_RESULTS))) if os.path.exists(LIVE_RESULTS) else []
for r in live:            # rows not (yet) recorded under D1 are drawn as pending core-change points, not fitted (load_train)
    if r.get("rule") != "D5":
        r["flag"] = r.get("flag") or "pending"
pred_path = os.path.join(ROOT, "data", "al_live_predictions.csv")
done = {r["round"] for r in live}
preds = [p for p in csv.DictReader(open(pred_path)) if p["round"] in done and not p.get("status", "").startswith("voided")] \
    if os.path.exists(pred_path) else []                                                       # voided rows not drawn
g = np.linspace(-30, 30, 241)
m, s = make_model(name).fit(chi[:, None], tau).predict(g[:, None])
c = np.cos(np.radians(g))
fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
panels = ((axes[0], lambda x: 1.0, "τ_P (GPa)"), (axes[1], lambda x: np.cos(np.radians(x)) / TAU0, "r = τ_P cos χ / τ_P(0)"))
for ax, k, fl in panels:                     # k(chi): factor from tau_P to the panel's quantity
    f = lambda x, t: t * k(x)
    ax.fill_between(g, f(g, m - 2 * s), f(g, m + 2 * s), color=BLUE, alpha=0.18, lw=0, label=f"surrogate ±2 spread ({name}, all {len(chi)} points)")
    ax.plot(g, f(g, m), color=BLUE, lw=2, label="surrogate mean")
    ax.plot(g, f(g, TAU0 / c), color=MUTED, lw=1.5, ls="--", label="Schmid's law: τ_P(0) / cos χ")
    ax.plot([p[0] for p in v1], [f(p[0], p[1]) for p in v1], "o", ms=8, color=BLUE, mec="white", mew=1.5, label="χ sweep (LAMMPS, bracket midpoint)")
    for p in preds:
        x, mu, sp = float(p["chi_pick"]), float(p["pred_mean"]), float(p["pred_spread"])
        ax.errorbar(x, f(x, mu), yerr=2 * sp * k(x), fmt="_", ms=14, color=INK, capsize=4, lw=1.2,
                    label="pre-registered prediction ±2 spread" if p is preds[0] else None)
    for r in live:
        x, t = float(r["chi_deg"]), float(r["tau_mid"])
        ax.plot(x, f(x, t), "D", ms=9, color=ORANGE, mec="white", mew=1.5, label="active-learning pick (LAMMPS)" if r is live[0] else None)
        ax.annotate(r["round"] + ("*" if r.get("flag") else ""), (x, f(x, t)), **{"1": dict(xytext=(10, -12), ha="left", va="top"), "3": dict(xytext=(-10, 12), ha="right", va="bottom")}.get(
                        r["round"], dict(xytext=(-10, -6), ha="right", va="center")), textcoords="offset points", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel(fl, color=INK); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.axvline(0, color=MUTED, lw=0.8)
for x, txt in ((-15, "twinning side (χ < 0)"), (15, "anti-twinning side (χ > 0)")):
    axes[0].text(x, 0.97, txt, transform=axes[0].get_xaxis_transform(), ha="center", va="top", color=MUTED, fontsize=10)
axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=8.5, frameon=False)
axes[1].set_xlabel("MRSS angle χ (degrees)", color=INK)
axes[1].set_xticks(range(-30, 31, 6))
axes[0].set_title("Peierls stress vs χ: surrogate with live active learning (Mendelev 2003 Fe, 146.87 Å cell)", fontsize=10.5, color=INK)
flagged = [r for r in live if r.get("flag")]
if flagged:
    fig.text(0.01, 0.005, "* " + "; ".join(f"round {r['round']} (χ = {float(r['chi_deg']):+.0f}°): core-change bracket, no glide jump; "
             f"not in the fit, pending Ansh's decision" for r in flagged), fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02 if flagged else 0, 1, 1))
out = os.path.join(ROOT, "figures", "surrogate_chi_v1.png")
fig.savefig(out, dpi=150); print("saved", out)
