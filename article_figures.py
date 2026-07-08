"""
article_figures.py – publication-quality figures for the pedestrian dynamics article.
No titles; large axis labels and legends.
"""
import re, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ─── Config ───────────────────────────────────────────────────────────────────
TARGET_ANGLES  = [0, 30, 60, 90, 120, 150, 180]
DENSITY_LABELS = ["all", "fov_210_c5", "fov_210_c0"]   # rho_1, rho_0.5, rho_0

DLABEL_DISPLAY = {
    "all":        r"$\rho_1$",
    "fov_210_c5": r"$\rho_{0.5}$",
    "fov_210_c0": r"$\rho_0$",
}
DLABEL_YLABEL = {
    "all":        r"$\langle\rho_1\rangle$",
    "fov_210_c5": r"$\langle\rho_{0.5}\rangle$",
    "fov_210_c0": r"$\langle\rho_0\rangle$",
}
DLABEL_MARKER = {"all": "o", "fov_210_c5": "^", "fov_210_c0": "s"}

ANGLE_COLOR = {
    0: "#000000", 30: "#87CEEB", 60: "#009900", 90: "#7700CC",
    120: "#FEE12B", 150: "#0033CC", 180: "#CC0000",
}
ANGLE_MARKER = {0: "o", 30: "s", 60: "^", 90: "D", 120: "v", 150: "h", 180: "*"}

T_CMAP = LinearSegmentedColormap.from_list(
    "t_cmap", ["#FFE000", "#FF8000", "#CC0000", "#880088", "#00008B"])

TCOLOR_GROUPS = [[0, 30], [60, 90], [120, 150, 180]]

FS_LABEL  = 20
FS_TICK   = 17
FS_LEGEND = 17


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_angle_data(csv_dir, dlabel, alpha):
    angle_dir = csv_dir / dlabel / f"Angle_{alpha}"
    if not angle_dir.exists():
        return tuple(np.array([]) for _ in range(6))
    tp_l, v_l, acc_l, rho_l, da_l, dlt_l = [], [], [], [], [], []
    for fpath in sorted(glob.glob(str(angle_dir / "*_vrho.csv"))):
        df = pd.read_csv(fpath)
        tp = df["t_prime"].values
        indices = [int(m.group(1)) for c in df.columns
                   if (m := re.match(r"^v_(\d+)$", c))]
        for idx in indices:
            v   = pd.to_numeric(df[f"v_{idx}"],      errors="coerce").values
            rho = pd.to_numeric(df[f"rho_{idx}"],    errors="coerce").values
            da  = pd.to_numeric(df[f"dangle_{idx}"], errors="coerce").values
            dlt = pd.to_numeric(df[f"delta_{idx}"],  errors="coerce").values
            acc_col = f"acc_{idx}"
            acc = (pd.to_numeric(df[acc_col], errors="coerce").values
                   if acc_col in df.columns else np.full(len(v), np.nan))
            base = np.isfinite(v) & np.isfinite(rho)
            tp_l.append(tp[base]);   v_l.append(v[base])
            rho_l.append(rho[base]); da_l.append(da[base])
            dlt_l.append(dlt[base]); acc_l.append(acc[base])
    if not tp_l:
        return tuple(np.array([]) for _ in range(6))
    return (np.concatenate(tp_l), np.concatenate(v_l),
            np.concatenate(acc_l), np.concatenate(rho_l),
            np.concatenate(da_l),  np.concatenate(dlt_l))


def median_binned(x, y, edges, centres):
    xs, ys = [], []
    for i, c in enumerate(centres):
        m = (x >= edges[i]) & (x < edges[i + 1])
        yy = y[m]
        if len(yy) >= 200:
            xs.append(c); ys.append(np.median(yy))
    return np.array(xs), np.array(ys)


# ─── Plot helpers ─────────────────────────────────────────────────────────────
def style(ax):
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)


def save(fig, path_stem):
    """Save figure as PDF (vector, supports transparency)."""
    fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight")


def panel_tag(ax, txt, loc="tr"):
    """loc: 'tr' = top-right, 'br' = bottom-right."""
    if loc == "br":
        x, y, va = 0.97, 0.03, "bottom"
    else:
        x, y, va = 0.97, 0.97, "top"
    ax.text(x, y, txt, transform=ax.transAxes,
            ha="right", va=va, fontsize=FS_LABEL,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))


def manual_tcbar(fig, left=0.89, bottom=0.12, height=0.76):
    cax = fig.add_axes([left, bottom, 0.018, height])
    sm = plt.cm.ScalarMappable(cmap=T_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("$t'$", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    return cb


def manual_tcbar_h(fig, left=0.06, width=0.89, bottom=0.03):
    """Horizontal t' colorbar at the bottom of the figure."""
    cax = fig.add_axes([left, bottom, width, 0.022])
    sm = plt.cm.ScalarMappable(cmap=T_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("$t'$", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK)
    return cb


def make_232_axes(fig, sharey=False):
    """Seven equal-size axes in (2,3,2) layout.
    All panels span 2 of 6 columns; rows 0 and 2 are centred (cols 1-2 and 3-4).
    sharey=True collapses y-tick labels to the leftmost panel in each row.
    """
    wspace = 0.06 if sharey else 0.32
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.32, wspace=wspace,
                           left=0.06, right=0.95, top=0.98, bottom=0.12)
    ax0 = fig.add_subplot(gs[0, 1:3])
    sy = {"sharey": ax0} if sharey else {}
    axes = [
        ax0,
        fig.add_subplot(gs[0, 3:5], **sy),
        fig.add_subplot(gs[1, 0:2], **sy),
        fig.add_subplot(gs[1, 2:4], **sy),
        fig.add_subplot(gs[1, 4:6], **sy),
        fig.add_subplot(gs[2, 1:3], **sy),
        fig.add_subplot(gs[2, 3:5], **sy),
    ]
    if sharey:
        for pi, ax in enumerate(axes):
            if pi not in (0, 2, 5):
                ax.tick_params(labelleft=False)
    return axes


def angle_legend_handles():
    return [Line2D([0], [0], marker=ANGLE_MARKER[a], color=ANGLE_COLOR[a],
                   linestyle="None", markersize=8, label=f"α={a}°")
            for a in TARGET_ANGLES]


def outside_legend(fig, axes, ncol=1):
    """Place legend just outside the rightmost axis; tight_layout handles spacing."""
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=FS_LEGEND, ncol=ncol, frameon=True, borderaxespad=0)
    fig.tight_layout()


# ─── Main ─────────────────────────────────────────────────────────────────────
def make_figures(csv_dir="csv_data"):
    csv_dir = Path(__file__).parent / csv_dir
    out_dir = Path(__file__).parent / "article_figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_edges     = np.arange(0.0, 1.01, 0.01)
    n_tbins     = len(t_edges) - 1
    t_centres   = 0.5 * (t_edges[:-1] + t_edges[1:])
    rho_edges   = np.arange(0.0, 20.2, 0.2)
    rho_centres = 0.5 * (rho_edges[:-1] + rho_edges[1:])

    print("Loading data …")
    # raw[dl][alpha] = (tp, v, acc, rho, da, dlt)
    raw = {dl: {a: load_angle_data(csv_dir, dl, a)
                for a in TARGET_ANGLES}
           for dl in DENSITY_LABELS}

    # Time-binned means: tb[dl][alpha] keys:
    #   r, v, acc, tc
    #   dlt       = mean signed δ₁
    #   dlt_abs   = mean |δ₁|
    #   da        = mean |δ₂|
    #   da_signed = mean signed δ₂
    tb = {}
    for dl in DENSITY_LABELS:
        tb[dl] = {}
        for alpha in TARGET_ANGLES:
            tp, v, acc, rho, da, dlt = raw[dl][alpha]
            if len(v) == 0:
                continue
            mr       = np.full(n_tbins, np.nan)
            mv       = np.full(n_tbins, np.nan)
            ma       = np.full(n_tbins, np.nan)
            mda      = np.full(n_tbins, np.nan)
            mda_s    = np.full(n_tbins, np.nan)
            mdlt     = np.full(n_tbins, np.nan)
            mdlt_abs = np.full(n_tbins, np.nan)
            for b in range(n_tbins):
                mask = (tp >= t_edges[b]) & (tp < t_edges[b + 1])
                if not mask.any():
                    continue
                mr[b] = np.mean(rho[mask])
                mv[b] = np.mean(v[mask])
                da_m  = da[mask];  da_m  = da_m[np.isfinite(da_m)]
                if len(da_m):
                    mda[b]   = np.mean(np.abs(da_m))
                    mda_s[b] = np.mean(da_m)
                dlt_m = dlt[mask]; dlt_m = dlt_m[np.isfinite(dlt_m)]
                if len(dlt_m):
                    mdlt[b]     = np.mean(dlt_m)
                    mdlt_abs[b] = np.mean(np.abs(dlt_m))
                acc_m = acc[mask]; acc_m = acc_m[np.isfinite(acc_m)]
                if len(acc_m):
                    ma[b] = np.mean(acc_m)
            tb[dl][alpha] = dict(r=mr, v=mv, acc=ma,
                                 da=mda, da_signed=mda_s,
                                 dlt=mdlt, dlt_abs=mdlt_abs,
                                 tc=t_centres)
    print("Data ready.")

    # Black shape handles for density methods (figs 3/7/8)
    shape_handles = [
        Line2D([0], [0], marker=DLABEL_MARKER[dl], color="black",
               linestyle="None", markersize=9, label=DLABEL_DISPLAY[dl])
        for dl in DENSITY_LABELS
    ]

    # ─── Fig 1: rho vs t', 3 panels; legend outside ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for col, dl in enumerate(DENSITY_LABELS):
        ax = axes[col]
        for alpha in TARGET_ANGLES:
            if alpha not in tb[dl]:
                continue
            t = tb[dl][alpha]; ok = np.isfinite(t["r"])
            ax.plot(t["tc"][ok], t["r"][ok],
                    color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
        ax.set_xlabel("$t'$", fontsize=FS_LABEL)
        ax.set_ylabel(DLABEL_YLABEL[dl], fontsize=FS_LABEL)
        ax.tick_params(labelleft=True)
        ax.set_box_aspect(1)
        style(ax)
    outside_legend(fig, axes)
    save(fig, out_dir / "Fig01_rho_vs_tprime")
    plt.close(fig); print("Fig01 saved")

    # ─── Fig 2: v vs rho, 3 panels, per-angle medians with colors ────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, sharex=True)
    for col, dl in enumerate(DENSITY_LABELS):
        ax = axes[col]
        for alpha in TARGET_ANGLES:
            tp, v, acc, rho, da, dlt = raw[dl][alpha]
            if len(rho) == 0:
                continue
            mr, mv = median_binned(rho, v, rho_edges, rho_centres)
            if len(mr):
                ax.scatter(mr, mv, color=ANGLE_COLOR[alpha],
                           marker=ANGLE_MARKER[alpha],
                           s=64 if alpha not in [60, 120, 180] else 90,
                           label=f"α={alpha}°")
        ax.set_xlabel(DLABEL_DISPLAY[dl], fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel("$v$ [m/s]", fontsize=FS_LABEL)
        ax.set_box_aspect(1)
        style(ax)
    outside_legend(fig, axes)
    save(fig, out_dir / "Fig02_v_vs_rho_perangle")
    plt.close(fig); print("Fig02 saved")

    # ─── Fig 3: <v> vs <rho>, t' color, 7 panels (2,3,2), density=shape ──────
    fig3 = plt.figure(figsize=(16, 11))
    axes3 = make_232_axes(fig3, sharey=True)
    for pi, alpha in enumerate(TARGET_ANGLES):
        ax = axes3[pi]
        for dl in DENSITY_LABELS:
            if alpha not in tb[dl]:
                continue
            t = tb[dl][alpha]
            ok = np.isfinite(t["r"]) & np.isfinite(t["v"])
            if ok.sum() < 2:
                continue
            ax.scatter(t["r"][ok], t["v"][ok], c=t["tc"][ok],
                       cmap=T_CMAP, vmin=0, vmax=1,
                       s=50, marker=DLABEL_MARKER[dl], alpha=0.85)
        ax.set_xlabel(r"$\langle\rho\rangle$", fontsize=FS_LABEL)
        if pi in (0, 2, 5):
            ax.set_ylabel(r"$\langle v\rangle$ [m/s]", fontsize=FS_LABEL)
        ax.set_ylim(1.2, 1.55)
        panel_tag(ax, f"α={alpha}°")
        style(ax)
    manual_tcbar_h(fig3)
    fig3.legend(handles=shape_handles, loc="upper right",
                fontsize=FS_LEGEND, frameon=True, ncol=1,
                bbox_to_anchor=(0.95, 0.98))
    save(fig3, out_dir / "Fig03_v_rho_tcolor_perangle")
    plt.close(fig3); print("Fig03 saved")

    # ─── Fig 4a: delta1 and delta2 vs t' (signed, no abs) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for alpha in TARGET_ANGLES:
        if alpha not in tb["all"]:
            continue
        t = tb["all"][alpha]
        ok1 = np.isfinite(t["dlt"]); ok2 = np.isfinite(t["da_signed"])
        axes[0].plot(t["tc"][ok1], t["dlt"][ok1],
                     color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
        axes[1].plot(t["tc"][ok2], t["da_signed"][ok2],
                     color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
    axes[0].set_xlabel("$t'$", fontsize=FS_LABEL)
    axes[0].set_ylabel(r"$\langle\delta_1\rangle$ [°]", fontsize=FS_LABEL)
    axes[1].set_xlabel("$t'$", fontsize=FS_LABEL)
    axes[1].set_ylabel(r"$\langle\delta_2\rangle$ [°]", fontsize=FS_LABEL)
    for ax in axes:
        style(ax)
    outside_legend(fig, axes)
    save(fig, out_dir / "Fig04a_delta_vs_tprime_signed")
    plt.close(fig); print("Fig04a saved")

    # ─── Fig 4b: |delta1| and |delta2| vs t' with insets (excl. α=30°) ─────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for alpha in TARGET_ANGLES:
        if alpha not in tb["all"]:
            continue
        t = tb["all"][alpha]
        ok1 = np.isfinite(t["dlt_abs"]); ok2 = np.isfinite(t["da"])
        axes[0].plot(t["tc"][ok1], t["dlt_abs"][ok1],
                     color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
        axes[1].plot(t["tc"][ok2], t["da"][ok2],
                     color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
    axes[0].set_xlabel("$t'$", fontsize=FS_LABEL)
    axes[0].set_ylabel(r"$\langle|\delta_1|\rangle$ [°]", fontsize=FS_LABEL)
    axes[1].set_xlabel("$t'$", fontsize=FS_LABEL)
    axes[1].set_ylabel(r"$\langle|\delta_2|\rangle$ [°]", fontsize=FS_LABEL)
    # Insets: all angles except α=30°, zoomed y-range
    for ax, key, ylim in [(axes[0], "dlt_abs", (0, 6)), (axes[1], "da", (0, 2.2))]:
        axin = ax.inset_axes([0.36, 0.48, 0.54, 0.50])
        for alpha in TARGET_ANGLES:
            if alpha not in tb["all"]:
                continue
            t = tb["all"][alpha]
            ok = np.isfinite(t[key])
            axin.plot(t["tc"][ok], t[key][ok], color=ANGLE_COLOR[alpha], lw=1.5)
        axin.set_xlim(0, 1)
        axin.set_ylim(*ylim)
        axin.tick_params(labelsize=FS_TICK - 4)
        axin.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ind = ax.indicate_inset(
            bounds=[0, 0, 1, ylim[1]], inset_ax=axin, edgecolor="0.45", lw=1.0
        )
        for c in ind.connectors:
            c.set_visible(False)
    for ax in axes:
        style(ax)
    outside_legend(fig, axes)
    save(fig, out_dir / "Fig04b_delta_vs_tprime_abs")
    plt.close(fig); print("Fig04b saved")

    # ─── Fig 5: |delta| vs rho, (2,3); row0 ylim 0-6, row1 ylim 0-3 ──────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.85, top=0.97, bottom=0.08,
                        hspace=0.05, wspace=0.22)
    for col, dl in enumerate(DENSITY_LABELS):
        for alpha in TARGET_ANGLES:
            tp, v, acc, rho, da, dlt = raw[dl][alpha]
            if len(rho) == 0:
                continue
            c, mk = ANGLE_COLOR[alpha], ANGLE_MARKER[alpha]
            ok1 = np.isfinite(dlt)
            if ok1.any():
                mr, md = median_binned(rho[ok1], np.abs(dlt[ok1]),
                                       rho_edges, rho_centres)
                if len(mr):
                    axes[0, col].scatter(mr, md, color=c, marker=mk, s=64 if alpha not in [60,120,180] else 128,
                                         label=f"α={alpha}°")
            ok2 = np.isfinite(da)
            if ok2.any():
                mr2, md2 = median_binned(rho[ok2], np.abs(da[ok2]),
                                          rho_edges, rho_centres)
                if len(mr2):
                    axes[1, col].scatter(mr2, md2, color=c, marker=mk, s=64 if alpha not in [60,120,180] else 128,
                                          label=f"α={alpha}°")
        # x-label on both rows; y-label on left column only
        axes[0, col].set_xlabel(DLABEL_DISPLAY[dl], fontsize=FS_LABEL)
        axes[1, col].set_xlabel(DLABEL_DISPLAY[dl], fontsize=FS_LABEL)
        if col == 0:
            axes[0, col].set_ylabel(r"$|\delta_1|$ [°]", fontsize=FS_LABEL)
            axes[1, col].set_ylabel(r"$|\delta_2|$ [°]", fontsize=FS_LABEL)
        axes[0, col].set_ylim(0, 7)
        axes[1, col].set_ylim(0, 2.5)
        for row in range(2):
            axes[row, col].set_box_aspect(1)
            style(axes[row, col])
    # one legend outside right
    axes[0, -1].legend(handles=angle_legend_handles(), loc="upper left",
                       bbox_to_anchor=(1.02, 1.0), fontsize=FS_LEGEND,
                       frameon=True, borderaxespad=0)
    save(fig, out_dir / "Fig05_delta_vs_rho")
    plt.close(fig); print("Fig05 saved")

    # ─── Fig 6: delta vs v, t' color, grouped angles; one angle legend at bottom
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(left=0.09, right=0.87, top=0.97, bottom=0.09,
                        hspace=0.25, wspace=0.22)
    for col, group in enumerate(TCOLOR_GROUPS):
        for alpha in group:
            if alpha not in tb["all"]:
                continue
            t = tb["all"][alpha]; mk = ANGLE_MARKER[alpha]
            ok1 = np.isfinite(t["dlt_abs"]) & np.isfinite(t["v"])
            if ok1.sum() >= 2:
                axes[0, col].scatter(t["v"][ok1], t["dlt_abs"][ok1],
                                     c=t["tc"][ok1], cmap=T_CMAP, vmin=0, vmax=1,
                                     s=85, marker=mk, alpha=0.85)
                axes[0,col].set_xlim(1.2,1.5)
                axes[0,col].set_ylim(1,6)
            ok2 = np.isfinite(t["da"]) & np.isfinite(t["v"])
            if ok2.sum() >= 2:
                axes[1, col].scatter(t["v"][ok2], t["da"][ok2],
                                     c=t["tc"][ok2], cmap=T_CMAP, vmin=0, vmax=1,
                                     s=85, marker=mk, alpha=0.85)
                axes[1,col].set_xlim(1.2,1.5)
                axes[1,col].set_ylim(0.3,2.3)
        # x-label on both rows; y-label left column only
        axes[0, col].set_xlabel(r"$\langle v\rangle$ [m/s]", fontsize=FS_LABEL)
        axes[1, col].set_xlabel(r"$\langle v\rangle$ [m/s]", fontsize=FS_LABEL)
        if col == 0:
            axes[0, col].set_ylabel(r"$\langle|\delta_1|\rangle$ [°]", fontsize=FS_LABEL)
            axes[1, col].set_ylabel(r"$\langle|\delta_2|\rangle$ [°]", fontsize=FS_LABEL)
        for row in range(2):
            axes[row, col].set_box_aspect(1)
            style(axes[row, col])
        # per-panel shape legend on both rows; bottom-right for 120/150/180 col
        group_handles = [
            Line2D([0], [0], marker=ANGLE_MARKER[a], color="black",
                   linestyle="None", markersize=8, label=f"α={a}°")
            for a in group
        ]
        leg_loc = "lower right" if col == 2 else "upper right"
        for row in range(2):
            axes[row, col].legend(handles=group_handles, loc=leg_loc,
                                  fontsize=FS_LEGEND, frameon=True)
    manual_tcbar(fig)
    save(fig, out_dir / "Fig06_delta_vs_v_tcolor")
    plt.close(fig); print("Fig06 saved")

    # ─── Fig 7: <|delta1|> vs <rho>, t' color, 7 equal panels (2,3,2) ─────────
    fig7 = plt.figure(figsize=(16, 11))
    axes7 = make_232_axes(fig7, sharey=True)
    for pi, alpha in enumerate(TARGET_ANGLES):
        ax = axes7[pi]
        for dl in DENSITY_LABELS:
            if alpha not in tb[dl]:
                continue
            t = tb[dl][alpha]
            ok = np.isfinite(t["r"]) & np.isfinite(t["dlt_abs"])
            if ok.sum() < 2:
                continue
            ax.scatter(t["r"][ok], t["dlt_abs"][ok], c=t["tc"][ok],
                       cmap=T_CMAP, vmin=0, vmax=1,
                       s=50, marker=DLABEL_MARKER[dl], alpha=0.85)
            ax.set_ylim(1,6)
        ax.set_xlabel(r"$\langle\rho\rangle$", fontsize=FS_LABEL)
        if pi in (0, 2, 5):
            ax.set_ylabel(r"$\langle|\delta_1|\rangle$ [°]", fontsize=FS_LABEL)
        panel_tag(ax, f"α={alpha}°", loc="br" if alpha >= 60 else "tr")
        style(ax)
    manual_tcbar_h(fig7)
    fig7.legend(handles=shape_handles, loc="upper right",
                fontsize=FS_LEGEND, frameon=True, ncol=1,
                bbox_to_anchor=(0.95, 0.98))
    save(fig7, out_dir / "Fig07_delta1_rho_tcolor_perangle")
    plt.close(fig7); print("Fig07 saved")

    # ─── Fig 8: <|delta2|> vs <rho>, t' color, 7 equal panels (2,3,2) ────────
    fig8 = plt.figure(figsize=(16, 11))
    axes8 = make_232_axes(fig8, sharey=True)
    for pi, alpha in enumerate(TARGET_ANGLES):
        ax = axes8[pi]
        for dl in DENSITY_LABELS:
            if alpha not in tb[dl]:
                continue
            t = tb[dl][alpha]
            ok = np.isfinite(t["r"]) & np.isfinite(t["da"])
            if ok.sum() < 2:
                continue
            ax.scatter(t["r"][ok], t["da"][ok], c=t["tc"][ok],
                       cmap=T_CMAP, vmin=0, vmax=1,
                       s=50, marker=DLABEL_MARKER[dl], alpha=0.85)
            ax.set_ylim(0.3,2.5)
        ax.set_xlabel(r"$\langle\rho\rangle$", fontsize=FS_LABEL)
        if pi in (0, 2, 5):
            ax.set_ylabel(r"$\langle|\delta_2|\rangle$ [°]", fontsize=FS_LABEL)
        panel_tag(ax, f"α={alpha}°", loc="br" if alpha >= 60 else "tr")
        style(ax)
    manual_tcbar_h(fig8)
    fig8.legend(handles=shape_handles, loc="upper right",
                fontsize=FS_LEGEND, frameon=True, ncol=1,
                bbox_to_anchor=(0.95, 0.98))
    save(fig8, out_dir / "Fig08_delta2_rho_tcolor_perangle")
    plt.close(fig8); print("Fig08 saved")

    # ─── Fig 9: PDF(a) as KDE lines, all angles ───────────────────────────────
    all_acc_cat = np.concatenate([raw["all"][a][2] for a in TARGET_ANGLES])
    acc_fin     = all_acc_cat[np.isfinite(all_acc_cat)]
    lo, hi      = (np.percentile(acc_fin, [0.5, 99.5]) if len(acc_fin) else (-3.0, 3.0))
    x_pdf       = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(8, 6))
    for alpha in TARGET_ANGLES:
        fin = raw["all"][alpha][2]; fin = fin[np.isfinite(fin)]
        if len(fin) < 5:
            continue
        kde = gaussian_kde(fin)
        ax.plot(x_pdf, kde(x_pdf), color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
    ax.set_xlabel("$a$ [m/s²]", fontsize=FS_LABEL)
    ax.set_ylabel("PDF", fontsize=FS_LABEL)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=FS_LEGEND, frameon=True, borderaxespad=0)
    style(ax)
    fig.tight_layout()
    save(fig, out_dir / "Fig09_pdf_acc_perangle")
    plt.close(fig); print("Fig09 saved")

    # ─── Fig 10: a vs t' ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for alpha in TARGET_ANGLES:
        if alpha not in tb["all"]:
            continue
        t = tb["all"][alpha]
        ok_a = np.isfinite(t["acc"])
        axes[0].plot(t["tc"][ok_a], t["acc"][ok_a],
                     color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
        ok_v = np.isfinite(t["v"])
        axes[1].plot(t["tc"][ok_v], t["v"][ok_v],
                     color=ANGLE_COLOR[alpha], lw=2, label=f"α={alpha}°")
    axes[0].set_xlabel("$t'$", fontsize=FS_LABEL)
    axes[0].set_ylabel(r"$\langle a\rangle$ [m/s²]", fontsize=FS_LABEL)
    axes[1].set_xlabel("$t'$", fontsize=FS_LABEL)
    axes[1].set_ylabel(r"$\langle v\rangle$ [m/s]", fontsize=FS_LABEL)
    for ax in axes:
        style(ax)
    outside_legend(fig, axes)
    save(fig, out_dir / "Fig10_v_acc_vs_tprime")
    plt.close(fig); print("Fig10 saved")

    # ─── Fig 11: a vs rho, 3 panels, per-angle medians; ylim [-0.1, 0.1] ──────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for col, dl in enumerate(DENSITY_LABELS):
        ax = axes[col]
        for alpha in TARGET_ANGLES:
            tp, v, acc, rho, da, dlt = raw[dl][alpha]
            if len(rho) == 0:
                continue
            ok = np.isfinite(acc)
            if ok.any():
                mr, ma = median_binned(rho[ok], acc[ok], rho_edges, rho_centres)
                if len(mr):
                    ax.scatter(mr, ma, color=ANGLE_COLOR[alpha],
                               marker=ANGLE_MARKER[alpha], s=64 if alpha not in [60,120,180] else 90, label=f"α={alpha}°")
        ax.set_xlabel(DLABEL_DISPLAY[dl], fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel("$a$ [m/s²]", fontsize=FS_LABEL)
        ax.set_ylim(-0.15, 0.15)
        ax.set_box_aspect(1)
        style(ax)
    outside_legend(fig, axes)
    save(fig, out_dir / "Fig11_acc_vs_rho_perangle")
    plt.close(fig); print("Fig11 saved")

    print(f"\nAll article figures → {out_dir}")


if __name__ == "__main__":
    make_figures()
