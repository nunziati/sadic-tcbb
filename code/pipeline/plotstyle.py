"""Shared figure style: every plot is drawn at its printed size
(W inches wide, included in LaTeX with width=W in), 8 pt labels and 7 pt ticks."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

W, H = 3.25, 2.3            # printed size in inches
C0, C1, C2 = "#1f77b4", "#d62728", "#ff7f0e"   # blue, red, orange
HIST = dict(color=C0, edgecolor="white", linewidth=0.4)


def apply():
    mpl.rcParams.update({
        "figure.figsize": (W, H), "figure.constrained_layout.use": True,
        "font.family": "DejaVu Sans", "font.size": 8, "mathtext.default": "regular",
        "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.5,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.5, "axes.formatter.useoffset": False,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _k(v, _pos=None):
    a = abs(v)
    if a >= 1e6: return f"{v / 1e6:g}M"
    if a >= 1e3: return f"{v / 1e3:g}k"
    return f"{v:g}"


def kfmt(*axes):
    """Thousands/millions tick labels (10k, 1.5M) instead of long numbers or 1e8 offsets."""
    for axis in axes:
        axis.set_major_formatter(FuncFormatter(_k))


def labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def save(fig, path):
    fig.savefig(path); plt.close(fig)
