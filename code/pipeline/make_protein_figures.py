"""Protein renderings for the paper (requires PyMOL, pymol-open-source).

Style: PyMOL ray-traced spheres, soft ambient occlusion, white background, orthoscopic
(perspective only for the bounding-box figure); atoms coloured by the SADIC depth index
(RdYlBu_r, 0-2) or by DEPTH (RdYlBu, exposed = red). Panels are composed with matplotlib
at printed size (3.25 in, 8 pt text) and written to outputs/figures/notables/:

  fig_7rco_alignment.pdf   7RCO before (AABB) / after (OBB) the alignment heuristic
  fig_6k43_cutaway.pdf     6K43 with the front half removed (internal cavity)
  fig_notables_corr.pdf    2A3W / 5T4O (strongest / weakest correlation with DEPTH)
  fig_7ljs_sadic_depth.pdf 7LJS coloured by SADIC and by DEPTH, same view (two-column figure)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plotstyle
from sadicpipe.align import align
from sadicpipe.runner import protein_seed
from sadicpipe import vdw
from analysis_lib import load_depth_tool, atom_keys
from pymol import cmd, cgo

from paths import ANALYSIS, FIGURES, NPZ_DIR, PDB_DIR, RUNS
OUT = os.path.join(FIGURES, "notables")
TMP = os.path.join(ANALYSIS, "renders_pymol")
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
S = 1600  # render size in pixels
SADIC_CMAP, DEPTH_CMAP = "RdYlBu_r", "RdYlBu"


# ---------------------------------------------------------------- data
def load(pid):
    d = np.load(f"{NPZ_DIR}/{pid}.npz")
    s = np.load(f"{RUNS}/morph_transph_aligned/depth_indexes/{pid}.npy")
    return d, d["coords"].astype(float), s


def pca_frame(x):
    c = x - x.mean(0); _, V = np.linalg.eigh(np.cov(c.T)); V = V[:, ::-1]
    if np.linalg.det(V) < 0: V[:, 2] *= -1
    return c @ V


def write_pdb(pid, coords, vals, name):
    lines = [l for l in open(f"{PDB_DIR}/{pid}.pdb") if l.startswith(("ATOM", "HETATM"))]
    assert len(lines) == len(coords) == len(vals)
    path = f"{TMP}/{name}.pdb"
    with open(path, "w") as f:
        for l, (a, b, c), v in zip(lines, coords, vals):
            v = -1.0 if not np.isfinite(v) else v
            f.write(f"{l[:30]}{a:8.3f}{b:8.3f}{c:8.3f}{l[54:60]}{v:6.2f}{l[66:].rstrip()}\n")
        f.write("END\n")
    return path


# ---------------------------------------------------------------- pymol
def setup(pdb, perspective=False):
    cmd.reinitialize(); cmd.load(pdb, "p"); cmd.hide("everything"); cmd.show("spheres", "p")
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1)
    cmd.set("orthoscopic", 0 if perspective else 1); cmd.set("field_of_view", 30)
    cmd.set("depth_cue", 0); cmd.set("fog", 0); cmd.set("antialias", 2); cmd.set("hash_max", 300)
    cmd.set("ambient_occlusion_mode", 1); cmd.set("ambient_occlusion_scale", 15); cmd.set("ambient_occlusion_smooth", 15)
    cmd.set("ray_shadows", 0); cmd.set("specular", 0.15); cmd.set("ambient", 0.4); cmd.set("direct", 0.55); cmd.set("reflect", 0.35)


def colour(cmap, lo, hi, N=64):
    cm = matplotlib.colormaps[cmap]
    cmd.set_color("nodata", [0.75, 0.75, 0.75]); cmd.color("nodata", "p")
    for i in range(N):
        cmd.set_color(f"c{i}", list(cm(i / (N - 1))[:3]))
        a = lo + (hi - lo) * i / N; b = lo + (hi - lo) * (i + 1) / N
        if i == 0: a = min(lo, 0) - 0.001
        if i == N - 1: b = 1e9
        cmd.color(f"c{i}", f"p and b > {a:.4f} and b <= {b:.4f}")


def png(name):
    path = f"{TMP}/{name}.png"; cmd.png(path, width=S, height=S, ray=1); return path


EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def corners(lo, hi):
    return np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])


def box_cgo(c, edge_rgb, face_rgb, alpha=0.12, r=0.5):
    c = np.asarray(c); obj = []
    for i, j in EDGES: obj += [cgo.CYLINDER, *c[i], *c[j], r, *edge_rgb, *edge_rgb]
    obj += [cgo.ALPHA, alpha, cgo.BEGIN, cgo.TRIANGLES, cgo.COLOR, *face_rgb]
    ctr = c.mean(0)
    for f in FACES:
        q = c[list(f)]; n = np.cross(q[1] - q[0], q[2] - q[0]); n /= np.linalg.norm(n)
        if np.dot(n, q.mean(0) - ctr) < 0: n = -n
        for tri in ((0, 1, 2), (0, 2, 3)):
            obj += [cgo.NORMAL, *n]
            for t in tri: obj += [cgo.VERTEX, *q[t]]
    return obj + [cgo.END, cgo.ALPHA, 1.0]


# ---------------------------------------------------------------- composition
def crop(path, box=None, pad=20):
    im = Image.open(path).convert("RGB"); b = box or ImageOps.invert(im).getbbox()
    return np.asarray(im.crop((max(b[0] - pad, 0), max(b[1] - pad, 0), min(b[2] + pad, im.width), min(b[3] + pad, im.height))))


def bbox(path):
    return ImageOps.invert(Image.open(path).convert("RGB")).getbbox()


def colorbar(fig, rect, cmap, lo, hi, label, ticks, orientation="horizontal"):
    cax = fig.add_axes(rect)
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(lo, hi), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation=orientation, ticks=ticks); cb.set_label(label); cb.outline.set_linewidth(0.5)
    cax.tick_params(width=0.5, length=2)


def panels(images, titles, out, height=2.35, sadic_bar=True, legend=None):
    fig = plt.figure(figsize=(plotstyle.W, height))
    n = len(images); w = 1.0 / n
    for k, (im, t) in enumerate(zip(images, titles)):
        ax = fig.add_axes([0.01 + k * w, 0.27, w - 0.02, 0.63]); ax.imshow(im); ax.axis("off")
        if t: ax.set_title(t, fontsize=7.5, pad=2)
    if legend: fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0.05), frameon=False, fontsize=7)
    if sadic_bar:
        colorbar(fig, [0.2, 0.16, 0.6, 0.04], SADIC_CMAP, 0, 2, "SADIC depth index", [0, 0.5, 1, 1.5, 2])
    fig.savefig(f"{OUT}/{out}.pdf", dpi=300); fig.savefig(f"{TMP}/{out}.png", dpi=150); plt.close(fig)
    print("saved", f"{OUT}/{out}.pdf")


def main():
    plotstyle.apply(); plt.rcParams["figure.constrained_layout.use"] = False

    # 7RCO: original frame (AABB) and OBB frame from the paper's heuristic, same seed as the runs
    d, x, s = load("7rco")
    rad = np.array([vdw.radius_of(e, None) for e in d["element"]])
    a, _ = align(x, method="approximate", seed=protein_seed("7rco"))
    Rt, *_ = np.linalg.lstsq(x, a, rcond=None)
    ext = lambda p: ((p - rad[:, None]).min(0), (p + rad[:, None]).max(0))
    lo, hi = ext(x); c0 = (lo + hi) / 2
    la, ha = ext(a); ca = (la + ha) / 2
    perm = np.argsort(-(ha - la))[[0, 2, 1]]          # longest side along x, shortest side vertical
    red = 100 * (1 - np.prod(ha - la) / np.prod(hi - lo))
    before = write_pdb("7rco", x - c0, s, "7rco_before"); after = write_pdb("7rco", (a - ca)[:, perm], s, "7rco_after")
    setup(before, perspective=True); colour(SADIC_CMAP, 0, 2)
    cmd.load_cgo(box_cgo(corners(lo - c0, hi - c0), [0.55] * 3, [0.85] * 3), "aabb")
    cmd.reset(); cmd.turn("x", -28); cmd.turn("y", 35); cmd.zoom("all", 3, complete=1); view = cmd.get_view()
    p1 = png("7rco_before")
    setup(after, perspective=True); colour(SADIC_CMAP, 0, 2)
    cmd.load_cgo(box_cgo(corners((la - ca)[perm], (ha - ca)[perm]), [0.1] * 3, [0.45] * 3), "obb")
    cmd.set_view(view); p2 = png("7rco_after")
    b1, b2 = bbox(p1), bbox(p2); bb = (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))
    panels([crop(p1, bb), crop(p2, bb)], ["Before alignment (AABB)", f"After alignment (OBB, $-${red:.0f}%)"], "fig_7rco_alignment")
    print(f"7RCO box volume reduction: {red:.1f}%")

    # 6K43: front half removed
    d, x, s = load("6k43"); pdb = write_pdb("6k43", pca_frame(x), s, "6k43")
    setup(pdb); colour(SADIC_CMAP, 0, 2)
    cmd.alter_state(1, "p", "q = 1.0 if z > 0 else 0.0"); cmd.hide("spheres", "p and q > 0.5")
    cmd.reset(); cmd.zoom("p", 2, complete=1); p = png("6k43")
    fig = plt.figure(figsize=(plotstyle.W, 3.0))
    ax = fig.add_axes([0.15, 0.25, 0.7, 0.73]); ax.imshow(crop(p)); ax.axis("off")
    colorbar(fig, [0.2, 0.13, 0.6, 0.03], SADIC_CMAP, 0, 2, "SADIC depth index", [0, 0.5, 1, 1.5, 2])
    fig.savefig(f"{OUT}/fig_6k43_cutaway.pdf", dpi=300); fig.savefig(f"{TMP}/fig_6k43_cutaway.png", dpi=150); plt.close(fig)
    print("saved", f"{OUT}/fig_6k43_cutaway.pdf")

    # 2A3W / 5T4O
    ims = []
    for pid in ["2a3w", "5t4o"]:
        d, x, s = load(pid); setup(write_pdb(pid, pca_frame(x), s, pid)); colour(SADIC_CMAP, 0, 2)
        cmd.reset(); cmd.zoom("p", 2, complete=1); ims.append(crop(png(pid)))
    panels(ims, ["2A3W", "5T4O"], "fig_notables_corr")

    # 7LJS: SADIC and DEPTH, same view
    d, x, s = load("7ljs"); r = pca_frame(x)
    k = atom_keys("7ljs"); dep = load_depth_tool("7ljs")
    m = k.reset_index().merge(dep[["chain", "reskey", "atomname", "depth"]], on=["chain", "reskey", "atomname"], how="left").sort_values("index")
    dv = m.depth.values.astype(float); dlo, dhi = np.nanmin(dv), np.nanmax(dv)
    setup(write_pdb("7ljs", r, s, "7ljs_sadic")); colour(SADIC_CMAP, 0, 2); cmd.reset(); cmd.zoom("p", 2, complete=1); pa = png("7ljs_sadic")
    setup(write_pdb("7ljs", r, dv, "7ljs_depth")); colour(DEPTH_CMAP, dlo, dhi); cmd.reset(); cmd.zoom("p", 2, complete=1); pb = png("7ljs_depth")
    fig = plt.figure(figsize=(7.0, 3.0))   # two-column figure: SADIC left, DEPTH right
    for i, (pth, cm, lo_, hi_, lab, tk, t) in enumerate([(pa, SADIC_CMAP, 0, 2, "SADIC depth index", [0, 0.5, 1, 1.5, 2], "SADIC"),
                                                        (pb, DEPTH_CMAP, dlo, dhi, "DEPTH (Å)", [5, 10, 15, 20, 25], "DEPTH")]):
        x0 = 0.02 + i * 0.5
        ax = fig.add_axes([x0, 0.02, 0.38, 0.88]); ax.imshow(crop(pth)); ax.axis("off"); ax.set_title(t, fontsize=8, pad=2)
        colorbar(fig, [x0 + 0.40, 0.18, 0.012, 0.6], cm, lo_, hi_, lab, tk, orientation="vertical")
    fig.savefig(f"{OUT}/fig_7ljs_sadic_depth.pdf", dpi=300); fig.savefig(f"{TMP}/fig_7ljs_sadic_depth.png", dpi=150); plt.close(fig)
    print("saved", f"{OUT}/fig_7ljs_sadic_depth.pdf", f"(DEPTH range {dlo:.2f}-{dhi:.2f} A, unmatched {np.isnan(dv).mean():.3f})")


if __name__ == "__main__":
    main()
