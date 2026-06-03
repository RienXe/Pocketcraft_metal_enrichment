"""Generate the PocketCraft Metal Enrichment EGNN architecture diagram.

Design rules that prevent arrow-through-box problems:
  - All sequential main-column boxes connected by short straight vertical arrows.
  - Metal-token column sits at x=13.5 (well outside main-column x range 5.5-10.5).
  - Metal Embedding placed at the SAME y as Merge box → single horizontal arrow,
    no diagonal, no crossing.
  - Arrows are drawn at zorder=4; box backgrounds at zorder=3 so if they share
    a boundary only the arrowhead tip touches the edge, not hidden behind a box.
  - All arrow start/end coordinates are computed from the box's actual edge, not
    the box centre, keeping arrowheads at the boundary.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── canvas ────────────────────────────────────────────────────────────────────
FW, FH = 16, 22
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor("#F7F9FC")
ax.set_facecolor("#F7F9FC")

# ── palette ───────────────────────────────────────────────────────────────────
C_IN   = "#D6EAF8"
C_ENC  = "#D5F5E3"
C_EGNN = "#FFFBF0"
C_LAY  = "#FAD7A0"
C_HEAD = "#E8DAEF"
C_OUT  = "#FDEDEC"
C_NMS  = "#EBF5FB"
C_FIN  = "#D6EAF8"
CB     = "#2C3E50"   # border / text colour
CA     = "#34495E"   # arrow colour
CD     = "#C0984C"   # dashed EGNN border colour

# ── box dimensions ────────────────────────────────────────────────────────────
BH = 0.78    # standard box height
SH = 0.65    # short box height (merge, NMS, output CIF)
HH = 0.72    # EGNN-inner box height
MW = 5.0     # main-column box width
SW = 4.0     # side-column (metal) box width
IW = 8.5     # EGNN inner box width
OW = 4.6     # output-head box width
NW = 5.6     # NMS / output CIF box width

# ── column centres ────────────────────────────────────────────────────────────
XM  = 8.0    # main column
XS  = 13.5   # metal side column
# EGNN outer box: left=3.0, right=13.0  (centred on XM=8)

# ── y coordinates (top → bottom means decreasing y in matplotlib) ─────────────
Y_TITLE   = 21.3
Y_PROT    = 20.2    # Protein Structure  (also Metal Type Token)
Y_HEAVY   = 18.8    # Heavy Atom Extraction
Y_PROJ    = 17.5    # Protein Projection
Y_MERGE   = 16.0    # Merge "h←h+m"  (also Metal Embedding)
Y_RGRAPH  = 14.7    # Radius Graph

# EGNN outer box
EGNN_TOP  = 14.0
EGNN_BOT  = 7.4
# EGNN inner element centres
Y_ETITLE  = 13.65
Y_RBF     = 12.80
Y_EDGE    = 11.65
Y_COORD   = 10.50
Y_NODE    = 9.35
Y_RES     = 8.45   # residual text (not a box)

# Output section
Y_ESPLIT  = 6.95   # horizontal split line
Y_SHEAD   = 6.45   # Score Head / Offset Head centres
Y_PRED    = 5.30   # P(binding) / Predicted Coords
Y_NMS     = 4.10   # NMS
Y_OCIF    = 2.95   # Output CIF
Y_FOOT    = 2.20

XSH = 4.5    # Score Head x
XOH = 11.5   # Offset Head x

# ── helper: rounded box ───────────────────────────────────────────────────────
def rbox(cx, cy, w, h, label, sub="", color="#FFF",
         fs=9.5, bold=False, bz=3, tz=4, radius=0.18):
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        facecolor=color, edgecolor=CB, linewidth=1.3, zorder=bz,
    )
    ax.add_patch(p)
    yy = cy + (0.14 if sub else 0)
    ax.text(cx, yy, label, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=CB, zorder=tz)
    if sub:
        ax.text(cx, cy - 0.20, sub, ha="center", va="center",
                fontsize=7.8, color="#555", style="italic", zorder=tz)

# ── helper: vertical down-arrow between two box edges ────────────────────────
def varr(cx, y_from, y_to, z=4):
    """Straight vertical arrow from (cx, y_from) to (cx, y_to)."""
    ax.annotate("", xy=(cx, y_to + 0.03), xytext=(cx, y_from - 0.03),
                arrowprops=dict(arrowstyle="-|>", color=CA, lw=1.6,
                                mutation_scale=14),
                zorder=z)

# ── helper: horizontal arrow ──────────────────────────────────────────────────
def harr(x_from, y, x_to, z=4):
    """Horizontal arrow (y constant)."""
    dx = 1 if x_to > x_from else -1
    ax.annotate("", xy=(x_to + dx * 0.03, y), xytext=(x_from - dx * 0.03, y),
                arrowprops=dict(arrowstyle="-|>", color=CA, lw=1.6,
                                mutation_scale=14),
                zorder=z)

# ── helper: diagonal arrow ────────────────────────────────────────────────────
def darr(x0, y0, x1, y1, z=4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=CA, lw=1.5,
                                mutation_scale=13),
                zorder=z)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(XM, Y_TITLE, "PocketCraft Metal Enrichment — EGNN Architecture",
        ha="center", va="center", fontsize=13.5, fontweight="bold",
        color=CB, zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT ROW  (Protein Structure left, Metal Token right)
# ══════════════════════════════════════════════════════════════════════════════
rbox(XM,  Y_PROT, MW, BH, "Protein Structure", ".pdb / .cif",
     C_IN, bold=True, fs=10.5)
rbox(XS,  Y_PROT, SW, BH, "Metal Type Token",
     "ZN / CA / MG / FE / MN / CU / NI / CO / NA / K",
     C_IN, bold=True, fs=9.0)

# ══════════════════════════════════════════════════════════════════════════════
# PROTEIN PATH  (main column, straight verticals)
# ══════════════════════════════════════════════════════════════════════════════
varr(XM, Y_PROT - BH/2, Y_HEAVY + BH/2)
rbox(XM, Y_HEAVY, MW, BH, "Heavy Atom Extraction",
     "N heavy atoms  ×  (xyz  +  32-dim feature vector)", C_ENC)

varr(XM, Y_HEAVY - BH/2, Y_PROJ + BH/2)
rbox(XM, Y_PROJ, MW, BH, "Protein Projection",
     "Linear(32 → 96)  +  LayerNorm    →    h  [N × 96]", C_ENC)

varr(XM, Y_PROJ - BH/2, Y_MERGE + SH/2)

# ══════════════════════════════════════════════════════════════════════════════
# METAL PATH  (side column, one long vertical + one short horizontal)
# ══════════════════════════════════════════════════════════════════════════════
#  Long vertical: Metal Token bottom → Metal Embedding top
#  (x=13.5 is entirely outside main-column x range 5.5–10.5 → no crossing)
varr(XS, Y_PROT - BH/2, Y_MERGE + SH/2)
rbox(XS, Y_MERGE, SW, SH, "Metal Embedding",
     "Embedding(11, 96)    →    m  [96]", C_ENC)

#  Short horizontal: Metal Embedding left edge → Merge box right edge (same y)
harr(XS - SW/2, Y_MERGE, XM + MW/2)

# ══════════════════════════════════════════════════════════════════════════════
# MERGE BOX
# ══════════════════════════════════════════════════════════════════════════════
rbox(XM, Y_MERGE, MW, SH,
     "h  ←  h + m [ metal_type ]",
     "broadcast metal embedding to every node", C_ENC, fs=9.5)

# ══════════════════════════════════════════════════════════════════════════════
# RADIUS GRAPH
# ══════════════════════════════════════════════════════════════════════════════
varr(XM, Y_MERGE - SH/2, Y_RGRAPH + BH/2)
rbox(XM, Y_RGRAPH, MW + 0.6, BH, "Radius Graph Construction",
     "r = 4.0 Å,  max_neighbors = 24    →    edge_index  [2 × E]", C_ENC)

# ══════════════════════════════════════════════════════════════════════════════
# EGNN OUTER DASHED BOX  ← drawn FIRST (low zorder) so inner boxes sit on top
# ══════════════════════════════════════════════════════════════════════════════
egnn_patch = FancyBboxPatch(
    (3.0, EGNN_BOT), 10.0, EGNN_TOP - EGNN_BOT,
    boxstyle="round,pad=0.12,rounding_size=0.35",
    facecolor=C_EGNN, edgecolor=CD,
    linewidth=2.5, linestyle="--", zorder=2,
)
ax.add_patch(egnn_patch)

# Enter EGNN
varr(XM, Y_RGRAPH - BH/2, Y_RBF + HH/2)

ax.text(XM, Y_ETITLE, "× 6    EGNN  Layers",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#7D4E00", zorder=4)

# ── EGNN inner boxes (all at XM, zorder=3 so they sit above dashed background)
rbox(XM, Y_RBF, IW, HH, "RBF Distance Expansion",
     r"$d_{ij}\ \rightarrow\ \exp\!\left(-\,(d_{ij}-\mu_k)^2/\sigma^2\right)$"
     "    [32 Gaussian RBFs]",
     C_LAY, bz=3, tz=4)

varr(XM, Y_RBF - HH/2, Y_EDGE + HH/2)
rbox(XM, Y_EDGE, IW, HH, "Edge MLP    →    message  mᵢⱼ",
     r"$[\,h_i \;\|\; h_j \;\|\; \mathrm{rbf}_{ij}\,]$"
     "   →   Linear → SiLU → Linear   [96]",
     C_LAY, bz=3, tz=4)

varr(XM, Y_EDGE - HH/2, Y_COORD + HH/2)
rbox(XM, Y_COORD, IW, HH, "Coordinate Update",
     r"$\Delta p_i = \sum_j \tanh\!\left(\mathrm{CoordMLP}(m_{ij})\right)"
     r"\cdot (p_i - p_j)\,/\,\deg_i$   [bounded]",
     C_LAY, bz=3, tz=4)

varr(XM, Y_COORD - HH/2, Y_NODE + HH/2)
rbox(XM, Y_NODE, IW, HH, "Node MLP    →    feature update  Δhᵢ",
     r"$[\,h_i \;\|\; \mathrm{mean}_j(m_{ij})\,]$"
     "   →   Linear → SiLU → Dropout → Linear   [96]",
     C_LAY, bz=3, tz=4)

varr(XM, Y_NODE - HH/2, Y_RES + 0.12)
ax.text(XM, Y_RES,
        r"$h \leftarrow h + \Delta h$"
        r"              $p \leftarrow p + \Delta p$"
        "     (residual)",
        ha="center", va="center", fontsize=10, color=CB, zorder=4)

# ── Recurrence arc  (right side, INSIDE the dashed box gap)
REC_X = 12.55   # just inside EGNN right edge (inner box edge + gap)
ax.annotate(
    "", xy=(REC_X, Y_RBF), xytext=(REC_X, Y_NODE - HH/2),
    arrowprops=dict(
        arrowstyle="-|>", color=CD, lw=2.0, mutation_scale=14,
        connectionstyle="arc3,rad=-0.45",
    ),
    zorder=4,
)
ax.text(REC_X + 0.55, (Y_RBF + Y_NODE) / 2, "next\nlayer",
        ha="center", va="center", fontsize=9, color="#7D4E00",
        style="italic", zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# EXIT EGNN  →  horizontal split  →  Score Head  |  Offset Head
# ══════════════════════════════════════════════════════════════════════════════
varr(XM, EGNN_BOT, Y_ESPLIT + 0.04)

# Horizontal split line from XSH to XOH
ax.plot([XSH, XOH], [Y_ESPLIT, Y_ESPLIT], color=CA, lw=1.6, zorder=4)

# Down-arrows from split line to heads
varr(XSH, Y_ESPLIT, Y_SHEAD + HH/2)
varr(XOH, Y_ESPLIT, Y_SHEAD + HH/2)

rbox(XSH, Y_SHEAD, OW, HH, "Score Head",
     "LN → Linear → SiLU → Dropout → Linear → logit", C_HEAD, bz=3, tz=4)
rbox(XOH, Y_SHEAD, OW, HH, "Offset Head",
     "LN → Linear → SiLU → Linear → Tanh × 2.5 Å", C_HEAD, bz=3, tz=4)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION BOXES
# ══════════════════════════════════════════════════════════════════════════════
varr(XSH, Y_SHEAD - HH/2, Y_PRED + SH/2)
varr(XOH, Y_SHEAD - HH/2, Y_PRED + SH/2)

rbox(XSH, Y_PRED, OW, SH, "P(binding) per atom",
     "sigmoid(logit)", C_OUT, bold=True, bz=3, tz=4)
rbox(XOH, Y_PRED, OW, SH, "Predicted Coordinates",
     "updated_pos + offset", C_OUT, bold=True, bz=3, tz=4)

# ══════════════════════════════════════════════════════════════════════════════
# NMS  (two diagonal arrows converging)
# ══════════════════════════════════════════════════════════════════════════════
NMS_L = XM - NW/2   # left edge of NMS box
NMS_R = XM + NW/2   # right edge

darr(XSH, Y_PRED - SH/2, NMS_L + 0.4, Y_NMS + SH/2)
darr(XOH, Y_PRED - SH/2, NMS_R - 0.4, Y_NMS + SH/2)

rbox(XM, Y_NMS, NW, SH,
     "Non-Maximum Suppression",
     "greedy,  2.0 Å minimum distance per metal type",
     C_NMS, bz=3, tz=4)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT CIF
# ══════════════════════════════════════════════════════════════════════════════
varr(XM, Y_NMS - SH/2, Y_OCIF + SH/2)
rbox(XM, Y_OCIF, NW, SH,
     "Output CIF",
     "HETATM  chain M  |  B-factor = confidence × 100",
     C_FIN, bold=True, bz=3, tz=4)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
ax.text(XM, Y_FOOT,
        "Forward pass repeated for each of 10 metal types — results pooled before NMS",
        ha="center", va="center", fontsize=9, color="#555",
        style="italic", zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
handles = [
    mpatches.Patch(facecolor=C_IN,  edgecolor=CB, label="Input"),
    mpatches.Patch(facecolor=C_ENC, edgecolor=CB, label="Encoding / graph"),
    mpatches.Patch(facecolor=C_LAY, edgecolor=CB, label="EGNN layer ops"),
    mpatches.Patch(facecolor=C_HEAD,edgecolor=CB, label="Output heads"),
    mpatches.Patch(facecolor=C_OUT, edgecolor=CB, label="Predictions"),
]
ax.legend(handles=handles, loc="lower left", fontsize=9,
          framealpha=0.9, bbox_to_anchor=(0.0, 0.02))

plt.tight_layout(pad=0.2)
plt.savefig("architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved architecture.png")
