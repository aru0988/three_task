"""Figure 1: Experimental Framework — clean two-stage layout, NO overlapping elements.
Generated at large native resolution; LaTeX scales to \columnwidth proportionally."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Large canvas — will be scaled down proportionally by LaTeX \includegraphics[width=\columnwidth]
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
ax.axis('off')

# ── Colours ──
C_STAGE1 = '#E3F2FD'; C_STAGE2 = '#FFF8E1'
C_CONDA  = '#E8F5E9'; C_CONDB  = '#FFEBEE'
C_EMB    = '#78909C'; C_SHARED = '#37474F'
C_SPEC   = '#455A64'; C_PROJ   = '#7B1FA2'
C_T4      = '#E65100'; C_TOWER  = '#00838F'
C_E1 = '#1E88E5'; C_E2 = '#43A047'; C_E3 = '#FB8C00'
C_CAUSAL = '#D32F2F'
FONT = 11; FONT_S = 10; FONT_TITLE = 14

def box(ax, x, y, w, h, text, bg, tc='white', fw='bold', fs=FONT, ec='#555'):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                       facecolor=bg, edgecolor=ec, linewidth=1.0, zorder=2)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, linespacing=1.15, zorder=3)

def arrow(ax, x1, y1, x2, y2, c='#555', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle='->', color=c, lw=lw))

# ═══════════════════════════════════════════
# STAGE 1  (y: 8.5 — 14)
# ═══════════════════════════════════════════
ax.text(10, 14.2, 'Stage 1:  Multi-task Pre-training  (all parameters frozen afterwards)',
        ha='center', fontsize=FONT_TITLE, fontweight='bold', color='#1565C0',
        bbox=dict(boxstyle='round', facecolor=C_STAGE1, edgecolor='#1565C0',
                  pad=0.6, lw=1.8), zorder=2)

# Feature Embedding
box(ax, 0.5, 10.6, 2.2, 1.5, 'Feature\nEmbedding\n(Frozen)', C_EMB)

# Shared Expert
box(ax, 3.5, 10.6, 2.2, 1.5, 'Shared\nExpert', C_SHARED)

# Arrow: Embedding → Shared
arrow(ax, 2.7, 11.35, 3.45, 11.35)

# Task-Specific Experts (vertical stack)
spe_y  = [12.5, 10.55, 8.6]
spe_lbl = ['Spec. Expert T1\n(Income)', 'Spec. Expert T2\n(Marital)', 'Spec. Expert T3\n(Education)']
for i in range(3):
    box(ax, 6.8, spe_y[i], 2.6, 1.0, spe_lbl[i], C_SPEC)
    # Embedding → Spec Expert
    arrow(ax, 2.7, 11.35, 6.75, spe_y[i] + 0.5)

# Task Embeddings
emb_y = [12.5, 10.55, 8.6]
emb_lbl = [r'Task Emb  $\mathbf{E}_1$', r'Task Emb  $\mathbf{E}_2$', r'Task Emb  $\mathbf{E}_3$']
emb_clr = [C_E1, C_E2, C_E3]
for i in range(3):
    box(ax, 10.5, emb_y[i], 2.6, 1.0, emb_lbl[i], emb_clr[i])
    arrow(ax, 9.4, spe_y[i] + 0.5, 10.45, emb_y[i] + 0.5)

# GAN annotation
ax.annotate('GAN\ndisentanglement', xy=(9.5, 7.5), fontsize=FONT_S, color='#666',
            ha='center', va='center', zorder=2,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#aaa', pad=0.3))

# ── Bridge: Stage 1 → Stage 2 ──
ax.plot([10, 10], [8.2, 7.3], '-', color='#888', lw=1.8, dashes=(6, 4), zorder=1)
ax.text(10.5, 7.75, 'frozen &\ntransferred', fontsize=FONT_S, color='#888', ha='left', va='center')

# ═══════════════════════════════════════════
# STAGE 2  (y: 1 — 7)
# ═══════════════════════════════════════════
ax.text(10, 6.8, 'Stage 2:  New-task Prompt Tuning  —  Causal Manipulation',
        ha='center', fontsize=FONT_TITLE, fontweight='bold', color='#BF360C',
        bbox=dict(boxstyle='round', facecolor=C_STAGE2, edgecolor='#BF360C',
                  pad=0.6, lw=1.8), zorder=2)

# New Task T4
box(ax, 0.5, 4.0, 2.2, 1.2, 'New Task\n$T_4$ Label', C_T4)

# Projection Network
box(ax, 3.5, 4.1, 2.2, 1.0, 'Projection\nNetwork', C_PROJ)
arrow(ax, 2.7, 4.6, 3.45, 4.6)

# ── Condition A (top branch, y≈5) ──
ax.text(7.5, 5.95, 'Condition A  (2-source pool)', ha='center', fontsize=FONT,
        fontweight='bold', color='#2E7D32')
box(ax, 6.5, 4.7, 4.0, 1.0,
    r'Softmax$(\mathbf{h}_p \cdot [\mathbf{E}_1, \mathbf{E}_2] \,/\, \tau)$',
    C_CONDA, tc='#222', ec='#2E7D32', fw='bold', fs=FONT_S)
arrow(ax, 5.7, 4.6, 6.45, 5.2)

# Weight bars — Condition A
w_y_base_a = 3.85
ax.barh([w_y_base_a + 0.3, w_y_base_a - 0.15], [0.62, 0.38], height=0.35,
        left=6.5, color=[C_E1, C_E2], edgecolor='white', lw=0.6, zorder=3)
ax.text(7.0, w_y_base_a + 0.3, r'$W_1$ = 0.62', va='center', fontsize=FONT_S,
        color='white', fontweight='bold', zorder=4)
ax.text(6.72, w_y_base_a - 0.15, r'$W_2$ = 0.38', va='center', fontsize=FONT_S,
        color='white', fontweight='bold', zorder=4)

# Tower A
box(ax, 11.5, 4.7, 2.8, 1.0, r'New Tower  $\mathcal{G}^{new}$', C_TOWER)
arrow(ax, 10.5, 5.2, 11.45, 5.2)

# Output A
ax.text(14.8, 5.2, r'$\hat{y}_{T_4}^{(A)}$', ha='center', fontsize=FONT,
        fontweight='bold', color='#2E7D32')

# ── Condition B (bottom branch, y≈2) ──
ax.text(7.5, 2.95, 'Condition B  (3-source pool)', ha='center', fontsize=FONT,
        fontweight='bold', color='#C62828')
box(ax, 6.5, 1.7, 4.0, 1.0,
    r'Softmax$(\mathbf{h}_p \cdot [\mathbf{E}_1, \mathbf{E}_2, \mathbf{E}_3] \,/\, \tau)$',
    C_CONDB, tc='#222', ec='#C62828', fw='bold', fs=FONT_S)
arrow(ax, 5.7, 4.4, 6.45, 2.2)

# Weight bars — Condition B
w_y_base_b = 0.85
ax.barh([w_y_base_b + 0.3, w_y_base_b - 0.15, w_y_base_b - 0.6],
        [0.47, 0.30, 0.23], height=0.3, left=6.5,
        color=[C_E1, C_E2, C_E3], edgecolor='white', lw=0.6, zorder=3)
ax.text(7.0, w_y_base_b + 0.3, r'$W_1$ = 0.47', va='center', fontsize=FONT_S,
        color='white', fontweight='bold', zorder=4)
ax.text(6.72, w_y_base_b - 0.15, r'$W_2$ = 0.30', va='center', fontsize=FONT_S,
        color='white', fontweight='bold', zorder=4)
ax.text(6.68, w_y_base_b - 0.6, r'$W_3$ = 0.23', va='center', fontsize=FONT_S,
        color='white', fontweight='bold', zorder=4)

# Tower B
box(ax, 11.5, 1.7, 2.8, 1.0, r'New Tower  $\mathcal{G}^{new}$', C_TOWER)
arrow(ax, 10.5, 2.2, 11.45, 2.2)

# Output B
ax.text(14.8, 2.2, r'$\hat{y}_{T_4}^{(B)}$', ha='center', fontsize=FONT,
        fontweight='bold', color='#C62828')

# ── Causal comparison ──
ax.annotate('', xy=(16.5, 3.0), xytext=(16.5, 5.8),
            arrowprops=dict(arrowstyle='<->', color=C_CAUSAL, lw=3.0), zorder=3)
ax.text(17.5, 4.4, 'Causal\n' + r'$\Delta$(B$-$A)', ha='center', fontsize=FONT,
        color=C_CAUSAL, fontweight='bold')

# ── Annotation: same backbone ──
ax.annotate('Same frozen backbone\nacross both conditions', xy=(14.5, 7.0),
            fontsize=FONT_S, color='#555', ha='center', va='center', zorder=2,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#aaa', pad=0.3))

# ── Legend ──
ly = 0.15
for lbl, clr, xp in [('E1 (Income)', C_E1, 1.5), ('E2 (Marital)', C_E2, 6.5),
                       ('E3 (Education)', C_E3, 11.5)]:
    ax.add_patch(plt.Rectangle((xp, ly), 0.5, 0.3, color=clr, zorder=3))
    ax.text(xp + 0.7, ly + 0.15, lbl, fontsize=FONT_S, va='center')

plt.tight_layout(pad=0.5)
plt.savefig('D:/MPT-Rec-three_task/MPT-Rec/paper_lbr/figures/fig_framework.pdf',
            bbox_inches='tight', pad_inches=0.15, dpi=200)
plt.savefig('D:/MPT-Rec-three_task/MPT-Rec/paper_lbr/figures/fig_framework.png',
            bbox_inches='tight', pad_inches=0.15, dpi=300)
print("Figure 1 (framework) saved.")
