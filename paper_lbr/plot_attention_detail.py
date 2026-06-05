"""Figure 2: Attention Mechanism Detail — wide two-panel comparison, NO overlapping text/lines.
Generated at large native resolution; LaTeX scales to \columnwidth proportionally."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif',
                     'mathtext.fontset': 'stix'})

# Wide canvas with generous per-panel width
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 5.5))

C_E1 = '#1E88E5'; C_E2 = '#43A047'; C_E3 = '#FB8C00'
C_QUERY = '#7B1FA2'; C_SOFTMAX = '#37474F'
C_DILUTION = '#D32F2F'; C_FW = '#9E9E9E'; C_DOTPROD = '#ECEFF1'

FONT   = 11.5
FONT_S = 10
FONT_L = 13

def draw_box(ax, x, y, w, h, label, color, fs=FONT, tc='white', fw='bold',
             ec='#555', lw=1.0):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                       facecolor=color, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, linespacing=1.15, zorder=3)

def arrow(ax, x1, y1, x2, y2, c='#555', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle='->', color=c, lw=lw))

# ═══════════════════════════════════════════
# LEFT PANEL: 2-source
# ═══════════════════════════════════════════
ax = ax_l
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('A:  2-source pool  [$\\mathbf{E}_1,\\ \\mathbf{E}_2$]',
             fontsize=FONT_L, fontweight='bold', color='#2E7D32', pad=10)

# Query
draw_box(ax, 0.3, 4.2, 1.6, 1.3, 'Query\n$\\mathbf{h}_p$', C_QUERY, fs=FONT_S)

# Candidate embeddings
draw_box(ax, 2.8, 5.0, 2.0, 0.8, '$\\mathbf{E}_1$ (T1)', C_E1, fs=FONT_S)
draw_box(ax, 2.8, 3.7, 2.0, 0.8, '$\\mathbf{E}_2$ (T2)', C_E2, fs=FONT_S)

arrow(ax, 1.9, 4.85, 2.75, 5.4)
arrow(ax, 1.9, 4.55, 2.75, 4.1)

# Dot product block
draw_box(ax, 5.5, 3.7, 2.2, 1.8, '', C_DOTPROD, tc='#333', fw='normal', ec='#999')
ax.text(6.6, 5.1, r'$\mathbf{h}_p \cdot \mathbf{E}_i$', ha='center',
        fontsize=FONT_S, fontweight='bold', color='#333', zorder=3)
ax.text(6.6, 4.5, r'$\div\ \tau\ (150)$', ha='center',
        fontsize=FONT_S, color='#888', zorder=3)
ax.text(6.6, 3.95, 'Scaled dot-product\nattention', ha='center',
        fontsize=8.5, color='#aaa', zorder=3)

arrow(ax, 4.8, 5.4, 5.45, 4.85)
arrow(ax, 4.8, 4.1, 5.45, 4.3)

# Softmax
draw_box(ax, 8.5, 4.55, 1.5, 0.75, 'Softmax', C_SOFTMAX, fs=FONT_S)
arrow(ax, 7.7, 4.85, 8.45, 4.92)

# Weight bars — well separated on the right
bar_y = [7.0, 5.4]
bar_w = [0.62, 0.38]
bar_h = 0.7
bar_left = 0.3

for i in range(2):
    ax.barh(bar_y[i], bar_w[i], height=bar_h, left=bar_left,
            color=[C_E1, C_E2][i], edgecolor='white', lw=0.8, zorder=3)
    ax.text(bar_left + bar_w[i]/2, bar_y[i],
            f'$W_{i+1}$ = {bar_w[i]:.2f}', ha='center', va='center',
            fontsize=FONT, fontweight='bold', color='white', zorder=4)

# Arrows: softmax → bars
arrow(ax, 9.25, 4.92, 2.0, bar_y[0] + bar_h/2, c='#777')
arrow(ax, 9.25, 4.8, 2.0, bar_y[1] + bar_h/2, c='#777')

# FW reference
ax.axhline(y=bar_y[1] - bar_h/2 - 0.35, xmin=0.03, xmax=0.95,
           color=C_FW, ls='--', lw=1.5, alpha=0.5, zorder=1)
ax.text(11.0, bar_y[1] - bar_h/2 - 0.15, r'FW = 1/2', fontsize=FONT_S,
        color=C_FW, ha='right')

ax.text(0.15, 0.2, r'$K=2$', fontsize=FONT_S, color='#888',
        transform=ax.transAxes)

# ═══════════════════════════════════════════
# RIGHT PANEL: 3-source
# ═══════════════════════════════════════════
ax = ax_r
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('B:  3-source pool  [$\\mathbf{E}_1,\\ \\mathbf{E}_2,\\ \\mathbf{E}_3$]',
             fontsize=FONT_L, fontweight='bold', color='#C62828', pad=10)

# Query
draw_box(ax, 0.3, 4.2, 1.6, 1.3, 'Query\n$\\mathbf{h}_p$', C_QUERY, fs=FONT_S)

# Candidate embeddings (3 stacked with tighter spacing)
e3_y = [5.6, 4.5, 3.4]
e3_lbl = ['$\\mathbf{E}_1$ (T1)', '$\\mathbf{E}_2$ (T2)', '$\\mathbf{E}_3$ (T3)']
e3_clr = [C_E1, C_E2, C_E3]
for i in range(3):
    draw_box(ax, 2.8, e3_y[i], 2.0, 0.7, e3_lbl[i], e3_clr[i], fs=FONT_S)
    arrow(ax, 1.9, 4.85, 2.75, e3_y[i] + 0.35)

# "added" tag
ax.annotate('added', xy=(3.5, 3.1), fontsize=7.5, color='#C62828',
            fontweight='bold', ha='center', zorder=4,
            bbox=dict(boxstyle='round,pad=0.1', fc='#FFEBEE', ec='#C62828', lw=0.6))

# Dot product
draw_box(ax, 5.5, 3.7, 2.2, 1.8, '', C_DOTPROD, tc='#333', fw='normal', ec='#999')
ax.text(6.6, 5.1, r'$\mathbf{h}_p \cdot \mathbf{E}_i$', ha='center',
        fontsize=FONT_S, fontweight='bold', color='#333', zorder=3)
ax.text(6.6, 4.5, r'$\div\ \tau\ (150)$', ha='center',
        fontsize=FONT_S, color='#888', zorder=3)
ax.text(6.6, 3.95, 'Scaled dot-product\nattention', ha='center',
        fontsize=8.5, color='#aaa', zorder=3)

for i in range(3):
    arrow(ax, 4.8, e3_y[i] + 0.35, 5.45, 4.6 + (1-i) * 0.15)

# Softmax
draw_box(ax, 8.5, 4.55, 1.5, 0.75, 'Softmax', C_SOFTMAX, fs=FONT_S)
arrow(ax, 7.7, 4.85, 8.45, 4.92)

# Weight bars (3, with dilution annotations below)
bar_y3 = [7.5, 5.8, 4.1]
bar_w3 = [0.47, 0.30, 0.23]
bar_h3 = 0.55
bar_left3 = 0.3
old_w = [0.62, 0.38]
bar_clr3 = [C_E1, C_E2, C_E3]

for i in range(3):
    ax.barh(bar_y3[i], bar_w3[i], height=bar_h3, left=bar_left3,
            color=bar_clr3[i], edgecolor='white', lw=0.8, zorder=3)
    ax.text(bar_left3 + 0.05, bar_y3[i],
            f'$W_{i+1}$ = {bar_w3[i]:.2f}', ha='left', va='center',
            fontsize=FONT_S, fontweight='bold', color='white', zorder=4)

# Arrows: softmax → bars
for y in bar_y3:
    arrow(ax, 9.25, 4.92, 2.5, y + bar_h3/2, c='#777')

# ── Dilution annotations (below W1 and W2 bars, clearly separated) ──
for i in range(2):
    old_val = old_w[i]
    new_val = bar_w3[i]
    y_arrow = bar_y3[i] - bar_h3/2 - 0.45
    # Reduction arrow
    ax.annotate('', xy=(bar_left3 + new_val, y_arrow),
                xytext=(bar_left3 + old_val, y_arrow),
                arrowprops=dict(arrowstyle='<-,head_width=0.4,head_length=0.3',
                                color=C_DILUTION, lw=2.5), zorder=3)
    ax.text(bar_left3 + (old_val + new_val)/2, y_arrow - 0.28,
            f'$-${old_val - new_val:.2f}', fontsize=FONT_S, color=C_DILUTION,
            fontweight='bold', ha='center', zorder=3)

# Dilution summary label (in its own clear space)
ax.text(6.5, 2.3, r'$\mathbf{E}_3$ dilutes $W_1$ and $W_2$', ha='center',
        fontsize=FONT_S, fontweight='bold', color=C_DILUTION, zorder=3,
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF5F5', ec=C_DILUTION, lw=1.2))

# FW reference
ax.axhline(y=bar_y3[2] - bar_h3/2 - 0.45, xmin=0.03, xmax=0.95,
           color=C_FW, ls='--', lw=1.5, alpha=0.5, zorder=1)
ax.text(11.0, bar_y3[2] - bar_h3/2 - 0.25, r'FW = 1/3', fontsize=FONT_S,
        color=C_FW, ha='right')

ax.text(0.15, 0.2, r'$K=3$', fontsize=FONT_S, color='#888',
        transform=ax.transAxes)

# ── Global title ──
fig.suptitle('Attention Mechanism: How Pool Size Affects Weight Distribution',
             fontweight='bold', fontsize=15, y=1.02)

plt.tight_layout(pad=0.8, rect=[0, 0, 1, 0.96])
plt.savefig('D:/MPT-Rec-three_task/MPT-Rec/paper_lbr/figures/fig_attention_detail.pdf',
            bbox_inches='tight', pad_inches=0.15, dpi=200)
plt.savefig('D:/MPT-Rec-three_task/MPT-Rec/paper_lbr/figures/fig_attention_detail.png',
            bbox_inches='tight', pad_inches=0.15, dpi=300)
print("Figure 2 (attention detail) saved.")
