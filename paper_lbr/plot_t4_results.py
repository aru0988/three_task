"""Figure 3: T4 Causal Experiment Results — clean grouped bar chart, NO overlapping text.
Generated at large native resolution; LaTeX scales to \columnwidth proportionally."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

data = {
    'Sex  ($\\rho \\approx 0.015$)': {
        '2-source\n$[E_1,E_2]$': {
            'Prompt':       (0.8034, 0.0183),
            'KL-Prompt':    (0.8024, 0.0252),
            'FW':           (0.8009, 0.0213),
        },
        '3-source\n$[E_1,E_2,E_3]$': {
            'Prompt':       (0.8092, 0.0170),
            'KL-Prompt':    (0.8047, 0.0235),
            'FW':           (0.8006, 0.0207),
        },
    },
    'Race  (max $\\rho \\approx 0.02$)': {
        '2-source\n$[E_1,E_2]$': {
            'Prompt':       (0.8139, 0.0080),
            'KL-Prompt':    (0.8089, 0.0103),
            'FW':           (0.8129, 0.0119),
        },
        '3-source\n$[E_1,E_2,E_3]$': {
            'Prompt':       (0.8098, 0.0107),
            'KL-Prompt':    (0.8094, 0.0084),
            'FW':           (0.8122, 0.0109),
        },
    },
}

COLORS  = {'Prompt': '#1E88E5', 'KL-Prompt': '#8E24AA', 'FW': '#FB8C00'}
MODES   = ['Prompt', 'KL-Prompt', 'FW']

# Large canvas for readability after scaling
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.0))
conditions = list(list(data.values())[0].keys())

# Compute consistent y-range with headroom for annotations
all_vals = [v[0] for d in data.values() for c in d.values() for v in c.values()]
all_errs = [v[1] for d in data.values() for c in d.values() for v in c.values()]
y_bottom = 0.765
y_top    = max(all_vals) + 2.2 * max(all_errs) + 0.015

for col, (task_name, task_data) in enumerate(data.items()):
    ax = axes[col]
    x = np.arange(len(conditions))
    width = 0.26
    gap = 0.06

    for i, mode in enumerate(MODES):
        means = [task_data[c][mode][0] for c in conditions]
        stds  = [task_data[c][mode][1] for c in conditions]
        offset = (i - 1) * (width + gap)
        bars = ax.bar(x + offset, means, width, label=mode,
                      color=COLORS[mode], edgecolor='white', lw=0.8, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt='none',
                    ecolor='#444', capsize=4.5, lw=1.4, zorder=4)

    # ── Δ annotation: only for Prompt between A and B ──
    p_a = task_data[conditions[0]]['Prompt'][0]
    p_b = task_data[conditions[1]]['Prompt'][0]
    delta = p_b - p_a

    # X positions for the Prompt bars
    prompt_offset = (0 - 1) * (width + gap)
    bar_x_a = x[0] + prompt_offset
    bar_x_b = x[1] + prompt_offset

    # Y position for bracket: above the taller bar + its error + margin
    err_a = task_data[conditions[0]]['Prompt'][1]
    err_b = task_data[conditions[1]]['Prompt'][1]
    bracket_y = max(p_a + err_a, p_b + err_b) + 0.008

    ax.annotate('', xy=(bar_x_a, bracket_y), xytext=(bar_x_b, bracket_y),
                arrowprops=dict(arrowstyle='<->', color='#D32F2F', lw=2.8), zorder=5)
    ax.text((bar_x_a + bar_x_b) / 2, bracket_y + 0.003,
            f'$\\Delta$ = {delta:+.4f}',
            ha='center', fontsize=13, color='#D32F2F', fontweight='bold', zorder=5)

    # ── FW Δ footnote ──
    fw_a = task_data[conditions[0]]['FW'][0]
    fw_b = task_data[conditions[1]]['FW'][0]
    fw_delta = fw_b - fw_a
    ax.text(0.5, -0.18, f'FW $\\Delta$ = {fw_delta:+.4f}   (validity check passed)',
            ha='center', fontsize=10, color='#888', style='italic',
            transform=ax.transAxes)

    # ── Direction indicator ──
    if delta > 0:
        label = '3-source helps  ↑'
        clr   = '#2E7D32'
    else:
        label = '3-source hurts  ↓'
        clr   = '#C62828'
    ax.text(0.5, 0.93, label, ha='center', fontsize=11, fontweight='bold',
            color=clr, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=clr, alpha=0.9, lw=1.5))

    # ── Axis styling ──
    ax.set_title(task_name, fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylim(y_bottom, y_top)
    ax.set_yticks(np.arange(0.77, 0.85, 0.01))

    if col == 0:
        ax.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.2, zorder=0, ls='--')
    ax.set_axisbelow(True)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True,
           fontsize=12, bbox_to_anchor=(0.5, 1.08), edgecolor='#ccc')

fig.suptitle('T4 Experiment: Causal Effect of Attention Pool Size on New-Task AUC',
             fontweight='bold', fontsize=16, y=1.16)

plt.tight_layout(rect=[0, 0.02, 1, 0.90])
plt.savefig('D:/MPT-Rec-three_task/MPT-Rec/paper_lbr/figures/fig_t4_results.pdf',
            bbox_inches='tight', pad_inches=0.15, dpi=200)
plt.savefig('D:/MPT-Rec-three_task/MPT-Rec/paper_lbr/figures/fig_t4_results.png',
            bbox_inches='tight', pad_inches=0.15, dpi=300)
print("Figure 3 (T4 results) saved.")
