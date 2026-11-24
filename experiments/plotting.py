"""The VisPubData pairwise Spearman figure."""

import numpy as np


def plot_agreement(author_matrix, paper_matrix, n_authors, n_papers):
    """Draw the paper's two annotated panels on a shared Spearman scale."""
    import matplotlib.pyplot as plt

    names = list(author_matrix.columns)
    for matrix in (author_matrix, paper_matrix):
        if list(matrix.index) != names or list(matrix.columns) != names:
            raise ValueError("Both correlation matrices must use the same method order.")

    n = len(names)
    side = n * 0.46
    gap, margin, bottom = 1.5 * 16 / 14, 0.06, 2.16
    width, height = 2 * margin + 2 * side + gap, bottom + side + 0.45
    style = {"font.family": ["Times New Roman", "STIXGeneral"],
              "mathtext.fontset": "stix"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(width, height), dpi=100, facecolor="white")
        cmap = plt.get_cmap("RdYlGn").with_extremes(bad="#e0e0e0")
        axes = []
        panels = ((author_matrix, "Authors", n_authors), (paper_matrix, "Papers", n_papers))
        for panel, (matrix, title, count) in enumerate(panels):
            x = margin + panel * (side + gap)
            ax = fig.add_axes([x / width, bottom / height, side / width, side / height])
            values = matrix.to_numpy(dtype=float)
            mesh = ax.pcolormesh(values, cmap=cmap, vmin=-1, vmax=1, shading="flat",
                                 edgecolors="white", linewidth=0.45)
            ax.set(xlim=(0, n), ylim=(n, 0), aspect="equal")
            positions = np.arange(n) + 0.5
            angle, align = (-45, "left") if panel == 0 else (45, "right")
            ax.set_xticks(positions, names, rotation=angle, ha=align, rotation_mode="anchor")
            ax.set_yticks(positions, names)
            ax.tick_params(length=0, labelsize=16, pad=4, labelleft=False)
            ax.spines[:].set_visible(False)
            ax.set_title(f"{title} ($n$ = {count:,})", fontsize=18, pad=6)
            for row, col in np.ndindex(values.shape):
                value = values[row, col]
                label = "—" if np.isnan(value) else f"{value:.2f}".replace("-0.00", "0.00")
                rgb = np.array(cmap(mesh.norm(value))[:3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                lightness = linear @ [0.2126, 0.7152, 0.0722]
                ax.text(col + 0.5, row + 0.5, label, ha="center", va="center",
                        fontsize=10, color="white" if lightness < 0.179 else "black")
            axes.append(ax)
        axes[0].tick_params(axis="y", labelright=True, pad=gap * 72 / 2)
        for label in axes[0].get_yticklabels():
            label.set(ha="center", va="center")
        left = margin + (side + gap) / 2
        bar_ax = fig.add_axes([left / width, 0.76 / height, side / width, 0.13 / height])
        bar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
        bar.ax.tick_params(labelsize=14)
        bar.set_label(r"Spearman's $\rho$", fontsize=16, labelpad=7)
    return fig
