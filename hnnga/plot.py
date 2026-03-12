"""Script for results plotting."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

from matplotlib.patches import Rectangle

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.use("Agg")


def plot_crack(model, epoch=None, ax=None, color="red", show=True, save_path=None):
    """
    Display the crack defined by the two tips z1 and z2 in the model.

    :param model: enriched_PIHNN_finding model with parameters z1 and z2
    :param epoch: epoch number (optional), default is None    :param ax: existing matplotlib axes object (optional), default is None
    :param color: color of the crack, default is 'red'
    :param show: boolean, default is to display the plot
    :param save_path: path to save the image (without extension)
    """
    z1 = model.z1.detach().cpu().numpy()
    z2 = model.z2.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot([z1.real, z2.real], [z1.imag, z2.imag], "-", color=color, linewidth=2)
    ax.scatter([z1.real, z2.real], [z1.imag, z2.imag], color=color, s=30)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    title = f"Crack tip evolution"
    if epoch is not None:
        title += f" (epoch {epoch})"
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        fname = (
            f"{save_path}_epoch_{epoch:04d}.png"
            if epoch is not None
            else f"{save_path}.png"
        )
        plt.savefig(fname)
    if show and ax is None:
        plt.show()
    elif not show:
        plt.close()


def plot_population(
    population_dict,
    target,
    ax,
):

    plt.rcParams["lines.solid_capstyle"] = "round"
    colors = [
        "#e69f00",
        "#56b4e9",
        "#cc79a7",
        "#009e73",
        "#0072b2",
        "#d55e00",
        "#f0e442",
        "#000000",
    ]

    target_x_vals = np.array([target[0][0], target[1][0]])
    target_y_vals = np.array([target[0][1], target[1][1]])

    ax.plot(target_x_vals, target_y_vals, color=colors[1], linewidth=1.25)
    # ax.scatter(target_x_vals, target_y_vals, color=colors[1], marker=".", s=10)

    for key, value in population_dict.items():
        crack_x_vals = np.array([value[0][0][0], value[0][1][0]])
        crack_y_vals = np.array([value[0][0][1], value[0][1][1]])
        if key == 0:
            color = colors[0]
            zorder = 3
            lw = 1.25
            marker_size = 10
        else:
            color = "#e0e0e0"
            zorder = -1
            lw = 0.75
            marker_size = 7
        # ax.scatter(
        #     crack_x_vals,
        #     crack_y_vals,
        #     color=color,
        #     marker=".",
        #     s=marker_size,
        #     zorder=zorder,
        # )
        ax.plot(
            crack_x_vals,
            crack_y_vals,
            color=color,
            zorder=zorder,
            linewidth=lw,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    return


def create_custom_axes(
    ax,
    population_dict,
    target,
    gen,
    num_arrows=10,
    arrow_length=0.15,
    text=r"$\sigma_0$",
    num_ax=None,
):
    # Set limits so arrows are visible
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.4, 1.4)

    # Remove spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw quadratic frame around domain [-1,-1] to [1,1]
    frame = Rectangle((-1, -1), 2, 2, fill=False, color="black", lw=0.5)
    ax.add_patch(frame)

    plot_population(
        population_dict,
        target,
        ax,
    )

    # Compute arrow positions along top and bottom
    x_positions = np.linspace(-1, 1, num_arrows)

    # Bottom arrows (pointing down)
    for x in x_positions:
        ax.annotate(
            "",
            xy=(x, -1 - arrow_length),
            xytext=(x, -1),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
            clip_on=False,
        )

    # Top arrows (pointing up)
    for x in x_positions:
        ax.annotate(
            "",
            xy=(x, 1 + arrow_length),
            xytext=(x, 1),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
            clip_on=False,
        )

    # Line connecting top arrow tips
    # ax.plot(
    #     [-1, 1],
    #     [1 + arrow_length, 1 + arrow_length],
    #     color="black",
    #     lw=1.0,
    #     clip_on=False,
    # )
    # ax.plot(
    #     [-1, 1],
    #     [-1 - arrow_length, -1 - arrow_length],
    #     color="black",
    #     lw=1.0,
    #     clip_on=False,
    # )

    # Text at midpoint
    # ax.text(
    #     0,
    #     1 + arrow_length * 1.025,
    #     text,
    #     ha="center",
    #     va="bottom",
    #     fontsize=10,
    #     clip_on=False,
    # )
    # ax.text(
    #     0,
    #     -1 - arrow_length * 1.025,
    #     text,
    #     ha="center",
    #     va="top",
    #     fontsize=10,
    #     clip_on=False,
    # )

    # Text in lower left corner
    annotation_list = [r"a) ", r"b) ", r"c) ", r"d) ", r"e) ", r"f) "]
    if num_ax is not None:
        annotation = annotation_list[num_ax] + f"Gen. {gen}"
    else:
        annotation = f"Gen. {gen}"
    ax.text(
        0.0,
        -1.35,
        annotation,
        ha="center",
        va="center",
        fontsize=13,
        clip_on=False,
        # bbox=dict(
        #     boxstyle="round,pad=0.25",
        #     facecolor="white",
        #     edgecolor="black",
        #     linewidth=0.5,
        # ),
    )
    return


def plot_residual(y_values, out_file_name="residual_plot", **plot_kwargs):
    """Plot the residual values on a logarithmic scale."""
    x_values = [round(x, 1) for x in np.arange(-0.6, 0.4, 0.01) if round(x, 2) != 0.3]

    plt.plot(x_values, y_values, **plot_kwargs)

    plt.xlabel("Position x")
    plt.ylabel("Residual")

    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_file_name + ".png", dpi=300)
    plt.close()


def plot_crack_vs_target(
    crack,
    target,
    generation,
    crack_tips=None,
    residual=None,
    out_path="crack_vs_target.png",
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0),
):
    crack = np.array(crack)
    target = np.array(target)

    if crack_tips is None:
        crack_tips = crack

    crack_x_vals = np.array([crack[0][0], crack[1][0]])
    crack_y_vals = np.array([crack[0][1], crack[1][1]])
    target_x_vals = np.array([target[0][0], target[1][0]])
    target_y_vals = np.array([target[0][1], target[1][1]])

    fig = plt.figure(figsize=(8, 6))  # width=8 inches, height=6 inches
    ax = fig.add_subplot(111)
    ax.plot(target_x_vals, target_y_vals, "b-", label="Target crack")
    ax.plot(crack_x_vals, crack_y_vals, "r-", label="Calculated crack")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()

    title_str = f"Gen. {generation} | Residual: {residual:.6f}\nTarget: ({np.round(target[0][0], 2)}, {np.round(target[0][1], 2)}), ({np.round(target[1][0], 2)}, {np.round(target[1][1], 2)}) | Current: ({np.round(crack_tips[0][0], 2)}, {np.round(crack_tips[0][1], 2)}), ({np.round(crack_tips[1][0], 2)}, {np.round(crack_tips[1][1], 2)})"
    ax.set_title(title_str)
    ax.set_aspect("equal")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_final_result(
    crack,
    target,
    generation,
    crack_tips=None,
    residual=None,
    out_path="final_result.png",
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0),
):
    crack = np.array(crack)
    target = np.array(target)

    if crack_tips is None:
        crack_tips = crack

    crack_x_vals = np.array([crack[0][0], crack[1][0]])
    crack_y_vals = np.array([crack[0][1], crack[1][1]])
    target_x_vals = np.array([target[0][0], target[1][0]])
    target_y_vals = np.array([target[0][1], target[1][1]])

    fig = plt.figure(figsize=(8, 6))  # width=8 inches, height=6 inches
    ax = fig.add_subplot(111)
    ax.plot(target_x_vals, target_y_vals, "b-", label="Target crack")
    ax.plot(crack_x_vals, crack_y_vals, "r-", label="Calculated crack")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()

    title_str = f"Gen. {generation} | Residual: {residual:.6f}\nTarget: ({np.round(target[0][0], 2)}, {np.round(target[0][1], 2)}), ({np.round(target[1][0], 2)}, {np.round(target[1][1], 2)}) | Result: ({np.round(crack_tips[0][0], 2)}, {np.round(crack_tips[0][1], 2)}), ({np.round(crack_tips[1][0], 2)}, {np.round(crack_tips[1][1], 2)})"
    ax.set_title(title_str)
    ax.set_aspect("equal")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def update_rcParams():
    """
    Update the matplotlib rcParams to produce plots with the specified fontsizes
    """
    update_rcParams = {
        # Use xxpt font in plots, to match xxpt font in document
        "font.size": 10,
        "axes.labelsize": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    mpl.rcParams.update(update_rcParams)
    return
