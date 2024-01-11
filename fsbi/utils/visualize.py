from sbi.analysis import pairplot
from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _defaults_COBA_ISP():
    return {
        "conditions": [
            r"$\theta \sim \pi(\theta)$",
            r"$\theta \sim q_{\phi}(\theta\ |\ rate \in\ [5, 25]Hz)$",
            r"$\theta \sim q_{\phi}(\theta\ |\ rate \in\ [5, 25]Hz,\ CV(ISI)\ \in\ [0.9, 1.1])$",
        ],
        "samples_colors": ["grey", "tab:blue", "tab:orange"],
    }

def _make_pairplot(samples, limits, labels, title, samples_colors):
    fig, axes = pairplot(
        samples=samples,
        upper="contour",
        hist_diag={"alpha": 0.5, "bins": 50, "density": True, "histtype": "stepfilled"},
        contour_offdiag={"levels": [0.1, 0.68], "percentile": True},
        points_colors=["tab:red"],
        points_diag={"lw": 3},
        limits=limits,
        samples_colors=samples_colors,
    )
    for i in range(len(axes)):
        for j in range(i, len(axes)):
            if i == j:
                axes[i, j].set_xlabel(labels[i], fontsize=15)
    fig.suptitle(title, fontsize=15)
    # axes[3, 1].plot([], [], color="tab:blue", label=r"$\pi(\theta)$")
    # axes[3, 1].plot([], [], color="tab:orange", label=r"$q_1(\theta|x_o=25)$")
    # axes[3, 1].plot([], [], color="tab:green", label=r"$q_2(\theta|x_o=26) \approx p(x |\theta, x\ is\ even)\ q_1(\theta|x_o=25)$")
    # axes[3, 1].legend(frameon=False, fontsize=15)
    fig.patch.set_facecolor("white")

    return fig, axes

def posterior_plot_COBA_ISP(
    samples: List[torch.tensor],
    conditions: List[str],
    samples_colors: List[str] = None,
    save_path: str = None,
) -> tuple:
    """
    Make pairplot for COBA ISP net posterior.

    Args:
        samples: list of posterior samples.
        conditions: legend labels for each element in list of samples.
        samples_colors: colors for samples.
        save_path: path to save plot.
    """
    assert len(samples) == len(conditions)
    labels = [
        r"$\alpha$",
        r"$\beta$",
        r"$\gamma$",
        r"$\kappa$",
        r"$\gamma$",
        r"$\kappa$",
    ]
    title = ""
    limits = [[0.01, 0.05], [0.01, 0.05], [-2, 2.0], [-2, 2.0], [-2, 2.0], [-2, 2.0]]
    if samples_colors is None:
        samples_colors = ["tab:orange", "tab:purple"]
    else:
        assert len(samples_colors) >= len(samples)

    fig, axes = _make_pairplot(samples, limits, labels, title, samples_colors)

    for cond, color in zip(conditions, samples_colors):
        axes[-1, -3].plot([], [], color=color, label=cond)
    axes[-1, -3].legend(frameon=False)

    if save_path is not None:
        fig.savefig(save_path)
    return fig, axes

def plot_compare_metric(file_names, metric, y_label="", linewidth=3, fontsize=20, figsize=(10, 2), font = "arial"):
    n_points = len(file_names)
    xs = [i for i in range(n_points)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, metric, color = "#008080", linewidth=0, marker='o')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.set_xlabel('networks', fontsize=fontsize, fontname=font, labelpad = 5)
    ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
    ax.set_xticks(xs)
    ax.set_xticklabels(file_names, rotation='vertical')

    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    return(ax)

def plot_compare_params_metric(file_names, params_list, metric, sweep_name=["",""], y_label="", log=True, linewidth=3, fontsize=20, figsize=(10, 2), font="arial"):
    n_par = len(params_list)
    n_nets = len(file_names)
    xs = [i for i in range(n_nets)]
    #make color cycler
    color_start = (178/255,34/255,34/255)
    color_stop = (0/255,128/255,128/255)
    colors = [(color_start[0] + i*(color_stop[0]-color_start[0])/(n_par-1),\
              color_start[1] + i*(color_stop[1]-color_start[1])/(n_par-1),\
              color_start[2] + i*(color_stop[2]-color_start[2])/(n_par-1))\
              for i in range(n_par)]

    fig, ax = plt.subplots(figsize=figsize)
    for par_num in range(n_par):
        if log:
            ax.semilogy(xs, metric[:,par_num], label = params_list[par_num][sweep_name[0]], color=colors[par_num], linewidth=0, marker='o', alpha = 0.5)
        else:
            ax.plot(xs, metric[:,par_num], label = params_list[par_num][sweep_name[0]], color=colors[par_num], linewidth=0, marker='o', alpha = 0.5)
#         ax.plot([0,1], [par_num,par_num], color=colors[par_num]) #test color map
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.set_xlabel('networks', fontsize=fontsize, fontname=font, labelpad = 5)
    ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
    ax.set_xticks(xs)
    ax.set_xticklabels(file_names, rotation='vertical')
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    ax.legend(prop={'family':font, 'size':fontsize}, frameon=False, \
              bbox_to_anchor=(1, -0.1, 0.7, 1.2),mode="expand",ncol=1, \
              handlelength=0.3, handletextpad = 0.2, labelspacing = 0,\
              columnspacing=-100)
    ax.text(1.1,1.04,sweep_name[1], fontsize=fontsize, fontname=font, transform=ax.transAxes)
    return(ax)

def apply_1_condition(dataset, condition):
    return(np.logical_and(condition[1] <= dataset[condition[0]], dataset[condition[0]] <= condition[2]))

def apply_n_conditions(dataset, conditions):
    n_conditions = len(conditions)
    cond = apply_1_condition(dataset, conditions[0])
    for i in range(1, n_conditions):
        cond = np.logical_and(cond, apply_1_condition(dataset, conditions[i]))
    return(cond)

def load_and_merge(save_dir, paths):
    """
    paths: tuple with the files to merge (relative to save_dir). Epxecting a numpy structured array
    """
    n_files = len(paths)
    dataset = np.load(save_dir + paths[0])
    for i in range(1,n_files):
        dataset = np.append(dataset, np.load(save_dir + paths[i]), axis=0)
    print("retrieved", str(len(dataset))+"/"+str(len(np.unique(dataset))) ,"simulations")
    return(dataset)