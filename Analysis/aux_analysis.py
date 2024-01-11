import matplotlib.pyplot as plt
import numpy as np
import time
from synapsbi.utils import apply_n_conditions
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch import nn
from matplotlib import cm
from matplotlib import colors
from matplotlib import patches
from scipy.interpolate import splrep, BSpline
from scipy.ndimage import gaussian_filter

def plot_rule(thetas=None, n_bins=1000,
              x_lim=[0,1], logx=False, logy=False, x_label="", ax=None, 
              color=None, save_path=None, linewidth=3, fontsize=20, figsize=(5,3), font="arial",
              x_ticks=None, x_ticklabels=None, rotation=0, y_label=None,
              y_lim=None, y_ticks=None, y_ticklabels=None, axwidth=3, labelpad_xlabel=30,
              labelpad_ylabel=-40, dpi=200, xticks_pad=None, yticks_pad=None, color_ylabel='black'):

    fig, ax = plt.subplots(figsize=figsize, tight_layout=False, dpi=dpi)
    
    ts = np.linspace(x_lim[0], x_lim[1],num=n_bins)
    ind_t_pos = 0
    while ts[ind_t_pos] < 0:
        ind_t_pos += 1
    
    dws = np.array([thetas[2] + thetas[3] + thetas[5]*np.exp(-np.abs(ts[i])/thetas[1]) for i in range(ind_t_pos)])
    dws = np.append(dws, np.array([thetas[2] + thetas[3] + thetas[4]*np.exp(-np.abs(ts[i])/thetas[0]) for i in range(ind_t_pos, len(ts))]), axis=0)


    ax.plot(ts, dws, color=color, 
            linewidth=linewidth)
        
    # if logx:
    #     ax.set_xscale('log')
    #     ax.tick_params(axis='x', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
    
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation = rotation)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = labelpad_xlabel)
    
    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize, fontname=font, labelpad = labelpad_ylabel, color=color_ylabel)

    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    ax.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    # ax.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
    plt.show()

def get_pop_rate_square_window(spiketimes=None, t_start=None, t_stop=None, window_size=None, n_neurons=None):
    ts = np.arange(t_start, t_stop, window_size)
    rates = np.zeros((n_neurons, len(ts)-1)) 
    for neuron in range(n_neurons):
        inds_insert = np.searchsorted(spiketimes[str(neuron)], ts, side='left', sorter=None)
        rates[neuron] = np.diff(inds_insert)
    
    pop_rate = np.zeros(len(ts)-1)
    for i in range(len(ts)-1):
        pop_rate[i] = np.mean(rates[:,i])
    return(ts[:-1], pop_rate/window_size)

def plot_fft_pop_rate(x=None, y=None, x_lim=None, y_lim=None, logx=False, logy=False,
                      color="#008080", x_label=None, y_label=None, 
                      axwidth=3, fontsize=20, figsize=(4, 3), font = "arial",
                      x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, 
                      rotation=0, linewidth=3):
    fig = plt.figure(figsize=figsize, dpi=200)
    ax = plt.subplot()
    ax.plot(x, y, color=color, linewidth=axwidth)
    
    if logx:
        ax.set_xscale("log")
        ax.tick_params(axis='x', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1.5*axwidth)
        
    if logy:
        ax.set_yscale("log")
        ax.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1.5*axwidth)

    ax.set_xlabel(x_label, fontname=font, fontsize=fontsize , labelpad = 0)
    ax.set_ylabel(y_label, fontname=font, fontsize=fontsize , labelpad = 0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    if x_label is not None:
        ax.set_xlabel(x_label, fontname=font, fontsize=fontsize, labelpad = 0)
        
    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
    
    plt.show()

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[window_len//2-1:-window_len//2]

def plot_raster(sts, neuron_indices, t_lim=None, save_path=None, fontsize=20, color="black", x_label="", y_label="", markersize=0.05, 
figsize=(20, 3), font="arial", ax=None, x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, tickwidth=2,
axwidth=3, dpi=200, ylabel_xloc=-0.1, ylabel_yloc=0.15, xlabel_xloc=0.4, xlabel_yloc=-0.1):
    n_to_plot = len(neuron_indices)
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ct = 0
    for neuron_num in neuron_indices:
        # ax.plot(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linestyle='', marker='.',color=color, markersize=markersize)
        ax.scatter(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linewidths=0, color=color, s=markersize, edgecolors=None, marker='o')
        ct += 1
    if t_lim is not None:
        ax.set_xlim([t_lim[0], t_lim[1]])
    if (t_lim is None) and (x_ticks is not None):
        ax.set_xlim([x_ticks[0], x_ticks[-1]])
    if x_ticks is None:
        ax.set_xticks([t_lim[0], t_lim[1]])
    else:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)

    if x_label is not None:
        ax.plot((0.45, 0.55), (-0.05, -0.05), transform=ax.transAxes, color="black", clip_on=False, linewidth=axwidth)
        fig.text(xlabel_xloc, xlabel_yloc, x_label, fontsize=fontsize, fontname=font, ha='center')

    if y_label is not None:
        ax.plot((-0.05, -0.05), (0.45, 0.55), transform=ax.transAxes, color="black", clip_on=False, linewidth=axwidth)
        fig.text(ylabel_xloc, ylabel_yloc, y_label, fontsize=fontsize, fontname=font, rotation=90, ha='center')

    ax.set_ylim([-1, n_to_plot + 0.1])
    if y_ticks is None:
        ax.set_yticks([0, n_to_plot])
    else:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(width=tickwidth, labelsize=fontsize, length=tickwidth*2, pad = 10)
    ax.tick_params(axis='y', pad = 0)

    
    
    if save_path!=None and ax != None: 
        fig.savefig(save_path+".png", format='png', dpi=800, transparent=True, bbox_inches='tight')
    return(ax)

def plot_weights(ts = None, ws=None, n_to_plot="all", w_lim=None, w_max=None, t_lim=None, con_type="EE", save_path= "", ax=None, linewidth=3, 
fontsize=20, figsize=(5, 4), font="arial", log=False, color='black',
x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, axwidth=None, x_label="t (s)", y_label=None):
    """
    plot evolution of weights across time

    args: w_dict dict from get_weights, keys: 't' np.array(float) and 'w' np.array(np.array(float)) [syn_num, time]
          t_lim = [t_min,t_max] float in seconds
          w_lim = [w_min,w_up] float, window of view of ws. different of w_max which is the hard cap on sim
          w_max float or None, max weight of the simulation. if None, uses auto lim from matplotlib
          con_type" "EE" "IE" "II" "inp", changes the color of plot that's all.
          save_path str: if not "" will same the figure
          other args are cosmetic kwargs from matplotlib, see matplotlib documentation

    returns: matplotlib axis
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    if axwidth is None:
        axwidth=linewidth

    n_syn = len(ws)
    if con_type == 'EE':
        # color = (200/255,37/255,46/255)
        if log:
            ax.set_ylabel(r'$w^{EE}_{i} \ logscale$', fontname=font, fontsize=fontsize , labelpad = 0)
        else:
            ax.set_ylabel(r'$w^{EE}_{i}$', fontname=font, fontsize=fontsize , labelpad = 0)
    elif con_type == 'IE':
        # color = (43/255,118/255,184/255)
        if log:
            ax.set_ylabel(r'$w^{IE}_{i} \ logscale$', fontname=font, fontsize=fontsize , labelpad = 0)
        else:
            ax.set_ylabel(r'$w^{IE}_{i}$', fontname=font, fontsize=fontsize , labelpad = 0)
    elif con_type == 'inp':
        color = "black"
        if log:
            ax.set_ylabel(r'$w^{input_E}_{i} \ logscale$', fontname=font, fontsize=fontsize , labelpad = 0)
        else:
            ax.set_ylabel(r'$w^{input_E}_{i}$', fontname=font, fontsize=fontsize , labelpad = 0)
    elif con_type == 'II':
        # color = (85/255,139/255,47/255)
        if log:
            ax.set_ylabel(r'$w^{II}_{i} \ logscale$', fontname=font, fontsize=fontsize , labelpad = 0)
        else:
            ax.set_ylabel(r'$w^{II}_{i}$', fontname=font, fontsize=fontsize , labelpad = 0)
    else:
        print("unknown con_type (EE, II, IE, inp)")
        return
    
    if n_to_plot == "all":
        for syn_num in range(n_syn):
            if log:
                ax.semilogy(ts, ws[syn_num, :], color=color, linewidth=linewidth)
            else:
                ax.plot(ts, ws[syn_num, :], color=color, linewidth=linewidth)
    else:
        syn_to_plot = np.random.choice(n_syn, size=n_to_plot, replace=False)
        for syn_num in syn_to_plot:
            if log:
                ax.semilogy(ts, ws[syn_num, :], color=color, linewidth=linewidth)
            else:
                ax.plot(ts, ws[syn_num, :], color=color, linewidth=linewidth)

    if t_lim is not None:
        ax.set_xlim([t_lim[0],t_lim[1]])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)

    if w_lim is not None:
        ax.set_ylim([w_lim[0],w_lim[1]])
    else:
        ax.set_ylim([0,ax.set_ylim()[1]])
    if w_max is not None:
        ax.plot(ts, [w_max for i in range(len(ts))], color="black", linewidth=linewidth, linestyle='dashed')
        ax.set_ylim([w_lim[0]-0.01,w_max])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)

    if save_path!=None and ax != None: 
        fig.savefig(save_path+".png", format='png', dpi=800, transparent=True, bbox_inches='tight')
    return(ax)

def plot_rate_distr(rs=None, n_bins=None, x_lim=None, x_ticks=None, x_ticklabels=None, x_label=None,
                    y_lim=None, y_ticks=None, y_ticklabels=None, y_label=None, ax=None, 
                    figsize=None, linewidth=2, title=None, target=None,
                    fontsize=15,  font="arial", color="black", rotation=0):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    a= ax.hist(rs, bins=n_bins, color=color, histtype='bar')
    
    if target is not None:
        ax.vlines(target, 0, max(a[0]), color=color, linestyle='--', linewidth=linewidth)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_visible(False)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.minorticks_off()
    
    if x_lim is not None:
        ax.set_xlim([x_lim[0], x_lim[1]])
        ax.set_xticks([x_lim[0], x_lim[1]])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation = rotation)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
    if title is not None:
        ax.set_title(label=title, fontsize=fontsize*1.2)
    
    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks([])
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)

def get_individual_rates(spiketimes, n_neurons, l_record):
    rates = np.zeros(n_neurons)
    for neuron in range(n_neurons):
        rates[neuron] =  len(spiketimes[str(neuron)])/l_record
    return(rates)

def get_spiketimes_large_auryn(filename, t_start=0, t_stop=0, n_neurons=0):
    start = time.time()
    
    spiketimes = dict()
    for neuron in range(n_neurons):
        spiketimes[str(neuron)] = []
    
    with open(filename) as file:
        for line in file:
            aux = line.split(" ")
            if float(aux[0]) >= t_start:
                if float(aux[0]) >= t_stop:
                    break
                spiketimes[str(int(aux[1]))].append(float(aux[0]))
                
    for neuron in range(n_neurons):
        spiketimes[str(neuron)] = np.array(spiketimes[str(neuron)])
    end =  time.time()
    print("Spikes extracted between", t_start, "s and", t_stop, "s in", np.round(end-start, 2), "s")
    return(spiketimes)

def load_w_mat(filename):
    """
    loads a full weight matrix from file as a dictionnary. weight matrix itself stored as a structured array
    Format: np.array([int pre, int post, float weight])
    Careful: Matrix market format starts at 1
    w["n_from"] int
    w["n_to"] int
    w["n_w_non_zero"] int
    w["w"] array(dtype=('pre',np.uint16),('post',np.uint16),('w',np.float32))
    """    
    start = time.time()
    
    with open(filename) as file:
        for i, line in enumerate(file):
            aux = line.split(" ")
            if i >= 6:
                ws["w"][i-6] = (int(aux[0]), int(aux[1]), float(aux[2]))
            elif i < 5:
                pass
            elif i == 5:
                n_from = int(aux[0])
                n_to = int(aux[1])
                n_w_non_zero = int(aux[2])
                print("matrix found from", n_from, "to", n_to, "neurons.", np.round(n_w_non_zero/n_from/n_to*100, 2), "% sparsity")
                ws = dict()
                ws["n_from"]=n_from
                ws["n_to"]=n_to
                ws["n_w_non_zero"]=n_w_non_zero
                dt = np.dtype([('pre',np.uint16),('post',np.uint16),('w',np.float32)])
                ws["w"]=np.zeros( (ws["n_w_non_zero"],), dt )
            
    end =  time.time()
    return(ws)

def plot_pop_rate(rs=None, ts=None, t_lim=None, r_lim=None, color="#008080", x_label=None, y_label=r'$r_{pop} \; (Hz)$', 
save_path=None, ax=None, linewidth=3, axwidth=3, fontsize=20, figsize=(5, 2), font = "arial",
x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, target=None, tight_layout=True):
    """
    makes a plot of the popualtion firing rate across time

    args: rs np.array(float)
          ts np.array(float) same size as rs
          t_lim = [t_min,t_max] float: in seconds, otherwise "default"
          r_lim = [r_min, r_max] float: Hz, otherwise "default"
          y_label str: will be displayed on the y axis of the plot
          save_path str: if not "" will same the figure
          other args are cosmetic kwargs from matplotlib, see matplotlib documentation

    returns: matplotlib axis
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=200)

    ax.plot(ts, rs, color=color, linewidth=linewidth, marker='')

    if target is not None:
        ax.hlines(target,t_lim[0],t_lim[1], linestyles="--", linewidth=linewidth, color=color, zorder=0)

    if t_lim is not None:
        ax.set_xlim([t_lim[0], t_lim[1]])
        ax.set_xticks([t_lim[0], t_lim[1]])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
    
    if r_lim is not None:
        ax.set_ylim([r_lim[0], r_lim[1]])
        ax.set_yticks([r_lim[0], r_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if save_path!=None and ax != None: 
        fig.savefig(save_path+ ".png", format='png', dpi=800, transparent=True, bbox_inches='tight')
    return(ax)

# def plot_IandE_rate_distr(r_exc=None, r_inh=None, n_bins=None, 
#                           x_lim=None, x_ticks=None, x_ticklabels=None, x_label=None,
#                     y_lim=None, y_ticks=None, y_ticklabels=None, y_label=None, 
#                     figsize=None, linewidth=2, title=None, target=None,
#                     fontsize=15,  font="arial", color_exc=None, color_inh=None, rotation=45):
    
#     fig, ax = plt.subplots(figsize=figsize, dpi=200)
    
#     if target is not None:
#         line3 = ax.vlines(target, 0, 1, color=color_exc, linestyle='--', linewidth=linewidth,
#                  label=r'$r^{exc}_{pred}$')

#     range_hist = [min(min(r_exc), min(r_inh)), max(max(r_exc), max(r_inh))]
#     exc_hist, exc_bins = np.histogram(r_exc, bins=n_bins, density=False, range=range_hist)
#     inh_hist, inh_bins = np.histogram(r_inh, bins=n_bins, density=False, range=range_hist)
    
#     line2, = ax.plot(inh_bins[1:], inh_hist/max(inh_hist), color=color_inh, linewidth=linewidth,
#            label=r'$r_i^{inh}$')
#     line1, = ax.plot(exc_bins[1:], exc_hist/max(exc_hist), color=color_exc, linewidth=linewidth, label=r'$r_i^{exc}$')
    

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(linewidth)
#     ax.spines['left'].set_visible(False)
#     ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
#     ax.minorticks_off()
    
#     if x_lim is not None:
#         ax.set_xlim([x_lim[0], x_lim[1]])
#         ax.set_xticks([x_lim[0], x_lim[1]])
#     if x_ticks is not None:
#         ax.set_xticks(x_ticks)
#     if x_ticklabels is not None:
#         ax.set_xticklabels(x_ticklabels, rotation = rotation)
#     if x_label is not None:
#         ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
#     if title is not None:
#         ax.set_title(label=title, fontsize=fontsize*1.2)
    
#     if y_lim is not None:
#         ax.set_ylim([y_lim[0], y_lim[1]])
#         ax.set_yticks([y_lim[0], y_lim[1]])
#     if y_ticks is not None:
#         ax.set_yticks(y_ticks)
#     else:
#         ax.set_yticks([])
#     if y_ticklabels is not None:
#         ax.set_yticklabels(y_ticklabels)
#     if y_label is not None:
#         ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
        
#     legend = ax.legend(handles=[line1,  line2, line3], loc='upper left', bbox_to_anchor=(1, 0.9), 
#               fontsize=fontsize, ncol=1, frameon=False,
#               borderpad=0, labelspacing=1.2, handlelength=1, 
#               columnspacing=0, handletextpad=0.5, borderaxespad=0.6)
    
# def get_individual_rates(spiketimes, n_neurons, start=None, stop=None):
#     rates = np.zeros(n_neurons)
#     for neuron in range(n_neurons):
#         rates[neuron] =  np.sum(np.logical_and(start<=spiketimes[str(neuron)], spiketimes[str(neuron)]<=stop))/(stop-start)
#     return(rates)

# def plot_summary_simulation(spiketimes_inp=None, spiketimes_exc=None, spiketimes_inh=None, n_to_plot_raster=None, 
#                             n_exc=4096, n_inh=1024, n_input=5000,
#                             wee=None, wei=None, wie=None, wii=None, t_start=None, t_stop=None,
#                             window_pop_rate=0.1, axwidth=4, linewidth=4, n_to_plot_weights=100, ms_raster=0.5,
#                             fontsize=20, figsize=(5, 2), font = "arial",  linewidth_milestones=1,
#                             color_ee=None, color_ei=None, color_ie=None, color_ii=None, color_input=None,
#                             x_ticks=None, x_ticklabels=None, linewidth_weights=0.5, 
#                             y_lim_we=None, y_ticks_we=None, y_ticklabels_we=None,
#                             y_lim_wi=None, y_ticks_wi=None, y_ticklabels_wi=None,
#                             x_label=None, alpha_w=None, ylocleg=1.5,
#                             y_ticks_pop_rate=None, y_lim_pop_rate=None, x_milestones=None, alpha_milestone=0.5):
    
#     fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1,figsize=figsize, constrained_layout=True, dpi=600, 
#                                                   gridspec_kw={'height_ratios': [1.5, 1.5, 1.5, 1, 1.2, 1.2]})
# #     plt.subplots_adjust(wspace=0, hspace=0.05)
    
#     for x in x_milestones:
#         ax1.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones, zorder=1, alpha=alpha_milestone)
#     for ct, neuron_num in enumerate(np.linspace(0, n_input-1, num=n_to_plot_raster, dtype=int)):
#         ax1.scatter(spiketimes_inp[str(neuron_num)], np.full(len(spiketimes_inp[str(neuron_num)]), ct), linewidths=0, color=color_input, s=ms_raster, edgecolors=None, marker='o', zorder=10)
#     ax1.set_xlim([t_start, t_stop])
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_ylabel("Input", fontsize=fontsize, fontname=font)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['left'].set_visible(False)
    
#     for x in x_milestones:
#         ax2.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)
#     for ct, neuron_num in enumerate(np.linspace(0, n_exc-1, num=n_to_plot_raster, dtype=int)):
#         ax2.scatter(spiketimes_exc[str(neuron_num)], np.full(len(spiketimes_exc[str(neuron_num)]), ct), linewidths=0, color=color_ee, s=ms_raster, edgecolors=None, marker='o')
#     ax2.set_xlim([t_start, t_stop])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_ylabel("Exc", fontsize=fontsize, fontname=font)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['left'].set_visible(False)
    
#     for x in x_milestones:
#         ax3.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)
#     for ct, neuron_num in enumerate(np.linspace(0, n_inh-1, num=n_to_plot_raster, dtype=int)):
#         ax3.scatter(spiketimes_inh[str(neuron_num)], np.full(len(spiketimes_inh[str(neuron_num)]), ct), linewidths=0, color=color_ii, s=ms_raster, edgecolors=None, marker='o')
#     ax3.set_xlim([t_start, t_stop])
#     ax3.set_xticks([])
#     ax3.set_yticks([])
#     ax3.set_ylabel("Inh", fontsize=fontsize, fontname=font)
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_visible(False)
#     ax3.spines['left'].set_visible(False)
    
#     for x in x_milestones:
#         ax4.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)
#     ts, pop_rate = get_pop_rate_square_window(spiketimes=spiketimes_exc,t_start=t_start, t_stop=t_stop,window_size=window_pop_rate,n_neurons=n_exc)
#     ax4.plot(ts, pop_rate, color=color_ee, linewidth=linewidth, marker='')
#     ax4.set_xlim([t_start, t_stop])
#     ax4.set_xticks(x_ticks)
#     ax4.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax4.set_ylabel(r'$r^{pop}_{exc}$' + " (Hz)", fontsize=fontsize, fontname=font)
#     ax4.spines['top'].set_visible(False)
#     ax4.spines['right'].set_visible(False)
#     ax4.spines['bottom'].set_linewidth(axwidth)
#     ax4.spines['left'].set_linewidth(axwidth)
#     ax4.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     ax4.set_yticks(y_ticks_pop_rate)
#     ax4.set_ylim(y_lim_pop_rate)

#     for x in x_milestones:
#         ax5.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone,zorder=0.1)
#     for syn_num in range(n_to_plot_weights):
#         ax5.plot(wee['t'], wee['w'][syn_num, :], color=color_ee, linewidth=linewidth_weights, alpha=alpha_w,zorder=0)
#         ax5.plot(wei['t'], wei['w'][syn_num, :], color=color_ei, linewidth=linewidth_weights, alpha=alpha_w,zorder=0)
#     a2, =ax5.plot(wei['t'], np.mean(wei['w'], axis=0), label=r'$w_{ei}$', color=color_ei, linewidth=linewidth, alpha=1,zorder=0.8)
#     a1, =ax5.plot(wee['t'], np.mean(wee['w'], axis=0), label=r'$w_{ee}$', color=color_ee, linewidth=linewidth, alpha=1,zorder=0.9)
#     # ax5.set_yscale('log')
#     # ax5.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
#     ax5.set_xlim([t_start, t_stop])
#     ax5.set_ylim(y_lim_we)
#     ax5.set_yticks(y_ticks_we)
#     ax5.spines['top'].set_visible(False)
#     ax5.spines['right'].set_visible(False)
#     ax5.spines['bottom'].set_linewidth(axwidth)
#     ax5.spines['left'].set_linewidth(axwidth)
#     ax5.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     # ax5.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1*axwidth)
#     ax5.set_xticks(x_ticks)
#     ax5.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax5.set_yticklabels(y_ticklabels_we)
#     ax5.set_xlabel(None, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax5.legend(handles=[a1, a2], loc='upper center', bbox_to_anchor=(0.5, ylocleg), fontsize=fontsize, ncol=2, frameon=False,
#                  borderpad=0, labelspacing=3, handlelength=0.5, columnspacing=2)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

#     for x in x_milestones:
#         ax6.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone,zorder=0.2)
#     for syn_num in range(n_to_plot_weights):
#         ax6.plot(wie['t'], wie['w'][syn_num, :], color=color_ie, linewidth=linewidth_weights, alpha=alpha_w,zorder=0)
#         ax6.plot(wii['t'], wii['w'][syn_num, :], color=color_ii, linewidth=linewidth_weights, alpha=alpha_w,zorder=0.1)
#     a4, =ax6.plot(wii['t'], np.mean(wii['w'], axis=0), label=r'$w_{ii}$', color=color_ii, linewidth=linewidth, alpha=1,zorder=0.8)
#     a3, =ax6.plot(wie['t'], np.mean(wie['w'], axis=0), label=r'$w_{ie}$', color=color_ie, linewidth=linewidth, alpha=1,zorder=0.9)
#     # ax6.set_zorder(20)
#     ax6.set_xlim([t_start, t_stop])
#     ax6.set_ylim(y_lim_wi)
#     ax6.set_yticks(y_ticks_wi)
#     ax6.spines['top'].set_visible(False)
#     ax6.spines['right'].set_visible(False)
#     ax6.spines['bottom'].set_linewidth(axwidth)
#     ax6.spines['left'].set_linewidth(axwidth)
#     ax6.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     ax6.set_xticks(x_ticks)
#     ax6.set_yticklabels(y_ticklabels_wi)
#     ax6.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax6.legend(handles=[a3, a4], loc='upper center', bbox_to_anchor=(0.5, ylocleg), fontsize=fontsize, ncol=2, frameon=False,
#                  borderpad=0, labelspacing=3, handlelength=0.5, columnspacing=2)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

# def plot_summary_simulationMLP(spiketimes_inp=None, spiketimes_exc=None, spiketimes_inh=None, n_to_plot_raster=None, 
#                             n_exc=4096, n_inh=1024, n_input=5000,
#                             wee=None, wie=None, t_start=None, t_stop=None,
#                             window_pop_rate=0.1, axwidth=4, linewidth=4, n_to_plot_weights=100, ms_raster=0.5,
#                             fontsize=20, figsize=(5, 2), font = "arial",  linewidth_milestones=1,
#                             color_ee=None, color_ie=None, color_ii=None, color_input=None,
#                             x_ticks=None, x_ticklabels=None, linewidth_weights=0.5, y_lim_w=None, 
#                             y_ticks_w=None, y_ticklabels_w=None, x_label=None, alpha_w=None,
#                             y_ticks_pop_rate=None, y_lim_pop_rate=None, x_milestones=None, alpha_milestone=0.5):
    
#     fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=figsize, constrained_layout=True, dpi=600, 
#                                                   gridspec_kw={'height_ratios': [1.5, 1.5, 1.5, 1, 2.5]})
# #     plt.subplots_adjust(wspace=0, hspace=0.05)
    
#     for x in x_milestones:
#         ax1.axvline(x=x, ymin=0, ymax=n_to_plot_raster, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones, zorder=1, alpha=alpha_milestone)
#     for ct, neuron_num in enumerate(np.linspace(0, n_input-1, num=n_to_plot_raster, dtype=int)):
#         ax1.scatter(spiketimes_inp[str(neuron_num)], np.full(len(spiketimes_inp[str(neuron_num)]), ct), linewidths=0, color=color_input, s=ms_raster, edgecolors=None, marker='o', zorder=10)
#     ax1.set_xlim([t_start, t_stop])
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_ylabel("Input", fontsize=fontsize, fontname=font)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['left'].set_visible(False)
    
#     for x in x_milestones:
#         ax2.axvline(x=x, ymin=0, ymax=n_to_plot_raster, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)
#     for ct, neuron_num in enumerate(np.linspace(0, n_exc-1, num=n_to_plot_raster, dtype=int)):
#         ax2.scatter(spiketimes_exc[str(neuron_num)], np.full(len(spiketimes_exc[str(neuron_num)]), ct), linewidths=0, color=color_ee, s=ms_raster, edgecolors=None, marker='o')
#     ax2.set_xlim([t_start, t_stop])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_ylabel("Exc", fontsize=fontsize, fontname=font)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['left'].set_visible(False)
    
#     for x in x_milestones:
#         ax3.axvline(x=x, ymin=0, ymax=n_to_plot_raster, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)
#     for ct, neuron_num in enumerate(np.linspace(0, n_inh-1, num=n_to_plot_raster, dtype=int)):
#         ax3.scatter(spiketimes_inh[str(neuron_num)], np.full(len(spiketimes_inh[str(neuron_num)]), ct), linewidths=0, color=color_ie, s=ms_raster, edgecolors=None, marker='o')
#     ax3.set_xlim([t_start, t_stop])
#     ax3.set_xticks([])
#     ax3.set_yticks([])
#     ax3.set_ylabel("Inh", fontsize=fontsize, fontname=font)
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_visible(False)
#     ax3.spines['left'].set_visible(False)
    
#     for x in x_milestones:
#         ax4.axvline(x=x, ymin=y_lim_pop_rate[0], ymax=y_lim_pop_rate[1], linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)
#     ts, pop_rate = get_pop_rate_square_window(spiketimes=spiketimes_exc,t_start=t_start, t_stop=t_stop,window_size=window_pop_rate,n_neurons=n_exc)
#     ax4.plot(ts, pop_rate, color=color_ee, linewidth=linewidth, marker='')
#     ax4.set_xlim([t_start, t_stop])
#     ax4.set_xticks(x_ticks)
#     ax4.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax4.set_ylabel(r'$r^{pop}_{exc}$' + " (Hz)", fontsize=fontsize, fontname=font)
#     ax4.spines['top'].set_visible(False)
#     ax4.spines['right'].set_visible(False)
#     ax4.spines['bottom'].set_linewidth(axwidth)
#     ax4.spines['left'].set_linewidth(axwidth)
#     ax4.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     ax4.set_yticks(y_ticks_pop_rate)
#     ax4.set_ylim(y_lim_pop_rate)
    
#     for x in x_milestones:
#         ax5.axvline(x=x, ymin=y_lim_w[0], ymax=y_lim_w[1], linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)
#     for syn_num in range(n_to_plot_weights):
#         ax5.plot(wee['t'], wee['w'][syn_num, :], color=color_ee, linewidth=linewidth_weights, alpha=alpha_w)
#         ax5.plot(wie['t'], wie['w'][syn_num, :], color=color_ie, linewidth=linewidth_weights, alpha=alpha_w)
#     a1, =ax5.plot(wee['t'], np.mean(wee['w'], axis=0), label=r'$w_{ee}$', color=color_ee, linewidth=linewidth, alpha=1)
#     a3, =ax5.plot(wie['t'], np.mean(wie['w'], axis=0), label=r'$w_{ie}$', color=color_ie, linewidth=linewidth, alpha=1)
#     ax5.set_yscale('log')
#     ax5.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
#     ax5.set_xlim([t_start, t_stop])
#     ax5.set_ylim(y_lim_w)
#     ax5.set_yticks(y_ticks_w)
#     ax5.spines['top'].set_visible(False)
#     ax5.spines['right'].set_visible(False)
#     ax5.spines['bottom'].set_linewidth(axwidth)
#     ax5.spines['left'].set_linewidth(axwidth)
#     ax5.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     ax5.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1*axwidth)
#     ax5.set_xticks(x_ticks)
#     ax5.set_yticklabels(y_ticklabels_w)
#     ax5.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax5.legend(handles=[a1, a3], loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=fontsize, ncol=4, frameon=False,
#                  borderpad=0, labelspacing=3, handlelength=0.5, columnspacing=2)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

def accumulation_plot(fracs_list=None, colormap=plt.cm.copper, figsize=(2, 1), clip_frac_low=1e-4,
                      condition_labels=["stable\nactivity", "stable\nweights", "plausible\ncandidate"],
                      ylim=[1e-4,1.1], yticks=[1e-4,1e-3,1e-2,0.1,1], yticklabels=[r'$<10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'],
                      xlabel="round", xlim=None, labelspacing=0.3, xlabel_colors=None,
                      ylabel="fraction", x_loc_leg=1, y_loc_leg=-0.30, xticklabels=None, marker="_",
                      linewidth=2, axwidth=2, font = "Arial", fontsize=10, ms=0, l=0.2):
    n_datasets = len(fracs_list[0])
    n_conditions_chosen = len(fracs_list)
    rounds = [i for i in range(n_datasets)]
    if xticklabels is None:
        rounds_labels = ["prior"]
        for i in range(n_datasets-1):
            rounds_labels.append(str(i+1))
    else:
        rounds_labels=xticklabels
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colormap(np.linspace(0,1,n_conditions_chosen)))
    color_plots = colormap(np.linspace(0,1,n_conditions_chosen))

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = plt.subplot()

    for i in range(n_conditions_chosen):
        ax.semilogy(rounds, np.clip(fracs_list[i,:], clip_frac_low, 2), "--", linewidth = linewidth/3, ms=ms,
                    marker=marker,clip_on=False, color=color_plots[i])
    
    fracs_list = np.clip(fracs_list, clip_frac_low, 2)
    plots = []
    for d_num in range(n_datasets):
        for i in range(n_conditions_chosen):
            a, = ax.semilogy([rounds[d_num]-l, rounds[d_num]+l], [fracs_list[i,d_num], fracs_list[i,d_num]], "-", 
                    label=condition_labels[i], linewidth = linewidth, ms=ms,
                    marker=marker, color=color_plots[i], clip_on=False)
            plots.append(a)

    ax.set_xticks([i for i in range(n_datasets)])
    if xlim is None:
        xlim = [0,n_datasets-1]
    ax.set_xlim(xlim)
    ax.set_xticklabels(rounds_labels, ha="center")
    ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize , labelpad = 0)

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize , labelpad = 0)
    if xlabel_colors is not None:
        for i in range(n_datasets):
            plt.gca().get_xticklabels()[i].set_color(xlabel_colors[i])

    ax.legend(handles=[plots[i] for i in range(n_conditions_chosen)], loc=(x_loc_leg
    ,y_loc_leg), fontsize=fontsize, frameon=False, borderpad = 0.7, labelspacing = labelspacing,
             handlelength=0.5, handletextpad=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    ax.tick_params(axis='x', width=0, labelsize=fontsize, length=0)

    plt.show()

def plot_evol_dim_post(exp_var_ratios=None, font = "Arial", fontsize=10, linewidth=1.5, axwidth=1.5,
                      dpi=200, colors=None, labels=None, figsize=(2, 1), labelspacing=0.3,
                      xticks=[1,24], xlim=[1,24], xlabel=r'$\theta$' + " dimensions",
                       ylim=[0,0.1], ylabel="explained\nvariance", y_loc_leg=-0.2):

    n_thetas = len(exp_var_ratios[0])
    n_datasets = len(exp_var_ratios)
    
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colormap(np.linspace(0,1,n_datasets)))
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()
    for i in range(n_datasets):
        ax.plot([i+1 for i in range(n_thetas)], 
                exp_var_ratios[i], "-", 
                label = labels[i], 
                linewidth = linewidth,
                color=colors[i])
        
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)

    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize , labelpad=0)
    ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize , labelpad=0)

    ax.legend(loc=(1,y_loc_leg), fontsize=fontsize, frameon=False, borderpad = 0.7, labelspacing=labelspacing,
             handlelength=1, handletextpad=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    plt.show()

# def plot_summary_simulation_bg(spiketimes_exc=None, spiketimes_inh=None, n_to_plot_raster=None, 
#                             n_exc=4096, n_inh=1024, n_input=5000,
#                             wee=None, wei=None, wie=None, wii=None, t_start=None, t_stop=None,
#                             window_pop_rate=0.1, axwidth=4, linewidth=4, n_to_plot_weights=100, ms_raster=0.5,
#                             fontsize=20, figsize=(5, 2), font = "arial",
#                             color_ee=None, color_ei=None, color_ie=None, color_ii=None,
#                             x_ticks=None, x_ticklabels=None, x_label=None, alpha_w=None,  xlim=None,
#                             linewidth_weights=0.5,
#                             y_ticks_we=None, y_ticklabels_we=None, y_lim_we=None, 
#                             y_ticks_wi=None, y_ticklabels_wi=None, y_lim_wi=None,
#                             y_ticks_pop_rate=None, y_lim_pop_rate=None):
    
#     fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=figsize, constrained_layout=True, dpi=600, 
#                                                   gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1.2, 1.2]})
    
#     for ct, neuron_num in enumerate(np.linspace(0, n_exc-1, num=n_to_plot_raster, dtype=int)):
#         ax1.scatter(spiketimes_exc[str(neuron_num)], np.full(len(spiketimes_exc[str(neuron_num)]), ct), linewidths=0, color=color_ee, s=ms_raster, edgecolors=None, marker='o')
#     if xlim is None:
#         ax1.set_xlim([t_start, t_stop])
#     else:
#         ax1.set_xlim(xlim)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_ylabel("Exc", fontsize=fontsize, fontname=font)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['left'].set_visible(False)
    
#     for ct, neuron_num in enumerate(np.linspace(0, n_inh-1, num=n_to_plot_raster, dtype=int)):
#         ax2.scatter(spiketimes_inh[str(neuron_num)], np.full(len(spiketimes_inh[str(neuron_num)]), ct), linewidths=0, color=color_ii, s=ms_raster, edgecolors=None, marker='o')
#     if xlim is None:
#         ax2.set_xlim([t_start, t_stop])
#     else:
#         ax2.set_xlim(xlim)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_ylabel("Inh", fontsize=fontsize, fontname=font)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['left'].set_visible(False)
    
#     ts, pop_rate = get_pop_rate_square_window(spiketimes=spiketimes_exc,t_start=t_start, t_stop=t_stop,window_size=window_pop_rate,n_neurons=n_exc)
#     ax3.plot(ts, pop_rate, color=color_ee, linewidth=linewidth, marker='')
#     if xlim is None:
#         ax3.set_xlim([t_start, t_stop])
#     else:
#         ax3.set_xlim(xlim)
#     ax3.set_xticks(x_ticks)
#     ax3.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax3.set_ylabel(r'$r^{pop}_{exc}$' + " (Hz)", fontsize=fontsize, fontname=font)
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_linewidth(axwidth)
#     ax3.spines['left'].set_linewidth(axwidth)
#     ax3.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     ax3.set_yticks(y_ticks_pop_rate)
#     ax3.set_ylim(y_lim_pop_rate)
    
#     for syn_num in range(n_to_plot_weights):
#         ax4.plot(wee['t'], wee['w'][syn_num, :], color=color_ee, linewidth=linewidth_weights, alpha=alpha_w)
#         ax4.plot(wei['t'], wei['w'][syn_num, :], color=color_ei, linewidth=linewidth_weights, alpha=alpha_w)
#     a2, =ax4.plot(wei['t'], np.mean(wei['w'], axis=0), label=r'$w_{ei}$', color=color_ei, linewidth=linewidth, alpha=1)
#     a1, =ax4.plot(wee['t'], np.mean(wee['w'], axis=0), label=r'$w_{ee}$', color=color_ee, linewidth=linewidth, alpha=1)
#     # ax4.set_yscale('log')
#     # ax4.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
#     if xlim is None:
#         ax4.set_xlim([t_start, t_stop])
#     else:
#         ax4.set_xlim(xlim)
#     ax4.set_ylim(y_lim_we)
#     ax4.set_yticks(y_ticks_we)
#     ax4.spines['top'].set_visible(False)
#     ax4.spines['right'].set_visible(False)
#     ax4.spines['bottom'].set_linewidth(axwidth)
#     ax4.spines['left'].set_linewidth(axwidth)
#     ax4.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     # ax4.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1*axwidth)
#     ax4.set_xticks(x_ticks)
#     ax4.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax4.set_yticklabels(y_ticklabels_we)
#     ax4.set_xlabel(None, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax4.legend(handles=[a1, a2], loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=fontsize, ncol=2, frameon=False,
#                  borderpad=0, labelspacing=-0.1, handlelength=0.1, columnspacing=0.5, handletextpad=0.1)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

#     for syn_num in range(n_to_plot_weights):
#         ax5.plot(wie['t'], wie['w'][syn_num, :], color=color_ie, linewidth=linewidth_weights, alpha=alpha_w)
#         ax5.plot(wii['t'], wii['w'][syn_num, :], color=color_ii, linewidth=linewidth_weights, alpha=alpha_w)
#     a4, =ax5.plot(wii['t'], np.mean(wii['w'], axis=0), label=r'$w_{ii}$', color=color_ii, linewidth=linewidth, alpha=1)
#     a3, =ax5.plot(wie['t'], np.mean(wie['w'], axis=0), label=r'$w_{ie}$', color=color_ie, linewidth=linewidth, alpha=1)
#     # ax5.set_yscale('log')
#     # ax5.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
#     if xlim is None:
#         ax5.set_xlim([t_start, t_stop])
#     else:
#         ax5.set_xlim(xlim)
#     ax5.set_ylim(y_lim_wi)
#     ax5.set_yticks(y_ticks_wi)
#     ax5.spines['top'].set_visible(False)
#     ax5.spines['right'].set_visible(False)
#     ax5.spines['bottom'].set_linewidth(axwidth)
#     ax5.spines['left'].set_linewidth(axwidth)
#     ax5.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     # ax5.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1*axwidth)
#     ax5.set_xticks(x_ticks)
#     ax5.set_yticklabels(y_ticklabels_wi)
#     ax5.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax5.legend(handles=[a3, a4], loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=fontsize, ncol=2, frameon=False,
#                  borderpad=0, labelspacing=0.1, handlelength=0.1, columnspacing=0.5, handletextpad=0.1)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

# def plot_summary_simulation_bgMLP(spiketimes_exc=None, spiketimes_inh=None, n_to_plot_raster=None, 
#                             n_exc=4096, n_inh=1024,
#                             wee=None, wie=None, t_start=None, t_stop=None,
#                             window_pop_rate=0.1, axwidth=4, linewidth=4, n_to_plot_weights=100, ms_raster=0.5,
#                             fontsize=20, figsize=(5, 2), font = "arial",
#                             color_ee=None, color_ie=None, color_ii=None,
#                             x_ticks=None, x_ticklabels=None, linewidth_weights=0.5, xlim=None,
#                             y_ticks_we=None, y_ticklabels_we=None, y_lim_we=None, 
#                             y_ticks_wi=None, y_ticklabels_wi=None, y_lim_wi=None,
#                             x_label=None, alpha_w=None,
#                             y_ticks_pop_rate=None, y_lim_pop_rate=None):
    
#     fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=figsize, constrained_layout=True, dpi=600, 
#                                                   gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1.2, 1.2]})
    
#     for ct, neuron_num in enumerate(np.linspace(0, n_exc-1, num=n_to_plot_raster, dtype=int)):
#         ax1.scatter(spiketimes_exc[str(neuron_num)], np.full(len(spiketimes_exc[str(neuron_num)]), ct), linewidths=0, color=color_ee, s=ms_raster, edgecolors=None, marker='o')
#     if xlim is None:
#         ax1.set_xlim([t_start, t_stop])
#     else:
#         ax1.set_xlim(xlim)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_ylabel("Exc", fontsize=fontsize, fontname=font)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['left'].set_visible(False)
    
#     for ct, neuron_num in enumerate(np.linspace(0, n_inh-1, num=n_to_plot_raster, dtype=int)):
#         ax2.scatter(spiketimes_inh[str(neuron_num)], np.full(len(spiketimes_inh[str(neuron_num)]), ct), linewidths=0, color=color_ii, s=ms_raster, edgecolors=None, marker='o')
#     if xlim is None:
#         ax2.set_xlim([t_start, t_stop])
#     else:
#         ax2.set_xlim(xlim)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_ylabel("Inh", fontsize=fontsize, fontname=font)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['left'].set_visible(False)
    
#     ts, pop_rate = get_pop_rate_square_window(spiketimes=spiketimes_exc,t_start=t_start, t_stop=t_stop,window_size=window_pop_rate,n_neurons=n_exc)
#     ax3.plot(ts, pop_rate, color=color_ee, linewidth=linewidth, marker='')
#     if xlim is None:
#         ax3.set_xlim([t_start, t_stop])
#     else:
#         ax3.set_xlim(xlim)
#     ax3.set_xticks(x_ticks)
#     ax3.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax3.set_ylabel(r'$r^{pop}_{exc}$' + " (Hz)", fontsize=fontsize, fontname=font)
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_linewidth(axwidth)
#     ax3.spines['left'].set_linewidth(axwidth)
#     ax3.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     ax3.set_yticks(y_ticks_pop_rate)
#     ax3.set_ylim(y_lim_pop_rate)
    
#     for syn_num in range(n_to_plot_weights):
#         ax4.plot(wee['t'], wee['w'][syn_num, :], color=color_ee, linewidth=linewidth_weights, alpha=alpha_w)
#     a1, =ax4.plot(wee['t'], np.mean(wee['w'], axis=0), label=r'$w_{ee}$', color=color_ee, linewidth=linewidth, alpha=1)
#     # ax4.set_yscale('log')
#     # ax4.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
#     if xlim is None:
#         ax4.set_xlim([t_start, t_stop])
#     else:
#         ax4.set_xlim(xlim)
#     ax4.set_ylim(y_lim_we)
#     ax4.set_yticks(y_ticks_we)
#     ax4.spines['top'].set_visible(False)
#     ax4.spines['right'].set_visible(False)
#     ax4.spines['bottom'].set_linewidth(axwidth)
#     ax4.spines['left'].set_linewidth(axwidth)
#     ax4.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     # ax4.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1*axwidth)
#     ax4.set_xticks(x_ticks)
#     ax4.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax4.set_yticklabels(y_ticklabels_we)
#     ax4.set_xlabel(None, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax4.legend(handles=[a1], loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=fontsize, ncol=2, frameon=False,
#                  borderpad=0, labelspacing=3, handlelength=0.5, columnspacing=2)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

#     for syn_num in range(n_to_plot_weights):
#         ax5.plot(wie['t'], wie['w'][syn_num, :], color=color_ie, linewidth=linewidth_weights, alpha=alpha_w)
#     a3, =ax5.plot(wie['t'], np.mean(wie['w'], axis=0), label=r'$w_{ie}$', color=color_ie, linewidth=linewidth, alpha=1)
#     # ax5.set_yscale('log')
#     # ax5.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
#     if xlim is None:
#         ax5.set_xlim([t_start, t_stop])
#     else:
#         ax5.set_xlim(xlim)
#     ax5.set_ylim(y_lim_wi)
#     ax5.set_yticks(y_ticks_wi)
#     ax5.spines['top'].set_visible(False)
#     ax5.spines['right'].set_visible(False)
#     ax5.spines['bottom'].set_linewidth(axwidth)
#     ax5.spines['left'].set_linewidth(axwidth)
#     ax5.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     # ax5.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1*axwidth)
#     ax5.set_xticks(x_ticks)
#     ax5.set_yticklabels(y_ticklabels_wi)
#     ax5.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax5.legend(handles=[a3], loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=fontsize, ncol=2, frameon=False,
#                  borderpad=0, labelspacing=3, handlelength=0.5, columnspacing=2)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

def plot_2data_1heatmap(data1, data2, data3, xlabel, ylabel, heatmap_label, 
                        font = "Arial", 
                        fontsize = 20, 
                        linewidth = 3, 
                        xlim = None, 
                        ylim = None,
                        figsize=(10, 6),
                        xticks=None,
                        yticks=None,
                        cbarticks=None,
                        cbarticklabels=None,
                        xhandlepad=None,
                        yhandlepad=None,
                        cbarhandlepad=None,
                        s=1,
                        savepath=None,
                        ordering=False,
                        center_axes = False,
                        dpi=200,
                        clim=None,
                        cmap="Spectral_r",
                        ax=None,
                        fig=None,
                        color_xlabel="black",
                        color_ylabel="black"):
    
    if ax == None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot()

    if ordering is None:
        ind_sort_z = np.random.shuffle(np.array([i for i in range(len(data3))]))
    elif ordering:
        ind_sort_z = np.argsort(data3)
    else:
        ind_sort_z = np.flip(np.argsort(data3))

    
    if clim is not None:
        img = ax.scatter(data1[ind_sort_z], data2[ind_sort_z], c=data3[ind_sort_z], cmap=cmap, s=s, marker='o', edgecolors=None, vmin=clim[0], vmax=clim[1],linewidths = 0) #cividis
    else:
        img = ax.scatter(data1[ind_sort_z], data2[ind_sort_z], c=data3[ind_sort_z], cmap=cmap, s=s, marker='o', edgecolors=None, linewidths = 0) #cividis
    cbar = fig.colorbar(img, label = heatmap_label, aspect=15, ax=ax)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)
    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    
    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if yticks != None:
        ax.set_yticks(yticks)
    if cbarticks != None:
        cbar.set_ticks(cbarticks)
    if cbarticklabels is not None:
        cbar.set_ticklabels(cbarticklabels)
    if center_axes:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.label.set_color(color_xlabel)
    ax.yaxis.label.set_color(color_ylabel)
    
    if savepath != None:
        fig.savefig(savepath,dpi=300,format='png', bbox_inches='tight')

def plot_mean_field_emergence(dataset_list=None,
                              condition_list=None,
                                font = "Arial", 
                                figsize=(10, 6),
                              
                                xlim_EE = [-4,4],
                                ylim_EE = [-0.4,0.4],
                                xticks_EE = [-4,4],
                                yticks_EE = [-0.4,0.4],
                              
                                xlim_EI = [-100, 100],
                                ylim_EI = [-20,20],
                                xticks_EI = [-100, 100],
                                yticks_EI = [-20,20],
                              
                                xlim_IE = [-100,100],
                                ylim_IE = [-22,22],
                                xticks_IE = [-100,100],
                                yticks_IE = [-22,22],
                              
                                xlim_II = [-4,4],
                                ylim_II = [-0.4,0.4],
                                xticks_II = [-4,4],
                                yticks_II = [-0.4,0.4],
                                s=0.5,
                                center_axes=True,
                                dpi=200,
                                clim=[0,50],
                                cmap='nipy_spectral',
                                fontsize=10,
                                linewidth=1.5,
                                xhandlepad=20,
                                yhandlepad=10,
                                cbarhandlepad=0,
                                ordering=True,
                                figsize_small=(1.5,1)
                                ): 
    fig, axs = plt.subplots(5,4,figsize=figsize, constrained_layout=True, dpi=dpi)
    
    for dataset, cond_num in zip(dataset_list, [0,1,2,3]):
        
        condition = apply_n_conditions(dataset, condition_list[cond_num])
        
        tau_pre_EE = dataset['theta'][:,0]
        tau_post_EE = dataset['theta'][:,1]
        alpha_EE = dataset['theta'][:,2]
        beta_EE = dataset['theta'][:,3]
        gamma_EE = dataset['theta'][:,4]
        kappa_EE = dataset['theta'][:,5]
        lambd_EE = kappa_EE*tau_post_EE+gamma_EE*tau_pre_EE

        tau_pre_EI = dataset['theta'][:,6]
        tau_post_EI = dataset['theta'][:,7]
        alpha_EI = dataset['theta'][:,8]
        beta_EI = dataset['theta'][:,9]
        gamma_EI = dataset['theta'][:,10]
        kappa_EI = dataset['theta'][:,11]
        lambd_EI = kappa_EI*tau_post_EI+gamma_EI*tau_pre_EI

        tau_pre_IE = dataset['theta'][:,12]
        tau_post_IE = dataset['theta'][:,13]
        alpha_IE = dataset['theta'][:,14]
        beta_IE = dataset['theta'][:,15]
        gamma_IE = dataset['theta'][:,16]
        kappa_IE = dataset['theta'][:,17]
        lambd_IE = kappa_IE*tau_post_IE+gamma_IE*tau_pre_IE

        tau_pre_II = dataset['theta'][:,18]
        tau_post_II = dataset['theta'][:,19]
        alpha_II = dataset['theta'][:,20]
        beta_II = dataset['theta'][:,21]
        gamma_II = dataset['theta'][:,22]
        kappa_II = dataset['theta'][:,23]
        lambd_II = kappa_II*tau_post_II+gamma_II*tau_pre_II

        r = dataset['rate']
        ri = dataset['rate_i']
        
        axs[0, cond_num] = plot_2data_1heatmap(-alpha_EE[condition] -beta_EE[condition], lambd_EE[condition], r[condition], 
                        r'$-\alpha_{EE}-\beta_{EE}$', 
                        r'$\lambda_{EE}$', 
                        r'$r_{exc}$', 
                        xlim=xlim_EE,
                        ylim=ylim_EE,
                        xticks=xticks_EE,
                        yticks=yticks_EE,
                        clim=clim,
                        cbarticks=clim,
                        center_axes=center_axes,
                        s=s,
                        fontsize=fontsize,
                        linewidth=linewidth,
                        xhandlepad=xhandlepad,
                        yhandlepad=yhandlepad,
                        cbarhandlepad=cbarhandlepad,
                        cmap=cmap, #terrain  nipy_spectral
                        ordering=ordering,
                        figsize=figsize_small,
                        ax=axs[0, cond_num],
                        fig=fig)
        
        axs[1, cond_num] = plot_2data_1heatmap(-alpha_EI[condition]*r[condition], beta_EI[condition] + lambd_EI[condition]*r[condition], ri[condition],
                        r'$-\alpha_{EI}r_{exc}$',
                        r'$\beta_{EI} + r_{exc} \lambda_{EI}$',
                        r'$r_{inh}$', 
                        xlim=xlim_EI,
                        ylim=ylim_EI,
                        xticks=xticks_EI,
                        yticks=yticks_EI,
                        clim=clim,
                        cbarticks=clim,
                        center_axes=center_axes,
                        s=s,
                        fontsize=fontsize,
                        linewidth=linewidth,
                        xhandlepad=xhandlepad,
                        yhandlepad=yhandlepad,
                        cbarhandlepad=cbarhandlepad,
                        cmap=cmap,
                        ordering=ordering,
                        figsize=figsize_small,
                        ax=axs[1, cond_num],
                        fig=fig)
        
        axs[2, cond_num] = plot_2data_1heatmap(-alpha_IE[condition]*ri[condition], beta_IE[condition] + lambd_IE[condition]*ri[condition], r[condition], 
                              r'$-\alpha_{IE}r_{inh}$',
                              r'$\beta_{IE} + r_{inh} \lambda_{IE}$',
                              r'$r_{exc}$',
                        xlim=xlim_IE,
                        ylim=ylim_IE,
                        xticks=xticks_IE,
                        yticks=yticks_IE,
                        clim=clim,
                        cbarticks=clim,
                        center_axes=center_axes,
                        s=s,
                        fontsize=fontsize,
                        linewidth=linewidth,
                        xhandlepad=xhandlepad,
                        yhandlepad=yhandlepad,
                        cbarhandlepad=cbarhandlepad,
                        cmap=cmap, #terrain  nipy_spectral
                        ordering=ordering,
                        figsize=figsize_small,
                        ax=axs[2, cond_num],
                        fig=fig)
        
        axs[3, cond_num] = plot_2data_1heatmap(-alpha_II[condition] -beta_II[condition], lambd_II[condition], ri[condition],
                              r'$-\alpha_{II}-\beta_{II}$',
                              r'$\lambda_{II}$',
                              r'$r_{inh}$',
                        xlim=xlim_II,
                        ylim=ylim_II,
                        xticks=xticks_II,
                        yticks=yticks_II,
                        clim=clim,
                        cbarticks=clim,
                        center_axes=center_axes,
                        s=s,
                        fontsize=fontsize,
                        linewidth=linewidth,
                        xhandlepad=xhandlepad,
                        yhandlepad=yhandlepad,
                        cbarhandlepad=cbarhandlepad,
                        cmap=cmap, #terrain  nipy_spectral
                        ordering=ordering,
                        figsize=figsize_small,
                        ax=axs[3, cond_num],
                        fig=fig)
        
        axs[4, cond_num] = plot_2data_1heatmap(alpha_EE[condition], beta_EE[condition], r[condition],
                              r'$\alpha_{EE}$',
                              r'$\beta_{EE}$',
                              r'$r_{exc}$',
                        xlim=[-2,2],
                        ylim=[-2,2],
                        xticks=[-2,2],
                        yticks=[-2,2],
                        clim=clim,
                        cbarticks=clim,
                        center_axes=center_axes,
                        s=s,
                        fontsize=fontsize,
                        linewidth=linewidth,
                        xhandlepad=xhandlepad,
                        yhandlepad=yhandlepad,
                        cbarhandlepad=cbarhandlepad,
                        cmap=cmap, #terrain  nipy_spectral
                        ordering=ordering,
                        figsize=figsize_small,
                        ax=axs[4, cond_num],
                        fig=fig)
    plt.show()

def plot_mat_poly(mat, save = False, name = "", linewidth = 1.5, fontsize = 10, 
             font = "Arial", figsize=(3,3), labelsx=None, labelsy=None, cmap="Spectral", dpi=200, rotation=0,
            color_ee=None, color_ei=None, color_ie=None, color_ii=None, heatmap_label=None, cbarhandlepad=None,):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    matrice = ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap)
    
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    
    ax.set_xticks([i for i in range(mat.shape[0])])
    ax.set_xticklabels(labelsx)
    ax.set_yticks([i for i in range(mat.shape[0])])
    ax.set_yticklabels(labelsy)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.tick_params(axis = 'x', rotation=rotation, pad=0)


    colors = [color_ee, color_ei, color_ie, color_ii]
    colors = np.repeat(colors, 6)

    for i in range(24):
        plt.gca().get_xticklabels()[i].set_color(colors[i])
        plt.gca().get_yticklabels()[i].set_color(colors[i])

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax.xaxis.set_minor_formatter(ticker.FixedFormatter([labelsx[i*2+1] for i in range(len(labelsx)//2)]))
    # ax.tick_params(axis='x', which='minor', length=5*linewidth, color='black', width=linewidth, labelsize=fontsize, pad=10)
    # ax.tick_params(axis='x', which='both', color='lightgrey')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    
    cbar = fig.colorbar(matrice, cax = cax, drawedges=False)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)
    cbar.ax.set_yticks([-1,0,1])

    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    
    plt.show()

def plot_BND_subplot_rnrf(data1, data2, data3, xlabel, ylabel, heatmap_label, 
                        font = "Arial", 
                        fontsize = 20, 
                        linewidth = 3, 
                        xlim = None, 
                        ylim = None,
                        figsize=(10, 6),
                        xticks=None,
                        yticks=None,
                        cbarticks=None,
                        xhandlepad=None,
                        yhandlepad=None,
                        cbarhandlepad=None,
                        cbarticklabels=None,
                        s=1,
                        savepath=None,
                        ordering=False,
                        center_axes = False,
                        dpi=200,
                        clim=None,
                        cmap="Spectral_r",
                        ax=None,
                        fig=None,
                        color_xlabel="black",
                        color_ylabel="black",
                        l_subplot=None,
                        b_subplot=None,
                        h_subplot=None,
                        w_subplot=None,
                        linewidth_small=None,
                        xticks_small=None,
                        xlim_small=None):
    
    if ax == None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot()

    if ordering is None:
        ind_sort_z = np.random.shuffle(np.array([i for i in range(len(data3))]))
    elif ordering:
        ind_sort_z = np.argsort(data3)
    else:
        ind_sort_z = np.flip(np.argsort(data3))

    
    if clim is not None:
        img = ax.scatter(data1[ind_sort_z], data2[ind_sort_z], c=data3[ind_sort_z], cmap=cmap, s=s, marker='o', edgecolors=None, vmin=clim[0], vmax=clim[1]) #cividis
    else:
        img = ax.scatter(data1[ind_sort_z], data2[ind_sort_z], c=data3[ind_sort_z], cmap=cmap, s=s, marker='o', edgecolors=None) #cividis
    cbar = fig.colorbar(img, label = heatmap_label, aspect=15, ax=ax)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)
    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    
    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if yticks != None:
        ax.set_yticks(yticks)
    if cbarticks != None:
        cbar.set_ticks(cbarticks)
    if cbarticklabels is not None:
        cbar.set_ticklabels(cbarticklabels)
    if center_axes:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.label.set_color(color_xlabel)
    ax.yaxis.label.set_color(color_ylabel)
    
    ax2 = fig.add_axes([l_subplot, b_subplot, w_subplot, h_subplot])
    to_plot, bins = np.histogram(data3, bins=100)
    ax2.plot(bins[1:], to_plot, color='black', linewidth=linewidth_small)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(linewidth_small)
    ax2.spines['left'].set_linewidth(linewidth_small)
    ax2.set_xticks(xticks_small)
    ax2.set_yticks([])
    if xlim_small is not None:
        ax2.set_xlim(xlim_small)
    ax2.tick_params(width=linewidth_small, labelsize=fontsize/2, length=2*linewidth_small,pad=0)
    
    if savepath != None:
        fig.savefig(savepath,dpi=300,format='png', bbox_inches='tight')

class MLP(nn.Module):
    def __init__(self, ni, nh1, nh2, eta):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(ni, nh1, bias=False)
        self.nl1 = torch.nn.Sigmoid()
        self.l2 = nn.Linear(nh1, nh2, bias=False)
        self.nl2 = torch.nn.Sigmoid()
        self.l3 = nn.Linear(nh2, 1, bias = True)
        self.rs = torch.nn.Flatten(0, 1)
        self.eta = eta

    def forward(self, x):
        x = self.l1(x)
        x = self.nl1(x)
        x = self.l2(x)
        x = self.nl2(x)
        x = self.l3(x)
        x = self.rs(x)
        return self.eta*x
    
def plot_rule_MLP(net_pre=None, net_post=None, var_dict=None,
              tau_pre1=None, tau_pre2=None, tau_post1=None, tau_post2=None,
              n_bins=1000, dtype=None,
              x_lim=[0,1], logx=False, logy=False, x_label="", ax=None, 
              color=None, save_path=None, linewidth=3, fontsize=20, figsize=(5,3), font="arial",
              x_ticks=None, x_ticklabels=None, rotation=0, y_label=None,
              y_lim=None, y_ticks=None, y_ticklabels=None, axwidth=3, labelpad_xlabel=30,
              labelpad_ylabel=-40, dpi=200, xticks_pad=None, yticks_pad=None, color_ylabel='black',
              rescale_trace=2, rescale_v=15., rescale_w=1., rescale_cexc=15., rescale_cinh=15.,
              heatmap_label=None, cbarhandlepad=1, cmap=cm.copper, cbarticks=None, cbarticklabels=None):
    
    n_cond = len(var_dict["xpre"])  
    color_plots = cmap(np.linspace(0,1,n_cond))

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
    plt.rcParams["axes.axisbelow"] = False
    
    deltats = np.linspace(x_lim[0], x_lim[1],num=n_bins)
    ind_t_pos = 0
    while deltats[ind_t_pos] < 0:
        ind_t_pos += 1
        
    x_pres = np.array([0 for i in range(ind_t_pos)])
    x_pres = np.append(x_pres, np.array([np.exp(-np.abs(deltats[i])/tau_pre1) for i in range(ind_t_pos, len(deltats))]), axis=0)
    x_pres = rescale_trace*x_pres
    
    x_posts = np.array([np.exp(-np.abs(deltats[i])/tau_pre1) for i in range(ind_t_pos)])
    x_posts = np.append(x_posts, np.array([0 for i in range(ind_t_pos, len(deltats))]), axis=0)
    x_posts = rescale_trace*x_posts
    
    for cond_num in range(n_cond):
        Xs_on_pre = [[var_dict["xpre"][cond_num]*rescale_trace,
                      i,
                      rescale_w*var_dict["w"][cond_num],
                      rescale_v*var_dict["v"][cond_num],
                      rescale_cexc*var_dict["cexc"][cond_num],
                      rescale_cinh*var_dict["cinh"][cond_num]] for i in x_posts]
        Xs_on_post = [[i,
                       var_dict["xpost"][cond_num]*rescale_trace,
                       rescale_w*var_dict["w"][cond_num],
                       rescale_v*var_dict["v"][cond_num],
                       rescale_cexc*var_dict["cexc"][cond_num],
                       rescale_cinh*var_dict["cinh"][cond_num]] for i in x_pres]
        dws = net_pre(torch.tensor(Xs_on_pre, dtype=dtype)) + net_post(torch.tensor(Xs_on_post, dtype=dtype))
        ax.plot(deltats, dws.detach().numpy(), color=color_plots[cond_num],
                linewidth=linewidth)
            
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation = rotation)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = labelpad_xlabel)
    
    
    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize, fontname=font, labelpad = labelpad_ylabel, color=color_ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    norm = colors.Normalize(vmin=None, vmax=None, clip=False)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label=heatmap_label, cax=cax)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)
    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    if cbarticks != None:
        cbar.set_ticks(cbarticks)
    if cbarticklabels is not None:
        cbar.set_ticklabels(cbarticklabels)

    ax.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    plt.show()

def plot_2data(data1, data2, xlabel, ylabel,
                        font = "Arial", 
                        fontsize = 20, 
                        linewidth = 3, 
                        xlim = None, 
                        ylim = None,
                        figsize=(10, 6),
                        xticks=None,
                        yticks=None,
                        xhandlepad=None,
                        yhandlepad=None,
                        s=1,
                        savepath=None,
                        center_axes = False,
                        dpi=200,
                        n_to_plot=1000,
                        color='black',
                        ax=None,
                        fig=None,
                        color_xlabel="black",
                        color_ylabel="black",
                        marker='o'):
    
    ind_plot = np.array([i for i in range(len(data1))])
    np.random.shuffle(ind_plot)
    ind_plot = ind_plot[:n_to_plot]

    if ax == None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot()
    
    img = ax.scatter(data1[ind_plot], data2[ind_plot], s=s, marker=marker, edgecolors=None, color=color, linewidths = 0)
    
    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if yticks != None:
        ax.set_yticks(yticks)
    if center_axes:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.label.set_color(color_xlabel)
    ax.yaxis.label.set_color(color_ylabel)
    
    if savepath != None:
        fig.savefig(savepath,dpi=300,format='png', bbox_inches='tight')

def plot_mat_MLP(mat, save = False, name = "", linewidth = 1.5, fontsize = 10, 
             font = "Arial", figsize=(3,3), labelsx=None, labelsy=None, cmap="Spectral", dpi=200, rotation=0,
            color_ee=None, color_ie=None, heatmap_label=None, cbarhandlepad=None,):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    reordered_matrix = mat[:,[0,2,3,4,5,6,7,8,9,10,11,1,12,13,14,15,16,17,18,19,20,21]]
    reordered_matrix = reordered_matrix[[0,2,3,4,5,6,7,8,9,10,11,1,12,13,14,15,16,17,18,19,20,21],:]
    
    matrice = ax.imshow(reordered_matrix, vmin=-1, vmax=1, cmap=cmap)
    
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    
    # ax.set_xticks([0,3,8,11,14,19])
    # ax.set_yticks([0,3,8,11,14,19])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_patch(patches.Rectangle((-2,-0.5),1.5,1, edgecolor='black', facecolor=color_ee, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((-2,0.5),1.5,5, edgecolor='black', facecolor=color_ee, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((-2,5.5),1.5,5, edgecolor='black', facecolor=color_ee, lw=linewidth, clip_on=False))
    fig.text(0.1, 0.845, r'$\eta$', fontsize=fontsize, fontname=font, ha='center', color='white', rotation=-90)
    fig.text(0.098, 0.71, r'$W_{pre}$', fontsize=fontsize, fontname=font, ha='center', color='white', rotation=-90)
    fig.text(0.098, 0.54, r'$W_{post}$', fontsize=fontsize, fontname=font, ha='center', color='white', rotation=-90)

    ax.add_patch(patches.Rectangle((-2,10.5),1.5,1, edgecolor='black', facecolor=color_ie, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((-2,11.5),1.5,5, edgecolor='black', facecolor=color_ie, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((-2,16.5),1.5,5, edgecolor='black', facecolor=color_ie, lw=linewidth, clip_on=False))
    fig.text(0.1, 0.4755, r'$\eta$', fontsize=fontsize, fontname=font, ha='center', color='white', rotation=-90)
    fig.text(0.098, 0.35, r'$W_{pre}$', fontsize=fontsize, fontname=font, ha='center', color='white', rotation=-90)
    fig.text(0.098, 0.17, r'$W_{post}$', fontsize=fontsize, fontname=font, ha='center', color='white', rotation=-90)

    ax.add_patch(patches.Rectangle((-0.5,21.5),1,1.5, edgecolor='black', facecolor=color_ee, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((0.5,21.5),5,1.5, edgecolor='black', facecolor=color_ee, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((5.5,21.5),5,1.5, edgecolor='black', facecolor=color_ee, lw=linewidth, clip_on=False))
    fig.text(0.14, 0.092, r'$\eta$', fontsize=fontsize, fontname=font, ha='center', color='white')
    fig.text(0.245, 0.092, r'$W_{pre}$', fontsize=fontsize, fontname=font, ha='center', color='white')
    fig.text(0.415, 0.092, r'$W_{post}$', fontsize=fontsize, fontname=font, ha='center', color='white')

    ax.add_patch(patches.Rectangle((10.5,21.5),1,1.5, edgecolor='black', facecolor=color_ie, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((11.5,21.5),5,1.5, edgecolor='black', facecolor=color_ie, lw=linewidth, clip_on=False))
    ax.add_patch(patches.Rectangle((16.5,21.5),5,1.5, edgecolor='black', facecolor=color_ie, lw=linewidth, clip_on=False))
    fig.text(0.14+0.37, 0.092, r'$\eta$', fontsize=fontsize, fontname=font, ha='center', color='white')
    fig.text(0.245+0.37, 0.092, r'$W_{pre}$', fontsize=fontsize, fontname=font, ha='center', color='white')
    fig.text(0.415+0.37, 0.092, r'$W_{post}$', fontsize=fontsize, fontname=font, ha='center', color='white')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    
    cbar = fig.colorbar(matrice, cax = cax, drawedges=False)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)
    cbar.ax.set_yticks([-1,0,1])

    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    
    plt.show()

# def ISP_pre(x_post, eta, target, tau_post):
#     alpha = target*2*tau_post
#     return(eta*(-alpha + x_post))

# def ISP_post(x_pre, eta):
#     return(eta*x_pre)

# def get_training_data(ns, xrange, dtype, device, eta, tau, target, name):
#     x_train = np.linspace(xrange[0], xrange[1], num = ns)
#     if name == "pre":
#         dw_train = [ISP_pre(x, eta, target, tau) for x in x_train]
#         x_train = [[0, i, 1, -1, 1, 1] for i in x_train]
#     elif name == "post":
#         dw_train = [ISP_post(x, eta) for x in x_train]
#         x_train = [[i, 0, 1, -1, 1, 1] for i in x_train]
#     else:
#         print("unrecognised name")
    
#     return(torch.reshape(torch.tensor(x_train, device = device, dtype = dtype), (ns, 6)), \
#            torch.tensor(dw_train, device = device, dtype = dtype))

# def plot_rule_vs_ground_truth(model, model_untrained, ns, xrange, dtype, device, eta, tau, target, name, criterion,
#                               color_gt=None, color_model=None, linewidth=2, dpi=200, figsize=(1.5,1),
#                               x_lim=None, x_ticks=None, x_ticklabels=None, x_label=None, fontsize=10, font="Arial",
#                               y_lim=None, y_ticks=None, y_ticklabels=None, y_label=None, axwidth=2, ):
    
#     x = np.linspace(xrange[0], xrange[1], num = ns)
#     x_train, dw_truth = get_training_data(ns, xrange, dtype, device, eta, tau, target, name)
#     dw_pred = model(x_train)
#     dw_pred_untrained = model_untrained(x_train)
#     print(criterion(dw_truth, dw_pred))
#     # plt.plot(x, dw_truth.detach().numpy())
#     # plt.plot(x, dw_pred.detach().numpy())
    
#     fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=dpi)

#     line1, = ax.plot(x, dw_truth.detach().numpy(), linestyle='-', color=color_gt, linewidth=linewidth, marker='', label="ground-truth")
#     line2, = ax.plot(x, dw_pred.detach().numpy(), linestyle='dotted', color=color_model, linewidth=linewidth, marker='',label="trained MLP")
#     # ax.plot(x, dw_pred_untrained.detach().numpy(), linestyle='dotted', color=color_model, linewidth=linewidth, marker='')


    

#     if x_lim is not None:
#         ax.set_xlim([x_lim[0], x_lim[1]])
#         ax.set_xticks([x_lim[0], x_lim[1]])
#     if x_ticks is not None:
#         ax.set_xticks(x_ticks)
#     if x_ticklabels is not None:
#         ax.set_xticklabels(x_ticklabels)
#     ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
    
#     if y_lim is not None:
#         ax.set_ylim([y_lim[0], y_lim[1]])
#         ax.set_yticks([y_lim[0], y_lim[1]])
#     if y_ticks is not None:
#         ax.set_yticks(y_ticks)
#     if y_ticklabels is not None:
#         ax.set_yticklabels(y_ticklabels)
#     ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(axwidth)
#     ax.spines['left'].set_linewidth(axwidth)
#     ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     for tick in ax.get_yticklabels():
#         tick.set_fontname(font)
#     for tick in ax.get_yticklabels():
#         tick.set_fontname(font)

#     legend = ax.legend(handles=[line1,  line2], loc='upper left', bbox_to_anchor=(1, 0.9), 
#               fontsize=fontsize, ncol=1, frameon=False,
#               borderpad=0, labelspacing=1.2, handlelength=1, 
#               columnspacing=0, handletextpad=0.5, borderaxespad=0.6)
    
#     plt.show()
#     return()

# def plot_summary_simulation_bgMLPIE(spiketimes_exc=None, spiketimes_inh=None, n_to_plot_raster=None, 
#                             n_exc=4096, n_inh=1024,
#                             wie=None, t_start=None, t_stop=None,
#                             window_pop_rate=0.1, axwidth=4, linewidth=4, n_to_plot_weights=100, ms_raster=0.5,
#                             fontsize=20, figsize=(5, 2), font = "arial",
#                             color_ee=None, color_ie=None, color_ii=None,
#                             x_ticks=None, x_ticklabels=None, linewidth_weights=0.5, xlim=None,
#                             y_ticks_we=None, y_ticklabels_we=None, y_lim_we=None, 
#                             y_ticks_wi=None, y_ticklabels_wi=None, y_lim_wi=None,
#                             x_label=None, alpha_w=None,
#                             y_ticks_pop_rate=None, y_lim_pop_rate=None):
    
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=figsize, constrained_layout=True, dpi=600, 
#                                                   gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1.2]})
    
#     for ct, neuron_num in enumerate(np.linspace(0, n_exc-1, num=n_to_plot_raster, dtype=int)):
#         ax1.scatter(spiketimes_exc[str(neuron_num)], np.full(len(spiketimes_exc[str(neuron_num)]), ct), linewidths=0, color=color_ee, s=ms_raster, edgecolors=None, marker='o')
#     if xlim is None:
#         ax1.set_xlim([t_start, t_stop])
#     else:
#         ax1.set_xlim(xlim)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax1.set_ylabel("Exc", fontsize=fontsize, fontname=font)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['left'].set_visible(False)
    
#     for ct, neuron_num in enumerate(np.linspace(0, n_inh-1, num=n_to_plot_raster, dtype=int)):
#         ax2.scatter(spiketimes_inh[str(neuron_num)], np.full(len(spiketimes_inh[str(neuron_num)]), ct), linewidths=0, color=color_ii, s=ms_raster, edgecolors=None, marker='o')
#     if xlim is None:
#         ax2.set_xlim([t_start, t_stop])
#     else:
#         ax2.set_xlim(xlim)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_ylabel("Inh", fontsize=fontsize, fontname=font)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['left'].set_visible(False)
    
#     ts, pop_rate = get_pop_rate_square_window(spiketimes=spiketimes_exc,t_start=t_start, t_stop=t_stop,window_size=window_pop_rate,n_neurons=n_exc)
#     ax3.plot(ts, pop_rate, color=color_ee, linewidth=linewidth, marker='')
#     if xlim is None:
#         ax3.set_xlim([t_start, t_stop])
#     else:
#         ax3.set_xlim(xlim)
#     ax3.set_xticks(x_ticks)
#     ax3.set_xticklabels(["" for i in range(len(x_ticks))])
#     ax3.set_ylabel(r'$r^{pop}_{exc}$' + " (Hz)", fontsize=fontsize, fontname=font)
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_linewidth(axwidth)
#     ax3.spines['left'].set_linewidth(axwidth)
#     ax3.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     ax3.set_yticks(y_ticks_pop_rate)
#     ax3.set_ylim(y_lim_pop_rate)

#     for syn_num in range(n_to_plot_weights):
#         ax4.plot(wie['t'], wie['w'][syn_num, :], color=color_ie, linewidth=linewidth_weights, alpha=alpha_w)
#     a3, =ax4.plot(wie['t'], np.mean(wie['w'], axis=0), label=r'$w_{ie}$', color=color_ie, linewidth=linewidth, alpha=1)
#     # ax5.set_yscale('log')
#     # ax5.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
#     if xlim is None:
#         ax4.set_xlim([t_start, t_stop])
#     else:
#         ax4.set_xlim(xlim)
#     ax4.set_ylim(y_lim_wi)
#     ax4.set_yticks(y_ticks_wi)
#     ax4.spines['top'].set_visible(False)
#     ax4.spines['right'].set_visible(False)
#     ax4.spines['bottom'].set_linewidth(axwidth)
#     ax4.spines['left'].set_linewidth(axwidth)
#     ax4.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     # ax5.tick_params(axis='y', which="minor", width=0.5*axwidth, labelsize=0, labelcolor='w', length=1*axwidth)
#     ax4.set_xticks(x_ticks)
#     ax4.set_yticklabels(y_ticklabels_wi)
#     ax4.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
#     leg = ax4.legend(handles=[a3], loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=fontsize, ncol=2, frameon=False,
#                  borderpad=0, labelspacing=3, handlelength=0.5, columnspacing=2)
#     for line in leg.get_lines():
#         line.set_linewidth(linewidth)

# def plot_loss(loss_hist, n_meta_it, fontsize = 15, linewidth = 2, font = 'Arial', alpha = 1, dpi=600, color='black', figsize=(2.5, 1.5)):
#     fig, ax = plt.subplots(figsize=figsize, dpi=600)
#     ax.semilogy(loss_hist, linewidth = linewidth, color = color, alpha = alpha)
#     ax.set_xlabel('meta-iterations', fontsize=fontsize, fontname=font)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(linewidth)
#     ax.spines['left'].set_linewidth(linewidth)
#     ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
#     ax.tick_params(which = 'minor', width=0, labelsize=0, length=0)
#     ax.set_xticks([0,int(n_meta_it/2), n_meta_it])
#     ax.set_yticks([1e-2,1,100])
#     ax.set_ylabel(r'$L$', fontsize=fontsize, fontname=font, labelpad = -6)
#     ax.set_xlim([0,n_meta_it])
#     plt.show()   

def frac_metrics_longevity(fracs_list1=None, fracs_list2=None, colors_plot=["black","red"], figsize=(2, 1), rotation_xlabel=45,
                      ylim=[1e-4,1.1], yticks=[1e-4,1e-3,1e-2,0.1,1], yticklabels=[r'$<10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'],
                      xlabel="round", xlim=None, labelspacing=0.3, xlabel_colors=None,
                      ylabel="fraction", x_loc_leg=1, y_loc_leg=-0.30, xticklabels=None, marker="_",
                      linewidth=2, axwidth=2, font = "Arial", fontsize=10, ms=0, l=0.2):
    
    n_fractions = len(fracs_list1)

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = plt.subplot()

    for i in yticks:
        ax.plot(xlim, [i,i], color="black", alpha=0.2, linewidth = linewidth*0.3, linestyle="--")
    
    for i in range(n_fractions):
        a, = ax.plot([i-l, i+l], [fracs_list1[i], fracs_list1[i]], "-", linewidth = linewidth, ms=ms,
                marker=marker, color=colors_plot[0], label="Polynomial")
        b, = ax.plot([i-l, i+l], [fracs_list2[i], fracs_list2[i]], "-", linewidth = linewidth, ms=ms,
                marker=marker, color=colors_plot[1], label="MLP")
        

    ax.set_xticks([i for i in range(n_fractions)])
    if xlim is None:
        xlim = [0,n_fractions-1]
    ax.set_xlim(xlim)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, ha="center", va="top", rotation = rotation_xlabel)
    ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize , labelpad = 0)

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize , labelpad = 5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    ax.tick_params(axis='x', width=0, labelsize=fontsize, length=0)
    if xlabel_colors is not None:
        for i in range(n_fractions):
            plt.gca().get_xticklabels()[i].set_color(xlabel_colors[i])

    ax.legend(handles=[a,b], loc=(x_loc_leg
    ,y_loc_leg), fontsize=fontsize, frameon=False, borderpad = 0.7, labelspacing = labelspacing,
             handlelength=0.5, handletextpad=0.5, ncol=2, columnspacing=3)

    plt.show()

def frac_metrics_robustness(fracs_list=None, colors_plot=["black","red"], figsize=(2, 1), rotation_xlabel=45,
                      ylim=[1e-4,1.1], yticks=[1e-4,1e-3,1e-2,0.1,1], yticklabels=[r'$<10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$'],
                      xlabel="round", xlim=None, labelspacing=0.3, xlabel_colors=None, legend_labels=None,
                      ylabel="fraction", x_loc_leg=1, y_loc_leg=-0.30, xticklabels=None, marker="_",
                      linewidth=2, axwidth=2, font = "Arial", fontsize=10, ms=0, l=0.2):
    
    n_fractions = len(fracs_list)
    n_datasets = len(fracs_list[0])

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = plt.subplot()

    for i in yticks:
        ax.plot(xlim, [i,i], color="black", alpha=0.2, linewidth = linewidth*0.3, linestyle="--")
    
    plots=[]
    for d in range(n_datasets):
        for i in range(n_fractions):
            a, = ax.plot([i-l, i+l], [fracs_list[i,d], fracs_list[i,d]], "-", linewidth = linewidth, ms=ms,
                    marker=marker, color=colors_plot[d], label=legend_labels[d])
        plots.append(a)
        

    ax.set_xticks([i for i in range(n_fractions)])
    if xlim is None:
        xlim = [0,n_fractions-1]
    ax.set_xlim(xlim)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, ha="center", va="top", rotation = rotation_xlabel)
    ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize , labelpad = 0)

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize , labelpad = 5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    ax.tick_params(axis='x', width=0, labelsize=fontsize, length=0)
    if xlabel_colors is not None:
        for i in range(n_fractions):
            plt.gca().get_xticklabels()[i].set_color(xlabel_colors[i])

    ax.legend(handles=plots, loc=(x_loc_leg
    ,y_loc_leg), fontsize=fontsize, frameon=False, borderpad = 0.7, labelspacing = labelspacing,
             handlelength=0.5, handletextpad=0.5, ncol=1, columnspacing=3)

    plt.show()

def plot_2_w_distr(xs=None, n_bins=100, labels=None, x_lim=[0,1], logx=False, logy=False, x_label="", ax=None, xhandlepad=0, yhandlepad=0,
            colors=["black", "black", "black"], save_path=None, linewidth=2, fontsize=10, figsize=(1.5, 1), font="arial", xlabel_color="black",
            x_ticks=None, x_ticklabels=None, rotation=0, range_hist=None, dpi=600, labelspacing=1.2, bbox_to_anchor=(1, 1)):
   
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.minorticks_off()

    hist, bin_edges = np.histogram(np.clip(xs[0], range_hist[0], range_hist[1]), bins=np.linspace(range_hist[0], range_hist[1],num=n_bins))
    ax.plot(bin_edges[1:], hist, color=colors[0], label=labels[0], linewidth=linewidth, zorder=0.9)
    
    hist, bin_edges = np.histogram(np.clip(xs[1], range_hist[0], range_hist[1]), bins=np.linspace(range_hist[0], range_hist[1],num=n_bins))
    ax.plot(bin_edges[1:], hist, color=colors[1], label=labels[1], linewidth=linewidth, zorder=0.8)

    if len(xs) > 2:
        hist, bin_edges = np.histogram(np.clip(xs[2], range_hist[0], range_hist[1]), bins=np.linspace(range_hist[0], range_hist[1],num=n_bins))
        ax.plot(bin_edges[1:], hist, color=colors[2], label=labels[2], linewidth=linewidth, zorder=0.7)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, color=xlabel_color)
    if logx:
        ax.set_xscale('log')
        ax.tick_params(axis='x', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
    if logy:
        ax.set_yscale('log')
        ax.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
        ax.set_ylabel("frequency \nlog-scale", fontname=font, fontsize=fontsize , labelpad=yhandlepad)
    else:
        ax.set_ylabel("frequency", fontname=font, fontsize=fontsize , labelpad=yhandlepad)
    if x_lim != "default":
        ax.set_xlim([x_lim[0], x_lim[1]])
    ax.set_yticks([])
    
    if x_lim is not None:
        ax.set_xlim([x_lim[0], x_lim[1]])
        ax.set_xticks([x_lim[0], x_lim[1]])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation = rotation)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad=xhandlepad)
        
    ax.legend(loc='upper left', bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, ncol=1, frameon=False,
                 borderpad=0, labelspacing=labelspacing, handlelength=0.5, columnspacing=1, handletextpad=0.5, borderaxespad=0.6)
    
    if save_path!=None and ax != None: 
        fig.savefig(save_path+ ".png", format='png', dpi=800, transparent=True, bbox_inches='tight')
    
    return(ax)

def plot_2data_robustnessBND(data1, data2, xlabel, ylabel,
                        font = "Arial", 
                        fontsize = 10, 
                        linewidth = 1.5, 
                        xlim = None, 
                        ylim = None,
                        figsize=(1.5, 1),
                        xticks=None,
                        yticks=None,
                        xhandlepad=None,
                        yhandlepad=None,
                        s=1,
                        center_axes = False,
                        dpi=600,
                        color='black',
                        color_xlabel="black",
                        color_ylabel="black",
                        marker='o',
                        linewidth_line=1.5,
                        color_line="black",
                        alpha_line=0.5,
                        xticklabels=None,
                        yticklabels=None,
                        zorder_line=0.1):

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()
    
    img = ax.scatter(data1, data2, s=s, marker=marker, edgecolors=None, color=color, linewidths = 0, zorder=0.2)

    start=min(xlim[0], ylim[0])
    stop=max(xlim[1], ylim[1])

    ax.plot([start, stop], [start, stop], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
    
    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if xticklabels != None:
        ax.set_xticklabels(xticklabels)
    if yticks != None:
        ax.set_yticks(yticks)
    if yticklabels != None:
        ax.set_yticklabels(yticklabels)
    if center_axes:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.label.set_color(color_xlabel)
    ax.yaxis.label.set_color(color_ylabel)

def plot_3data_BND(data1, data2, data3, xlabel, ylabel, heatmap_label, 
                        font = "Arial", 
                        fontsize = 20, 
                        linewidth = 3, 
                        xlim = None, 
                        ylim = None,
                        figsize=(10, 6),
                        xticks=None,
                        yticks=None,
                        cbarticks=None,
                        cbarticklabels=None,
                        xhandlepad=None,
                        yhandlepad=None,
                        cbarhandlepad=None,
                        s=1,
                        savepath=None,
                        ordering=False,
                        center_axes = False,
                        dpi=200,
                        clim=None,
                        cmap="Spectral_r",
                        color_xlabel="black",
                        color_ylabel="black",
                        linewidth_line=1.5,
                        color_line="black",
                        alpha_line=0.5,
                        xticklabels=None,
                        yticklabels=None,
                        plot_ax=False,
                        plot_diag=False):
    

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()

    if ordering is None:
        ind_sort_z = np.random.shuffle(np.array([i for i in range(len(data3))]))
    elif ordering:
        ind_sort_z = np.argsort(data3)
    else:
        ind_sort_z = np.flip(np.argsort(data3))

    start=min(xlim[0], ylim[0])
    stop=max(xlim[1], ylim[1])

    if plot_diag:
        ax.plot([start, stop], [start, stop], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=0.1)
    if plot_ax:
        ax.plot([start, stop], [0, 0], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=0.1)
        ax.plot([0, 0], [start, stop], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=0.1)


    
    if clim is not None:
        img = ax.scatter(data1[ind_sort_z], data2[ind_sort_z], c=data3[ind_sort_z], cmap=cmap, s=s, marker='o', edgecolors=None, vmin=clim[0], vmax=clim[1],linewidths = 0) #cividis
    else:
        img = ax.scatter(data1[ind_sort_z], data2[ind_sort_z], c=data3[ind_sort_z], cmap=cmap, s=s, marker='o', edgecolors=None, linewidths = 0) #cividis
    cbar = fig.colorbar(img, label = heatmap_label, aspect=15, ax=ax)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)
    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    
    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if yticks != None:
        ax.set_yticks(yticks)
    if cbarticks != None:
        cbar.set_ticks(cbarticks)
    if cbarticklabels is not None:
        cbar.set_ticklabels(cbarticklabels)
    if center_axes:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.label.set_color(color_xlabel)
    ax.yaxis.label.set_color(color_ylabel)
    
    if savepath != None:
        fig.savefig(savepath,dpi=300,format='png', bbox_inches='tight')

def plot_r_distr(xs=None, n_bins=100, labels=None, x_lim=[0,1], logx=False, logy=False, x_label="", ax=None, xhandlepad=0, yhandlepad=0,
            colors=["black", "black", "black"], save_path=None, linewidth=2, fontsize=10, figsize=(1.5, 1), font="arial", xlabel_color="black",
            x_ticks=None, x_ticklabels=None, rotation=0, range_hist=None, dpi=600, labelspacing=1.2, bbox_to_anchor=(1, 1),
            y_lim=None, y_ticks=None):
   
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.minorticks_off()

    hist, bin_edges = np.histogram(np.clip(xs[0], range_hist[0], range_hist[1]), density=True, bins=np.linspace(range_hist[0], range_hist[1],num=n_bins))
    ax.plot(bin_edges[1:], hist, color=colors[0], label=labels[0], linewidth=linewidth, zorder=0.9)

    hist, bin_edges = np.histogram(np.clip(xs[1], range_hist[0], range_hist[1]), density=True, bins=np.linspace(range_hist[0], range_hist[1],num=n_bins))
    ax.plot(bin_edges[1:], hist, color=colors[1], label=labels[1], linewidth=linewidth, zorder=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, color=xlabel_color)
    if logx:
        ax.set_xscale('log')
        ax.tick_params(axis='x', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
    if logy:
        ax.set_yscale('log')
        ax.tick_params(axis='y', which="minor", width=0.5*linewidth, labelsize=0, labelcolor='w', length=1.5*linewidth)
        ax.set_ylabel("frequency \nlog-scale", fontname=font, fontsize=fontsize , labelpad=yhandlepad)
    else:
        ax.set_ylabel("frequency", fontname=font, fontsize=fontsize , labelpad=yhandlepad)
    if x_lim != "default":
        ax.set_xlim([x_lim[0], x_lim[1]])
    
    if x_lim is not None:
        ax.set_xlim([x_lim[0], x_lim[1]])
        ax.set_xticks([x_lim[0], x_lim[1]])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation = rotation)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad=xhandlepad)
    
    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks([])

    ax.legend(loc='upper left', bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, ncol=1, frameon=False,
                 borderpad=0, labelspacing=labelspacing, handlelength=0.5, columnspacing=1, handletextpad=0.5, borderaxespad=0.6)

    
    return(ax)

# def plot_2Dhist(data1, data2,
#                 xlabel=None,
#                 ylabel=None,
#                 n_bins=100,
#                 range_2Dhist=[[-0.5, 1.5], [-0.5, 1.5]],
#                 font = "Arial", 
#                 fontsize = 10, 
#                 linewidth = 1.5, 
#                 xlim = None, 
#                 ylim = None,
#                 figsize=(1.5, 1),
#                 xticks=None,
#                 yticks=None,
#                 xhandlepad=None,
#                 yhandlepad=None,
#                 s=1,
#                 center_axes = False,
#                 dpi=600,
#                 color='black',
#                 color_xlabel="black",
#                 color_ylabel="black",
#                 marker='o',
#                 linewidth_line=1.5,
#                 color_line="black",
#                 alpha_line=0.5,
#                 xticklabels=None,
#                 yticklabels=None,
#                 zorder_line=0.1,
#                 cmap="magma_r"):
    
#     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
#     H, xedges, yedges = np.histogram2d(data1, data2, bins=n_bins, range=range_2Dhist, density=None, weights=None)
#     ax.imshow(H.T, interpolation='nearest', origin='lower',
#             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, aspect='auto')
#     ax.plot(range_2Dhist[0], range_2Dhist[1], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
#     ax.plot(range_2Dhist[0], [0,0], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
#     ax.plot([0,0], range_2Dhist[1], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)


#     if xhandlepad != None:
#         ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
#     else:
#         ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
#     if yhandlepad != None:
#         ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
#     else:
#         ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(linewidth)
#     ax.spines['left'].set_linewidth(linewidth)
#     ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
#     for tick in ax.get_yticklabels():
#         tick.set_fontname(font)
#     for tick in ax.get_yticklabels():
#         tick.set_fontname(font)
#     if xlim != None:
#         ax.set_xlim(xlim)
#     if ylim != None:
#         ax.set_ylim(ylim)
#     if xticks != None:
#         ax.set_xticks(xticks)
#     if xticklabels != None:
#         ax.set_xticklabels(xticklabels)
#     if yticks != None:
#         ax.set_yticks(yticks)
#     if yticklabels != None:
#         ax.set_yticklabels(yticklabels)
#     if center_axes:
#         ax.spines['left'].set_position(('data', 0))
#         ax.spines['bottom'].set_position(('data', 0))
#     ax.xaxis.label.set_color(color_xlabel)
#     ax.yaxis.label.set_color(color_ylabel)
#     plt.show()

# def plot_2Dhist_contour(dataref, list_data,
#                 xlabel=None,
#                 ylabel=None,
#                 n_bins=100,
#                 range_2Dhist=[[-0.5, 1.5], [-0.5, 1.5]],
#                 font = "Arial", 
#                 fontsize = 10, 
#                 linewidth = 1.5, 
#                 xlim = None, 
#                 ylim = None,
#                 figsize=(1.5, 1),
#                 xticks=None,
#                 yticks=None,
#                 xhandlepad=None,
#                 yhandlepad=None,
#                 s=1,
#                 center_axes = False,
#                 dpi=600,
#                 color='black',
#                 color_xlabel="black",
#                 color_ylabel="black",
#                 marker='o',
#                 linewidth_line=1.5,
#                 color_line="black",
#                 alpha_line=0.5,
#                 xticklabels=None,
#                 yticklabels=None,
#                 zorder_line=0.1,
#                 colors=None,
#                 axwidth=1.5,
#                        level=3,
#                        gaussian_filt_sigma=0.71,
#                        labels_data=None):
    
#     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
#     n_data = len(list_data)
#     for i in range(n_data):
#         H, xedges, yedges = np.histogram2d(dataref, list_data[i], bins=n_bins, range=range_2Dhist, density=True, weights=None)
        
#         ax.contour(xedges[:-1], yedges[:-1], gaussian_filter(H.T, gaussian_filt_sigma), [level], 
#                    colors=[colors[i]], linewidths=linewidth)
#     ax.plot(range_2Dhist[0], range_2Dhist[1], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
#     ax.plot(range_2Dhist[0], [0,0], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
#     ax.plot([0,0], range_2Dhist[1], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)


#     if xhandlepad != None:
#         ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
#     else:
#         ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
#     if yhandlepad != None:
#         ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
#     else:
#         ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(axwidth)
#     ax.spines['left'].set_linewidth(axwidth)
#     ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
#     for tick in ax.get_yticklabels():
#         tick.set_fontname(font)
#     for tick in ax.get_yticklabels():
#         tick.set_fontname(font)
#     if xlim != None:
#         ax.set_xlim(xlim)
#     if ylim != None:
#         ax.set_ylim(ylim)
#     if xticks != None:
#         ax.set_xticks(xticks)
#     if xticklabels != None:
#         ax.set_xticklabels(xticklabels)
#     if yticks != None:
#         ax.set_yticks(yticks)
#     if yticklabels != None:
#         ax.set_yticklabels(yticklabels)
#     if center_axes:
#         ax.spines['left'].set_position(('data', 0))
#         ax.spines['bottom'].set_position(('data', 0))
#     ax.xaxis.label.set_color(color_xlabel)
#     ax.yaxis.label.set_color(color_ylabel)

#     proxy = [plt.Rectangle((0,0),1,1,fc = color) for color in colors] 

#     legend = ax.legend(proxy, labels_data, loc='upper left', bbox_to_anchor=(0.98, 0.9), 
#               fontsize=fontsize, ncol=1, frameon=False,
#               borderpad=0, labelspacing=0.5, handlelength=1, 
#               columnspacing=0, handletextpad=0.3, borderaxespad=0.6)
    
#     plt.show()