import numpy as np
import h5py
from itertools import chain
from scipy.special import rel_entr
from scipy.signal import correlate
from scipy.fft import fft, fftfreq
import sys
import torch
import time
import matplotlib.pyplot as plt

def default_x():
        """
        Make dictionary to sample values per metric to get filtered posterior samples.

        The key for each entry corresponds to a metric defined in ComputeMetrics.
        The item of each entry is a function that takes an integer `num` as input
            and returns `num` values of the metric.
        We can use these values as conditional values corresponding to which we
            sample parameters from the density estimator.
        """
        # TODO need entry here for every metric we intend to use for filtering
        # ["rate","cv_isi","kl_isi","spatial_Fano","temporal_Fano","auto_cov","fft",
#                    "w_blow", "std_rate_temporal","std_rate_spatial","std_cv"]


        return {"rate": lambda num: torch.rand(num, 1) * 29 + 1, #]1,50]
                "cv_isi": lambda num: torch.rand(num, 1) * 2 + .7, #[0.7,2.7]
                "kl_isi": lambda num: torch.rand(num, 1) * .5, #[0,0.5]
                "spatial_Fano": lambda num: torch.rand(num, 1) * 2 + 0.5, #[0.5,2.5]
                "temporal_Fano": lambda num: torch.rand(num, 1) * 2 + 0.5, #[0.5,2.5]
                "auto_cov": lambda num: torch.rand(num, 1) * .1, #[0,0.1]
                "fft": lambda num: torch.rand(num, 1), #[0,1]
                "w_blow": lambda num: torch.rand(num, 1) * .1, #[0,0.1]
                "std_rate_temporal": lambda num: torch.rand(num, 1) * .05, #[0,0.05]
                "std_rate_spatial": lambda num: torch.rand(num, 1) * 5, #[0,5]
                "std_cv": lambda num: torch.rand(num, 1)*0.2, #[0,0.2]
                "w_creep": lambda num: torch.rand(num, 1)*0.05, #[0,0.05]
                "rate_i": lambda num: torch.rand(num, 1)*49 + 1, #[1,50]
                "weef": lambda num: torch.rand(num, 1)*0.5, #[0,0.5]
                "weif": lambda num: torch.rand(num, 1)*0.5, #[0,0.5]
                "wief": lambda num: torch.rand(num, 1)*5, #[0,5]
                "wiif": lambda num: torch.rand(num, 1)*5, #[0,5]
                "r_fam": lambda num: torch.rand(num, 1)*1 + 13, #[13,14]
                "r_nov": lambda num: torch.rand(num, 1)*1 + 15, #[15, 16]
                "std_fam": lambda num: torch.rand(num, 1)*2 + 15, #[15,17]
                "std_nov": lambda num: torch.rand(num, 1)*2 + 19, #[19, 21]
                "ratio_nov_fa3": lambda num: torch.rand(num, 1)*0.08 + 0.12} #[0.12, 0.2]

def condition():
    """
    Make dictionary of metric-specific conditions to rejection sample simulations.

    The key for each entry corresponds to a metric defined in ComputeMetrics.
    The item of each entry is a function that takes a metric value as input
        and returns boolean indicating if it satisfies a specified condition.
    We can use these to rejection sample and reuse simulations from previous rounds.
    """
    # TODO need entry here for every metric we intend to use for filtering
    return {"rate": lambda x: np.logical_and(1 <= x, x <= 50),
            "cv_isi": lambda x: np.logical_and(0.7 <= x, x <= 2.7),
            "kl_isi": lambda x: np.logical_and(0 <= x, x <= 0.5),
            "spatial_Fano": lambda x: np.logical_and(0.5 <= x, x <= 2.5),
            "temporal_Fano": lambda x: np.logical_and(0.5 <= x, x <= 2.5),
            "auto_cov": lambda x: np.logical_and(0 <= x, x <= 0.1),
            "fft": lambda x: np.logical_and(0 <= x, x <= 1),
            "w_blow": lambda x: np.logical_and(0 <= x, x <= 0.1),
            "std_rate_temporal": lambda x: np.logical_and(0 <= x, x <= 0.05),
            "std_rate_spatial": lambda x: np.logical_and(0 <= x, x <= 5),
            "std_cv": lambda x: np.logical_and(0 <= x, x <= 0.2),
            "w_creep": lambda x: np.logical_and(0 <= x, x <= 0.05),
            "rate_i": lambda x: np.logical_and(1 <= x, x <= 50),
            "weef": lambda x: np.logical_and(0 <= x, x <= 0.5),
            "weif": lambda x: np.logical_and(0 <= x, x <= 0.5),
            "wief": lambda x: np.logical_and(0 <= x, x <= 5),
            "wiif": lambda x: np.logical_and(0 <= x, x <= 5),
            "r_fam": lambda x: np.logical_and(13 <= x, x <= 14),
            "r_nov": lambda x: np.logical_and(15 <= x, x <= 16),
            "std_fam": lambda x: np.logical_and(15 <= x, x <= 17),
            "ratio_nov_fam": lambda x: np.logical_and(0.12 <= x, x <= 0.5)}


class ComputeMetrics:
    """Compute metrics given a simulation."""

    def __init__(self, spiketimes: dict, sim_params: dict, weights: dict=None, spiketimes_i: dict=None) -> None:
        """Set up class to compute metrics."""
        # initialise an object directly with spiketimes dict, weight dict and
        # params dict
        self.spiketimes = spiketimes
        self.spiketimes_i = spiketimes_i
        self.params = sim_params
        self.weights = weights

        self.binned_spikes_small_computed = False
        self.binned_spikes_small = np.array([])
        self.ts_small = np.array([])
        self.binned_spikes_medium_computed = False
        self.binned_spikes_medium = np.array([])
        self.binned_spikes_big_computed = False
        self.binned_spikes_big = np.array([])

        self.isis_computed = False
        self.isis = dict()

        self.cvs_computed = False
        self.cvs = np.zeros(self.params["n_recorded"])

    def _check(self, keys):
        assert np.all([k in self.params.keys() for k in keys])

    def _return_nan(metric_func):
        def modify_metric_to_return_nan(self):
            if self.spiketimes is None or self.weights is None:
                return np.nan
            else:
                return metric_func(self)
        return modify_metric_to_return_nan

    def get_binned_spikes_small(self):
        if not self.binned_spikes_small_computed:
            bins = np.arange(self.params["t_start_rec"], self.params["t_stop_rec"], self.params["bin_size_small"])
            self.binned_spikes_small = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                    for neuron_num in range(self.params["n_recorded"])])
            self.ts_small = bins[:-1]
            self.binned_spikes_small_computed = True
        return(self.binned_spikes_small)
    
    def get_binned_spikes_medium(self):
        if not self.binned_spikes_medium_computed:
            bins = np.arange(self.params["t_start_rec"], self.params["t_stop_rec"], self.params["bin_size_medium"])
            self.binned_spikes_medium = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                    for neuron_num in range(self.params["n_recorded"])])
            self.binned_spikes_medium_computed = True
        return(self.binned_spikes_medium)
    
    def get_binned_spikes_big(self):
        if not self.binned_spikes_big_computed:
            bins = np.arange(self.params["t_start_rec"], self.params["t_stop_rec"], self.params["bin_size_big"])
            self.binned_spikes_big = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                    for neuron_num in range(self.params["n_recorded"])])
            self.binned_spikes_big_computed = True
        return(self.binned_spikes_big)
    
    def get_autocov(self, neuron_num):
        lags = int(self.params["window_view_auto_cov"]/self.params["bin_size_medium"])
        x =  self.binned_spikes_medium[neuron_num, :]
        if np.std(x) > 0.05: #tuned so that a spiketrain of 1Hz poisson with bin 10ms ish is not detected
            xcorr = correlate(x - x.mean(), x - x.mean(), 'full')
            xcorr = np.abs(xcorr[:]) / xcorr.max()
            return(np.mean(xcorr[(xcorr.size//2-(lags)):(xcorr.size//2+(lags+1))]))
        else: #signal is almost constant
            return(1)

    def get_isis(self):
        if not self.isis_computed:
            for neuron_num in range(self.params["n_recorded"]):
                self.isis[str(neuron_num)] = np.diff(self.spiketimes[str(neuron_num)])
            self.isis_computed = True
        return(self.isis)

    def get_cvs(self):
        self.get_isis()
        if not self.cvs_computed:
            for neuron_num in range(self.params["n_recorded"]):
                if len(self.isis[str(neuron_num)]) > 2:
                    self.cvs[neuron_num] = np.std(self.isis[str(neuron_num)])/np.mean(self.isis[str(neuron_num)])
            self.cvs_computed = True
        return(self.cvs)

    def get_isi_aggregate(self):
        return(np.array(list(chain(*self.get_isis().values()))))

    def compute_fano(self, counts):
        if np.sum(counts) <= 3:
            return(0)
        else: 
            return(np.var(counts)/np.mean(counts))

    @property
    @_return_nan
    def rate(self):
        """Total population rate."""
        self._check(["n_recorded", "ls"])
        n_tot_spikes = np.sum([i.shape[0] for i in self.spiketimes.values()])
        return(n_tot_spikes/self.params["n_recorded"]/self.params["ls"])

    @property
    @_return_nan
    def cv_isi(self):
        """Coefficient of variation of exc neurons' interspike-interval distribution, averaged over neurons"""
        self._check(["n_recorded"])
        return(np.mean(self.get_cvs()))

    @property
    @_return_nan
    def kl_isi(self):
        """KL divergence between Poisson and simulated spike ISI distribution (single isi distr aggregated over all neurons)"""
        self._check(
            ["n_recorded",
             "t_start_rec",
             "t_stop_rec",
             "n_bins_kl_isi",
             "isi_lim_kl_isi"])

        isis_agg = self.get_isi_aggregate()
        xs = np.linspace(self.params["isi_lim_kl_isi"][0], self.params["isi_lim_kl_isi"][1],num=self.params["n_bins_kl_isi"])
        binned_isi,_ = np.histogram(np.clip(isis_agg, self.params["isi_lim_kl_isi"][0], self.params["isi_lim_kl_isi"][1]),
                                                bins=xs,
                                                density=False)
        binned_isi = binned_isi/max(len(isis_agg), 1)
        if self.rate <= 0.1:
            rate = 0.01
        else:
            rate = 1/np.mean(isis_agg)
        binned_poisson_isi = np.array([np.exp(-rate*xs[i-1]) - np.exp(-rate*xs[i]) for i in range(1, len(xs))])
        kl = np.sum(rel_entr(binned_isi, binned_poisson_isi))
        return(np.abs(kl))

    @property
    @_return_nan
    def spatial_Fano(self):
        """Fano factor: computed over neurons during a single timebin, then returns an average over timebins"""
        self._check(
            ["n_recorded", "t_start_rec", "t_stop_rec", "bin_size_big"])

        self.get_binned_spikes_big()
        return(np.mean([self.compute_fano(self.binned_spikes_big[:,i]) for i in range(len(self.binned_spikes_big[0]))]))

    @property
    @_return_nan
    def temporal_Fano(self):
        """Fano factor: computed over timebins in single neurons (temporal), then returns an average over neurons"""
        self._check(
            ["n_recorded", "t_start_rec", "t_stop_rec", "bin_size_big"])

        self.get_binned_spikes_big()
        return(np.mean([self.compute_fano(self.binned_spikes_big[i,:]) for i in range(self.params["n_recorded"])]))

    @property
    @_return_nan
    def auto_cov(self):
        """fraction of non zero elements in the autocovariance of spiketrains, averaged over neurons"""
        self._check(
            ["n_recorded",
             "t_start_rec",
             "t_stop_rec",
             "bin_size_medium",
             "window_view_auto_cov"])

        self.get_binned_spikes_medium()
        return(np.mean([self.get_autocov(i) for i in range(self.params["n_recorded"])]))

    @property
    @_return_nan
    def fft(self):
        """area under the curve in fourier transform, computed over pop firing rate with small bin"""
        self._check(["n_recorded", "t_start_rec", "t_stop_rec", "bin_size_small"])

        self.get_binned_spikes_small()
        xf = fftfreq(len(self.ts_small), self.params["bin_size_small"])[:len(self.ts_small)//2]
        yf = 2.0/len(self.ts_small) * np.abs( fft(np.mean(self.binned_spikes_small, axis=0))[0:len(self.ts_small)//2])
        return(np.sum(yf[1:]))

    @property
    @_return_nan
    def w_blow(self):
        """Indicate if synaptic weights have exploded."""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded", "t_start_rec", "t_stop_rec", "wmax"])

        if self.weights is None:
            return(-1000)

        f_blow = 0
        for key in self.weights.keys():
            if key != "t":
                w_distr = get_w_distr(w_dict={"w": self.weights[key], "t": self.weights["t"]}, t_start=self.params["t_start_rec"], t_stop=self.params["t_stop_rec"])
                f_blow += np.sum([i == 0 or i == self.params["wmax"] for i in w_distr]) / len(w_distr)
        return (f_blow / (len(self.weights.keys()) - 1))
    
    @property
    @_return_nan
    def std_rate_temporal(self):
        """compute std of population firing rate across time"""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded", "t_start_rec", "t_stop_rec"])
        self.get_binned_spikes_small()
        return(np.std(np.mean(self.binned_spikes_small, axis=0)))
    
    @property
    @_return_nan
    def std_rate_spatial(self):
        """compute std of individual firing rates"""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded", "ls"])
        all_rates = np.array( [len(self.spiketimes[str(j)])/self.params["ls"] for j in range(self.params["n_recorded"])] ) 
        return( np.std(all_rates) )
    
    @property
    @_return_nan
    def std_cv(self):
        """compute std of individual neurons cv_isi"""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded"])
        return( np.std(self.get_cvs()) )
    
    @property
    @_return_nan
    def w_creep(self):
        """compute change of mean weight between start and finish (as percentage), max amount all weights considered"""
        # check that the simulation has the params for the metric to be computed
        self._check(["t_start_rec", "t_stop_rec"])
        w_creep_metric = 0
        for key in self.weights.keys():
            if key != "t":
                start_w = np.mean(self.weights[key][:,0])
                end_w = np.mean(self.weights[key][:,-1])
                if start_w + end_w > 0.1:
                    candidate = np.abs(2*(end_w - start_w)/(end_w + start_w))
                    if candidate > w_creep_metric:
                        w_creep_metric = candidate
        return( w_creep_metric )
    
    @property
    @_return_nan
    def rate_i(self):
        """Total population rate of inh population"""
        self._check(["n_recorded_i", "ls"])
        if self.spiketimes_i is None:
            return(np.nan)
        n_tot_spikes = np.sum([i.shape[0] for i in self.spiketimes_i.values()])
        return(n_tot_spikes/self.params["n_recorded_i"]/self.params["ls"])
    
    @property
    @_return_nan
    def weef(self):
        """final mean EE weight"""
        return(np.mean(self.weights["ee"][:,-1]))
    
    @property
    @_return_nan
    def weif(self):
        """final mean EI weight"""
        return(np.mean(self.weights["ei"][:,-1]))
    
    @property
    @_return_nan
    def wief(self):
        """final mean IE weight"""
        return(np.mean(self.weights["ie"][:,-1]))
    
    @property
    @_return_nan
    def wiif(self):
        """final mean II weight"""
        return(np.mean(self.weights["ii"][:,-1]))
    
    @property
    @_return_nan
    def r_nov(self):
        """pop rate in response to nov stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"]
        stop = start + self.params["lp"]
        return(np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def r_fam(self):
        """pop rate in response to familiar stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp", "lb2"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"] + self.params["lp"] + self.params["lb2"]
        stop = start + self.params["lp"]
        return(np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def std_nov(self):
        """std of individual neurons rate in response to nov stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"]
        stop = start + self.params["lp"]
        return(np.std([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def std_fam(self):
        """std of individual neurons rate in response to familiar stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp", "lb2"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"] + self.params["lp"] + self.params["lb2"]
        stop = start + self.params["lp"]
        return(np.std([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def ratio_nov_fam(self):
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp", "lb2"])
        return(self.r_nov - self.r_fam)/(self.r_fam+0.0001)

def get_w_distr(w_dict=None, t_start=0, t_stop=60, n_bins=100, w_min=0, w_max=10):
    # check that t_start and t_stop are legit
    if t_start > w_dict['t'][-1] or t_stop < w_dict['t'][0]:
        raise ValueError('t_start or t_stop are outside the recorded range')
    valid_ts = [t_start<i<t_stop for i in w_dict['t']]
    return(w_dict['w'][:, valid_ts].flatten())