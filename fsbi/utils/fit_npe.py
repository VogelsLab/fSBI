"""Job script to fit density estimator."""
from fsbi.analyse import ComputeMetrics, default_x, condition
from fsbi.analyse import ComputeMetrics
import numpy as np
import h5py
from typing import List
from concurrent import futures

default_x_dict = default_x()
condition_dict = condition()
# metrics = ["rate", "cv_isi"]


def worker_get_density_estim_data(seeds_list, h5_path, metrics):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_keys = len(keys)
        n_keys_to_process = len(seeds_list)

        n_thetas = f[keys[0]]['theta'].shape[0]
        aux_dt = [('seed', 'U64'), ('theta', np.float64, n_thetas)]
        for metric in metrics:
            aux_dt.append((metric, np.float64))
        dt = np.dtype(aux_dt)
        
        data = np.zeros(n_keys_to_process, dtype=dt)
        
        for i, k in enumerate(seeds_list):
            data[i]['seed'] = k
            data[i]['theta'] = np.array(f[k]["theta"])
            ## we have a blow-up: no spiketimes or weights to gather
            if f[k].attrs["blow_up"] >= 0:
                comp_metrics = ComputeMetrics(spiketimes=None,
                                          sim_params=dict(f[k].attrs.items()),
                                          weights=None)
            else:
                spiketimes = {str(j): f[k]["spiketimes"][str(j)][()] for j in range(0, f[k].attrs["n_recorded"])}
                if f[k].attrs["record_i"]:
                    spiketimes_i = {str(j): f[k]["spiketimes_i"][str(j)][()] for j in range(0, f[k].attrs["n_recorded_i"])}
                else:
                    spiketimes_i = None
                weights = {i: f[k]["weights"][i][()] for i in f[k]["weights"].keys()}
                comp_metrics = ComputeMetrics(spiketimes=spiketimes,
                                          sim_params=dict(f[k].attrs.items()),
                                          spiketimes_i=spiketimes_i,
                                          weights=weights)
            for j, metric in enumerate(metrics):
                data[i][metric] = getattr(comp_metrics, metric)
    return(data)    

def get_density_estim_data(h5_path: str, metrics: List[str], parallel=False, n_workers=2):
    """
    computes metrics over a list of h5 files. does not check for existence of a previous file
    """
    n_metrics = len(metrics)
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_keys = len(keys)
        n_thetas = f[keys[0]]['theta'].shape[0]
        aux_dt = [('seed', 'U64'), ('theta', np.float64, n_thetas)]
        for metric in metrics:
            aux_dt.append((metric, np.float64))
        dt = np.dtype(aux_dt)
        print("Found h5 file with", n_keys, "simulations")

    if not parallel:
        data = worker_get_density_estim_data(keys, h5_path, metrics)

    else:
        data = np.zeros(n_keys, dtype=dt)
        all_seeds_list = np.array_split(keys, n_workers) #divide the keys in equal parts
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            jobs = [executor.submit(worker_get_density_estim_data, seeds_list, h5_path, metrics)\
                    for seeds_list in all_seeds_list]

        data = np.zeros(0, dtype=dt)
        for job in jobs:
            data = np.append(data, job.result(), axis=0)
    return(data)

def save_metric(path=None, data=None):
    while True:
        try:
            save_file = str(input("Do you want to save metric file as " + path + "? (y/n):"))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        else:
            break
    if save_file == "y":
        np.save(path, data, allow_pickle=True)
        print('file saved')
    else:
        print('file NOT saved')
    return

def resample_old_data(
    path_to_data: str, filter: int, refine: int, metric: str
                            ):
    """
    Resample data from previous rounds (be sample efficient and for data augmentation).

    The current round is defined by the input `filter` and `refine` numbers.
    This method loads raw data from previous filtering rounds, computes the metrics
    required for the current round, and appends to current round data.
    It also loads metrics from previous refining rounds and appends to the current
    round data.
    Args:
        path_to_data: path to data (data from all rounds is expected to be in the same
            directory).
        filter: current filtering round number.
        refine: current refining round number.
        metric: the metric to compute for the current round.
    """
    all_data = []
    for filt in range(filter+1):
        for ref in range(refine+1):
            data = get_density_estim_data(
                path_to_data % (filt, ref), metric
                                           )
            inds = range(len(data))
            # If not the current filtering / refining round, filter data by
            # current condition
            if filt != filter and ref != refine:
                inds = condition[metrics[filter]](data[:, -1])
            all_data.append(data[inds])
    if len(all_data) > 1:
        return np.concatenate(all_data, 0)
    else:
        return all_data[0]
