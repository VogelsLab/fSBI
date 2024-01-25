import h5py
import torch
import numpy as np
import os
from typing import Optional, List
# from tqdm import tqdm_notebook


def load_data_from_h5py(path_to_file: str):
    """
    Load data from h5py file.

    Args:
        path_to_file: path to load file.
    """
    with h5py.File(path_to_file, "a") as f:
        keys = list(f.keys())

        thetas = torch.FloatTensor(len(keys), 4)
        obs = torch.FloatTensor(len(keys), 2)

        for i, k in enumerate(keys):
            thetas[i] = torch.FloatTensor(f[k]["thetas"])[2:]
            ob = [
                torch.FloatTensor(np.asarray(f[k]["cv"]).reshape(1, 1)),
                torch.FloatTensor(np.asarray(f[k]["rate"]).reshape(1, 1)),
            ]
            obs[i] = torch.cat(ob, -1)
    return thetas, obs


def read_monitor_spiketime_files(
    workdir: str,
    seed: str,
    num_neurons: Optional[int] = 500,
    remove_file: Optional[bool] = False,
    which: str = "e"
):
    """
    Read monitor files generated from auryn code.

    return dictionary of spiketimes.
    Args:
        workdir: working directory for auryn, where simulation output is written.
        seed: seed or ID for simulation.
        num_neurons: number of neurons in simulation.
        remove_file: if True, delete monitor file after reading from it.
    """
    spiketimes = {str(neuron): [] for neuron in range(num_neurons)}

    filename = workdir + "/out.e." + str(seed) + ".0.ras"
    if which == "i":
        filename = workdir + "/out.i." + str(seed) + ".0.ras"

    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        aux = line.split(" ")
        spiketimes[str(int(aux[1]))].append(float(aux[0]))

    spiketimes["id"] = seed  # unique id for the simulation

    if remove_file:
        # delete monitor files to make space
        os.remove(filename)
    return spiketimes


def read_monitor_weights_files(
    workdir: str,
    seed: str,
    plastic_connections: List[float],
    remove_file: Optional[bool] = False,
):
    """
    Read monitor files generated from auryn.

    Return dictionary of dictionnary of weight traces.
    w["ei"]["ts] and w["ii"]["ws"]

    Args:
        workdir: working directory for auryn, where simulation output is written.
        seed: seed or ID for simulation.
        plastic_connections: "ee" "ei" "ie" "ii" which connections are recorded and
            to be stored
        remove_file: if True, delete monitor file after reading from it.
    """
    w_dict = dict()

    for con_type in plastic_connections:
        filename = workdir + "/con_%s." % (con_type) + str(seed) + ".0.syn"

        f = open(filename, "r")
        lines = f.readlines()
        n_neurons = len(lines[0].split(" "))-2
        n_ts = len(lines)

        ts = np.zeros(n_ts)
        weights = np.zeros((n_neurons, n_ts))
        for i in range(n_ts):
            aux_ar = np.array(lines[i].split(" ")[:-1]).astype(float)
            ts[i] = aux_ar[0]
            weights[:, i] = aux_ar[1:]

        w_dict[con_type] = dict()
        w_dict[con_type]['t'] = ts
        w_dict[con_type]['w'] = weights

        w_dict["id"] = seed  # unique id for the simulation

        if remove_file:
            # delete monitor files to make space
            os.remove(filename)
    return(w_dict)


def _create_h5_entry(
    h5_file, id, theta, params, spiketimes=None, spiketimes_i = None, weights=None, blow_up=-1
                     ):
    try:
        group = h5_file.create_group(id)
    except ValueError as e:
        print("Skipping seed:", id, "Error:", e)
        return(h5_file)

    group["theta"] = theta

    for k, v in params.items():
        if k not in [
            "auryn_sim_dir", "workdir", "prior_params", "n_dim_params", "con_type"
                     ]:
            group.attrs[k] = v
    group.attrs["t_start_rec"] = params["lns"]
    group.attrs["t_stop_rec"] = params["lns"] + params["ls"]
    group.attrs["blow_up"] = blow_up

    if spiketimes is not None:
        spiketimes_grp = group.create_group("spiketimes")
        for k, v in spiketimes.items():
            if k not in ["theta", "id"]:
                spiketimes_grp[k] = v

    if spiketimes_i is not None:
        spiketimes_i_grp = group.create_group("spiketimes_i")
        for k, v in spiketimes_i.items():
            if k not in ["theta", "id"]:
                spiketimes_i_grp[k] = v

    if weights is not None:
        weights_grp = group.create_group("weights")
        ts_stored = False
        for k, v in weights.items():
            if k not in ["theta", "id"]:
                for i in weights[k].keys():
                    if i == "t" and not ts_stored:
                        weights_grp["t"] = v["t"]
                        ts_stored = True
                    elif i == "w":
                        weights_grp[k] = v[i]

    return h5_file


def h5_group_to_dict(grp):
    """
    Transform a hdf5 group into a regular python dict.
    DOES NOT ACCEPT NESTED GROUP YET!! (FEATURE NEEDED)?
    """
    grp_dict = {}
    for k in grp.keys():
        grp_dict[k] = grp[k][()]
    return grp_dict


def save_data_to_hdf5(
    path_to_file: str,
    data: Optional[List[dict]] = None,
    thetas: Optional[list] = None,
    seeds: Optional[list] = None,
    spiketimes_workdir: Optional[str] = "",
    con_type: Optional[List[str]] = None,
    outputs: Optional[List[float]] = None,
    params: Optional[List[dict]] = None
):
    """
    Save simulated data to hdf5 file.

    Args:
        path_to_file: path to hdf5 file.
        data: list of dictionaries to save to file. Each dictionary should contain a
        unique seed, corresponding meta-parameters and simulated spiketimes.
        thetas: list of meta parameters, required if data is None.
        seeds: list of uniques seeds corresponding to thetas, required if data is None.
        spiketimes_workdir: directory where spiketime and weights monitor files are
            saved, required if data is None.
        con_type: ["ee","ei","ie","ii"] which connections have been recorded from, to
            store weights. If empty, no weights will be stored.
        outputs: list of outputs from C++ execution (know if a simulation blew up
            or not)
    """
    h5_file = h5py.File(
        path_to_file,
        "a",
        #    driver='mpio',
        #    comm=MPI.COMM_WORLD
    )
    if data is not None:
        for dat in data:
            assert "id" in dat.keys() and "theta" in dat.keys()
            h5_file = _create_h5_entry(h5_file, dat["id"], dat["theta"], dat)

    else: #need to fetch the auryn monitor files
        for id, theta, output in zip(seeds, thetas, outputs):
#         for id, theta, output in zip(seeds, thetas, outputs):
            if output < 0:  # simulation did not blow up, gather and store monitor files store spiketimes
                spiketimes = read_monitor_spiketime_files(spiketimes_workdir, id, which="e", num_neurons=params['n_recorded'])
                weights = read_monitor_weights_files(spiketimes_workdir, id, con_type)
                if params['record_i']:
                    spiketimes_i = read_monitor_spiketime_files(spiketimes_workdir, id, which="i", num_neurons=params['n_recorded_i'])
                    h5_file = _create_h5_entry(h5_file, 
                                               id, 
                                               theta, 
                                               params, 
                                               spiketimes=spiketimes, 
                                               spiketimes_i=spiketimes_i,
                                               weights=weights,
                                               blow_up=output)
                else:
                    h5_file = _create_h5_entry(h5_file,
                                               id,
                                               theta,
                                               params,
                                               spiketimes=spiketimes,
                                               spiketimes_i=None,
                                               weights=weights, 
                                               blow_up=output)
            else:  # simulation blew up
                h5_file = _create_h5_entry(h5_file,
                                           id,
                                           theta,
                                           params,
                                           spiketimes=None,
                                           spiketimes_i=None,
                                           weights=None,
                                           blow_up=output)
                
        #delete the monitor files?
        while True:
            try:
                delete_files = str(input("Do you want to remove all auryn monitor files? (y/n):"))
            except ValueError:
                print("Sorry, I didn't understand that.")
                continue
            else:
                break
        if delete_files == "y":
            print("deleting all monitor files")
            for id, theta, output in zip(seeds, thetas, outputs):
                filename = spiketimes_workdir + "/out.e." + str(id) + ".0.ras"
                if os.path.exists(filename):
                    os.remove(filename)
                filename = spiketimes_workdir + "/out.i." + str(id) + ".0.ras"
                if os.path.exists(filename):
                    os.remove(filename)
                for con_str in con_type:
                    filename = spiketimes_workdir + "/con_%s." % (con_str) + str(id) + \
                        ".0.syn"
                    if os.path.exists(filename):
                        os.remove(filename)
        else:
            print("did not remove any monitor files")
    h5_file.close()
    return()


def get_output_cluster(seeds, array_job_num, output_dir):
    n_sims = len(seeds)
    output = np.zeros(n_sims)
    for sim_num in range(n_sims):
        seed = seeds[sim_num]
        index_array = sim_num + 1 #array job indexed 1-n_jobs, 1 increment right now
        filename = output_dir + "array_" + str(array_job_num) + "-" + str(index_array) + ".log"
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                text = f.read()
                if text.find(str(seed)) < 0:
                    print(filename, "does not seem to be the right file for seed", seed)
                    break
                output[sim_num] = float(text.split("cynthia")[1])
    return(output)