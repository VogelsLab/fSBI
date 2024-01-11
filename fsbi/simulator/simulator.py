"""Implementation for auryn simulator wrapped in python."""
import os
import subprocess
import torch
import numpy as np
from fsbi.utils import read_monitor_spiketime_files, read_monitor_weights_files

class Simulator_bg_IF_EEEIIEII_6pPol:
    """
    Network of conductance-based neurons with nmda currents and relative refractoriness.
    All connections plastic following 6 params polynomial.
    background homogenous inputs to both E and I
    """

    def __init__(self, simulator_params: dict):
        """
        Set up the network.

        Args:
            simulator_params: dictionary of parameters for simulator.
        """
        self.simulator_params = simulator_params
        # Required arguments to make call string
        self.req_cls_str_args = [
            "NE",
            "NI",
            "tau_ampa",
            "tau_gaba",
            "tau_nmda",
            "ampa_nmda_ratio",
            "wmax",
            "eta",
            "wee",
            "wei",
            "wie",
            "wii",
            "sparseness",
            "N_input",
            "sparseness_poisson",
            "weight_poisson",
            "max_rate_checker",
            "tau_checker",
            "lns",
            "ls",
            "workdir",
            "n_recorded",
            "record_i",
            "n_recorded_i"
        ] ######## note that RATE POISSON IS A NUISANCE SO CONSIDERED A THETA and therefore not in the default arguments (SEE RULE_STR)

        # Check if required arguments are in param dictionary
        assert np.all(
            [
                k in simulator_params.keys()
                for k in (self.req_cls_str_args + ["auryn_sim_dir", "name"])
            ]
        )

    @property
    def cl_str(self):
        """Call string to pass to C++ simulator."""
        class_string = (
            self.simulator_params["auryn_sim_dir"]
            + str(self.simulator_params["name"])
            + " --ID "
            + "%s"
            + "%s"
        )  # Rule string here

        for arg in self.req_cls_str_args:
            class_string += " --%s %s" % (arg, str(self.simulator_params[arg]))
        return class_string

    @property
    def rule_str(self):
        """Part of call string for parametrized rule."""
        rule_string = " --ruleEE a" + "{}" + "a" + "{}"
        rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
        rule_string += " --ruleEI a" + "{}" + "a" + "{}"
        rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
        rule_string += " --ruleIE a" + "{}" + "a" + "{}"
        rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
        rule_string += " --ruleII a" + "{}" + "a" + "{}"
        rule_string += "a" + "{}" + "a" + "{}" + "a" + "{}" + "a" + "{}" + "a"
        rule_string += " --rate_poisson {}"
        return rule_string

    def sample(self, thetas, seeds=None, return_data=False, verbose=0):
        """Forward pass through simulator."""
        if seeds is None: #should not happen
            print("no seed was passed, generating some, but careful")
            seeds = [42 + i for i, _ in enumerate(thetas)]
        else:
            assert len(thetas) == len(seeds)

        all_spiketimes = []
        all_weights = []

        for th, seed in zip(thetas, seeds):
            # run auryn simulation
            rule_str = self.rule_str.format(*list(th))
            if verbose > 0:
                print(self.cl_str % (str(seed), rule_str))
            output = subprocess.run(
                self.cl_str % (str(seed), rule_str), shell=True, capture_output=True
            )

            if return_data:
                # Read spiketimes from monitor files
                spiketimes = read_monitor_spiketime_files(
                    workdir=self.simulator_params["workdir"],
                    seed=seed,
                    num_neurons=self.simulator_params["n_recorded"],
                    remove_file=True,
                    which='e'
                )

                # Collect spiketimes
                all_spiketimes.append(spiketimes)

                if self.simulator_params['record_i']:
                    spiketimes_i = read_monitor_spiketime_files(
                        workdir=self.simulator_params["workdir"],
                        seed=seed,
                        num_neurons=self.simulator_params["n_recorded_i"],
                        remove_file=True,
                        which='i'
                    )
                    all_spiketimes.append(spiketimes_i)

                # Read weight traces from monitor files
                weights = read_monitor_weights_files(
                    workdir=self.simulator_params["workdir"],
                    seed=seed,
                    plastic_connections=["ee", "ei", "ie", "ii"],
                    remove_file=True
                )

                # Collect weights
                all_weights.append(weights)

        # Return collected spiketimes
        if len(all_spiketimes) > 0:
            return(float(output.stdout.decode().split("cynthia")[1]),
                   all_spiketimes,
                   all_weights)
        else:
            return(float(output.stdout.decode().split("cynthia")[1]))

        
class Simulator_bg_CVAIF_EEIE_T4wvceciMLP:
    """
    Network of conductance-based neurons with nmda currents and relative refractoriness, and spike triggered adaption.
    EE connections are plastic using the CVAIF_EEIE_T4wvceciMLP.
    background homogenous inputs to both E and I
    """

    def __init__(self, simulator_params: dict):
        """
        Set up the network.

        Args:
            simulator_params: dictionary of parameters for simulator.
        """
        self.simulator_params = simulator_params

        # Required arguments to make call string
        self.req_cls_str_args = [
            "NE",
            "NI",
            "tau_ampa",
            "tau_gaba",
            "tau_nmda",
            "ampa_nmda_ratio",
            "nh1",
            "nh2",
            "wmax",
            "wee",
            "wei",
            "wie",
            "wii",
            "sparseness",
            "N_input",
            "sparseness_poisson",
            "w_poisson",
            "max_rate_checker",
            "tau_checker",
            "lns",
            "ls",
            "workdir",
            "n_recorded",
            "record_i",
            "n_recorded_i"
        ] ######## note that RATE POISSON IS A NUISANCE SO CONSIDERED A THETA and therefore not in the default arguments (SEE RULE_STR)

        # Check if required arguments are in param dictionary
        assert np.all(
            [
                k in simulator_params.keys()
                for k in (self.req_cls_str_args + ["auryn_sim_dir", "name", "rule_cst_part"])
            ]
        )

        self.make_rule_cst_str()

    @property
    def cl_str(self):
        """Call string to pass to C++ simulator."""
        class_string = (
            self.simulator_params["auryn_sim_dir"]
            + str(self.simulator_params["name"])
            + " --ID "
            + "%s"
            + "%s"
        )  # Rule string here

        for arg in self.req_cls_str_args:
            class_string += " --%s %s" % (arg, str(self.simulator_params[arg]))
        return class_string
    
    def make_rule_cst_str(self):
        """
        make one string with [taus, Wh1, Wh2]. Remains to be added eta first, and the last layers (Wpre and Wpost).
        EE and IE share their constant params
        """
        self.rule_cst_str = "a"
        for i in range(4): #same taus for EE and IE
            self.rule_cst_str += str(self.simulator_params["rule_cst_part"][i]) + "a"
        for i in range(4):
            self.rule_cst_str += str(self.simulator_params["rule_cst_part"][i]) + "a"
        for i in range(4, len(self.simulator_params["rule_cst_part"])):
            self.rule_cst_str += str(self.simulator_params["rule_cst_part"][i]) + "a"

    @property
    def rule_str(self):
        """Part of call string for parametrized rule."""
        rule_string = " --rule a" + "{}" + "a" + "{}" + self.rule_cst_str
        for i in range(2*2*(self.simulator_params["nh2"]+1)): #2 rules
            rule_string += "{}" + "a"
        rule_string += " --poisson_rate {}"
        return rule_string

    def sample(self, thetas, seeds=None, return_data=False, verbose=0):
        """Forward pass through simulator."""

        ##################################################################################################
        # print("hello")
        ##################################################################################################

        if seeds is None: #should not happen
            print("no seed was passed, generating some, but careful")
            seeds = [42 + i for i, _ in enumerate(thetas)]
        else:
            assert len(thetas) == len(seeds)

        all_spiketimes = []
        all_weights = []

        for th, seed in zip(thetas, seeds):
            # run auryn simulation
            rule_str = self.rule_str.format(*list(th))
            if verbose > 0:
                print(self.cl_str % (str(seed), rule_str))

            ##################################################################################################
            # print(self.cl_str % (str(seed), rule_str))
            ##################################################################################################

            output = subprocess.run(
                self.cl_str % (str(seed), rule_str), shell=True, capture_output=True
            )

            if return_data:
                # Read spiketimes from monitor files
                spiketimes = read_monitor_spiketime_files(
                    workdir=self.simulator_params["workdir"],
                    seed=seed,
                    num_neurons=self.simulator_params["n_recorded"],
                    remove_file=True,
                    which='e'
                )

                # Collect spiketimes
                all_spiketimes.append(spiketimes)

                if self.simulator_params['record_i']:
                    spiketimes_i = read_monitor_spiketime_files(
                        workdir=self.simulator_params["workdir"],
                        seed=seed,
                        num_neurons=self.simulator_params["n_recorded_i"],
                        remove_file=True,
                        which='i'
                    )
                    all_spiketimes.append(spiketimes_i)

                # Read weight traces from monitor files
                weights = read_monitor_weights_files(
                    workdir=self.simulator_params["workdir"],
                    seed=seed,
                    plastic_connections=["ee", "ie"],
                    remove_file=True
                )

                # Collect weights
                all_weights.append(weights)

        # Return collected spiketimes
        if len(all_spiketimes) > 0:
            return(float(output.stdout.decode().split("cynthia")[1]),
                   all_spiketimes,
                   all_weights)
        else:
            return(float(output.stdout.decode().split("cynthia")[1]))
   