# fSBI
Companion code to Confavreux*, Ramesh*, Gonçalves, Macke* & Vogels*, NeurIPS 2023, *Meta-learning families of plasticity rules in recurrent spiking networks using simulation-based inference*  
https://openreview.net/forum?id=FLFasCFJNo
___
### Checklist before starting the tutorial:
- Set-up the repo and dependencies with `setup.py` (see section below for more details).
- Install and test Auryn (see section below for more details).
- Compile the Auryn network simulations (see section below for more details).

### Tutorial:
Go through the notebooks:
- 1/ `Fit_posterior.ipynb`
- 2/ `Sample_from_posterior.ipynb`
- 3/ `Simulate_samples.ipynb`
- 4/ `Compute_metrics.ipynb`
- 5/ `Analysis/example.ipynb`

### Setting up the repository:
- Create and activate a new conda virtual environment
https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
- Install python, pip and setuptools: `conda install python`  
Note: this may install a python version that is too recent for some of the dependencies (e.g. pytorch), in which case force install an earlier version: e.g. `conda install python=3.11`
- Install fSBI with setup.py: `python3 -m pip install -e .`
- Install jupyter notebook and start the tutorial

### Installing Auryn:
All spiking network simulations in this repo use Auryn, a fast, C++ simulator developped by Friedemann Zenke.
To install, please refer to https://fzenke.net/auryn/doku.php?id=start  
Note that installing Auryn with MPI support is not required for the tutorial.

### Compile Auryn simulations:
- Compile the auryn simulations `sim_bg_IF_EEEIIEII_6pPol.cpp` and `sim_bg_CVAIF_EEIE_T4wvceciMLP.cpp` located in `synapsbi/simulator/cpp_simulators/`. First, edit the `Makefile` in the same directory, you should only need to change AURYNPATH there.  
For troubleshooting, refer to https://fzenke.net/auryn/doku.php?id=manual:compileandrunaurynsimulations
- Go to tasks_configs/ and update `auryn_sim_dir` and `workdir` inside the 2 yaml files (these variables control where Auryn will write output spike trains).

### Structure of data provided:
The main data release alongside this paper are all the plasticity rules we simulated to obtain the posteriors for generally plausible plasticity rules.
- `data_synapsesbi/bg_IF_EEEIIEII_6pPol/bg_IF_EEEIIEII_6pPol_all.npy` are all the rules from the polynomial search space (Fig 2).
- `data_synapsesbi/bg_CVAIF_EEIE_T4wvceciMLP/bg_CVAIF_EEIE_T4wvceciMLP_all.npy` are all the rules from the MLP search space (Fig 4).  

These are numpy structured arrays containing the plasticity parameters and the network metrics computed after a simulation of a spiking network evolving with the rule in question.  

Fields:  
- `theta`: the values of the plasticity parameters.  
For the polynomial rules, this array is: [τ<sub>pre EE</sub>, τ<sub>post EE</sub>, α<sub>EE</sub>, β<sub>EE</sub>, γ<sub>EE</sub>, κ<sub>EE</sub>, τ<sub>pre EI</sub>, etc... same for EI, IE and II].  
For the MLP rules: [η<sub>EE</sub>, η<sub>IE</sub>, W<sub>pre EE</sub>, W<sub>post EE</sub>, W<sub>pre IE</sub>, W<sub>post IE</sub>,].
- `seed`: unique identifier for each rule
- metrics: `rate`, `cv_isi`, `kl_isi'`, `spatial_Fano`, `temporal_Fano`, `auto_cov`, `fft`, `w_blow`, `std_rate_temporal'`, `std_rate_spatial`, `std_cv`, `w_creep`, `rate_i`, `weef`, `weif`, `wief`, `wiif`. Refer to the paper for the definition of each metric
