# fSBI
Companion code to Confavreux*, Ramesh*, Goncalves, Macke* & Vogels*, NeurIPS 2023, Meta-learning families of plasticity rules in recurrent spiking networks using simulation-based inference
https://openreview.net/forum?id=FLFasCFJNo
___

### Tutorial:
Go through the notebooks:
1/ Fit_posterior
2/ Sample_from_posterior
3/ Simulate_samples
4/ Compute_metrics
5/ Analysis

### Checklist before you start the tutorials:
1/ set-up the python library with all the dependencies (see setup.py).
2/ install and test auryn (see the relevant subsection).
3/ go to tasks_configs/ and change the "auryn_sim_dir" and "workdir" inside the 2 yaml files (where auryn will write the output spike trains for simulations on local hardware).
4/ compile the c++ network simulations in cpp-simulator (tutorial missing rn).

### Installing Auryn:
All spiking network simulations use Auryn, a fast, C++ simulator developped by Friedemann Zenke.
To install, please refer to https://fzenke.net/auryn/doku.php?id=start

### Structure of data provided:

### Metrics implemented and their names:

### Repository structure:
- `setup.py`: set up for installing package files.
- `run_submitit.py`: job script for generating simulations using `submitit`.
- `sample_simulator.py`: job script containing pipeline for simulations.
- `synapsbi`:
    - `__init__.py`: load all modules.
    - `prior.py`: code for prior over simulator parameters.
    - `simulator`:
        - `cpp_simulators`: C++ code for auryn simulators.
        - `__init__.py`: load all simulator modules.
        - `simulator.py`: python wrapper for C++ simulator code.
    - `density_estimator.py`: code for settinng up, training and sampling SBI density estimators.
    - `analyse.py`: code to compute metrics from simulated spike trains.
    - `utils`:
        - `__init__.py`: load all utility modules.
        - `load_data.py`: code for loading data from memory.
        - `visualize.py`: plotting code.
- `science_debug`: notebooks for visualising and debugging inner loops.
- `notebooks`: `.ipynb` notebooks containing testing / development code.