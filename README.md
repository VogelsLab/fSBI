# fSBI
Companion code to Confavreux*, Ramesh*, Goncalves, Macke* & Vogels*, NeurIPS 2023, *Meta-learning families of plasticity rules in recurrent spiking networks using simulation-based inference*  
https://openreview.net/forum?id=FLFasCFJNo
___

### Tutorial:
Go through the notebooks:
- `Fit_posterior.ipynb`
- `Sample_from_posterior.ipynb`
- `Simulate_samples.ipynb`
- `Compute_metrics.ipynb`
- `Analysis/example.ipynb`

### Checklist before you start the tutorial:
- Set-up the repo (see `setup.py`).
- Install and test Auryn (see the relevant subsection).
- Go to tasks_configs/ and change the "auryn_sim_dir" and "workdir" inside the 2 yaml files (where auryn will write the output spike trains for simulations on local hardware).
- Compile the Auryn network simulations (tutorial missing rn).

### Installing Auryn:
All spiking network simulations in this repo use Auryn, a fast, C++ simulator developped by Friedemann Zenke.
To install, please refer to https://fzenke.net/auryn/doku.php?id=start
Note that installing Auryn with MPI support is not required for the tutorial.

### Installing Auryn:
- Compile the auryn simulations `sim_bg_IF_EEEIIEII_6pPol.cpp` and `sim_bg_CVAIF_EEIE_T4wvceciMLP.cpp` in `synapsbi/simulator/cpp_simulators/`, adapting the `Makefile` in the repo, you should only need to change AURYNPATH in the makefile.  
For troubleshooting, refer to https://fzenke.net/auryn/doku.php?id=manual:compileandrunaurynsimulations

### Structure of data provided:
The main data we release alongside this paper are 

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
