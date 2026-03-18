# Holomorphic crack detection
This project is an extension of the original [PIHNN project](https://github.com/teocala/pihnn). It combines holomorphic neural networks (HNN) with a genetic algorithm (GA) approach to detect single cracks in 2D elastic bodies based on strain/displacement measurements.

The extension is the work of Nicolas Cuenca, Jonas Hund, and Tito Andriollo. The full article can be found here: https://doi.org/10.1016/j.engfracmech.2026.112050

## Setting up the Python environment
We recommend to create a [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#installing-conda) environment based on the [`environment.yml`](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) file provided.

In the folder with the `environment.yml` execute 
```
conda env create -f environment.yml
```
This will create an environment `holomorphic_crack_detection` where all packages necessary to execute the example script are provided. Before executing the script, activate the environment through
```
conda activate holomorphic_crack_detection
```

In the `experiment_1` folder, then execute the script `exp_1_central_crack.py`  through
```
python exp_1_central_crack.py
```
