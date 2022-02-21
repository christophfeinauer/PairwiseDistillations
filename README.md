# Interpretable Pairwise Distillations for Generative Protein Sequence Models

This code accompanies _Interpretable Pairwise Distillations for Generative Protein Sequence Models_ by Feinauer C; Meynard-Piganeau, B; Lucibello C [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.14.464358v1).


## Requirements

- Julia version 1.6 or higher
- Conda 4.11 or higher
- Code assumes at various places that a GPU is present


## Setup

### Clone repository and enter the directory

On the shell, execute

```bash
$git clone git@github.com:christophfeinauer/PairwiseDistillations.git && cd PairwiseDistillations
```

### Julia Environment

In the repository directory,  run 

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(;verbose=true)'
```

### Conda Environment

In the repository directory, run

```bash
conda env create -f environment.yml
```

Then, activate it using

```bash
conda activate PairwiseDistillations
```

All subsequent instructions assume that you are in the correct environment.

### EVMutation

In the repository directory, run

```bash
git submodule update
```

to pull the `plmc` submodule. Then, enter the subdirectory of plmc and build it with

```bash
pushd evmutation/plmc/ && make all && popd
```

*NOTE*: The `plmc` code also allows to compile with parallelization enabled (see readme in the subdirectory). This can be achieved by substiting `make all` in the command above with `make all-openmp`. However, the parallel implementation crashed our machine (not just the program, but the complete server) repeatedly. We did not invest any time in debugging it and just used the single-core version. 


## Reproducing Results


The complete pipeline is 

```
Preprocess Data -> Train Original Model and Create Samples -> Extract Independent and Paiwise Models using Samples -> Evaluate (calculate energies) -> Plot
```

There is a bash script called `run_all.sh` in the repository directory which contains the complete pipeline. However, due to the complex nature of the pipeline it is recommended to run the following commands one by one.

## Train Original Models


### ArDCA

In the repostitory directory, run





