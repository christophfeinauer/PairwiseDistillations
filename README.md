# Interpretable Pairwise Distillations for Generative Protein Sequence Models

Software accompanying _Feinauer, Christoph, Barthelemy Meynard-Piganeau, and Carlo Lucibello. "Interpretable pairwise distillations for generative protein sequence models." PLOS Computational Biology 18.6 (2022): e1010219_

[Link to Paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010219)

Please cite the paper if you use this code (see below).

## Requirements

- Linux (code has been tested on Ubuntu and Manjaro, not on anything else)
- Julia version 1.6 or higher
- Conda 4.11 or higher
- Git LFS
- Code assumes at various places that a GPU is present
- If you want to reproduce the contact prediction results, [hmmer](http://hmmer.org/) needs to be installed.

## Setup

### Clone repository and enter the directory

Make sure `git-lfs` is installed by running

```bash
$ git lfs install --skip-repo
```

on the shell.

Then, execute


```bash
$ git clone git@github.com:christophfeinauer/PairwiseDistillations.git && cd PairwiseDistillations
```

This downloads the necessary code files and the data. 

Please make sure that the `git-lfs` objects were pulled correctly by executing in the repository directory

```bash
head ./data/YAP1_HUMAN_1_b0.5.a2m
```

This should show you protein sequences. If not, something has gone wrong.


### Julia Environment

In the repository directory,  run 

```bash
$ julia --project=. -e 'using Pkg; Pkg.instantiate(;verbose=true)'
```

### Conda Environment

In the repository directory, run

```bash
$ conda env create -f environment.yml
```

Then, activate it using

```bash
$ conda activate PairwiseDistillations
```

All subsequent instructions assume that you are in the correct environment.

### EVMutation

The EVMutation code comes from the [plmc repostory](https://github.com/debbiemarkslab/plmc), which is used as a submodule here.

In the repository directory, run

```bash
$ git submodule update
```

to pull the `plmc` submodule. Then, enter the subdirectory of plmc and build it with

```bash
$ pushd evmutation/plmc/ && make all && popd
```

*NOTE*: The `plmc` code also allows to compile with parallelization enabled (see readme in the subdirectory). This can be achieved by substituting `make all` in the command above with `make all-openmp`. However, the parallel implementation crashed our machine (not just the program, but the complete server) repeatedly. We did not invest any time in debugging it and just used the single-core version. 


## Reproducing Results


The complete pipeline is 

```
1. Preprocess Data
2. Train Original Model and Create Samples
3. Extract Independent and Paiwise Models using Samples
4. Evaluate (calculate energies)
5. Plot
```

There is a bash script called `run_all.sh` in the repository directory which contains the complete pipeline. However, due to the complex nature of the pipeline it is recommended to run the following commands one by one.

### Preprocessing Data

In the repository directoy, run

```bash
./data/preprocess_data.sh
```

This creates several new files in the data directory:

1. For every `.a2m` file, it creates a set of training and testing sequences (suffix `.train` and `.test`)
2. For every test sequence, it calculates the mean and minimum Hamming distance (suffix `.train_test_hamming`)
3. For every mutational dataset, it creates an MSA that contains the mutated sequence and the corresponding experimental value in the annotation (suffix `.exp`)

### Train Original Models

The following steps train the original models and create samples from them (except for EVMutation). The later scripts ignore as far as possible cases where original models are missing, so you can also train for example only ArDCA models or just a subset of the VAE models and go on to the later stages.


#### ArDCA

The ArDCA code comes from the [ArDCA repository](https://github.com/pagnani/ArDCA.jl). It is installed as a registered Julia package (see above).

In the repostitory directory, run

```bash
ardca/train_ardca_models.sh
```

This calls the training code found in `ardca/ardca.jl` on all training datasets and saves the model in `models/`. The naming of the models follows the scheme `{DATASET_FILENAME}.train.ardca.jld`. Training is done using threads. The default is to set the number of threads equal to the number of cores in the machine - if that is too much, it can be changed by modifying the option `-t $(nproc -all)` in the `julia` invocation in `ardca/train_ardca_models.sh`.

The code also creates samples from the `T`, `U` and `M` distribution. They are saved in `samples/` and follow the naming scheme `{MODEL_FILENAME}.samples_{M,T,U}.h5`.


### VAE

Most of the VAE code comes from the [repository](https://github.com/xqding/PEVAE_Paper) accompanying the paper "Deciphering protein evolution and fitness landscapes with latent space models". A reduced and modified version is included directly as repository files here.


In the repository directory, run

```bash
vae/train_vae_models.sh
```

The code creates the same files as for `ArDCA`, with the same naming scheme except that the filenames for the models and samples also contain the settings for the latent dimension, the number of hidden units and the weight decays.

Note that creating the samples for the VAE can create a _very_ long time (up to several hours for a single model), depending on the machine/GPU. Most of that time is not spent in creating the sequences, butevaluating their enegies/negative log probabilities due to the fact that we use a very large number of ELBO samples (`5000`). If you want to quickly reproduce the result you can try reducing this number by modfying the line `elbo_samples=5000` in `vae/train_vae_models.sh`.


### EVMutation

In the repository directory, run

```bash
evmutation/train_evmutation.sh
```

This trains EVMutation models on all train datasets. The output of EVMutation is a binary file foraatm, which is then transformed into the same format as we use for the extracted pairwise models by the script `evmutation/plmc_to_extracted.jl`.


## Extract Pairwise and Independent Models

After having trained the original models, run in the repository directory

```bash
extracted/extract_couplings.sh
```

This extracts independent and pairwise models from the samples and saves them to the `models/` folder, adding a `extracted_{D}` tag, where `D` is the distribution used (`M`, `T` or `U`). Note that while we also have code that extracts models using samples from the `T` (*T*raining) distribution, these typically do not perform well and are not reported in the paper.


The extraction code is in `extracted/extract_couplings_from_samples.py`.

## Evaluate all Models


After having trained the original models and extracted pairwise and independent models, run

```bash
evaluations/evaluate_all.sh
```

in the repository directory. This evaluates the the energies/log probabilities of all models found in the `models/` folder on the datasets that end in `.test` or `.exp`. These are plain text files, adding a `eval_TYPE` tag to the filenames, where `TYPE` indicates whether the values correspond to test or experimental (mutational) sequences.

The naming scheme is cumulative, so for example the evluation file called `YAP1_HUMAN_1_b0.5.a2m.train.vae.latentdim_5.weightdecay_0.1.numhiddenunits_40.pt.extracted_PW_M.h5.eval_test` would contain the

1. Energy values on the test set...
2. ... of the pairwise model extracted using samples from the original model distribution
3. ... of a VAE model trained with `40` hidden units, weight decay set to `0.1` and a latent dimension of `5`
4. ... trained on the file `YAP1_HUMAN_1_b0.5.a2m.train`.


Independent models have `IND` instead of `PW` in the name. Note also that original `ArDCA` and `VAE` models report the log probability and not the eneriges. Extracted models and EVMutation models on the other hand report the energy. When comparing the evaluations of these model types the sign of one has to be switched.


## Contact Prediction

The script for using extracted models for contact prediction can be found in `extracted/predict_contacts.sh`. Note that [hmmer](http://hmmer.org/) needs to be installed for this to work.

## Plotting

Plotting scripts can be found in `plots/`.

## Credits

This is a list of repositories from which some of the code in this repository has been taken. Please cite appropriately if you use code from this repository.

- plmc (EVMutation): https://github.com/debbiemarkslab/plmc
- ArDCA: https://github.com/pagnani/ArDCA.jl
- VAE: https://github.com/xqding/PEVAE_Paper

## Citation

Please use these citations if you use this code. They will be updated as soon as the paper is accepted.

### Text

Feinauer, Christoph, Barthelemy Meynard-Piganeau, and Carlo Lucibello. "Interpretable pairwise distillations for generative protein sequence models." PLOS Computational Biology 18.6 (2022): e1010219.

### Bibtex

@article{feinauer2022interpretable,
  title={Interpretable pairwise distillations for generative protein sequence models},
  author={Feinauer, Christoph and Meynard-Piganeau, Barthelemy and Lucibello, Carlo},
  journal={PLOS Computational Biology},
  volume={18},
  number={6},
  pages={e1010219},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}

