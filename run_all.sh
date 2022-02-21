#/bin/bash

set -e

# preprocess data (creates train/test split and alignments corresponding to mutation effects)
data/preprocess_data.sh

# train ardca models and sample from them
ardca/train_ardca_models.sh

# train VAE models
vae/train_vae_models.sh

# extract pairwise and independent models
extracted/extract_couplings.sh

# evaluate original and extracted models
evaluations/evaluate_all.sh
