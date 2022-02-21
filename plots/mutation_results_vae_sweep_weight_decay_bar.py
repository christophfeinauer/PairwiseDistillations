import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import seaborn as sns
import pandas as pd
from itertools import chain
from tqdm import tqdm
sns.set_theme()
sns.set_style("white")


def parse_exp_files(dirname="../data"):

    expval_dict = {}
    evmutation_lit_dict = {}
    # loop through all files
    for fname in os.listdir(dirname):
        if not fname.endswith("exp"):
            continue
        expval_protein = []
        evmutation_lit_protein = []
        # save effects for protein
        for line in open(os.path.join(dirname, fname)):
            if not line.startswith(">"):
                continue
            expval_protein.append(float(line.split("/")[-1]))
            evmutation_lit_protein.append(float(line.split("/")[-2]))

        # save under protein name
        protein = fname.split("_")[0]
        expval_dict[protein] = expval_protein
        evmutation_lit_dict[protein] = evmutation_lit_protein

    return expval_dict, evmutation_lit_dict


if __name__ == '__main__':

    # parse effects
    expval_dict, evmutation_lit_dict = parse_exp_files()

    # set proteins
    proteins = sorted(list(expval_dict.keys()))

    # dists
    dists = ["O", "PW_U", "PW_M", "IND_U", "IND_M"]

    # set colors
    palette = sns.color_palette("Set2", len(dists))

    # loop through all evaluation files
    dirname = "../evaluations"
    result_dict = dict()

    print("parsing files..")
    for filename in tqdm(os.listdir(dirname)):

        if "vae" not in filename and "evmutation" not in filename:
            continue

        # want only experimental evaluations files
        if not filename.endswith("eval_exp"):
            continue
        path = os.path.join(dirname, filename)

        # check which protein the file belongs to
        protein = filename.split("_")[0]

        # figure out dist, exclude training dists for this plot
        if "_T" in filename:
            continue
        if "extracted" not in filename:
            dist = "O"
        else:
            dist = None
            for _dist in dists[1:]:
                if _dist in filename:
                    dist = _dist
                    break
        if dist is None:
            raise ValueError("cannot determine dist in {}".format(filename))

        # load data and apply sign if necessary (extracted models report energy)
        data = np.loadtxt(path)

        if len(data.shape) > 1:
            data = data[:, 0]
        sign = 1 if (dist == "O" and "evmutation" not in filename) else -1
        sr = sign * spearmanr(data, expval_dict[protein])[0]

        # if evmutation, we just add it to the result_dict and go on
        if "evmutation" in filename:
            result_dict[protein, "evmutation"] = sr
            continue

        # figure out weight_decay and latentdim
        weight_decay = None
        latent_dim = None
        num_hidden_units = None
        data = filename.split(".")
        for i, token in enumerate(data):
            if token.startswith("weightdecay"):
                weight_decay = float(data[i].split("_")[1]+"."+data[i+1])
            if token.startswith("latentdim"):
                latent_dim = int(token.split("_")[1])
            if token.startswith("numhiddenunits"):
                num_hidden_units = int(token.split("_")[1])

        if weight_decay is None or latent_dim is None or num_hidden_units is None:
            continue
        if latent_dim != 5 or num_hidden_units != 40:
            continue

        protein_ind = proteins.index(protein)
        result_dict[protein, weight_decay, dist] = sr

    # get list of possible weight decays
    weight_decays = []
    for key in result_dict.keys():
        if len(key) == 3:
            if key[1] not in weight_decays:
                weight_decays.append(key[1])

    weight_decays = sorted(weight_decays)

    # create subplots
    fig, axes = plt.subplots(len(weight_decays), len(proteins))


    print("plotting...")
    for protein_ind, protein in enumerate(proteins):


        for i, weight_decay in enumerate(weight_decays):

            ax = axes[i, protein_ind]

            for dist_ind, dist in enumerate(dists):

                if (protein, weight_decay, dist) not in result_dict.keys():
                    continue
                sr = result_dict[(protein, weight_decay, dist)]
                x = dist_ind
                ax.bar(x, sr, color=palette[dists.index(dist)])

            # plot evmutation
            if (protein, "evmutation") in result_dict.keys():
                sr = result_dict[protein , "evmutation"]
                ax.plot([-0.5, len(dists)-0.5], [sr, sr], color="red", lw=0.5)

            # delete all axes except left-most
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            ax.grid(True)
            ax.set_yticks(np.arange(0.1, 0.8, 0.1))
            ax.set_yticks(np.arange(0.1, 0.8, 0.01), minor=True)
 
            ax.set_xticks([])
            if protein_ind != 0:
                ax.spines['left'].set_visible(False)
                ax.set_yticklabels([])
            else:
                ax.set_yticks(np.arange(0.1, 0.8, 0.1))
                ax.set_yticks(np.arange(0.1, 0.8, 0.01), minor=True)
                ax.yaxis.set_ticks_position('left')
                ax.tick_params(axis='y', labelsize=8)
            if  i == 2 and protein_ind == 0:
                ax.set_ylabel("Spearman Correlation", fontsize=8)

            if protein_ind == len(proteins)-1:
                ax.set_ylabel("Weight Decay\n{}".format(weight_decay), fontsize=8, rotation=0, labelpad=40)
                ax.yaxis.set_label_position("right")

            # set ylim
            ax.set_ylim(0.3, 0.7)

            if i == 0:
                ax.set_title(protein)




# create color legend
handles = []
for dist_ind, dist in enumerate(dists):
    handle = mlines.Line2D([], [], color=palette[dist_ind], marker='s', linestyle='None', markersize=10, label=dist.replace("_", "/"))
    handles.append(handle)
# add handle for evmutation
handle = mlines.Line2D([], [], color='red', linestyle='-', label="EVMutation")
handles.append(handle)

lgd = fig.legend(handles=handles, ncol=len(proteins)+1, bbox_to_anchor=(0.5, .05), loc='center', frameon=False, fontsize=8)

plt.tight_layout()

fig.subplots_adjust(bottom=0.1)
plt.savefig("mutation_results_vae_sweep_weight_decay_bar.pdf", bbox_extra_artists=(lgd, ))
