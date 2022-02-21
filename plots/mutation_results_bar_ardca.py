import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import seaborn as sns
import pandas as pd
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

    # set model classes
    model_classes = ["ardca", "evmutation"]
    model_classes_nice = ["ArDCA", "EVMutation"]

    # set dists
    dists = ["O", "PW_U", "PW_M", "IND_U", "IND_M"]

    # set colors
    palette = sns.color_palette("Set2", len(dists))

    # set markers
    markers = ["d", "o", "x", "s"]

    # parse effects
    expval_dict, evmutation_lit_dict = parse_exp_files()

    # set proteins
    proteins = sorted(list(expval_dict.keys()))

    # create subplots (one model class less since we plot evmutation differently)
    fig, axes = plt.subplots(len(model_classes)-1, len(proteins), figsize=(7,2))

    # create evmutation dict
    evmutation_dict = {}

    # create table for mutation results on original models
    mutation_table = pd.DataFrame(index=model_classes, columns=proteins)

    # loop through all evaluation files
    dirname = "../evaluations"
    results = defaultdict(dict)
    for filename in os.listdir(dirname):

        # want only experimental evaluations files
        if not filename.endswith("eval_exp"):
            continue
        path = os.path.join(dirname, filename)

        # check which protein the file belongs to
        protein = filename.split("_")[0]

        # find dist
        if "extracted" not in filename:
            dist = "O"
        else:
            tokens = list(filter(lambda s: "extracted_" in s, filename.split(".")))
            if len(tokens) != 1:
                raise ValueError("cannot determine dist in {}".format(filename))
            dist = tokens[0].replace("extracted_", "")
            # check if we have a dist that we want to plot
            if dist not in dists:
                continue

        # parse model class
        model_class = None
        for _model_class in model_classes:
            if _model_class in filename:
                model_class = _model_class
        if model_class is None:
            continue

        # load data and apply sign if necessary (extracted models report energy)
        data = np.loadtxt(path)
        if len(data.shape) > 1:
            data = data[:, 0]
        sign = -1 if (dist != "O" or model_class == "evmutation") else 1
        sr = sign * spearmanr(data, expval_dict[protein])[0]

        # add result to mutation_table if original or evmutation
        if dist == "O":
            mutation_table.loc[model_class, protein] = sr

        if protein in evmutation_lit_dict.keys():
            mutation_table.loc["evmutation_lit", protein] = spearmanr(evmutation_lit_dict[protein], expval_dict[protein])[0]

        # figure out axes index
        model_class_index = model_classes.index(model_class)
        protein_index = proteins.index(protein)
        ax = axes[protein_index] if model_class != "evmutation" else None

        # figure out dist_ind
        dist_ind = dists.index(dist)


        # in case of evmutation we simply plot a horizontal line for all model classes and go one
        if model_class == "evmutation":
            for model_class_index in range(len(model_classes)-1):
                ax = axes[protein_index]
                ax.plot([0,len(dists)], [sr,sr], c='red', lw=0.5)
                continue
        else:
            ax.bar(dist_ind, [sr], color=palette[dist_ind], align='edge')


        # delete all axes except left-most
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.grid(True)
        ax.set_yticks(np.arange(0.1, 0.8, 0.1))
        ax.set_yticks(np.arange(0.1, 0.8, 0.01), minor=True)
 

        ax.set_xticks([])
        if protein_index != 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
        else:
            ax.set_yticks(np.arange(0.1, 0.8, 0.1))
            ax.set_yticks(np.arange(0.1, 0.8, 0.01), minor=True)
            ax.yaxis.set_ticks_position('left')

        # set ylim
        ax.set_ylim(0.3, 0.7)

    # create labels
    #for model_class_index, model_class in enumerate(model_classes_nice[:-1]):
    #    ax = axes[model_class_index, -1]
    #    ax.yaxis.set_label_position("right")
    #    ax.set_ylabel(model_class, rotation=0, ha='left')

    for protein_index, protein in enumerate(proteins):
        ax = axes[protein_index]
        ax.set_title(protein)

    # create color legend
    handles = []
    for dist_ind, dist in enumerate(dists):
        handle = mlines.Line2D([], [], color=palette[dist_ind], marker='s', linestyle='None', markersize=10, label=dist.replace("_", "/"))
        handles.append(handle)
    # add handle for evmutation
    handle = mlines.Line2D([], [], color='red', linestyle='-', label="EVMutation")
    handles.append(handle)
 

    lgd = fig.legend(handles=handles, ncol=len(proteins)+1, bbox_to_anchor=(0.5, .09), loc='center', frameon=False)
    fig.subplots_adjust(bottom=0.2)

    #plt.tight_layout()
    plt.savefig("mutation_results_bar_ardca.pdf", bbox_extra_artists=(lgd,))
    print(mutation_table)
