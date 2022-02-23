import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import rankdata
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import seaborn as sns
from copy import copy
import os
from tqdm import tqdm
from math import ceil, floor
sns.set_theme()
sns.set_style('white')
sns.set_palette("pastel")

def parse_effects(path):


    if not os.path.isfile(path):
        raise Exception("cannot find file {}".format(path))

    vals = []
    with open(path) as fid:
        for line in fid:
            if not line.startswith(">"):
                continue
            val = float(line.split("/")[-1])
            vals.append(val)
    return np.array(vals)


if __name__ == '__main__':

    val_dict = dict()

    extracted_types = ["PW_M", "PW_U", "IND_M", "IND_U"]

    markers = ['s', 'x', 'd', 'o', '^']


    # dict from protein to exp file
    exp_dict = dict()

    # parse all evaluations
    print("parsing..")
    for file in tqdm(os.listdir("../evaluations")):

        protein = file.split("_")[0]

        if not file.endswith("exp"):
            continue

        if protein not in exp_dict.keys():
            ind = file.find(".train")
            basename = file[:ind]
            exp_path = os.path.join("../data", basename+".exp")
            exp_dict[protein] = parse_effects(exp_path)


        model_class = None
        if "vae" in file:
            if not "latentdim_5" in file or not "numhiddenunits_40" in file or not "weightdecay_0.01" in file:
                continue
            model_class = "vae"
        elif "ardca" in file:
            model_class = "ardca"

        if model_class is None:
            continue

        model_type = None
        if "extracted" not in file:
            model_type = "original"
        else:
            for token in extracted_types:
                if token in file:
                    model_type = token
                    break
        if model_type is None:
            continue

        sign = -1 if model_type == "original" else 1

        key = (model_class, model_type, protein)
        val_dict[key] = sign*np.loadtxt(os.path.join("../evaluations", file))

    print("plotting...")
    keys = val_dict.keys()
    proteins = sorted(list(set([k[2] for k in keys])))
    model_classes = sorted(list(set([k[0] for k in keys])))
    model_types = sorted(list(set([k[1] for k in keys])))

    palette = sns.color_palette("Set2", len(model_types))

    print(proteins)
    print(model_classes)

    for protein_ind, protein in enumerate(tqdm(proteins)):

        if protein not in exp_dict.keys():
            continue

        vals_exp = copy(exp_dict[protein])

        fig, axes = plt.subplots(2, len(model_classes), figsize=(10, 6))
        plt.suptitle(protein)
        max_samples = 500

        sample_inds = None
        if len(vals_exp) > max_samples:

            sample_inds = np.random.choice(range(len(vals_exp)), max_samples, replace=False)
            vals_exp = vals_exp[sample_inds]

        for model_class_ind, model_class in enumerate(model_classes):

            model_class_nice = model_class.replace("ardca", "ArDCA").replace("vae", "VAE")

            for k, model_type in enumerate(model_types):

                key = (model_class, model_type, protein)

                if key not in keys:
                    continue

                vals_model = copy(val_dict[key])

                if sample_inds is not None:
                    vals_model = vals_model[sample_inds]
                vals_model = vals_model - np.mean(vals_model)



                ranks_exp = rankdata(vals_exp, method='dense')
                ranks_model = rankdata(vals_model, method='dense')

                nsamples = len(vals_model)//10

                nzorders = len(vals_model)//nsamples + 1
                for zorder in range(0, nzorders):
                    inds = list(range(zorder*nsamples, min((zorder+1)*nsamples, len(vals_model))))
                    if len(inds) == 0:
                        continue
                    # plot energies vs values
                    sns.scatterplot(x=vals_exp[inds], y=vals_model[inds], color=palette[k], marker=markers[k], ax=axes[0, model_class_ind], zorder=zorder, s=10)
                    axes[0, model_class_ind].set_title(model_class_nice)
                    axes[0, model_class_ind].set_xlabel("Experimental Values")
                    axes[0, model_class_ind].set_ylabel("Model Energies")

                    # plot ranks vs ranks
                    sns.scatterplot(x=ranks_exp[inds], color=palette[k], y=ranks_model[inds], marker=markers[k], ax=axes[1, model_class_ind], zorder=zorder, s=10)
                    axes[1, model_class_ind].set_title(model_class_nice)
                    axes[1, model_class_ind].set_xlabel("Experimental Rank")
                    axes[1, model_class_ind].set_ylabel("Model Energy Rank")


                # plot in batches for zorder
                #for zorder in range(0, nsamples//10):
                #    r = range(zorder*10,(zorder+1)*10)
                #    sns.scatterplot(x=vals_org[r], y=vals_ex[r], zorder=zorder, ax=ax, marker=marker, color=palette[k], s=s)
                #ax.set_aspect('equal')
                #ax.set_title(model_class.replace("ardca", "ArDCA").replace("vae", "VAE"))



        handles = []
        labels = []
        for k, t in enumerate(model_types):
            marker = markers[k]
            handle = Line2D([], [], marker=marker, color=palette[k], lw=0)
            handles.append(handle)
            label = t.replace("_", "/").replace("original", "Original")
            labels.append(label)

        lgd = fig.legend(handles, labels, ncol=len(handles), bbox_to_anchor=(0.5, 0.07), loc='center', frameon=False)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)

        plt.savefig("mutational_scatter_{}.pdf".format(protein))
