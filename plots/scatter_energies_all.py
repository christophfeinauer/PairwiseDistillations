import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import seaborn as sns
import os
from tqdm import tqdm
from math import ceil, floor
sns.set_theme()
sns.set_style('white')
sns.set_palette("pastel")


if __name__ == '__main__':

    val_dict = dict()

    extracted_types = ["PW_M", "PW_U", "IND_M", "IND_U"]

    markers = ['s', 'x']

    palette = sns.color_palette("Set2", len(extracted_types))

    data_types = ["exp", "test"]

    # parse all evaluations
    print("parsing..")
    for file in tqdm(os.listdir("../evaluations")):

        data_type = None
        for dt in data_types:
            if file.endswith("eval_{}".format(dt)):
                data_type = dt
                break
        if data_type is None:
            continue

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

        protein = file.split("_")[0]

        sign = -1 if model_type == "original" else 1

        val_dict[model_class, model_type, data_type, protein] = sign*np.loadtxt(os.path.join("../evaluations", file))

    print("plotting...")
    keys = val_dict.keys()
    proteins = sorted(list(set([k[3] for k in keys])))
    model_classes = sorted(list(set([k[0] for k in keys])))

    print(proteins)
    print(model_classes)



    for protein_ind, protein in enumerate(tqdm(proteins)):

        fig, axes = plt.subplots(1, len(model_classes), figsize=(10, 6))
        nsamples = 100
        plt.suptitle(protein)
        for model_class_ind, model_class in enumerate(model_classes):

            ax = axes[model_class_ind]
            minval = 100000
            maxval= -100000
            for data_type_ind, data_type in enumerate(data_types):

                key_org = (model_class, "original", data_type, protein)
                if key_org not in keys:
                    continue
                vals_org = val_dict[key_org]

                inds = None
                if len(vals_org) > nsamples:
                    inds = np.random.choice(range(len(vals_org)), nsamples)
                    vals_org = vals_org[inds]

                for k, extracted_type in enumerate(extracted_types):
                    key_ex = (model_class, extracted_type, data_type, protein)

                    if key_ex not in keys:
                        continue

                    vals_ex = val_dict[key_ex]
                    if inds is not None:
                        vals_ex = vals_ex[inds]


                    minval = np.minimum(minval, np.minimum(np.min(vals_org), np.min(vals_ex)) - 10)
                    maxval = np.maximum(maxval, np.maximum(np.max(vals_org), np.max(vals_ex)) + 10)

                    ax.set_ylabel("Energies in Extracted Model")
                    ax.set_xlabel("Energies in Original Model")


                    # plot in batches for zorder
                    for zorder in range(0, nsamples//10):
                        r = range(zorder*10,(zorder+1)*10)
                        marker = markers[data_type_ind]
                        s = 60 if data_type_ind == 0 else 20
                        sns.scatterplot(x=vals_org[r], y=vals_ex[r], zorder=zorder, ax=ax, marker=marker, color=palette[k], s=s)
                    ax.set_aspect('equal')
                    ax.set_title(model_class.replace("ardca", "ArDCA").replace("vae", "VAE"))

            ax.plot([minval, maxval], [minval, maxval], c="red", lw=0.5, zorder=0)

            ax.set_xlim(minval, maxval)
            ax.set_ylim(minval, maxval)
            lt = ceil(minval/10)*10
            up = floor(maxval/10)*10
            step = round((up-lt)/5/10)*10
            ax.set_xticks(np.arange(lt, up, step=step))
            ax.set_yticks(np.arange(lt, up, step=step))

        handles = []
        labels = []
        for k, t in enumerate(extracted_types):
            for l, d in enumerate(data_types):
                marker = markers[l]
                handle = Line2D([], [], marker=marker, color=palette[k], lw=0)
                handles.append(handle)
                label = t.replace("_", "/")+" {}".format(d.replace("test", "Test").replace("exp", "Mutational"))
                labels.append(label)

        lgd = fig.legend(handles, labels, ncol=len(handles)//2, bbox_to_anchor=(0.5, 0.07), loc='center', frameon=False)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)

        plt.savefig("energy_scatter_{}.pdf".format(protein))
