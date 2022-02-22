import pickle as pkl
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
sns.set_theme()
sns.set_style('white')
sns.set_palette("pastel")


if __name__ == '__main__':

    # set model classes
    model_classes = ['ardca']

    # set data types
    data_types = ["test_close", "test_far", "exp"]

    # set markers
    markers = ['*', 's', 'o', 'X', 'v']

    # create figure & axes
    fig, axes = plt.subplots(len(model_classes), len(data_types), figsize=(5,3))

    fig_scatter, axis_scatter = plt.subplots()

    # set extraction dist_pairs
    dist_pairs = [("PW_U", "PW_M"), ("IND_U", "IND_M")]

    # set colors
    palette_dist_pairs = sns.color_palette("Set2", len(dist_pairs))

    fasta_files = list(filter(lambda s: s.endswith("a2m.train"), os.listdir("../data/")))

    proteins = sorted(list(set([fasta_file.split("_")[0] for fasta_file in fasta_files])))

    for fasta_file in fasta_files:

        protein = fasta_file.split("_")[0]

        for model_class_index, model_class in enumerate(model_classes):

            # set model filename
            model_filename = None
            if model_class == "ardca":
                model_filename = fasta_file + ".ardca.jld"
            elif model_class == "vae":
                model_filename = fasta_file + ".vae.pt"
            else:
                raise ValueError("cannot find model class")

            if model_class == "vae":
                continue


            for data_type_index, data_type in enumerate(data_types):

                data_file_suffix = data_type if "test" not in data_type else data_type.split("_")[0]

                # read original values
                data_inds = None
                org_vals_file = os.path.join("../evaluations/", model_filename) + ".eval_{}".format(data_file_suffix)
                if not os.path.isfile(org_vals_file):
                    print("skipping {} - {} since {} does not exist".format(model_filename, data_type, org_vals_file))
                    continue
                if "test" not in data_type:
                    org_vals = np.loadtxt(org_vals_file)
                else:
                    hamming_file = fasta_file.replace(".train", ".train_test_hamming")
                    hamming_dists = np.loadtxt(os.path.join("../data", hamming_file))[:,3]
                    hamming_dists_sortperm = np.argsort(hamming_dists)
                    ndata = len(hamming_dists)
                    split = int(round(ndata*0.9))
                    if data_type == "test_far":
                        data_inds = hamming_dists_sortperm[-(ndata-split):]
                    else:
                        data_inds = hamming_dists_sortperm[:split]
                    org_vals = np.loadtxt(org_vals_file)[data_inds]


                # errors will contain errors for the extraction distributions
                for dist_pair in dist_pairs:
                    errors = []
                    for dist in dist_pair:
                        ex_vals_file = os.path.join("../evaluations/", model_filename) + ".extracted_{}.h5".format(dist) + ".eval_{}".format(data_file_suffix)
                        if not os.path.isfile(ex_vals_file):
                            print("skipping {} - {} since {} does not exist".format(model_filename, data_type, ex_vals_file))
                            continue
                        if data_inds is None:
                            ex_vals = -np.loadtxt(ex_vals_file)
                        else:
                            ex_vals = -np.loadtxt(ex_vals_file)[data_inds]
                        er_vec = np.sqrt((org_vals-ex_vals)**2) / np.abs(np.max(org_vals) - np.min(org_vals))
                        er = np.mean(er_vec)
                        if "test" in data_type and "PW" in dist_pair[0]:
                            c = ["red" if "far" in data_type else "blue"]
                            axis_scatter.scatter(hamming_dists[data_inds], er_vec, c=c)
                        errors.append(er)

                    ax = axes[data_type_index]
                    ax.plot([0, 1], errors, c=palette_dist_pairs[dist_pairs.index(dist_pair)], marker=markers[proteins.index(protein)], ls='--')
                    if data_type == "test_far":
                        ax.set_title("Test Distant")
                    if data_type == "test_close":
                        ax.set_title("Test Close")
                    if data_type == "exp":
                        ax.set_title("Mutational Data")
                    ax.set_yscale("log")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(0.001, 100)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([0, 1])
        ax.grid(True)
        if ax == axes[0]:
            ax.set_ylabel("Normalized RMSE")
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["U", "M"])


#        markers = ['*', 's', 'o', 'X', 'v']
#
#        # plot lines
#        for model_class_id, model_class in enumerate(model_classes):
#            axis = axes[model_class_id]
#            axis.set_title(model_class.replace("vae", "VAE").replace("ardca", "ArDCA").replace("mlp", "MLP"))
#            for fasta in fastas:
#                if '.random' in fasta:
#                    continue
#                # key for original values
#                k_org = Descr(model_class, 'original', fasta, None)
#                # key for U values
#                k_U = Descr(model_class, 'extracted (FULL)', fasta, 0.0)
#                # key for M values
#                k_M = Descr(model_class, 'extracted (FULL)', fasta, 0.99)
#
#                try:
#                    # correlation between original and U
#                    cc_U = results[k_org, k_U][1]
#                    # correlation between original and M
#                    cc_M = results[k_org, k_M][1]
#                except:
#                    print("missing {} or {}".format(k_U, k_M))
#                    continue
#                data_type_id = [data_type in fasta for data_type in data_types].index(True)
#                marker = [fasta_prefix in fasta for fasta_prefix in fasta_prefixes].index(True)
#                axis.plot([0,1], [cc_U, cc_M], '--', c = palette_data_type[data_type_id], marker=markers[marker])
#                axis.set_yscale("log")
#                axis.set_ylim(0.0001, 10000)
#                axis.spines['top'].set_visible(False)
#                axis.spines['right'].set_visible(False)
#                axis.spines['bottom'].set_visible(False)
#                axis.spines['left'].set_visible(False)
#                #axis.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#                if model_class_id == 0:
#                    axis.set_ylabel("Error")
#                else:
#                    axis.set_yticklabels([])
#                    pass
#                axis.set_xticks([0, 1])
#                axis.set_xticklabels(["U", "M"])
#                axis.grid(True)
#
#                #axis.axis('off')

marker_handles = []
for m in markers:
    handle = plt.plot([], [], marker=m, c=[0,0,0])[0]
    marker_handles.append(handle)
marker_text = proteins
marker_legend = fig.legend(marker_handles, marker_text, loc="center left", bbox_to_anchor=(0.9, 0.35), frameon=False)

line_handles = []
for i in range(len(dist_pairs)):
    handle = plt.plot([], [], c = palette_dist_pairs[i], ls="--")[0]
    line_handles.append(handle)
line_text = ["PW" if "PW" in dist_pair[0] else "IND" for dist_pair in dist_pairs]
line_legend = fig.legend(line_handles, line_text, loc="center left", bbox_to_anchor=(0.9, 0.65), frameon=False)

fig.add_artist(marker_legend)

fig.savefig("line_plot_mse_ardca.pdf", bbox_extra_artists=(line_legend, marker_legend), bbox_inches='tight')
