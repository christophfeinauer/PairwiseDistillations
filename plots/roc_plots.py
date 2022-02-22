import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import re

sns.set_theme()
sns.set_style('white')

if __name__ == '__main__':

    cmap_max_pred = 30

    model_classes = ['ardca', 'vae']

    mosaic = """
    AABBCC
    DDDEEE
    """

    fig = plt.figure(constrained_layout=True)
    axd = fig.subplot_mosaic(mosaic)

    eval_dir = "../evaluations"

    # model_labels
    dists = ["O", "PW_M", "PW_U"]

    palette = sns.color_palette("Set2", len(dists))

    # the sorting is just to keep the labels ordered (the plotting is order agnostic)
    sort_key = lambda s: max([dists.index(dist) if dist in s else 0 for dist in dists])

    roc_file_names = sorted(list(filter(lambda f: f.endswith("roc"), os.listdir("../evaluations/"))), key=sort_key)

    for roc_file_name in roc_file_names:

        if "_T" in roc_file_name:
            continue


        if "vae" in roc_file_name:
            model_class = "vae"
            model_class_ind = model_classes.index(model_class)
        elif "ardca" in roc_file_name:
            model_class = "ardca"
            model_class_ind = model_classes.index(model_class)
        else:
            continue

        # plot vae only for one type
        if model_class == "vae":
            if "weightdecay_0.01" not in roc_file_name:
                continue

        dist = "O"
        for _dist in dists:
            if _dist in roc_file_name:
                dist = _dist
                break

        roc_data = np.loadtxt(os.path.join(eval_dir, roc_file_name))

        # plot rocs
        if model_class == "ardca":
            print(roc_file_name)
            axis = axd["D"]
            axis.plot(roc_data[:, -1], c=palette[dists.index(dist)], label=dist.replace("_", "/"))
            axis.set_ylabel("PPV")
            axis.set_xlabel("Number of Predictions")
            axis.legend()
        else:
            axis = axd["E"]
            axis.plot(roc_data[:, -1], c=palette[dists.index(dist)], label=dist.replace("_", "/"))
            axis.set_ylabel("PPV")
            axis.set_xlabel("Number of Predictions")
            axis.legend()


        # plot 2d maps
        title = model_class.replace("ardca", "ArDCA").replace("vae", "VAE")
        if model_class == "ardca":
            if dist == "O":
                axis = axd["A"]
                title += " (O)"
            else:
                axis = axd["B"]
                title += " (U/M)"
        elif model_class == "vae":
            axis = axd["C"]
            title += " (U/M)"

        axis.set_title(title)


        cmap_file_name = roc_file_name[:-3]+"cmap"
        cmap_data = np.loadtxt(os.path.join(eval_dir, cmap_file_name))

        N = int(max(cmap_data[:, 1]))

        cmap = np.zeros((N, N), dtype=int)

        for i, j, d in cmap_data:
            cmap[int(i)-1, int(j)-1] = int(d<8)

        x_pos = []
        y_pos = []
        x_neg = []
        y_neg = []

        for i, j, _ in roc_data[:N]:
            if cmap[int(i)-1,int(j)-1] == 1:
                x_pos.append(i)
                y_pos.append(j)
            else:
                x_neg.append(i)
                y_neg.append(j)


        # plot ground truth
        x_true, y_true = np.nonzero(cmap)
        axis.scatter(x_true, y_true, marker='s', s=4, alpha=0.1,  facecolor=[0.4, 0.4, 0.4])
        axis.scatter(y_true, x_true, marker='s', s=4, alpha=0.1,  facecolor=[0.4, 0.4, 0.4])


        if "M" in dist:
            x_pos, y_pos = y_pos, x_pos
            x_neg, y_neg = y_neg, x_neg
        axis.scatter(x_pos, y_pos, s=4, marker='s', facecolor=[0, 1, 0])
        axis.scatter(x_neg, y_neg, s=4, marker='s', facecolor=[1, 0, 0])
        axis.set_xticks([0, 10, 20, 30])
        axis.set_yticks([0, 10, 20, 30])
        axis.set_aspect('equal')
    plt.savefig("YAP_roc.pdf")
