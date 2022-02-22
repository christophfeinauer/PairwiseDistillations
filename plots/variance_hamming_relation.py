import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('white')


if __name__ == "__main__":

    hamming_file = "../data/BRCA1_HUMAN_1_b0.5.a2m.train_test_hamming"
    eval_file = "../evaluations/BRCA1_HUMAN_1_b0.5.a2m.train.ardca.jld.eval_test"
    eval_extracted_file = "../evaluations/BRCA1_HUMAN_1_b0.5.a2m.train.ardca.jld.extracted_PW_M.h5.eval_test"

    hamming_data = np.loadtxt(hamming_file)
    eval_data = -np.loadtxt(eval_file)
    eval_extracted_data = np.loadtxt(eval_extracted_file)

    palette = sns.color_palette("Set2", 4)

    seq_len = int(round(hamming_data[0, 2]/hamming_data[0,3]))

    hamming_data = np.array([int(h) for h in hamming_data[:, 2]])

    fig, axes = plt.subplots(2, 2, figsize=(12,8))

    sns.scatterplot(x=hamming_data/seq_len, y=eval_data, ax=axes[0,0], color=palette[0], alpha=0.5)
    sns.scatterplot(x=hamming_data/seq_len, y=np.sqrt((eval_data-eval_extracted_data)**2), ax=axes[1,0], color=palette[1], alpha=0.5)
    axes[0,0].set_ylabel("Original ArDCA Energies")
    axes[0,0].set_xlabel("Normalized Hamming Distance")
    axes[0,0].set_title("Original ArDCA Energies")
    axes[1,0].set_title("Single Sequence Absolute Error, PW/M vs. Original ArDCA Energies")
    axes[1,0].set_xlabel("Normalized Hamming Distance")
    axes[1,0].set_ylabel("Absolute Error")

    variances = []
    hamming = []
    errors = []


    for i in range(seq_len):

        inds = hamming_data == i

        if sum(inds) == 0:
            continue

        hamming.append(i)
        variances.append(np.std(eval_data[inds]))
        errors.append(np.sqrt(np.mean((eval_data[inds] - eval_extracted_data[inds])**2)))


sns.scatterplot(x=np.array(hamming)/seq_len, y=variances, ax=axes[0,1], color=palette[2], alpha=0.5)
axes[0,1].set_ylabel("Standard Deviation")
axes[0,1].set_xlabel("Normalized Hamming Distance")
axes[0,1].set_title("Original ArDCA Energies Standard Deviation")

sns.scatterplot(x=np.array(hamming)/seq_len, y=errors, ax=axes[1,1], color=palette[3], alpha=0.5)

axes[1,1].set_title("RMSE, PW/M vs. Original ArDCA Energies")
axes[1,1].set_xlabel("Normalized Hamming Distance")
axes[1,1].set_ylabel("RMSE")


plt.suptitle("BRCA1 ArDCA Energies and Errors")
plt.tight_layout()
plt.savefig("variance_hamming_relation.pdf")
