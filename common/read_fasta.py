import numpy as np
from collections import defaultdict
import sys
import argparse

# we rely on ordered dictionaries here
assert sys.version_info >= (3, 6)


def read_fasta(fasta_path, alphabet='ACDEFGHIKLMNPQRSTVWY-', default_index=20):

    # read all the sequences into a dictionary
    seq_dict = {}
    with open(fasta_path, 'r') as file_handle:
        seq_id = None
        for line in file_handle:
            line = line.strip()
            if line.startswith(">"):
                seq_id = line
                seq_dict[seq_id] = ""
                continue
            assert seq_id is not None
            line = ''.join([c for c in line if c.isupper() or c == '-'])
            seq_dict[seq_id] += line

    aa_index = defaultdict(lambda: default_index, {alphabet[i]: i for i in range(len(alphabet))})

    seq_msa = []
    keys_list = []
    for k in seq_dict.keys():
        seq_msa.append([aa_index[s] for s in seq_dict[k]])
        keys_list.append(k)

    seq_msa = np.array(seq_msa, dtype=int)

    # reweighting sequences
    seq_weight = np.zeros(seq_msa.shape)
    for j in range(seq_msa.shape[1]):
        aa_type, aa_counts = np.unique(seq_msa[:, j], return_counts=True)
        num_type = len(aa_type)
        aa_dict = {}
        for a in aa_type:
            aa_dict[a] = aa_counts[list(aa_type).index(a)]
        for i in range(seq_msa.shape[0]):
            seq_weight[i, j] = (1.0 / num_type) * (1.0 / aa_dict[seq_msa[i, j]])
    tot_weight = np.sum(seq_weight)
    seq_weight = seq_weight.sum(1) / tot_weight

    return seq_msa, seq_weight, len(alphabet)


def get_seq_len(fasta_path):

    seq_len = 0
    first_line = True

    with open(fasta_path, 'r') as file_handle:

        for line in file_handle:

            if first_line:
                if not line.startswith(">"):
                    raise ValueError("Expect first line to start with >")
                first_line = False
                continue

            if (first_line is False and line.startswith(">")):
                return seq_len

            seq_len += len([c for c in line if c.isupper() or c == '-'])

    raise ValueError("Could not determine sequence length")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--print_seq_len', action='store_true')
    parser.add_argument('--fasta_path', type=str, default=None)

    args = parser.parse_args()

    if args.fasta_path is None and args.print_seq_len:
        raise ValueError("need fasta_path if printing seq_len")

    if args.print_seq_len:
        seq_len = get_seq_len(args.fasta_path)
        print(seq_len)
