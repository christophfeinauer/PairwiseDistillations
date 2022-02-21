import numpy as np
import argparse
import sys
from copy import copy
import os
from tqdm import tqdm

# we rely on ordered dicts
assert sys.version_info >= (3, 7)


def parse_fasta(fasta_path, alphabet, check_len=True):

    seq_dict = {}
    first_key = None
    with open(fasta_path) as fid:
        key = None
        for line in fid:
            line = line.strip()
            if line.startswith(">"):
                if first_key is None:
                    first_key = key
                key = line
                if key not in seq_dict.keys():
                    seq_dict[key] = ""
                continue
            else:
                seq = []
                for c in line:
                    if c in alphabet:
                        seq.append(c)
                    elif c.isupper():
                        seq.append("-")
                seq_dict[key] = seq_dict[key] + ''.join(seq)

    # check if all sequences have same length
    if check_len:
        assert len(set(map(lambda seq: len(seq), seq_dict.values()))) == 1

    return seq_dict, first_key


def remove_dups(seq_dict):

    # we want to keep the first appearance (since else we might remove first key)
    rev_seq_dict = {val: key for key, val in reversed(seq_dict.items())}
    seq_dict_nodups = {val: key for key, val in reversed(rev_seq_dict.items())}

    return seq_dict_nodups


def create_train_test(seq_dict, test_ratio, first_key = None):

    # Creates random train test split, ensuring that 'first_key' is in train
    if first_key is not None:
        assert first_key in seq_dict

    keys = np.array(list(seq_dict.keys()))
    nkeys = len(keys)
    perm = np.random.permutation(nkeys)

    cut_ind = int(args.test_ratio * nkeys)

    ind_test = perm[:cut_ind]
    ind_train = perm[cut_ind:]

    seq_dict_train = {}
    seq_dict_test = {}
    if first_key is not None:
        seq_dict_train[first_key] = seq_dict[first_key]

    for key in keys[ind_train]:
        if key == first_key:
            continue
        seq_dict_train[key] = seq_dict[key]

    for key in keys[ind_test]:
        if key == first_key:
            continue
        seq_dict_test[key] = seq_dict[key]

    assert len(seq_dict_train.keys() & seq_dict_test.keys()) == 0
    assert len(set(seq_dict_train.values()) & set(seq_dict_test.values())) == 0

    return seq_dict_train, seq_dict_test


def write_seq_dict(seq_dict, fname, overwrite=False):

    if os.path.isfile(fname) and overwrite is False:
        print("{} exists - skipping".format(fname))
        return

    with open(fname, "w") as fid:
        for key, seq in seq_dict.items():
            fid.write(key)
            fid.write("\n")
            fid.write(seq)
            fid.write("\n")


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_path", type=str)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--no_remove_dups", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nrandom", type=int, default=1000)
    parser.add_argument("--alphabet", type=str, default="ACDEFGHIKLMNPQRSTVWY-")
    parser.add_argument("--overwrite", action='store_false')

    args = parser.parse_args()
    alphabet = [c for c in args.alphabet]

    assert args.test_ratio <= 1.0 and args.test_ratio >= 0.0

    np.random.seed(args.seed)

    seq_dict, first_key = parse_fasta(args.fasta_path, args.alphabet)
    print("Parsed {} sequences".format(len(seq_dict)))
    seq_len = len(next(iter(seq_dict.values())))
    print("seq_len = {}".format(seq_len))

    if not args.no_remove_dups:
        len_before_remove_dups = len(seq_dict)
        seq_dict = remove_dups(seq_dict)
        removed = len_before_remove_dups - len(seq_dict)
        assert first_key in seq_dict.keys()
        print("Removed {} sequences, left with {}".format(removed, len(seq_dict)))

    seq_dict_train, seq_dict_test = create_train_test(seq_dict, args.test_ratio, first_key)

    suffices = ["train", "test"]
    seq_dicts = [seq_dict_train, seq_dict_test]

    for seq_dict, suffix in zip(seq_dicts, suffices):
        write_seq_dict(seq_dict, args.fasta_path + "." + suffix)
