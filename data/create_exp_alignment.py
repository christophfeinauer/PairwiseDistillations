import argparse

def read_fasta_info(fasta_path):

    position_map = {}

    first_sequence = ""
    first_key = None
    with open(fasta_path) as fid:
        for line in fid:
            if line.startswith(">"):
                if first_sequence:
                    break
                first_key = line.strip()
                # get uniprot start and end positions
                pos_start, pos_end = [int(s) for s in line.split("/")[-1].split("-")]
                continue
            first_sequence = first_sequence + line.strip()

    # create a map uniprot position -> alignment position
    # attention: uniprot positions start at 1, alignment indices here are 0-based
    uniprot_ind = pos_start
    alignment_ind = 0
    for c in first_sequence:
        if c == ".":
            continue
        if c == "-":
            raise ValueError("did not expect gap in first sequence")
        if c.islower():
            uniprot_ind += 1
            continue
        position_map[uniprot_ind] = alignment_ind
        uniprot_ind += 1
        alignment_ind += 1

    return first_sequence, first_key, position_map


def create_mutated_sequences(wt_sequence, wt_key, csv_file, position_map, experimental_column, out_file):

    # remove inserts
    wt_sequence = ''.join([c for c in wt_sequence if c.isupper() or c == "-"])

    # wt_sequence without inserts should correspond to all mapped alignment positions
    assert len(wt_sequence) == len(position_map)

    # some mutations are listed several times in the csv files, so we keep track of what we already added in order to avoid duplicates
    mutations_added = dict()

    with open(csv_file) as fid, open(out_file, "w") as ofid:
        for line_id, line in enumerate(fid):

            data = line.split(";")

            if line_id == 0:
                assert data[0] == "mutant"
                continue

            assert len(data) > 1

            # get experimental value and predictions for independent and pairwise model
            exp_val = data[1].strip()
            pw_val = data[-2].strip()
            ind_val = data[-1].strip()


            # check if exp_val is actually there
            if exp_val == "":
                continue

            if data[0] in mutations_added.keys():
                assert mutations_added[data[0]] == [ind_val, pw_val, exp_val]
                continue

            mutations_added[data[0]] = [ind_val, pw_val, exp_val]

            mutated_seq = wt_sequence

            for mut in data[0].split(":"):

                aa1 = mut[0]
                aa2 = mut[-1]
                uniprot_pos = int(mut[1:-1])

                if uniprot_pos not in position_map.keys():
                    continue

                alignment_ind = position_map[uniprot_pos]

                assert mutated_seq[alignment_ind] == aa1

                mutated_seq = mutated_seq[:alignment_ind] + aa2 + mutated_seq[alignment_ind + 1:]

            # assemble mutated sequence and print
            mutated_key = '/'.join([wt_key, mut, ind_val, pw_val, exp_val])
            print(mutated_key, file=ofid)
            print(mutated_seq, file=ofid)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--fasta_path", required=True)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--experimental_column", type=int, default=1)
    parser.add_argument("--out_file", type=str, required=True)

    args = parser.parse_args()

    wt_sequence, wt_key, position_map = read_fasta_info(args.fasta_path)
    create_mutated_sequences(wt_sequence, wt_key, args.csv_file, position_map, args.experimental_column, args.out_file)
