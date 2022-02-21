# Original Author: Xinqiang Ding (xqding@umich.edu)

import torch
import h5py
import numpy as np
from VAE_model import VAE
import argparse
from tqdm import tqdm
from read_fasta import read_fasta
from torch.nn.functional import one_hot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--out_file', type=str, default="stdout")
    parser.add_argument('--num_res_type', type=int, default=21)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--num_hidden_units', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    model_path = args.model_path
    batch_size = args.batch_size
    samples = args.samples
    out_file = args.out_file
    fasta_path = args.fasta_path

    # read data
    seq_msa, seq_weight, num_res_type = read_fasta(fasta_path)

    # build a VAE model
    nseq, len_protein = seq_msa.shape
    parameters = torch.load(model_path)
    vae = VAE(21, args.latent_dim, len_protein * num_res_type, [args.num_hidden_units])
    vae.load_state_dict(parameters)

    # move the VAE onto a GPU
    if not args.no_cuda:
        vae.cuda()

    seq_msa = torch.from_numpy(seq_msa)
    if args.no_cuda:
        train_msa = one_hot(seq_msa, num_classes=num_res_type)
    else:
        train_msa = one_hot(seq_msa, num_classes=num_res_type).cuda()
    train_msa = train_msa.view(train_msa.shape[0], -1).float()

    logp_all = []

    with torch.no_grad():
        for k in tqdm(range(nseq // batch_size + 1)):
            idx_begin = k * batch_size
            idx_end = min((k + 1) * batch_size, nseq)
            if idx_end == idx_begin:
                break
            data = train_msa[idx_begin:idx_end, :]
            logp = vae.compute_elbo_with_multiple_samples(data, samples)

            logp_all.append(logp)

        logp = torch.cat(logp_all)
        if args.out_file != "stdout":
            with h5py.File(args.out_file, "w") as fid:
                fid.create_dataset("samples", data=np.int8(seq_msa.detach().cpu().numpy()))
                fid.create_dataset("logp", data=logp.detach().cpu().numpy())
                fid.close()
        else:
            for m in range(nseq):
                print("{}".format(logp[m]))
