# Original Author: Xinqiang Ding (xqding@umich.edu)

import numpy as np
import torch
from VAE_model import VAE
import argparse
from torch.nn.functional import one_hot
import h5py
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--nsamples', type=int, default=10000000)
    parser.add_argument('--num_res_type', type=int, default=21)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--elbo_samples', type=int, default=100)
    parser.add_argument('--sample_dist', type=str, default="M")
    parser.add_argument('--num_hidden_units', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # read data
    parameters = torch.load(args.model_path)

    # figure out dimensions
    len_protein = parameters['encoder_linears.0.weight'].shape[1] // args.num_res_type

    # load VAE
    vae = VAE(21, args.latent_dim, len_protein * args.num_res_type, [args.num_hidden_units])
    vae.load_state_dict(parameters)

    device = "cpu" if args.no_cuda else "cuda"

    # move the VAE onto a GPU
    if not args.no_cuda:
        vae.cuda()

    samples = []
    logp = []
    with torch.no_grad():
        for _ in tqdm(range(args.nsamples // args.batch_size)):
            if args.sample_dist == "M":
                samples_batch = vae.sample(args.batch_size)
            elif args.sample_dist == "U":
                samples_batch = torch.randint(0, args.num_res_type, (args.batch_size, len_protein))
            samples_batch_one_hot = one_hot(samples_batch, num_classes=args.num_res_type).float().view(args.batch_size, -1).to(device)
            logp_batch = vae.compute_elbo_with_multiple_samples(samples_batch_one_hot, args.elbo_samples)
            samples.append(samples_batch.cpu())
            logp.append(logp_batch.cpu())

    samples = torch.cat(samples)
    logp = torch.cat(logp)

    with h5py.File(args.out_file, "w") as fid:
        fid.create_dataset("samples", data=np.int8(samples.numpy()))
        fid.create_dataset("logp", data=logp.numpy())
        fid.close()
