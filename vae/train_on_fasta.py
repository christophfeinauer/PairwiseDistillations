# Original Author: Xinqiang Ding (xqding@umich.edu)


import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from VAE_model import VAE
import argparse
from read_fasta import read_fasta
from torch.nn.functional import one_hot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--fasta_train_path', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--num_hidden_units', type=int, default=100)

    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # read data
    seq_msa, seq_weight, num_res_type = read_fasta(args.fasta_train_path)
    nseq, len_protein = seq_msa.shape

    # build a VAE model with random parameters
    vae = VAE(21, args.latent_dim, len_protein * num_res_type, [args.num_hidden_units]).cuda()

    # build the Adam optimizer
    optimizer = optim.Adam(vae.parameters(),
                           weight_decay=args.weight_decay)

    seq_msa = torch.from_numpy(seq_msa)
    train_msa = one_hot(seq_msa, num_classes=num_res_type).cuda()
    train_msa = train_msa.view(train_msa.shape[0], -1).float()

    train_weight = torch.from_numpy(seq_weight)
    train_weight = (train_weight / torch.sum(train_weight)).cuda()

    train_loss_list = []
    with tqdm(range(args.epochs)) as t:
        for epoch in t:
            loss = -vae.compute_weighted_elbo(train_msa, train_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description("train loss: {}".format(loss))
            train_loss_list.append(loss.item())

    vae.cpu()
    torch.save(vae.state_dict(), args.out_file)
