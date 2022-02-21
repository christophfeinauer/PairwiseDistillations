import torch
import numpy as np
from argparse import ArgumentParser
from torch.nn.functional import one_hot
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
import h5py


def extract_couplings_fitting(samples_dist,
                              logp_dist,
                              samples_U,
                              logp_U,
                              learning_rate,
                              q,
                              batch_size,
                              alpha,
                              smoothing_discount,
                              no_cuda,
                              model_type):

    device = "cpu" if no_cuda else "cuda"

    if samples_U is None:
        assert alpha == 1.0

    ndist = len(logp_dist)
    nU = len(logp_U) if logp_U is not None else 0

    _, N = samples_dist.shape

    J = torch.zeros(N * q, N * q).to(device)
    h = torch.zeros(N * q).to(device)
    c = torch.tensor(0.0)
    J.requires_grad = True if model_type == "PW" else False
    h.requires_grad = True
    c.requires_grad = True
    mask = torch.ones_like(J)
    for i in range(N):
        for j in range(0, i + 1):
            mask[i * q:(i + 1) * q, j * q:(j + 1) * q] = 0
    with torch.no_grad():
        J.data = J.data * mask.data
    mask.requires_grad = False
    mse_loss = MSELoss()

    energy_mean = 0

    parameters = (J, h, c) if model_type == "PW" else (h, c)

    optimizer = Adam(parameters, weight_decay=args.weight_decay)
    tbar = tqdm(float("inf"))
    min_loss = None
    steps_since_min_loss = 0
    loss_smooth = 0

    step = 0
    while True:

        nsamples_U_batch = np.sum(np.random.rand(batch_size) > alpha)
        nsamples_dist_batch = batch_size - nsamples_U_batch

        if nsamples_U_batch:
            # the additional calls to .numpy() guard against cases where inds_U_batch = 1
            inds_U_batch = torch.randint(0, nU, (nsamples_U_batch, )).numpy()
            samples_U_batch = torch.from_numpy(samples_U[inds_U_batch, :]).to(device).long()
            energy_U_batch = -torch.from_numpy(logp_U[inds_U_batch]).to(device)
        else:
            samples_U_batch = torch.zeros((0, N)).to(device).long()
            energy_U_batch = torch.zeros((0,)).to(device)

        if nsamples_dist_batch:
            inds_dist_batch = torch.randint(0, ndist, (nsamples_dist_batch, )).numpy()
            samples_dist_batch = torch.from_numpy(samples_dist[inds_dist_batch, :]).to(device).long()
            energy_dist_batch = -torch.from_numpy(logp_dist[inds_dist_batch]).to(device)
        else:
            samples_dist_batch = torch.zeros((0, N)).to(device).long()
            energy_dist_batch = torch.zeros((0,)).to(device)

        data = torch.cat((samples_U_batch, samples_dist_batch))
        data = one_hot(data, 21).view(batch_size, -1).float()

        energy = torch.cat((energy_U_batch, energy_dist_batch))
        energy_mean = step / (step + 1) * energy_mean + 1 / (step + 1) * torch.mean(energy)

        J_masked = J * mask
        J_masked = J_masked + torch.transpose(J_masked, 0, 1)
        energy_model = -data @ h - 0.5 * torch.sum(data * (data @ J_masked), -1) - c

        loss = mse_loss(energy_model, energy)
        optimizer.zero_grad()
        loss.backward()

        loss_smooth = loss.item() if step == 0 else smoothing_discount * loss.item() + (1 - smoothing_discount) * loss_smooth
        if min_loss is None or loss_smooth < min_loss:
            min_loss = loss_smooth
            steps_since_min_loss = 0
        else:
            steps_since_min_loss += 1
        if args.max_steps_since_min_loss > 0 and steps_since_min_loss >= args.max_steps_since_min_loss:
            break

        optimizer.step()

        # log 
        if step > 0:
            cor = np.corrcoef(energy_model.detach().cpu().numpy(), energy.detach().cpu().numpy())[0, 1]
            descr = "step = {3} loss = {0:.3f}, cor = {1:.3f}, plateau={2}".format(loss_smooth, cor, steps_since_min_loss, step)

            tbar.set_description(descr)

        step += 1



    # change h to Nxq format
    h = h.view(N, q)

    h = h.detach().cpu().numpy()
    J = J.detach().cpu().numpy()
    c = c.detach().cpu().numpy()

    # change J format to Lxqxq
    _J = np.zeros(((N * (N - 1)) // 2, q, q))
    site = 0
    for i in range(N):
        for j in range(i + 1, N):
            _J[site, :, :] = J[i * q:(i + 1) * q, j * q:(j + 1) * q]
            site += 1
    J = _J

    return J, h, c


def extract_couplings_zero_sum(samples,
                               logp,
                               q,
                               batch_size,
                               no_cuda,
                               model_type):

    N = samples.shape[1]

    device = "cpu" if no_cuda else "cuda"

    nsamples = len(logp)

    # note that below we actually extract the negative parameters - that's why we return with a negative sign
    J = torch.zeros(N * q, N * q).cuda().double()
    counts = torch.zeros(N * q, N * q).cuda().double()
    energy_mean = -np.mean(logp)

    def extract(J, energy_mean):

        with torch.no_grad():

            h = torch.diag(J) / torch.diag(counts) - energy_mean

            J = J / counts - h.unsqueeze(dim=1).repeat(1, N * q) \
                - h.repeat(N * q, 1) \
                - energy_mean

            return J, h, energy_mean

    with torch.no_grad():
        for k in tqdm(range(nsamples // batch_size)):

            inds_batch = range(k*batch_size, (k+1)*batch_size)
            samples_batch = one_hot(torch.from_numpy(samples[inds_batch, :]).long().to(device), num_classes=q).view(batch_size, -1)
            energy_batch = -torch.from_numpy(logp[inds_batch]).to(device)

            samples_batch = samples_batch.double()
            counts_this = samples_batch.T @ samples_batch
            counts = counts + counts_this
            samples_batch_energy = energy_batch.unsqueeze(axis=1).expand(-1, N * q) * samples_batch
            J_this = samples_batch_energy.T @ samples_batch
            J = J + J_this

        J, h, c = extract(J, energy_mean)

        # change h to Nxq format
        h = h.view(N, q)

        h = h.cpu().numpy()
        J = J.cpu().numpy()

        # change J format to Lxqxq
        _J = np.zeros(((N * (N - 1)) // 2, q, q))
        site = 0
        for i in range(N):
            for j in range(i + 1, N):
                _J[site, :, :] = J[i * q:(i + 1) * q, j * q:(j + 1) * q]
                site += 1
        J = _J

        if model_type == "IND":
            J = 0 * J

        return -J, -h, -c


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--q', type=int, default=21)
    parser.add_argument('--model_type', type=str, choices=["IND", "PW"], default="PW")
    parser.add_argument('--batch_size_sgd', type=int, default=1000)
    parser.add_argument('--batch_size_zs', type=int, default=1000)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--samples_file_dist')
    parser.add_argument('--samples_file_U')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_steps_since_min_loss', type=int, default=1000)
    parser.add_argument('--smoothing_discount', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # checks args
    if args.samples_file_dist is None and args.samples_file_U is None:
        raise ValueError("set either samples_file_dist or samples_file_U")

    if args.samples_file_U is None and args.alpha < 1.0:
        raise ValueError("need samples_file_U for alpha < 1.0")

    samples_dist = None
    logp_dist = None
    if args.samples_file_dist:
        with h5py.File(args.samples_file_dist, "r") as fid:
            samples_dist = fid['samples'][()]
            logp_dist = fid['logp'][()].astype(np.float32)

    samples_U = None
    logp_U = None
    if args.samples_file_U:
        with h5py.File(args.samples_file_U, "r") as fid:
            samples_U = fid['samples'][()]
            logp_U = fid['logp'][()].astype(np.float32)

    # if we have only U samples we can do zero sum directly
    if samples_dist is None:
        J, h, c = extract_couplings_zero_sum(samples_U, logp_U, args.q, args.batch_size_zs, args.no_cuda, args.model_type)
    else:
        J, h, c = extract_couplings_fitting(samples_dist, logp_dist, samples_U, logp_U, args.learning_rate, args.q, args.batch_size_sgd, args.alpha, args.smoothing_discount, args.no_cuda, args.model_type)

    # write out; save in column-major order
    with h5py.File(args.out_file, "w") as fid:
        fid.create_dataset("couplings", data=J.T)
        fid.create_dataset("fields", data=h.T)
        fid.create_dataset("constant", data=c)
