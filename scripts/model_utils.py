from collections import OrderedDict

import numpy as np
import torch


class MomentLearn(torch.nn.Module):
    def __init__(self, number_of_moments, hidden_layer_dim, cont_dim):
        super(MomentLearn, self).__init__()
        self.linear_segment = torch.nn.Sequential(
            torch.nn.Linear(number_of_moments, hidden_layer_dim),
            torch.nn.BatchNorm1d(hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, cont_dim),
            torch.nn.BatchNorm1d(cont_dim),
        )

    def forward(self, x, y, z):
        return self.linear_segment(x), self.linear_segment(y), z

    def forward_single_lab(self, x):
        return self.linear_segment(x)

    def forward_single_segment(self, x):
        return self.linear_segment(x)


def loss_func(out, distant, y):
    dist_sq = torch.sum(torch.pow(out - distant, 2), 1)
    dist = torch.sqrt(dist_sq + 1e-10)
    mdist = 1 - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / out.size()[0]
    return loss


def get_embedding(prot_rep, bins):
    return np.histogram(prot_rep, bins=bins)[0]


def moment_tensors_to_bits(list_of_moment_tensors, nbits=16):
    bits = []
    for segment in list_of_moment_tensors:
        x = (segment < 0).astype("uint8")
        bits.append(np.array("".join([str(bit) for bit in x])))
    return np.array(bits, dtype=f"|S{nbits}").flatten()


def moments_to_tensors(segments, model):
    if torch.cuda.is_available():
        return model.forward_single_segment(torch.tensor(segments).cuda()).cpu().detach().numpy()
    else:
        return model.forward_single_segment(torch.tensor(segments)).cpu().detach().numpy()


def moments_to_bit_list(list_of_moments, model, nbits=16):
    if torch.cuda.is_available():
        moment_tensors = (
            model.forward_single_segment(torch.tensor(list_of_moments).cuda())
            .cpu()
            .detach()
            .numpy()
        )
    else:
        moment_tensors = (
            model.forward_single_segment(torch.tensor(list_of_moments))
            .cpu()
            .detach()
            .numpy()
        )
    return list(moment_tensors_to_bits(moment_tensors, nbits=nbits))


def get_all_keys(list_of_moment_hashes, model, nbits=16):
    all_keys = set()
    for prot in list_of_moment_hashes:
        all_keys |= set(moments_to_bit_list(prot, model, nbits=nbits))
    return list(all_keys)


def count_with_keys(prot_hashes, keys):
    d = OrderedDict.fromkeys(keys, 0)
    for prot_hash in prot_hashes:
        d[prot_hash] += 1
    return np.array([d[x] for x in keys])


def get_hash_embeddings(protein_moments, model, nbits=16, with_counts=False):
    ind_moments_compressed = [
        moments_to_bit_list(x, model, nbits=nbits) for x in protein_moments
    ]
    all_keys = get_all_keys(protein_moments, model, nbits=nbits)
    print(f"Total shapemers: {len(all_keys)}")
    protein_embeddings = [count_with_keys(x, all_keys) for x in ind_moments_compressed]
    if with_counts:
        return [x / x.sum() for x in protein_embeddings], protein_embeddings
    return protein_embeddings
