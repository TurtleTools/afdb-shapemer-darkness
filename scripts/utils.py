import typing
from pathlib import Path

import numba as nb
import numpy as np


def get_sequences_from_fasta_yield(fasta_file: typing.Union[str, Path]) -> tuple:
    """
    Returns (accession, sequence) iterator
    Parameters
    ----------
    fasta_file
    Returns
    -------
    (accession, sequence)
    """
    with open(fasta_file) as f:
        current_sequence = ""
        current_key = None
        for line in f:
            if not len(line.strip()):
                continue
            if "==" in line:
                continue
            if line.startswith("#"):
                break
            if ">" in line:
                if current_key is None:
                    current_key = line.split(">")[1].strip()
                else:
                    yield current_key, current_sequence
                    current_sequence = ""
                    current_key = line.split(">")[1].strip()
            else:
                current_sequence += line.strip()
        yield current_key, current_sequence


def get_sequences_from_fasta(fasta_file: typing.Union[str, Path]) -> dict:
    """
    Returns dict of accession to sequence from fasta file
    Parameters
    ----------
    fasta_file
    Returns
    -------
    {accession:sequence}
    """
    return {
        key: sequence for (key, sequence) in get_sequences_from_fasta_yield(fasta_file)
    }


@nb.njit
def get_rmsd(coords_1: np.ndarray, coords_2: np.ndarray) -> float:
    """
    RMSD of paired coordinates = normalized square-root of sum of squares of euclidean distances
    """
    return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / coords_1.shape[0])


@nb.njit
def get_rmsd_neighbors(coords_1, coords_2, center_1, neighbors_1, mapping):
    distances = np.array([get_rmsd(coords_1[center_1], coords_1[n]) for n in neighbors_1])
    max_distance, min_distance = np.max(distances), np.min(distances)
    weights = 1 - ((distances - min_distance) / (max_distance - min_distance))
    weights = (0.5 * weights) + 0.5
    rmsd = 0
    for i in range(len(neighbors_1)):
        c = neighbors_1[i]
        if c != -1:
            rmsd += get_rmsd(coords_1[c], coords_2[mapping[c]]) * weights[i]
    return rmsd / sum(weights)


def alignment_to_numpy(alignment):
    aln_np = {}
    for n in alignment:
        aln_seq = []
        index = 0
        for a in alignment[n]:
            if a == "-":
                aln_seq.append(-1)
            else:
                aln_seq.append(index)
                index += 1
        aln_np[n] = np.array(aln_seq)
    return aln_np
