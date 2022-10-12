import gzip
import io
import tarfile
from dataclasses import dataclass
from time import time

import numpy as np
import prody as pd
import torch
from geometricus import MomentInvariants, SplitType, MomentType
from scipy import ndimage

import model_utils

MOMENT_TYPES = list(MomentType)


@dataclass
class SplitInfo:
    """
    Class to store information about structural fragmentation type.
    """
    split_type: SplitType
    split_size: int


SPLIT_TYPES = (SplitInfo(SplitType.RADIUS, 5),
               SplitInfo(SplitType.RADIUS, 10),
               SplitInfo(SplitType.KMER_CUT, 8),
               SplitInfo(SplitType.KMER_CUT, 16))
RANGE_SPLIT_TYPE = {"5_16": [0, 1000000], "2_10": [8, -8], "5_8": [4, -4], "2_5": [8, -8]}


def split_alphafold_protein(prody_protein, plddt_threshold=70, sigma=5):
    """
    Splits an AlphaFold protein into fragments based on a Gaussian-smoothed version of the PLDDT score.
    Parameters
    ----------
    prody_protein
        ProDy protein object.
    plddt_threshold
        Fragments will be split according to residues with a (smoothed) PLDDT score below this threshold.
    sigma
        Sigma for the smoothing of the PLDDT score.

    Returns
    -------
    (start, end) indices for each split
    """
    beta_list = prody_protein.getBetas()
    beta_list = ndimage.gaussian_filter1d(beta_list, sigma=sigma)
    indices = np.ones(beta_list.shape[0], dtype=int)
    indices[np.where(beta_list < plddt_threshold)] = 0
    slices = ndimage.find_objects(ndimage.label(indices)[0])
    slices = [(s[0].start, s[0].stop) for s in slices]
    return slices


def split_pdb_protein(prody_protein):
    """
    Splits an PDB protein into fragments based on chain.
    Parameters
    ----------
    prody_protein
        ProDy protein object.

    Returns
    -------
    (start, end) indices for each split - the last split is the whole protein
    """
    slices = []
    for chain in set(a.getChid() for a in prody_protein):
        if not len(chain.strip()):
            chain = prody_protein
        else:
            chain = prody_protein.select(f"chain {chain}")
        slices.append((chain[0].getResindex(), chain[-1].getResindex() + 1))
    return sorted(slices)


def get_shapemers_all(calpha_protein, model, num_bits):
    """
    Retrieves the moments of the protein.
    Parameters
    ----------
    calpha_protein
        prody object

    Returns
    -------
    """
    coords = calpha_protein.getCoords()
    hash_vectors = []
    moments = []
    for split_info in SPLIT_TYPES:
        if split_info.split_type == SplitType.KMER_CUT:
            split_type = SplitType.KMER
        else:
            split_type = split_info.split_type
        moments.append(normalized_moments(MomentInvariants.from_coordinates(
            "name",
            coords,
            None,
            split_type=split_type,
            split_size=split_info.split_size,
            moment_types=MOMENT_TYPES
        )))
    indices = list(range(0, len(coords)))
    moment_vector = np.hstack(moments)
    shapemers = model_utils.moments_to_bit_list(moment_vector,
                                                model,
                                                nbits=num_bits)
    hash_vectors += list(model_utils.moments_to_tensors(moment_vector, model))
    if len(shapemers):
        assert len(shapemers) == len(indices) == len(hash_vectors)
        return indices, shapemers, list(np.mean(hash_vectors, axis=0))
    return [], [], []


def get_shapemers(calpha_protein,
                  model, num_bits,
                  length_threshold=20,
                  plddt_threshold=70, sigma=5,
                  is_af=True, do_try=True):
    """
    Retrieves the moments of the protein.
    Parameters
    ----------
    calpha_protein
        prody object
    length_threshold
        Proteins with fewer (filtered) residues than this threshold will be ignored.
    plddt_threshold
        Residues with a (smoothed) PLDDT score below this threshold will be ignored.
    sigma
        Sigma for the smoothing of the PLDDT score.

    Returns
    -------
    """
    if is_af:
        residue_slices = split_alphafold_protein(calpha_protein, plddt_threshold, sigma)
    else:
        residue_slices = split_pdb_protein(calpha_protein)
    coords = calpha_protein.getCoords()
    shapemers = []
    hash_vectors = []
    indices = []
    if do_try:
        try:
            for start_index, end_index in residue_slices:
                if end_index - start_index > length_threshold:
                    moments = []
                    for split_info in SPLIT_TYPES:
                        k = f"{split_info.split_type}_{split_info.split_size}"
                        moments.append(normalized_moments(MomentInvariants.from_coordinates(
                            "name",
                            coords[start_index: end_index],
                            None,
                            split_type=split_info.split_type,
                            split_size=split_info.split_size,
                            moment_types=MOMENT_TYPES
                        ))[RANGE_SPLIT_TYPE[k][0]:RANGE_SPLIT_TYPE[k][1]])
                    indices += list(range(start_index, end_index))[8:-8]
                    moment_vector = np.hstack(moments)
                    shapemers += model_utils.moments_to_bit_list(moment_vector,
                                                                 model,
                                                                 nbits=num_bits)
                    hash_vectors += list(model_utils.moments_to_tensors(moment_vector, model))
            if len(shapemers):
                assert len(shapemers) == len(indices) == len(hash_vectors)
                return indices, shapemers, list(np.mean(hash_vectors, axis=0))
        except Exception as e:
            print(f"Error {e}")
    else:
        for start_index, end_index in residue_slices:
            if end_index - start_index > length_threshold:
                moments = []
                for split_info in SPLIT_TYPES:
                    k = f"{split_info.split_type}_{split_info.split_size}"
                    moments.append(normalized_moments(MomentInvariants.from_coordinates(
                        "name",
                        coords[start_index: end_index],
                        None,
                        split_type=split_info.split_type,
                        split_size=split_info.split_size,
                        moment_types=MOMENT_TYPES
                    ))[RANGE_SPLIT_TYPE[k][0]:RANGE_SPLIT_TYPE[k][1]])
                indices += list(range(start_index, end_index))[8:-8]
                moment_vector = np.hstack(moments)
                shapemers += model_utils.moments_to_bit_list(moment_vector,
                                                             model,
                                                             nbits=num_bits)
                hash_vectors += list(model_utils.moments_to_tensors(moment_vector, model))
        if len(shapemers):
            assert len(shapemers) == len(indices) == len(hash_vectors)
            return indices, shapemers, list(np.mean(hash_vectors, axis=0))
    return [], [], []


def get_shapemers_from_coords(coords, model, num_bits):
    shapemers = []
    hash_vectors = []
    indices = []
    try:
        moments = []
        for split_info in SPLIT_TYPES:
            k = f"{split_info.split_type}_{split_info.split_size}"
            moments.append(normalized_moments(MomentInvariants.from_coordinates(
                "name",
                coords,
                None,
                split_type=split_info.split_type,
                split_size=split_info.split_size,
                moment_types=MOMENT_TYPES
            ))[RANGE_SPLIT_TYPE[k][0]:RANGE_SPLIT_TYPE[k][1]])
        indices += list(range(0, len(coords)))[8:-8]
        moment_vector = np.hstack(moments)
        shapemers += model_utils.moments_to_bit_list(moment_vector,
                                                     model,
                                                     nbits=num_bits)
        hash_vectors += list(model_utils.moments_to_tensors(moment_vector, model))
        if len(shapemers):
            assert len(shapemers) == len(indices) == len(hash_vectors)
            return indices, shapemers, list(np.mean(hash_vectors, axis=0))
    except Exception as e:
        print(e)
    return [], [], []


def normalized_moments(m):
    return ((np.sign(m.moments) * np.log1p(np.abs(m.moments))) / m.split_size).astype("float32")


def make_corpus_proteome(taxid, db_folder, output_folder, num_bits=16):
    if torch.cuda.is_available():
        model = torch.load(f"data/models/model{num_bits}.pth")
    else:
        model = torch.load(f"data/models/model{num_bits}.pth", map_location=torch.device("cpu"))
    model.eval()
    start = time()
    index = 0
    f_s = open(output_folder / f"{taxid}_{num_bits}_shapemers.txt", "w")
    f_t = open(output_folder / f"{taxid}_{num_bits}_tensors.txt", "w")
    f_i = open(output_folder / f"{taxid}_{num_bits}_indices.txt", "w")
    for f in db_folder.glob(f"proteome-tax_id-{taxid}-*_v3.tar"):
        with tarfile.open(f) as tar:
            for fh in tar.getmembers():
                if '.cif' in fh.name:
                    if index % 1000 == 0:
                        print(f"{index} proteins processed in {time() - start} seconds")
                    uniprot_ac = '-'.join(fh.name.split('-')[1:3])
                    mmcif = tar.extractfile(fh)
                    with gzip.open(mmcif, 'r') as mmcif:
                        with io.TextIOWrapper(mmcif, encoding='utf-8') as decoder:
                            protein = pd.parseMMCIFStream(decoder)
                            protein = protein.select("protein and calpha")
                            indices, shapemers, tensors = get_shapemers(protein, model, num_bits)
                            if len(shapemers):
                                shapemers = " ".join(str(int(b.decode(), base=2)) for b in shapemers)
                                f_i.write(f"{uniprot_ac}\t{' '.join(str(s) for s in indices)}\n")
                                f_s.write(f"{uniprot_ac}\t{shapemers}\n")
                                f_t.write(f"{uniprot_ac}\t{' '.join(str(s) for s in tensors)}\n")
                    index += 1
    f_s.close()
    f_t.close()
    f_i.close()


def make_corpus_afdb_swissprot_v2(db_folder, output_folder, num_bits=10):
    if torch.cuda.is_available():
        model = torch.load(f"data/models/model{num_bits}.pth")
    else:
        model = torch.load(f"data/models/model{num_bits}.pth", map_location=torch.device("cpu"))
    model.eval()
    start = time()
    index = 0
    f_s = open(output_folder / f"swissprot_v2_{num_bits}_shapemers.txt", "w")
    f_t = open(output_folder / f"swissprot_v2_{num_bits}_tensors.txt", "w")
    f_i = open(output_folder / f"swissprot_v2_{num_bits}_indices.txt", "w")
    for filename in db_folder.glob(f"*.pdb.gz"):
        if index % 1000 == 0:
            print(f"{index} proteins processed in {time() - start} seconds")
        uniprot_ac = '-'.join(filename.stem.split('-')[1:3])
        protein = pd.parsePDB(str(filename))
        protein = protein.select("protein and calpha")
        indices, shapemers, tensors = get_shapemers(protein, model, num_bits)
        if len(shapemers):
            shapemers = " ".join(str(int(b.decode(), base=2)) for b in shapemers)
            f_i.write(f"{uniprot_ac}\t{' '.join(str(s) for s in indices)}\n")
            f_s.write(f"{uniprot_ac}\t{shapemers}\n")
            f_t.write(f"{uniprot_ac}\t{' '.join(str(s) for s in tensors)}\n")
        index += 1
    f_s.close()
    f_t.close()
    f_i.close()


def make_corpus_pdb(db_folder, output_folder, num_bits=10):
    if torch.cuda.is_available():
        model = torch.load(f"data/models/model{num_bits}.pth")
    else:
        model = torch.load(f"data/models/model{num_bits}.pth", map_location=torch.device("cpu"))
    model.eval()
    start = time()
    index = 0
    f_s = open(output_folder / f"pdb_{num_bits}_shapemers.txt", "w")
    f_t = open(output_folder / f"pdb_{num_bits}_tensors.txt", "w")
    f_i = open(output_folder / f"pdb_{num_bits}_indices.txt", "w")
    for filename in db_folder.glob(f"*.cif"):
        try:
            if index % 1000 == 0:
                print(f"{index} proteins processed in {time() - start} seconds")
            uniprot_ac = filename.stem
            protein = pd.parseMMCIF(str(filename))
            protein = protein.select("protein and calpha")
            if protein is None:
                continue
            indices, shapemers, tensors = get_shapemers(protein, model, num_bits, is_af=False)
            if len(shapemers):
                shapemers = " ".join(str(int(b.decode(), base=2)) for b in shapemers)
                f_i.write(f"{uniprot_ac}\t{' '.join(str(s) for s in indices)}\n")
                f_s.write(f"{uniprot_ac}\t{shapemers}\n")
                f_t.write(f"{uniprot_ac}\t{' '.join(str(s) for s in tensors)}\n")
            index += 1
        except Exception as e:
            print(e)
    f_s.close()
    f_t.close()
    f_i.close()


def make_corpus_pdb_chain(db_folder, output_folder, num_bits=10):
    length_threshold = 20
    if torch.cuda.is_available():
        model = torch.load(f"data/models/model{num_bits}.pth")
    else:
        model = torch.load(f"data/models/model{num_bits}.pth", map_location=torch.device("cpu"))
    model.eval()
    start = time()
    index = 0
    f_s = open(output_folder / f"pdb_chain_{num_bits}_shapemers.txt", "w")
    f_t = open(output_folder / f"pdb_chain_{num_bits}_tensors.txt", "w")
    f_i = open(output_folder / f"pdb_chain_{num_bits}_indices.txt", "w")
    for filename in db_folder.glob(f"*.cif"):
        try:
            if index % 1000 == 0:
                print(f"{index} proteins processed in {time() - start} seconds")
            uniprot_ac = filename.stem
            protein = pd.parseMMCIF(str(filename))
            protein = protein.select("protein and calpha")
            if protein is None:
                continue
            for chid in set(a.getChid() for a in protein):
                chid = chid.strip()
                if not len(chid):
                    chain = protein
                else:
                    chain = protein.select(f"chain {chid}")
                start_index, end_index = 0, len(chain)
                if end_index - start_index > length_threshold:
                    coords = chain.getCoords()
                    moments = []
                    for split_info in SPLIT_TYPES:
                        k = f"{split_info.split_type}_{split_info.split_size}"
                        moments.append(normalized_moments(MomentInvariants.from_coordinates(
                            "name",
                            coords,
                            None,
                            split_type=split_info.split_type,
                            split_size=split_info.split_size,
                            moment_types=MOMENT_TYPES
                        ))[RANGE_SPLIT_TYPE[k][0]:RANGE_SPLIT_TYPE[k][1]])
                    indices = list(range(start_index, end_index))[8:-8]
                    moment_vector = np.hstack(moments)
                    shapemers = model_utils.moments_to_bit_list(moment_vector,
                                                                model,
                                                                nbits=num_bits)
                    hash_vectors = list(model_utils.moments_to_tensors(moment_vector, model))
                    if len(shapemers):
                        assert len(shapemers) == len(indices) == len(hash_vectors)
                        tensors = list(np.mean(hash_vectors, axis=0))
                        shapemers = " ".join(str(int(b.decode(), base=2)) for b in shapemers)
                        f_i.write(f"{uniprot_ac}_{chid}\t{' '.join(str(s) for s in indices)}\n")
                        f_s.write(f"{uniprot_ac}_{chid}\t{shapemers}\n")
                        f_t.write(f"{uniprot_ac}_{chid}\t{' '.join(str(s) for s in tensors)}\n")
                index += 1
        except Exception as e:
            print(f"{filename}\t{e}")
    f_s.close()
    f_t.close()
    f_i.close()
