import subprocess

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import prody as pd
from caretta.superposition_functions import paired_svd_superpose, apply_rotran
from geometricus import MomentType
from portein import get_best_transformation, apply_transformation, find_size
from scipy.signal import resample
from tqdm.notebook import tqdm

from utils import get_rmsd

MOMENT_TYPES = list(MomentType)


@nb.njit
def nb_mean_axis_0(array: np.ndarray) -> np.ndarray:
    """
    Same as np.mean(array, axis=0) but njitted
    """
    mean_array = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        mean_array[i] = np.mean(array[:, i])
    return mean_array


def align_shapemers_svd(inds_list, coords_list):
    reference_inds = inds_list[0]
    reference_coords = coords_list[0]
    reference_coords_inds = np.array([reference_coords[i] for i in reference_inds])
    reference_coords_all = np.array(
        [reference_coords[i] for i in reference_inds] + [reference_coords[i] for i in reference_coords if
                                                         i not in reference_inds])
    reference_center = nb_mean_axis_0(reference_coords_inds)
    reference_coords_inds -= reference_center
    reference_coords_all -= reference_center
    superposed_coords = [reference_coords_all]
    for inds, coords in tqdm(zip(inds_list[1:], coords_list[1:])):
        coords_inds = np.array([coords[i] for i in inds])
        center = nb_mean_axis_0(coords_inds)
        coords_inds -= center
        coords_all = np.array([coords[i] for i in inds] + [coords[i] for i in coords if i not in inds])
        coords_all -= center
        rot, tran = paired_svd_superpose(reference_coords_inds, coords_inds)
        superposed_coords.append(apply_rotran(coords_all, rot, tran))
    return superposed_coords


def align_and_write_shapemers(centers_list, proteins_list, folder):
    for index, (center, protein) in enumerate(zip(centers_list, proteins_list)):
        pdb_alpha = protein.select("protein and calpha")
        resindex_to_index = {x.getResindex(): i for i, x in enumerate(pdb_alpha)}
        indices = set(range(center - 8, center + 8))
        radius = pdb_alpha.select(f"within 10 of resindex {center}")
        if radius is not None:
            for x in radius:
                indices.add(resindex_to_index[x.getResindex()])
        matrix = get_best_transformation(pdb_alpha.getCoords()[sorted(indices)])
        protein = pd.applyTransformation(pd.Transformation(matrix), protein)
        pd.moveAtoms(protein, to=np.zeros(3))
        pd.moveAtoms(protein, to=-protein.select("protein and calpha").getCoords()[center])
        for i, res in enumerate(protein.iterResidues()):
            if i in indices:
                res.setBetas([1] * len(res))
            else:
                res.setBetas([0] * len(res))
        pd.writePDB(str(folder / f"{index}.pdb"), protein)
        with open(folder / f"{index}.txt", "w") as f:
            f.write(
                f"{index} and resi " + "+".join(str(pdb_alpha[i].getResnum()) for i in range(center - 8, center + 8)))


def plot_shapemers_pymol_grid(folder, trim=False):
    subprocess.call(["pymol", "-cq", "pymol_shapemer_grid.py", "--", folder])
    if trim:
        name = folder.stem
        im = plt.imread(folder.parent / f'{name}_grid.png')
        im_z = ((im[:, :, :3] * 255) - 255).astype(np.int64)
        keep_rows = []
        n_per_g = im_z.shape[0] // 3
        for i in range(n_per_g):
            for x in range(3):
                keep_rows_x = 0
                if np.sum(im_z[i + (x * n_per_g)]) != 0 or np.sum(im_z[:, i + (x * n_per_g)]) != 0:
                    keep_rows_x += 1
            if keep_rows_x > 0:
                keep_rows += [i, i + n_per_g, i + 2 * n_per_g]
        keep_rows = sorted(keep_rows)
        im_n = im[keep_rows, :][:, keep_rows]
        im_n = im_n.copy(order='C')
        plt.imsave(folder.parent / f'{name}_grid.png', im_n)


def plot_shapemers_scatter(superposed_coords, plot_representative=False):
    coords = np.vstack(superposed_coords)
    matrix = get_best_transformation(coords)
    superposed_coords = [apply_transformation(s, matrix) for s in superposed_coords]
    plt.figure(figsize=find_size(coords, height=7, width=None))
    for c in superposed_coords:
        plt.plot(c[:, 0][:16], c[:, 1][:16], "-", alpha=0.1, c="grey")
        plt.scatter(c[:, 0][16:], c[:, 1][16:], alpha=0.1, c="grey")
    if plot_representative:
        c = superposed_coords[0]
        plt.plot(c[:, 0][:16], c[:, 1][:16], "o-", alpha=1, c="black")
        plt.scatter(c[:, 0][16:], c[:, 1][16:], alpha=0.7, c="black")
    plt.axis("off")


def resample_coords(coords, resample_rate=1):
    return np.vstack([resample(coords[:16], resample_rate * 16), coords[16:]])


def plot_shapemers_hexbin(superposed_coords, resample_rate=2, plot_representative=False):
    x1 = np.vstack(superposed_coords)
    matrix = get_best_transformation(x1)
    superposed_coords = [apply_transformation(s, matrix) for s in superposed_coords]

    superposed_coords = [np.vstack([resample_coords(c[:16], resample_rate=resample_rate),
                                    c[16:]]) for c in superposed_coords]
    coords = np.vstack(superposed_coords)
    plt.figure(figsize=find_size(coords, width=7, height=None))
    plt.hexbin(coords[:, 0], coords[:, 1], cmap="inferno", gridsize=100)
    if plot_representative:
        c = superposed_coords[0]
        plt.plot(c[:, 0][:16], c[:, 1][:16], "o-", alpha=1, c="black")
        plt.scatter(c[:, 0][16:], c[:, 1][16:], alpha=0.7, c="black")
    plt.axis("off")


def color_protein(cif_file, indices):
    protein = pd.parseMMCIF(cif_file)
    coords = protein.select("protein and calpha").getCoords()
    matrix = get_best_transformation(coords)
    protein = pd.applyTransformation(pd.Transformation(matrix), protein)
    for i, res in enumerate(protein.iterResidues()):
        if i in indices:
            res.setBetas([1] * len(res))
        else:
            res.setBetas([0] * len(res))
    return protein


def get_topic_scores(protein, index_to_score):
    protein_calpha = protein.select("protein and calpha")
    coords = protein_calpha.getCoords()
    assert len(coords) == len(protein_calpha), protein.name
    scores = np.zeros(len(coords))
    resindex_to_index = {x.getResindex(): i for i, x in enumerate(protein_calpha)}
    for i, residue in enumerate(protein_calpha):
        score = index_to_score.get(i, 0)
        indices = set([x for x in range(i - 8, i + 8) if 0 < x < len(protein_calpha)])
        radius = protein_calpha.select(f"within 10 of resindex {i}")
        if radius is not None:
            for x in radius:
                indices.add(resindex_to_index[x.getResindex()])
        indices = list(indices)
        distances = np.array([get_rmsd(coords[i], coords[j]) for j in indices])
        max_distance, min_distance = np.max(distances), np.min(distances)
        weights = 1 - ((distances - min_distance) / (max_distance - min_distance))
        weights = (0.5 * weights) + 0.5
        for index, j in enumerate(indices):
            scores[j] += score * weights[index]
    scores = 100 * scores / np.max(scores)
    matrix = get_best_transformation(coords)
    protein = protein.select("protein")
    pd.writePDB("tmp.pdb", protein)
    protein = pd.parsePDB("tmp.pdb")
    protein = pd.applyTransformation(pd.Transformation(matrix), protein)
    for i, res in enumerate(protein.iterResidues()):
        if i < len(scores):
            res.setBetas([scores[i]] * len(res))
    return protein
