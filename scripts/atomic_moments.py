import typing as ty
from dataclasses import dataclass

import numpy as np
import prody as pd
from geometricus import MomentInvariants, MomentType, moment_utility, protein_utility, SplitType

MOMENT_TYPES = tuple(m.name for m in MomentType)


@dataclass
class AtomicMoments(MomentInvariants):
    # For a given atom type -> point cloud coordinates and their corresponding moment descriptors. provided a protein
    # structure
    calpha_coordinates: ty.Union[np.ndarray, None] = None

    @classmethod
    def from_prody_atomgroup(
            cls, name: protein_utility.ProteinKey, atom_group: pd.AtomGroup,
            split_size: int = 16, selection: str = "calpha",
            moment_types: ty.List[MomentType] = MOMENT_TYPES,
            split_type: SplitType = SplitType.RADIUS):

        sequence: str = str(atom_group.select("protein and calpha").getSequence())

        calpha_coordinates: np.ndarray = atom_group.select("protein and calpha").getCoords()

        residue_splits = protein_utility.group_indices(atom_group.select(selection).getResindices())

        shape = cls(
            name,
            len(residue_splits),
            atom_group.select(selection).getCoords(),
            residue_splits,
            atom_group.select(selection).getIndices(),
            sequence=sequence,
            split_type=split_type,
            split_size=split_size,
            calpha_coordinates=calpha_coordinates,
            moment_types=[m.name for m in moment_types]
        )
        shape.length = len(calpha_coordinates)
        shape._split(shape.split_type)
        return shape

    def _split(self, split_type: SplitType):
        if split_type == SplitType.KMER:
            split_indices, moments = self._kmerize()
        elif split_type == SplitType.RADIUS:
            split_indices, moments = self._split_radius()
        else:
            raise Exception("split_type must a RADIUS or KMER SplitType object")
        self.split_indices = split_indices
        self.moments = moments

    def _kmerize(self):
        split_indices = []
        for i in range(self.length):
            kmer = []
            for j in range(
                    max(0, i - self.split_size // 2),
                    min(len(self.residue_splits), i + self.split_size // 2),
            ):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _split_radius(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)
        for i in range(self.length):
            kd_tree.search(
                center=self.calpha_coordinates[i],
                radius=self.split_size,
            )
            split_indices.append(kd_tree.getIndices())
        return self._get_moments(split_indices)

    def _get_moments(self, split_indices):
        moments = np.zeros((len(split_indices), len(self.moment_types)))
        for i, indices in enumerate(split_indices):
            if indices is None:
                moments[i] = np.NaN
            else:
                moments[i] = moment_utility.get_moments_from_coordinates(
                    self.coordinates[indices], [MomentType[m] for m in self.moment_types]
                )
        return split_indices, moments

    @property
    def normalized_moments(self):
        return ((np.sign(self.moments) * np.log1p(np.abs(self.moments))) / self.split_size).astype("float32")


@dataclass
class MultipleAtomicMoments:
    multiple_atomic_moments: ty.List[AtomicMoments]

    @classmethod
    def from_prody_atomgroup(
            cls,
            name: protein_utility.ProteinKey,
            atom_group: pd.AtomGroup,
            radii: ty.List[int] = [12, 10, 8, 5],
            kmer_sizes: ty.List[int] = [15, 10, 20],
            selection: str = ["calpha", "carbon", "nitrogen", "oxygen"],
            moment_types: ty.List[str] = MOMENT_TYPES):
        multi: ty.List[AtomicMoments] = []
        for sel in selection:
            for radius in radii:
                multi.append(AtomicMoments.from_prody_atomgroup(name, atom_group, split_size=radius, selection=sel,
                                                                moment_types=moment_types, split_type=SplitType.RADIUS))
            for kmer_len in kmer_sizes:
                multi.append(AtomicMoments.from_prody_atomgroup(name, atom_group, split_size=kmer_len, selection=sel,
                                                                moment_types=moment_types, split_type=SplitType.KMER))
        return cls(multi)

    @property
    def moments(self):
        return np.hstack([x.moments for x in self.multiple_atomic_moments])

    @property
    def normalized_moments(self):
        return np.hstack([x.normalized_moments for x in self.multiple_atomic_moments])

    def get_neighbors(self):
        neighbors = [set() for i in range(len(self.multiple_atomic_moments[0].split_indices))]
        for moment in self.multiple_atomic_moments:
            for i, indices in enumerate(moment.split_indices):
                neighbors[i] |= set(indices)
        return neighbors
