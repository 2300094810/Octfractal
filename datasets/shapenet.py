
import os
from typing import Dict, List

import numpy as np
import torch
from thsolver import Dataset


def cfg_get(flags, key, default=None):
    if flags is None:
        return default
    if hasattr(flags, "get"):
        return flags.get(key, default)
    return getattr(flags, key, default)


class ReadVoxelCacheV8:
    def __init__(self, flags):
        self.voxel_cache_root = cfg_get(flags, "voxel_cache_root", None)
        if not self.voxel_cache_root:
            raise ValueError("DATA.voxel_cache_root is required for pure voxel v8.")

    @staticmethod
    def uid_from_filename(filename: str) -> str:
        parts = os.path.normpath(filename).split(os.sep)
        if len(parts) < 2:
            raise ValueError(f"Cannot infer <synset>/<model_id> from: {filename}")
        return os.path.join(parts[-2], parts[-1])

    def cache_path_from_filename(self, filename: str) -> str:
        uid = self.uid_from_filename(filename)
        return os.path.join(self.voxel_cache_root, uid + ".npz")

    def __call__(self, filename: str) -> Dict[str, np.ndarray]:
        cache_path = self.cache_path_from_filename(filename)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Voxel cache not found: {cache_path}\n"
                f"Run tools/preprocess_shapenet_voxels_v8.py first."
            )
        raw = np.load(cache_path)
        out: Dict[str, np.ndarray] = {}
        for key in raw.files:
            if key.startswith("occ_coords_d"):
                out[key] = raw[key].astype(np.int64)
        if len(out) == 0:
            raise ValueError(f"No occ_coords_d* arrays found in {cache_path}")
        return out


class TransformVoxelCacheV8:
    def __init__(self, flags):
        self.flags = flags
        self.full_depth = int(cfg_get(flags, "full_depth", 2))
        self.depth_stop = int(cfg_get(flags, "depth_stop", cfg_get(flags, "depth", 6)))

    def __call__(self, sample: Dict[str, np.ndarray], idx: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for depth in range(self.full_depth, self.depth_stop + 1):
            key = f"occ_coords_d{depth}"
            if key not in sample:
                raise KeyError(f"Missing {key} in voxel cache sample.")
            coords = torch.from_numpy(sample[key]).long()
            if coords.numel() == 0:
                coords = torch.zeros(0, 3, dtype=torch.long)
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f"{key} must have shape (M, 3), got {tuple(coords.shape)}")
            out[key] = coords
        return out


def collate_func_v8(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    all_keys = sorted({key for sample in batch for key in sample.keys() if key.startswith("occ_coords_d")})

    for key in all_keys:
        coords_all = []
        for batch_id, sample in enumerate(batch):
            coords = sample[key]
            if coords.numel() == 0:
                continue
            bcol = torch.full((coords.shape[0], 1), batch_id, dtype=torch.long)
            coords_all.append(torch.cat([coords.long(), bcol], dim=1))
        if coords_all:
            output[key] = torch.cat(coords_all, dim=0)
        else:
            output[key] = torch.zeros(0, 4, dtype=torch.long)
    return output


def get_shapenet_dataset(flags):
    transform = TransformVoxelCacheV8(flags)
    read_file = ReadVoxelCacheV8(flags)
    dataset = Dataset(cfg_get(flags, "location"), cfg_get(flags, "filelist"), transform, read_file)
    return dataset, collate_func_v8
