import os
import sys
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_ROOT, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from tqdm import tqdm
from thsolver import Dataset, Solver

from fractal_models.fractal_generator import FractalGenerator
from datasets.shapenet import ReadVoxelCacheV8
from vis_occ_utils import coords_to_points, save_pointcloud_ply
from occupancy_to_sdf import occupancy_to_sdf_pipeline

def cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def cfg_to_dict(obj):
    if obj is None:
        return {}
    if hasattr(obj, "items"):
        return dict(obj.items())
    keys = [k for k in dir(obj) if not k.startswith("_")]
    return {k: getattr(obj, k) for k in keys if not callable(getattr(obj, k))}


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    for base in [os.getcwd(), _PROJECT_ROOT]:
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path))


def count_filelist(filelist: Optional[str]) -> int:
    filelist = resolve_path(filelist)
    if not filelist or not os.path.exists(filelist):
        raise FileNotFoundError(f"Cannot infer num_latents because filelist was not found: {filelist}")
    count = 0
    with open(filelist, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    if count <= 0:
        raise ValueError(f"Filelist is empty: {filelist}")
    return count


class TransformVoxelCacheV13:
    def __init__(self, flags):
        self.flags = flags
        self.full_depth = int(cfg_get(flags, "full_depth", 2))
        self.depth_stop = int(cfg_get(flags, "depth_stop", cfg_get(flags, "depth", 6)))

    def __call__(self, sample: Dict[str, object], idx: int) -> Dict[str, torch.Tensor]:
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
        out["sample_idx"] = torch.tensor(int(idx), dtype=torch.long)
        return out


def collate_func_v13(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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

    output["sample_idx"] = torch.stack([sample["sample_idx"].long() for sample in batch], dim=0)
    return output


def get_shapenet_dataset_v13(flags):
    transform = TransformVoxelCacheV13(flags)
    read_file = ReadVoxelCacheV8(flags)
    dataset = Dataset(cfg_get(flags, "location"), cfg_get(flags, "filelist"), transform, read_file)
    return dataset, collate_func_v13


class FractalSolver(Solver):
    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.depth = int(FLAGS.MODEL.depth)
        self.depth_stop = int(FLAGS.MODEL.depth_stop)
        self.full_depth = int(FLAGS.MODEL.full_depth)

    def get_model(self, flags):
        model_kwargs = cfg_to_dict(flags.FractalGen)
        use_autodecoder = bool(model_kwargs.get("use_autodecoder", True))
        if use_autodecoder and int(model_kwargs.get("num_latents", 0)) <= 0:
            data_flags = self.FLAGS.DATA
            train_flags = data_flags.train if hasattr(data_flags, "train") else data_flags["train"]
            model_kwargs["num_latents"] = count_filelist(cfg_get(train_flags, "filelist", None))
        model = FractalGenerator(**model_kwargs)
        model.cuda(device=self.device)
        self.model_module = model
        return model

    def get_dataset(self, flags):
        data_flags = flags.DATA if hasattr(flags, "DATA") else flags
        return get_shapenet_dataset_v13(data_flags)

    def batch_to_cuda(self, batch):
        for key, value in list(batch.items()):
            if (key.startswith("occ_coords_d") or key == "sample_idx") and torch.is_tensor(value):
                batch[key] = value.cuda(device=self.device, non_blocking=True)

    def collect_voxel_occ(self, batch) -> Dict[int, torch.Tensor]:
        voxel_occ: Dict[int, torch.Tensor] = {}
        for key, value in batch.items():
            if not key.startswith("occ_coords_d"):
                continue
            depth = int(key.replace("occ_coords_d", ""))
            voxel_occ[depth] = value.long()
        if len(voxel_occ) == 0:
            raise ValueError(
                "No occ_coords_d{depth} tensors found in batch. "
                "Run the voxel-cache preprocessing script and use octgpt_new.datasets."
            )
        required = list(range(self.full_depth, self.depth_stop + 1))
        missing = [d for d in required if d not in voxel_occ]
        if missing:
            raise KeyError(
                f"Missing required occupancy cache depths: {missing}. "
                f"Expected occ_coords_d{self.full_depth} ... occ_coords_d{self.depth_stop}."
            )
        return voxel_occ

    def _current_epoch(self) -> Optional[int]:
        # thsolver versions expose epoch under different names. If none exists,
        # the model falls back to the static scheduled_sampling_prob in YAML.
        for name in ["epoch", "current_epoch", "cur_epoch", "_epoch", "global_epoch"]:
            value = getattr(self, name, None)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return None

    def model_forward(self, batch, is_train: bool = True):
        self.batch_to_cuda(batch)
        voxel_occ = self.collect_voxel_occ(batch)
        sample_idx = batch.get("sample_idx", None)
        epoch = self._current_epoch() if is_train else None
        return self.model(
            voxel_occ=voxel_occ,
            epoch=epoch,
            sample_idx=sample_idx,
            use_latent_table=is_train,
        )

    def train_step(self, batch):
        output = self.model_forward(batch, is_train=True)
        return {"train/" + k: v for k, v in output.items()}

    def test_step(self, batch):
        with torch.no_grad():
            output = self.model_forward(batch, is_train=False)
        return {"test/" + k: v for k, v in output.items()}

    def test_epoch(self, epoch):
        # Validation / visualization is less frequent because free-running AR generation is slower.
        if epoch % 50 != 0:
            return
        super().test_epoch(epoch)
        if self.is_master:
            self.generate_step(epoch)

    def _save_generated_pointcloud(self, final_coords, sample_index: int):
        save_dir = os.path.join(self.logdir, "visuals")
        os.makedirs(save_dir, exist_ok=True)
        points, _ = coords_to_points(final_coords, self.depth_stop)
        ply_path = os.path.join(save_dir, f"sample_{sample_index:04d}_points.ply")
        save_pointcloud_ply(points, ply_path)
        print(f"Saved voxel-center point cloud: {ply_path}")

    def _generation_kwargs_from_flags(self):
        gen = self.FLAGS.get("GENERATION", {}) if hasattr(self.FLAGS, "get") else {}
        if not hasattr(gen, "get"):
            return {}
        return dict(
            temperature=gen.get("temperature", None),
            sample_mode=gen.get("sample_mode", None),
            max_nodes_per_depth=gen.get("max_nodes_per_depth", None),
            min_keep_per_batch=gen.get("min_keep_per_batch", None),
        )

    @torch.no_grad()
    def generate_step(self, index):
        model = self.model_module
        model.eval()
        gen_kwargs = self._generation_kwargs_from_flags()

        with torch.autocast("cuda", enabled=self.use_amp):
            stats, final_coords = model.generate(
                batch_size=1,
                device=self.device,
                **gen_kwargs,
            )

        print("=" * 80)
        print(f"[v13 Parent-Parallel Sibling-AR Octree Generation] sample index = {index}")
        print(f"full_depth = {self.full_depth}, depth_stop = {self.depth_stop}")
        for depth in range(self.full_depth, self.depth_stop + 1):
            n = stats.get(f"depth_{depth}_num_nodes", 0)
            occ = stats.get(f"depth_{depth}_num_occ", 0)
            split = stats.get(f"depth_{depth}_num_split", 0)
            occ_mean = stats.get(f"depth_{depth}_occ_prob_mean", 0.0)
            occ_max = stats.get(f"depth_{depth}_occ_prob_max", 0.0)
            split_mean = stats.get(f"depth_{depth}_split_prob_mean", 0.0)
            split_max = stats.get(f"depth_{depth}_split_prob_max", 0.0)
            print(
                f"Depth {depth}: nodes={n}, split/continue={split}, occ/active={occ}, "
                f"split_p_mean={split_mean:.4f}, split_p_max={split_max:.4f}, "
                f"occ_p_mean={occ_mean:.4f}, occ_p_max={occ_max:.4f}"
            )
        print(f"Final depth {self.depth_stop} occupied voxels = {int(final_coords.shape[0])}")
        self._save_generated_pointcloud(final_coords, index)

        occupancy_to_sdf_pipeline(
            final_coords,
            depth=self.depth_stop,
            save_dir=os.path.join(self.logdir, "sdf"),
            name=f"sample_{index:04d}",
            resolution=64,
            save_mesh=True,
        )
        print("=" * 80)

    def generate(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.load_checkpoint()
        self.model.eval()

        num_samples = self.FLAGS.get("num_generate", 20)
        for i in tqdm(range(num_samples), ncols=80):
            self.generate_step(i)


if __name__ == "__main__":
    FractalSolver.main()
