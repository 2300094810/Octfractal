import argparse
import os
from typing import Dict

import numpy as np
from tqdm import tqdm


def read_filelist(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_model_dir(location: str, item: str) -> str:
    if os.path.isabs(item):
        return item
    return os.path.join(location, item)


def uid_from_model_dir(model_dir: str) -> str:
    parts = os.path.normpath(model_dir).split(os.sep)
    return os.path.join(parts[-2], parts[-1])


def voxelize_points(points: np.ndarray, depth: int) -> np.ndarray:
    """Voxelize points in [-1, 1] into unique integer coords at a given depth."""
    scale = 2 ** depth
    cell = ((points + 1.0) / 2.0 * scale).astype(np.int64)
    cell = np.clip(cell, 0, scale - 1)
    # Unique rows.
    if cell.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    cell = np.unique(cell, axis=0)
    return cell.astype(np.int64)


def preprocess_one(model_dir: str, points_scale: float, full_depth: int, depth_stop: int, max_points: int = 0) -> Dict[str, np.ndarray]:
    pc_path = os.path.join(model_dir, "pointcloud.npz")
    if not os.path.exists(pc_path):
        raise FileNotFoundError(pc_path)
    raw = np.load(pc_path)
    points = raw["points"].astype(np.float32) / float(points_scale)
    points = np.clip(points, -1.0, 1.0)

    if max_points and points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]

    output = {}
    for depth in range(full_depth, depth_stop + 1):
        output[f"occ_coords_d{depth}"] = voxelize_points(points, depth)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True, help="Root ShapeNet folder.")
    parser.add_argument("--filelist", required=True, help="Train/test filelist.")
    parser.add_argument("--output", required=True, help="Output cache root, e.g. /data/cache/shapenet_airplane_voxels_v8")
    parser.add_argument("--points_scale", type=float, default=0.5)
    parser.add_argument("--full_depth", type=int, default=2)
    parser.add_argument("--depth_stop", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=0, help="Optional point subsampling for faster preprocessing. 0 means use all points.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    items = read_filelist(args.filelist)
    os.makedirs(args.output, exist_ok=True)

    failed = []
    for item in tqdm(items, ncols=80):
        model_dir = resolve_model_dir(args.location, item)
        uid = uid_from_model_dir(model_dir)
        save_path = os.path.join(args.output, uid + ".npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path) and not args.overwrite:
            continue
        try:
            data = preprocess_one(
                model_dir=model_dir,
                points_scale=args.points_scale,
                full_depth=args.full_depth,
                depth_stop=args.depth_stop,
                max_points=args.max_points,
            )
            np.savez_compressed(save_path, **data)
        except Exception as e:
            failed.append((item, str(e)))

    if failed:
        fail_path = os.path.join(args.output, "failed.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for item, err in failed:
                f.write(f"{item}\t{err}\n")
        print(f"Finished with {len(failed)} failed items. See {fail_path}")
    else:
        print("Finished preprocessing with no failures.")


if __name__ == "__main__":
    main()
