import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage import measure

from vis_occ_utils import coords_to_points


# =========================================================
# Occupancy (voxel coords) -> Approximate SDF
# =========================================================
class OccupancyToSDF:
    def __init__(self, depth, resolution=64, bounds=(-1.0, 1.0)):
        self.depth = depth
        self.resolution = resolution
        self.bounds = bounds

    # -------------------------
    # build point cloud
    # -------------------------
    def _coords_to_points(self, final_coords):
        points, _ = coords_to_points(final_coords, self.depth)
        return points

    # -------------------------
    # build grid
    # -------------------------
    def _make_grid(self):
        x = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        z = np.linspace(self.bounds[0], self.bounds[1], self.resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        grid = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        return grid

    # -------------------------
    # core SDF computation
    # -------------------------
    def convert(self, final_coords):
        points = self._coords_to_points(final_coords)

        if points is None or len(points) == 0:
            return np.zeros(
                (self.resolution, self.resolution, self.resolution),
                dtype=np.float32,
            )

        tree = cKDTree(points)

        grid = self._make_grid()

        dist, _ = tree.query(grid, k=1)
        sdf = dist.astype(np.float32)

        # -------------------------
        # sign approximation
        # -------------------------
        voxel_size = 2.0 / (2 ** self.depth)
        threshold = 1.5 * voxel_size

        inside = dist < threshold
        sdf[inside] *= -1.0

        sdf = sdf.reshape(self.resolution, self.resolution, self.resolution)

        return sdf


# =========================================================
# Save utilities
# =========================================================
def save_sdf(sdf, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, sdf)


def save_mesh_from_sdf(sdf, path, level=0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    verts, faces, normals, values = measure.marching_cubes(sdf, level=level)

    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# =========================================================
# Full pipeline API
# =========================================================
def occupancy_to_sdf_pipeline(
    final_coords,
    depth,
    save_dir="./sdf_output",
    name="sample",
    resolution=64,
    save_mesh=True,
):
    os.makedirs(save_dir, exist_ok=True)

    converter = OccupancyToSDF(depth=depth, resolution=resolution)

    sdf = converter.convert(final_coords)

    sdf_path = os.path.join(save_dir, f"{name}_sdf.npy")
    save_sdf(sdf, sdf_path)

    print(f"[OK] Saved SDF -> {sdf_path}")

    if save_mesh:
        mesh_path = os.path.join(save_dir, f"{name}_mesh.obj")
        save_mesh_from_sdf(sdf, mesh_path)
        print(f"[OK] Saved mesh -> {mesh_path}")

    return sdf


# =========================================================
# Example test
# =========================================================
if __name__ == "__main__":
    # fake test data (replace with your model output)
    dummy = torch.randint(0, 20, (2000, 4))

    sdf = occupancy_to_sdf_pipeline(
        dummy,
        depth=6,
        save_dir="./test_sdf",
        name="demo",
        resolution=64,
        save_mesh=True,
    )

    print("SDF shape:", sdf.shape)