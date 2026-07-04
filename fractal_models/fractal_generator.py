import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for binary classification with class target 0/1."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(targets.long() == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt).pow(self.gamma)
        ce = F.cross_entropy(logits, targets.long(), reduction="none")
        return (focal_weight * ce).mean()


class ConditionalBinaryHead(nn.Module):
    """Binary head conditioned on a node/local feature and the sample latent z."""

    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + latent_dim),
            nn.Linear(feature_dim + latent_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 2),
        )

    def forward(self, feats: torch.Tensor, z_node: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feats, z_node], dim=-1))


class FractalGenerator(nn.Module):
    """Sibling-autoregressive octree generator with optional auto-decoder.

    Training path:
        z comes from a per-shape latent table when sample indices are provided,
        otherwise z ~ N(0, I)
        -> predict full_depth active nodes
        -> for each active parent, autoregressively predict its 8 child masks
        -> scheduled sampling mixes GT and predicted child masks for both
           child-token history and next-level expansion.

    Generation path:
        z ~ N(0, I)
        -> predict full_depth active nodes
        -> repeatedly generate 8-child masks for active parents
        -> prune zeros and continue ones until depth_stop.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        full_depth: int = 2,
        depth_stop: int = 6,
        expander_num_heads: int = 4,  # kept only for old YAML compatibility
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        split_focal_alpha: float = 0.75,
        split_focal_gamma: float = 2.0,
        occ_threshold: float = 0.58,
        split_threshold: float = 0.55,
        occ_weight: float = 1.0,
        split_weight: float = 1.5,
        pattern_weight: float = 0.0,  # ignored; old YAML compatibility
        rate_weight: float = 0.1,
        kl_weight: float = 0.0,
        num_freqs: int = 6,
        pos_emb_scale: float = 1.0,
        latent_dim: int = 128,
        film_scale: float = 0.5,
        latent_depth_scale: float = 0.8,
        default_temperature: float = 0.8,
        default_sample_mode: str = "topk_threshold",
        default_max_nodes_per_depth: Optional[Union[str, Dict[int, int]]] = None,
        min_keep_per_batch: int = 1,
        min_children_per_parent: int = 1,
        max_children_per_parent: int = 6,
        min_final_children_per_parent: int = 0,
        max_final_children_per_parent: int = 6,
        scheduled_sampling_prob: float = 0.0,
        scheduled_sampling_start_epoch: int = 100,
        scheduled_sampling_warmup_epochs: int = 300,
        scheduled_sampling_max_prob: float = 0.35,
        scheduled_sampling_force_gt_min: bool = True,
        use_autodecoder: bool = True,
        num_latents: int = 0,
        latent_noise_std: float = 0.03,
        generation_latent_mode: str = "table",
        generation_latent_noise_std: float = 0.05,
        use_pattern_loss: bool = False,
        **kwargs,
    ):
        super().__init__()
        if depth_stop < full_depth:
            raise ValueError(f"depth_stop must be >= full_depth, got {depth_stop=} {full_depth=}")

        self.feature_dim = int(feature_dim)
        self.full_depth = int(full_depth)
        self.depth_stop = int(depth_stop)
        self.num_depths = self.depth_stop - self.full_depth + 1
        self.num_expand_stages = max(self.depth_stop - self.full_depth, 0)

        self.occ_threshold = float(occ_threshold)
        self.split_threshold = float(split_threshold)
        self.occ_weight = float(occ_weight)
        self.split_weight = float(split_weight)
        self.rate_weight = float(rate_weight)
        self.pattern_weight = 0.0
        self.kl_weight = 0.0

        self.num_freqs = int(num_freqs)
        self.pos_emb_scale = float(pos_emb_scale)
        self.latent_dim = int(latent_dim)
        self.film_scale = float(film_scale)
        self.latent_depth_scale = float(latent_depth_scale)

        self.default_temperature = float(default_temperature)
        self.default_sample_mode = str(default_sample_mode)
        self.default_max_nodes_per_depth = self._parse_max_nodes(default_max_nodes_per_depth)
        self.min_keep_per_batch = int(min_keep_per_batch)
        self.min_children_per_parent = int(min_children_per_parent)
        self.max_children_per_parent = int(max_children_per_parent)
        self.min_final_children_per_parent = int(min_final_children_per_parent)
        self.max_final_children_per_parent = int(max_final_children_per_parent)

        self.scheduled_sampling_prob = float(scheduled_sampling_prob)
        self.scheduled_sampling_start_epoch = int(scheduled_sampling_start_epoch)
        self.scheduled_sampling_warmup_epochs = int(scheduled_sampling_warmup_epochs)
        self.scheduled_sampling_max_prob = float(scheduled_sampling_max_prob)
        self.scheduled_sampling_force_gt_min = bool(scheduled_sampling_force_gt_min)
        self.use_autodecoder = bool(use_autodecoder)
        self.num_latents = int(num_latents)
        self.latent_noise_std = float(latent_noise_std)
        self.generation_latent_mode = str(generation_latent_mode)
        self.generation_latent_noise_std = float(generation_latent_noise_std)

        self.occ_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.split_loss_fn = FocalLoss(alpha=split_focal_alpha, gamma=split_focal_gamma)

        # Base feature and positional / latent conditioning.
        self.root_embedding = nn.Parameter(torch.zeros(1, self.feature_dim))
        pos_in_dim = 3 * (1 + 2 * self.num_freqs)
        self.pos_proj = nn.Linear(pos_in_dim, self.feature_dim)
        self.depth_emb = nn.Embedding(self.depth_stop + 1, self.feature_dim)

        self.latent_to_root = nn.Sequential(
            nn.Linear(self.latent_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.latent_depth_proj = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.feature_dim) for _ in range(self.num_depths)]
        )
        self.global_film = nn.Sequential(
            nn.Linear(self.latent_dim, self.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
        )
        if self.use_autodecoder and self.num_latents > 0:
            self.latent_table = nn.Embedding(self.num_latents, self.latent_dim)
        else:
            self.latent_table = None

        # Coarse full_depth activation head. It selects the starting active nodes.
        self.coarse_head = ConditionalBinaryHead(self.feature_dim, self.latent_dim)

        # One child head per expansion stage. The head for stage d predicts the
        # child continue mask at depth d+1.
        self.child_heads = nn.ModuleList(
            [ConditionalBinaryHead(self.feature_dim, self.latent_dim) for _ in range(self.num_expand_stages)]
        )

        # Shared single-fractal-generator core. It is reused at every hierarchy
        # level. The only serial dimension is the 8 sibling octants; parents are
        # processed in parallel.
        self.child_pos_emb = nn.Parameter(torch.zeros(8, self.feature_dim))
        self.child_token_emb = nn.Embedding(3, self.feature_dim)  # 0, 1, START=2
        self.child_start_token_id = 2
        self.child_init = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh(),
        )
        self.child_in_norm = nn.LayerNorm(self.feature_dim)
        self.child_gru = nn.GRUCell(self.feature_dim, self.feature_dim)
        self.child_out_norm = nn.LayerNorm(self.feature_dim)
        self.child_ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.root_embedding, std=0.02)
        nn.init.normal_(self.child_pos_emb, std=0.02)
        if self.latent_table is not None:
            nn.init.normal_(self.latent_table.weight, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRUCell):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    param.data.zero_()

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        return numer / denom.clamp(min=1.0)

    @staticmethod
    def _parse_max_nodes(value: Optional[Union[str, Dict[int, int]]]) -> Optional[Dict[int, int]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return {int(k): int(v) for k, v in value.items()}
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            out: Dict[int, int] = {}
            for item in value.split(","):
                d, k = item.split(":")
                out[int(d.strip())] = int(k.strip())
            return out
        raise TypeError(f"Unsupported max_nodes_per_depth type: {type(value)}")

    def _scheduled_sampling_p(self, epoch: Optional[int] = None, override: Optional[float] = None) -> float:
        if override is not None:
            return float(max(0.0, min(1.0, override)))
        if not self.training:
            return 0.0
        if epoch is None:
            return float(max(0.0, min(1.0, self.scheduled_sampling_prob)))
        if epoch < self.scheduled_sampling_start_epoch:
            return 0.0
        if self.scheduled_sampling_warmup_epochs <= 0:
            return float(max(0.0, min(1.0, self.scheduled_sampling_max_prob)))
        ratio = (epoch - self.scheduled_sampling_start_epoch) / float(self.scheduled_sampling_warmup_epochs)
        ratio = max(0.0, min(1.0, ratio))
        return float(max(0.0, min(1.0, self.scheduled_sampling_max_prob * ratio)))

    def _fourier_encode(self, xyz: torch.Tensor) -> torch.Tensor:
        feats = [xyz]
        for k in range(self.num_freqs):
            freq = (2.0 ** k) * math.pi
            feats.append(torch.sin(freq * xyz))
            feats.append(torch.cos(freq * xyz))
        return torch.cat(feats, dim=-1)

    def _make_full_grid_coords(self, batch_size: int, depth: int, device: torch.device) -> torch.Tensor:
        scale = 2 ** depth
        xs = torch.arange(scale, device=device)
        ys = torch.arange(scale, device=device)
        zs = torch.arange(scale, device=device)
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        coords_all = []
        for b in range(batch_size):
            bcol = torch.full((xyz.shape[0], 1), b, device=device, dtype=torch.long)
            coords_all.append(torch.cat([xyz, bcol], dim=1))
        return torch.cat(coords_all, dim=0).long()

    def _expand_child_coords(self, parent_coords: torch.Tensor) -> torch.Tensor:
        if parent_coords.shape[0] == 0:
            return torch.zeros(0, 4, device=parent_coords.device, dtype=torch.long)
        xyz = parent_coords[:, :3]
        b = parent_coords[:, 3:4]
        offsets = torch.tensor(
            [
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
            ],
            device=parent_coords.device,
            dtype=torch.long,
        )
        child_xyz = xyz.unsqueeze(1) * 2 + offsets.unsqueeze(0)
        child_b = b.unsqueeze(1).expand(-1, 8, -1)
        return torch.cat([child_xyz, child_b], dim=-1).reshape(-1, 4).long()

    def _coords_to_pos_emb(self, coords: torch.Tensor, depth: int) -> torch.Tensor:
        if coords.shape[0] == 0:
            return torch.zeros(0, self.feature_dim, device=coords.device)
        scale = float(2 ** depth)
        xyz = coords[:, :3].float()
        xyz = (xyz + 0.5) / scale
        xyz = xyz * 2.0 - 1.0
        xyz = xyz * self.pos_emb_scale
        return self.pos_proj(self._fourier_encode(xyz))

    def _coords_to_keys(self, coords: torch.Tensor, depth: int) -> torch.Tensor:
        scale = 2 ** depth
        x = coords[:, 0].long()
        y = coords[:, 1].long()
        z = coords[:, 2].long()
        b = coords[:, 3].long()
        return b * (scale ** 3) + x * (scale ** 2) + y * scale + z

    def _coords_to_local_keys(self, coords: torch.Tensor, depth: int) -> torch.Tensor:
        scale = 2 ** depth
        x = coords[:, 0].long()
        y = coords[:, 1].long()
        z = coords[:, 2].long()
        return x * (scale ** 2) + y * scale + z

    def _sort_nodes(self, feats: torch.Tensor, coords: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if coords.shape[0] <= 1:
            return feats, coords
        keys = self._coords_to_keys(coords, depth)
        order = torch.argsort(keys)
        return feats[order], coords[order]

    def _match_candidate_keys(self, cand_keys: torch.Tensor, occ_keys: torch.Tensor) -> torch.Tensor:
        if cand_keys.numel() == 0:
            return torch.zeros(0, device=cand_keys.device, dtype=torch.long)
        if occ_keys.numel() == 0:
            return torch.zeros(cand_keys.numel(), device=cand_keys.device, dtype=torch.long)
        occ_keys = torch.unique(occ_keys)
        sorted_keys, _ = torch.sort(occ_keys)
        idx = torch.searchsorted(sorted_keys, cand_keys)
        idx = idx.clamp(0, sorted_keys.numel() - 1)
        return (sorted_keys[idx] == cand_keys).long()

    def _infer_batch_size(self, voxel_occ: Dict[int, torch.Tensor]) -> int:
        max_b = -1
        for coords in voxel_occ.values():
            if coords is None or coords.numel() == 0:
                continue
            max_b = max(max_b, int(coords[:, 3].max().item()))
        if max_b < 0:
            raise ValueError("voxel_occ is empty; cannot infer batch size.")
        return max_b + 1

    def _build_gt_occupancy(self, coords: torch.Tensor, depth: int, voxel_occ: Dict[int, torch.Tensor]) -> torch.Tensor:
        device = coords.device
        if coords.shape[0] == 0:
            return torch.zeros(0, device=device, dtype=torch.long)
        if depth not in voxel_occ:
            raise KeyError(f"Missing cached occupancy key for depth {depth}.")
        occ_coords = voxel_occ[depth].to(device=device, dtype=torch.long)
        if occ_coords.numel() == 0:
            return torch.zeros(coords.shape[0], device=device, dtype=torch.long)
        cand_keys = self._coords_to_keys(coords, depth)
        occ_keys = self._coords_to_keys(occ_coords, depth)
        return self._match_candidate_keys(cand_keys, occ_keys)

    def _sample_latent(
        self,
        batch_size: int,
        device: torch.device,
        sample_idx: Optional[torch.Tensor] = None,
        use_latent_table: bool = True,
    ) -> torch.Tensor:
        if self.latent_table is not None and use_latent_table and sample_idx is not None:
            sample_idx = sample_idx.to(device=device, dtype=torch.long).view(-1)
            if sample_idx.numel() != batch_size:
                raise ValueError(f"sample_idx has {sample_idx.numel()} ids, expected batch_size={batch_size}.")
            if sample_idx.numel() > 0:
                min_id = int(sample_idx.min().item())
                max_id = int(sample_idx.max().item())
                if min_id < 0 or max_id >= self.num_latents:
                    raise IndexError(f"sample_idx range [{min_id}, {max_id}] outside latent table size {self.num_latents}.")
            z = self.latent_table(sample_idx)
            if self.training and self.latent_noise_std > 0:
                z = z + torch.randn_like(z) * self.latent_noise_std
            return z
        return torch.randn(batch_size, self.latent_dim, device=device)

    def _sample_generation_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.latent_table is None or self.generation_latent_mode == "random":
            return torch.randn(batch_size, self.latent_dim, device=device)

        if self.generation_latent_mode in {"table", "random_table"}:
            ids = torch.randint(0, self.num_latents, (batch_size,), device=device)
            z = self.latent_table(ids)
        elif self.generation_latent_mode == "mean_table":
            z = self.latent_table.weight.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            raise ValueError(f"Unknown generation_latent_mode: {self.generation_latent_mode}")

        if self.generation_latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.generation_latent_noise_std
        return z

    def _node_latent(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if coords.shape[0] == 0:
            return torch.zeros(0, self.latent_dim, device=z.device)
        b = coords[:, 3].long()
        return z[b]

    def _add_context(self, feats: torch.Tensor, coords: torch.Tensor, depth: int, z: torch.Tensor) -> torch.Tensor:
        if feats.shape[0] == 0:
            return feats
        b = coords[:, 3].long()
        depth_idx = depth - self.full_depth
        feats = feats + self._coords_to_pos_emb(coords, depth)
        feats = feats + self.depth_emb(
            torch.full((feats.shape[0],), depth, device=feats.device, dtype=torch.long)
        )
        feats = feats + self.latent_depth_scale * self.latent_depth_proj[depth_idx](z[b])
        gamma_beta = self.global_film(z[b])
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return feats * (1.0 + self.film_scale * gamma) + beta

    def _init_features(self, coords: torch.Tensor, depth: int, z: torch.Tensor) -> torch.Tensor:
        b = coords[:, 3].long()
        root = self.root_embedding.expand(coords.shape[0], -1).contiguous()
        root = root + self.latent_to_root(z[b])
        return self._add_context(root, coords, depth, z)

    def _rate_loss(self, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.tensor(0.0, device=gt.device)
        probs = F.softmax(logits, dim=-1)[:, 1]
        return (probs.mean() - gt.float().mean()).pow(2)

    def _record_binary_metrics(self, output: dict, prefix: str, logits: torch.Tensor, gt: torch.Tensor, depth: int):
        if logits.numel() == 0:
            z0 = torch.tensor(0.0, device=gt.device)
            for name in ["acc", "precision", "recall", "gt_pos_rate", "pred_pos_rate", "num_nodes", "num_gt_pos", "num_pred_pos"]:
                output[f"{prefix}_{name}_d{depth}"] = z0
            return z0
        pred = logits.argmax(dim=-1)
        tp = ((pred == 1) & (gt == 1)).float().sum()
        fp = ((pred == 1) & (gt == 0)).float().sum()
        fn = ((pred == 0) & (gt == 1)).float().sum()
        gt_pos = (gt == 1).float().sum()
        pred_pos = (pred == 1).float().sum()
        total = torch.tensor(float(gt.numel()), device=logits.device)
        acc = (pred == gt).float().mean()
        output[f"{prefix}_acc_d{depth}"] = acc
        output[f"{prefix}_precision_d{depth}"] = self._safe_div(tp, tp + fp)
        output[f"{prefix}_recall_d{depth}"] = self._safe_div(tp, tp + fn)
        output[f"{prefix}_gt_pos_rate_d{depth}"] = gt_pos / total
        output[f"{prefix}_pred_pos_rate_d{depth}"] = pred_pos / total
        output[f"{prefix}_num_nodes_d{depth}"] = total
        output[f"{prefix}_num_gt_pos_d{depth}"] = gt_pos
        output[f"{prefix}_num_pred_pos_d{depth}"] = pred_pos
        return acc

    def _scheduled_mix_binary(self, gt: torch.Tensor, pred: torch.Tensor, coords: torch.Tensor, prob: float) -> torch.Tensor:
        if gt.numel() == 0:
            return gt.long()
        if prob <= 0.0:
            mixed = gt.long().clone()
        elif prob >= 1.0:
            mixed = pred.long().clone()
        else:
            use_pred = torch.rand(gt.shape, device=gt.device) < prob
            mixed = torch.where(use_pred, pred.long(), gt.long())

        if self.scheduled_sampling_force_gt_min:
            batch_ids = torch.unique(coords[:, 3].long(), sorted=True)
            for b in batch_ids.tolist():
                idx = torch.nonzero(coords[:, 3].long() == int(b), as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                if mixed[idx].sum() == 0 and gt[idx].sum() > 0:
                    first_gt_pos = idx[torch.nonzero(gt[idx] == 1, as_tuple=False).squeeze(1)[0]]
                    mixed[first_gt_pos] = 1
        return mixed.long()

    def _force_parent_gt_min(self, gt_view: torch.Tensor, mixed_view: torch.Tensor) -> torch.Tensor:
        if not self.scheduled_sampling_force_gt_min or gt_view.numel() == 0:
            return mixed_view.long()
        mixed_view = mixed_view.long().clone()
        missing = (mixed_view.sum(dim=1) == 0) & (gt_view.long().sum(dim=1) > 0)
        if missing.any():
            rows = torch.nonzero(missing, as_tuple=False).squeeze(1)
            first_gt = gt_view[rows].long().argmax(dim=1)
            mixed_view[rows, first_gt] = 1
        return mixed_view.long()

    def _child_step_feature(self, parent_features: torch.Tensor, hidden: torch.Tensor, prev_tokens: torch.Tensor, octant: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # parent_features: [N, C], hidden: [N, C], prev_tokens: [N]
        inp = parent_features + self.child_pos_emb[octant].view(1, -1) + self.child_token_emb(prev_tokens.long())
        hidden = self.child_gru(self.child_in_norm(inp), hidden)
        child_feat = parent_features + self.child_pos_emb[octant].view(1, -1) + hidden
        child_feat = child_feat + self.child_ffn(self.child_out_norm(child_feat))
        return child_feat, hidden

    def _sibling_ar_children_train(
        self,
        parent_features: torch.Tensor,
        parent_coords: torch.Tensor,
        parent_depth: int,
        z: torch.Tensor,
        voxel_occ: Dict[int, torch.Tensor],
        head: ConditionalBinaryHead,
        scheduled_sampling_prob: float,
    ):
        n = parent_features.shape[0]
        device = parent_features.device
        child_depth = parent_depth + 1
        if n == 0:
            zfeat = torch.zeros(0, self.feature_dim, device=device)
            zcoords = torch.zeros(0, 4, device=device, dtype=torch.long)
            zlogits = torch.zeros(0, 2, device=device)
            ztarget = torch.zeros(0, device=device, dtype=torch.long)
            return zfeat, zcoords, zlogits, ztarget, ztarget, ztarget.float()

        parent_features, parent_coords = self._sort_nodes(parent_features, parent_coords, parent_depth)
        child_coords = self._expand_child_coords(parent_coords)
        child_gt = self._build_gt_occupancy(child_coords, child_depth, voxel_occ).view(n, 8)
        child_coords_view = child_coords.view(n, 8, 4)
        z_parent = z[parent_coords[:, 3].long()]

        hidden = self.child_init(parent_features)
        prev_tokens = torch.full((n,), self.child_start_token_id, device=device, dtype=torch.long)
        all_feats = []
        all_logits = []
        all_mixed = []
        all_probs = []

        for octant in range(8):
            child_feat_t, hidden = self._child_step_feature(parent_features, hidden, prev_tokens, octant)
            logits_t = head(child_feat_t, z_parent)
            probs_t = F.softmax(logits_t, dim=-1)[:, 1]
            gt_t = child_gt[:, octant]
            pred_t = logits_t.argmax(dim=-1).detach()
            mixed_t = self._scheduled_mix_binary(gt_t, pred_t, child_coords_view[:, octant, :], scheduled_sampling_prob)

            all_feats.append(child_feat_t)
            all_logits.append(logits_t)
            all_mixed.append(mixed_t)
            all_probs.append(probs_t)
            prev_tokens = mixed_t.detach().long()

        mixed_view = torch.stack(all_mixed, dim=1).long()
        mixed_view = self._force_parent_gt_min(child_gt, mixed_view)
        child_features, child_coords, logits, probs = self._sibling_ar_children_forced(
            parent_features=parent_features,
            parent_coords=parent_coords,
            parent_depth=parent_depth,
            z=z,
            head=head,
            forced_tokens=mixed_view,
        )
        gt = child_gt.reshape(n * 8).long()
        mixed = mixed_view.reshape(n * 8).long()
        return child_features, child_coords, logits, gt, mixed, probs

    def _sibling_ar_children_forced(
        self,
        parent_features: torch.Tensor,
        parent_coords: torch.Tensor,
        parent_depth: int,
        z: torch.Tensor,
        head: ConditionalBinaryHead,
        forced_tokens: torch.Tensor,
    ):
        n = parent_features.shape[0]
        device = parent_features.device
        child_depth = parent_depth + 1
        if n == 0:
            return (
                torch.zeros(0, self.feature_dim, device=device),
                torch.zeros(0, 4, device=device, dtype=torch.long),
                torch.zeros(0, 2, device=device),
                torch.zeros(0, device=device, dtype=torch.float32),
            )

        parent_features, parent_coords = self._sort_nodes(parent_features, parent_coords, parent_depth)
        child_coords = self._expand_child_coords(parent_coords)
        forced_tokens = forced_tokens.to(device=device, dtype=torch.long).view(n, 8)
        z_parent = z[parent_coords[:, 3].long()]

        hidden = self.child_init(parent_features)
        prev_tokens = torch.full((n,), self.child_start_token_id, device=device, dtype=torch.long)
        all_feats = []
        all_logits = []
        all_probs = []

        for octant in range(8):
            child_feat_t, hidden = self._child_step_feature(parent_features, hidden, prev_tokens, octant)
            logits_t = head(child_feat_t, z_parent)
            probs_t = F.softmax(logits_t, dim=-1)[:, 1]
            all_feats.append(child_feat_t)
            all_logits.append(logits_t)
            all_probs.append(probs_t)
            prev_tokens = forced_tokens[:, octant].detach().long()

        child_features = torch.stack(all_feats, dim=1).reshape(n * 8, self.feature_dim)
        child_features = self._add_context(child_features, child_coords, child_depth, z)
        logits = torch.stack(all_logits, dim=1).reshape(n * 8, 2)
        probs = torch.stack(all_probs, dim=1).reshape(n * 8)
        return child_features, child_coords, logits, probs

    def _sample_binary_probs(self, logits: torch.Tensor, temperature: float, sample_mode: str, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        temperature = max(float(temperature), 1e-6)
        probs = F.softmax(logits / temperature, dim=-1)[:, 1]
        if sample_mode == "bernoulli":
            pred = torch.bernoulli(probs).long()
        elif sample_mode in {"threshold", "topk", "topk_threshold"}:
            # For top-k modes this is only the first-pass AR token. v13 runs a
            # second forced pass after local caps so final history is consistent.
            pred = (probs > threshold).long()
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")
        return pred, probs

    def _cap_existing_by_batch(
        self,
        pred: torch.Tensor,
        probs: torch.Tensor,
        coords: torch.Tensor,
        depth: int,
        max_nodes_per_depth: Optional[Dict[int, int]],
        min_keep_per_batch: int,
    ) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.long()
        capped = pred.long().clone()
        max_nodes = None if max_nodes_per_depth is None else max_nodes_per_depth.get(depth, None)
        batch_ids = torch.unique(coords[:, 3].long(), sorted=True)
        for b in batch_ids.tolist():
            idx = torch.nonzero(coords[:, 3].long() == int(b), as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            if max_nodes is not None and int(max_nodes) >= 0:
                keep_limit = min(int(max_nodes), idx.numel())
                pos_idx = idx[capped[idx] == 1]
                if pos_idx.numel() > keep_limit:
                    top_local = torch.topk(probs[pos_idx], k=keep_limit).indices
                    keep_idx = pos_idx[top_local]
                    capped[idx] = 0
                    capped[keep_idx] = 1
            if min_keep_per_batch > 0 and capped[idx].sum() < min_keep_per_batch:
                k = min(int(min_keep_per_batch), idx.numel())
                top_idx = idx[torch.topk(probs[idx], k=k).indices]
                capped[top_idx] = 1
        return capped.long()

    def _apply_depth_cap_by_batch(
        self,
        pred: torch.Tensor,
        probs: torch.Tensor,
        coords: torch.Tensor,
        depth: int,
        sample_mode: str,
        threshold: float,
        max_nodes_per_depth: Optional[Dict[int, int]],
        min_keep_per_batch: int,
    ) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.long()
        if sample_mode == "topk_threshold":
            capped = (probs > threshold).long()
        elif sample_mode == "topk":
            capped = torch.ones_like(pred, dtype=torch.long)
        else:
            capped = pred.long().clone()
        return self._cap_existing_by_batch(capped, probs, coords, depth, max_nodes_per_depth, min_keep_per_batch)

    def _apply_parent_child_constraints(
        self,
        pred: torch.Tensor,
        probs: torch.Tensor,
        child_coords: torch.Tensor,
        child_depth: int,
        sample_mode: str,
        threshold: float,
        max_nodes_per_depth: Optional[Dict[int, int]],
        min_keep_per_batch: int,
        is_final_depth: bool,
    ) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.long()
        n = pred.numel() // 8
        probs_view = probs.view(n, 8)
        if sample_mode == "topk_threshold":
            pred_view = (probs_view > threshold).long()
        elif sample_mode == "topk":
            pred_view = torch.ones_like(probs_view, dtype=torch.long)
        else:
            pred_view = pred.long().view(n, 8).clone()

        if is_final_depth:
            min_child = max(0, self.min_final_children_per_parent)
            max_child = self.max_final_children_per_parent
        else:
            min_child = max(0, self.min_children_per_parent)
            max_child = self.max_children_per_parent

        if max_child is not None and max_child >= 0:
            max_child = min(int(max_child), 8)
            if max_child == 0:
                pred_view.zero_()
            elif max_child < 8:
                for row in range(n):
                    pos = torch.nonzero(pred_view[row] == 1, as_tuple=False).squeeze(1)
                    if pos.numel() > max_child:
                        top_local = torch.topk(probs_view[row, pos], k=max_child).indices
                        keep = pos[top_local]
                        pred_view[row].zero_()
                        pred_view[row, keep] = 1

        if min_child > 0:
            min_child = min(int(min_child), 8)
            for row in range(n):
                if pred_view[row].sum() < min_child:
                    k = min_child
                    top = torch.topk(probs_view[row], k=k).indices
                    pred_view[row, top] = 1

        capped = pred_view.reshape(n * 8).long()
        return self._cap_existing_by_batch(
            capped,
            probs,
            child_coords,
            child_depth,
            max_nodes_per_depth,
            min_keep_per_batch,
        )

    @torch.no_grad()
    def _sibling_ar_children_generate(
        self,
        parent_features: torch.Tensor,
        parent_coords: torch.Tensor,
        parent_depth: int,
        z: torch.Tensor,
        head: ConditionalBinaryHead,
        temperature: float,
        sample_mode: str,
        max_nodes_per_depth: Optional[Dict[int, int]],
        min_keep_per_batch: int,
        is_final_depth: bool,
    ):
        n = parent_features.shape[0]
        device = parent_features.device
        child_depth = parent_depth + 1
        if n == 0:
            return (
                torch.zeros(0, self.feature_dim, device=device),
                torch.zeros(0, 4, device=device, dtype=torch.long),
                torch.zeros(0, device=device, dtype=torch.long),
                torch.zeros(0, device=device, dtype=torch.float32),
            )

        parent_features, parent_coords = self._sort_nodes(parent_features, parent_coords, parent_depth)
        child_coords = self._expand_child_coords(parent_coords)
        z_parent = z[parent_coords[:, 3].long()]
        threshold = self.occ_threshold if is_final_depth else self.split_threshold

        hidden = self.child_init(parent_features)
        prev_tokens = torch.full((n,), self.child_start_token_id, device=device, dtype=torch.long)
        all_feats = []
        all_pred = []
        all_probs = []

        for octant in range(8):
            child_feat_t, hidden = self._child_step_feature(parent_features, hidden, prev_tokens, octant)
            logits_t = head(child_feat_t, z_parent)
            pred_t, probs_t = self._sample_binary_probs(logits_t, temperature, sample_mode, threshold)
            all_feats.append(child_feat_t)
            all_pred.append(pred_t)
            all_probs.append(probs_t)
            prev_tokens = pred_t.detach().long()

        raw_pred = torch.stack(all_pred, dim=1).reshape(n * 8).long()
        raw_probs = torch.stack(all_probs, dim=1).reshape(n * 8)
        final_pred = self._apply_parent_child_constraints(
            raw_pred,
            raw_probs,
            child_coords,
            child_depth,
            sample_mode,
            threshold,
            max_nodes_per_depth,
            min_keep_per_batch,
            is_final_depth,
        )
        final_tokens = final_pred.view(n, 8)
        child_features, child_coords, _, final_probs = self._sibling_ar_children_forced(
            parent_features=parent_features,
            parent_coords=parent_coords,
            parent_depth=parent_depth,
            z=z,
            head=head,
            forced_tokens=final_tokens,
        )
        return child_features, child_coords, final_pred, final_probs

    def forward(
        self,
        voxel_occ: Dict[int, torch.Tensor],
        epoch: Optional[int] = None,
        scheduled_sampling_prob: Optional[float] = None,
        sample_idx: Optional[torch.Tensor] = None,
        use_latent_table: bool = True,
    ):
        if voxel_occ is None:
            raise ValueError("voxel_occ is required for v13 training.")

        first_depth = min(voxel_occ.keys())
        device = voxel_occ[first_depth].device
        batch_size = self._infer_batch_size(voxel_occ)
        z = self._sample_latent(batch_size, device, sample_idx=sample_idx, use_latent_table=use_latent_table)
        ss_prob = self._scheduled_sampling_p(epoch=epoch, override=scheduled_sampling_prob)

        output = {}
        total_occ_loss = torch.tensor(0.0, device=device)
        total_split_loss = torch.tensor(0.0, device=device)
        total_rate_loss = torch.tensor(0.0, device=device)
        total_occ_acc = torch.tensor(0.0, device=device)
        total_split_acc = torch.tensor(0.0, device=device)
        active_occ_terms = 0
        active_split_terms = 0

        # 1) Coarse full_depth activation prediction.
        current_coords = self._make_full_grid_coords(batch_size, self.full_depth, device)
        current_features = self._init_features(current_coords, self.full_depth, z)
        coarse_gt = self._build_gt_occupancy(current_coords, self.full_depth, voxel_occ)
        coarse_logits = self.coarse_head(current_features, self._node_latent(current_coords, z))
        coarse_loss = self.occ_loss_fn(coarse_logits, coarse_gt)
        coarse_rate = self._rate_loss(coarse_logits, coarse_gt)
        total_occ_loss = total_occ_loss + coarse_loss
        total_rate_loss = total_rate_loss + coarse_rate
        active_occ_terms += 1
        total_occ_acc = total_occ_acc + self._record_binary_metrics(output, "occ", coarse_logits, coarse_gt, self.full_depth)

        coarse_pred = coarse_logits.argmax(dim=-1).detach()
        coarse_mixed = self._scheduled_mix_binary(coarse_gt, coarse_pred, current_coords, ss_prob)
        keep_mask = coarse_mixed == 1
        current_features = current_features[keep_mask]
        current_coords = current_coords[keep_mask]
        current_features, current_coords = self._sort_nodes(current_features, current_coords, self.full_depth)

        # 2) Recursive parent-parallel, sibling-autoregressive child generation.
        for parent_depth in range(self.full_depth, self.depth_stop):
            child_depth = parent_depth + 1
            stage_idx = parent_depth - self.full_depth

            if current_features.shape[0] == 0:
                z0 = torch.tensor(0.0, device=device)
                for rem_depth in range(child_depth, self.depth_stop + 1):
                    for prefix in ["occ", "split"]:
                        for name in ["acc", "precision", "recall", "gt_pos_rate", "pred_pos_rate", "num_nodes", "num_gt_pos", "num_pred_pos"]:
                            output[f"{prefix}_{name}_d{rem_depth}"] = z0
                break

            child_features, child_coords, child_logits, child_gt, child_mixed, child_probs = self._sibling_ar_children_train(
                parent_features=current_features,
                parent_coords=current_coords,
                parent_depth=parent_depth,
                z=z,
                voxel_occ=voxel_occ,
                head=self.child_heads[stage_idx],
                scheduled_sampling_prob=ss_prob,
            )

            if child_depth == self.depth_stop:
                occ_loss = self.occ_loss_fn(child_logits, child_gt)
                rate_loss = self._rate_loss(child_logits, child_gt)
                total_occ_loss = total_occ_loss + occ_loss
                total_rate_loss = total_rate_loss + rate_loss
                active_occ_terms += 1
                total_occ_acc = total_occ_acc + self._record_binary_metrics(output, "occ", child_logits, child_gt, child_depth)
                break

            split_loss = self.split_loss_fn(child_logits, child_gt)
            rate_loss = self._rate_loss(child_logits, child_gt)
            total_split_loss = total_split_loss + split_loss
            total_rate_loss = total_rate_loss + rate_loss
            active_split_terms += 1
            total_split_acc = total_split_acc + self._record_binary_metrics(output, "split", child_logits, child_gt, child_depth)

            keep_mask = child_mixed == 1
            current_features = child_features[keep_mask]
            current_coords = child_coords[keep_mask]
            current_features, current_coords = self._sort_nodes(current_features, current_coords, child_depth)

        output["occ_loss"] = total_occ_loss / max(active_occ_terms, 1)
        output["split_loss"] = total_split_loss / max(active_split_terms, 1)
        output["pattern_loss"] = torch.tensor(0.0, device=device)
        output["rate_loss"] = total_rate_loss / max(active_occ_terms + active_split_terms, 1)
        output["kl_loss"] = torch.tensor(0.0, device=device)
        output["occ_accuracy"] = total_occ_acc / max(active_occ_terms, 1)
        output["split_accuracy"] = total_split_acc / max(active_split_terms, 1)
        output["scheduled_sampling_prob"] = torch.tensor(ss_prob, device=device)
        output["active_nodes_last"] = torch.tensor(float(current_coords.shape[0]), device=device)
        output["latent_z_abs_mean"] = z.abs().mean()
        output["latent_z_std"] = z.std()
        output["using_latent_table"] = torch.tensor(
            float(self.latent_table is not None and use_latent_table and sample_idx is not None),
            device=device,
        )
        if self.latent_table is not None:
            output["latent_table_norm"] = self.latent_table.weight.norm(dim=1).mean()
        output["loss"] = (
            self.occ_weight * output["occ_loss"]
            + self.split_weight * output["split_loss"]
            + self.rate_weight * output["rate_loss"]
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        device: Union[str, torch.device] = "cuda",
        temperature: Optional[float] = None,
        sample_mode: Optional[str] = None,
        max_nodes_per_depth: Optional[Union[str, Dict[int, int]]] = None,
        min_keep_per_batch: Optional[int] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, torch.Tensor]:
        """Free-running parent-parallel, sibling-autoregressive generation."""
        if isinstance(device, str):
            device = torch.device(device)
        temperature = self.default_temperature if temperature is None else float(temperature)
        sample_mode = self.default_sample_mode if sample_mode is None else str(sample_mode)
        max_nodes = self.default_max_nodes_per_depth if max_nodes_per_depth is None else self._parse_max_nodes(max_nodes_per_depth)
        min_keep = self.min_keep_per_batch if min_keep_per_batch is None else int(min_keep_per_batch)

        if z is None:
            z = self._sample_generation_latent(batch_size, device)
        else:
            z = z.to(device=device)
            batch_size = z.shape[0]

        stats = {}

        # 1) Select active full_depth nodes from the coarse grid.
        current_coords = self._make_full_grid_coords(batch_size, self.full_depth, device)
        current_features = self._init_features(current_coords, self.full_depth, z)
        coarse_logits = self.coarse_head(current_features, self._node_latent(current_coords, z))
        coarse_pred, coarse_probs = self._sample_binary_probs(coarse_logits, temperature, sample_mode, self.occ_threshold)
        coarse_pred = self._apply_depth_cap_by_batch(
            coarse_pred,
            coarse_probs,
            current_coords,
            self.full_depth,
            sample_mode,
            self.occ_threshold,
            max_nodes,
            min_keep,
        )
        stats[f"depth_{self.full_depth}_num_nodes"] = int(current_coords.shape[0])
        stats[f"depth_{self.full_depth}_num_occ"] = int(coarse_pred.sum().item())
        stats[f"depth_{self.full_depth}_num_split"] = int(coarse_pred.sum().item())
        stats[f"depth_{self.full_depth}_occ_prob_mean"] = float(coarse_probs.mean().item()) if coarse_probs.numel() else 0.0
        stats[f"depth_{self.full_depth}_occ_prob_max"] = float(coarse_probs.max().item()) if coarse_probs.numel() else 0.0
        stats[f"depth_{self.full_depth}_split_prob_mean"] = stats[f"depth_{self.full_depth}_occ_prob_mean"]
        stats[f"depth_{self.full_depth}_split_prob_max"] = stats[f"depth_{self.full_depth}_occ_prob_max"]

        current_features = current_features[coarse_pred == 1]
        current_coords = current_coords[coarse_pred == 1]
        current_features, current_coords = self._sort_nodes(current_features, current_coords, self.full_depth)
        final_coords = torch.zeros(0, 4, device=device, dtype=torch.long)

        # 2) Recursively expand active parents.
        for parent_depth in range(self.full_depth, self.depth_stop):
            child_depth = parent_depth + 1
            stage_idx = parent_depth - self.full_depth
            if current_features.shape[0] == 0:
                for rem_depth in range(child_depth, self.depth_stop + 1):
                    stats[f"depth_{rem_depth}_num_nodes"] = 0
                    stats[f"depth_{rem_depth}_num_occ"] = 0
                    stats[f"depth_{rem_depth}_num_split"] = 0
                    stats[f"depth_{rem_depth}_occ_prob_mean"] = 0.0
                    stats[f"depth_{rem_depth}_occ_prob_max"] = 0.0
                    stats[f"depth_{rem_depth}_split_prob_mean"] = 0.0
                    stats[f"depth_{rem_depth}_split_prob_max"] = 0.0
                break

            child_features, child_coords, child_pred, child_probs = self._sibling_ar_children_generate(
                parent_features=current_features,
                parent_coords=current_coords,
                parent_depth=parent_depth,
                z=z,
                head=self.child_heads[stage_idx],
                temperature=temperature,
                sample_mode=sample_mode,
                max_nodes_per_depth=max_nodes,
                min_keep_per_batch=min_keep,
                is_final_depth=(child_depth == self.depth_stop),
            )

            stats[f"depth_{child_depth}_num_nodes"] = int(child_coords.shape[0])
            stats[f"depth_{child_depth}_num_occ"] = int(child_pred.sum().item())
            stats[f"depth_{child_depth}_occ_prob_mean"] = float(child_probs.mean().item()) if child_probs.numel() else 0.0
            stats[f"depth_{child_depth}_occ_prob_max"] = float(child_probs.max().item()) if child_probs.numel() else 0.0

            if child_depth == self.depth_stop:
                stats[f"depth_{child_depth}_num_split"] = 0
                stats[f"depth_{child_depth}_split_prob_mean"] = 0.0
                stats[f"depth_{child_depth}_split_prob_max"] = 0.0
                final_coords = child_coords[child_pred == 1]
                break

            stats[f"depth_{child_depth}_num_split"] = int(child_pred.sum().item())
            stats[f"depth_{child_depth}_split_prob_mean"] = stats[f"depth_{child_depth}_occ_prob_mean"]
            stats[f"depth_{child_depth}_split_prob_max"] = stats[f"depth_{child_depth}_occ_prob_max"]

            current_features = child_features[child_pred == 1]
            current_coords = child_coords[child_pred == 1]
            current_features, current_coords = self._sort_nodes(current_features, current_coords, child_depth)

        return stats, final_coords
