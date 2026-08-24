"""HoloBrain-0 encoder wrapper.

Loads the pretrained HoloBrain-0 (GD/BIP3D variant) encoder modules from a
checkpoint exported from robo_orchard_lab, and exposes them for feature
extraction on states collected interactively (e.g. from RoboTwin):

- robot state encoder  (HoloBrainRobotStateEncoder): joint states -> per-link embeddings
- spatial enhancer     (DepthFusionSpatialEnhancer): 2D/3D visual features +
  camera parameters -> geometry-aware spatial features

The vision backbones (2D Swin + neck, depth Swin + neck) are also loaded since
the spatial enhancer consumes their features.

Checkpoint layout (from HuggingFace HorizonRobotics/HoloBrain_v0.0_GD):
    HoloBrain_v0.0_GD/
        pretrain/
            model.config.json
            model.safetensors
        urdf/...                       # robot URDFs for forward kinematics

Example:
    encoder = HoloBrainEncoder("HoloBrain/HoloBrain_v0.0_GD/pretrain")

    # robot state: joint angles -> FK -> [theta, x, y, z, qw, qx, qy, qz] per link
    kin = encoder.make_kinematics("path/to/robot.urdf")
    state_feat = encoder.encode_joint_angles(joint_state, kin)  # [B, n_link, T, 256]

    # spatial feature from multi-view RGB + depth + camera params
    feat = encoder.encode_visual(imgs, depths, image_wh, projection_mat)
"""

import json
import os

import torch
from torch import nn

try:
    from .build import build
except ImportError:  # top-level (non-package) import
    from main.network.build import build


def _load_state_dict_prefix(model: nn.Module, state: dict, prefix: str, name: str):
    """Load `state[prefix + k]` into `model`, tolerating a leading 'module.'."""
    sub = {}
    for k, v in state.items():
        for p in (prefix, "module." + prefix):
            if k.startswith(p):
                sub[k[len(p):]] = v
                break
    if not sub:
        raise KeyError(f"no weights found with prefix '{prefix}' for {name}")
    missing, unexpected = model.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{name}: missing={missing}, unexpected={unexpected}"
        )
    return model


class RobotInjection(nn.Module):
    """Robot feature fusion ("injection").

    The one place hand-chosen quantities enter the model: takes the frozen
    HoloBrain state tokens (pose only) and injects what a safety problem
    needs — the analytic body-token columns (velocities, chain depth, arm
    id, gripper opening) — then anchors two learned residual query slots at
    the gripper positions for the trunk's geometric bias.

    Rules it keeps: no positional embedding (position enters the trunk
    explicitly via rel), no pooling, nothing about the action, positions
    stay world metres (unnormalized — a fixed physical distance must read
    identically on every embodiment).

    Input contract:
        state_tokens [B, 14, 256]  encoder.encode_joint_angles(...), T squeezed
        body_tokens  [B, M, 16]    extractor, padded to max_tokens
        link_pos     [B, M, 3]     world metres
        body_mask    [B, M]        bool
        joint_index  [B, M]        action column, -1 for gripper/padding

    Output:
        tok  [B, M+2, 256]  fused body tokens + 2 residual query slots
        pos  [B, M+2, 3]    world-metre position per token (residuals anchored
                            at the gripper midpoints)
        mask [B, M+2]       bool; residuals are always valid
    """

    def __init__(self, state_dim: int = 256, body_dim: int = 16, n_residual: int = 2):
        super().__init__()
        self.n_residual = n_residual
        # Step 1: scale the frozen tokens to the analytic columns' ~unit range
        self.norm_state = nn.RMSNorm(state_dim)
        # Step 2: shared per-token projection (count-invariant across
        # embodiments — no per-joint parameters). Nonlinear because the
        # useful quantities are products (velocity x orientation), not sums.
        self.proj = nn.Sequential(
            nn.Linear(state_dim + body_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )
        self.norm_out = nn.RMSNorm(state_dim)
        # Step 3: learned CLS-style query slots for properties no single link
        # owns (arm-to-arm distance, grasped payload). State-independent at
        # input; they become state-dependent via trunk attention.
        self.residual_param = nn.Parameter(torch.randn(n_residual, state_dim) * 0.02)
        # Scene side: width adapter only — z_img is already pretrained +
        # spatially grounded; all scene work happens in cross-attention.
        self.scene_adapter = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.RMSNorm(state_dim),
        )

    def forward(self, state_tokens, body_tokens, link_pos, body_mask, joint_index):
        # one device transfer per batch (caller should normally pre-place)
        dev = self.residual_param.device
        state_tokens, body_tokens, link_pos, body_mask, joint_index = (
            t.to(dev)
            for t in (state_tokens, body_tokens, link_pos, body_mask, joint_index)
        )

        B, N, C = state_tokens.shape          # N = 14 (HoloBrain layout)
        M = body_tokens.shape[1]              # extractor's padded length

        # Align on the extractor's padded length (downstream indexes it).
        state_full = state_tokens.new_zeros(B, M, C)
        state_full[:, :N] = state_tokens
        # "has a frozen embedding" vs "is a real body part": a token is
        # state-valid iff it is a real body part AND inside HoloBrain's
        # N-entry layout. Same today; diverges when link count exceeds N.
        state_valid = body_mask & (
            torch.arange(M, device=dev) < N
        ).unsqueeze(0)

        # Step 1 — normalize the frozen half, mask padded slots explicitly
        s = self.norm_state(state_full)
        s = s * state_valid.unsqueeze(-1).to(s.dtype)

        # Step 2 — project to model width, shared weights per token
        x = torch.cat([s, body_tokens.to(s.dtype)], dim=-1)
        h = self.norm_out(self.proj(x))       # [B, M, 256]

        # Step 3 — residual tokens anchored at gripper positions, indexed by
        # ARM (token col 14), not by token order: the learned slots are not
        # interchangeable once trained. If an arm's gripper is absent
        # (single-arm episode, gripperless embodiment), fall back to that
        # arm's last body link (tip) with a warning — the slot count is an
        # architectural constant and must not shrink.
        grip = (joint_index == -1) & body_mask
        arm = body_tokens[..., 14]            # 0 left / 1 right
        res_pos = link_pos.new_zeros(B, self.n_residual, 3)
        for a in range(self.n_residual):
            sel = grip & (arm == a)
            has = sel.sum(dim=1)              # per batch: 0 or 1
            if not bool((has == 1).all()):
                import warnings

                warnings.warn(
                    f"RobotInjection: arm {a} gripper missing in "
                    f"{int((has == 0).sum())} batch element(s); "
                    f"anchoring residual at the arm tip link."
                )
            sel_pos = link_pos.new_zeros(B, 3)
            if bool((has == 1).any()):
                sel_pos[has.bool()] = link_pos[sel]
            # fallback: last valid body token of arm a (tip link)
            arm_tok = body_mask & (arm == a)
            last = (M - 1) - torch.flip(arm_tok, [1]).float().argmax(dim=1)
            tip_pos = link_pos.gather(
                1, last.view(B, 1, 1).expand(-1, -1, 3)
            ).squeeze(1)
            res_pos[:, a] = torch.where(
                (has == 1).unsqueeze(-1), sel_pos, tip_pos
            )

        res = self.residual_param.unsqueeze(0).expand(B, -1, -1).to(h.dtype)

        # Step 4 — concatenate and extend the masks (residuals always valid)
        tok = torch.cat([h, res], dim=1)                       # [B, M+2, 256]
        pos = torch.cat([link_pos, res_pos.to(link_pos.dtype)], dim=1)
        mask = torch.cat(
            [
                body_mask,
                torch.ones(B, self.n_residual, dtype=torch.bool,
                           device=body_mask.device),
            ],
            dim=1,
        )
        return tok, pos, mask

    def adapt_scene(self, z_img, z_pos=None):
        """Scene width adapter, shape-safe.

        Takes the raw 4-D outputs of encode_visual_tokens —
        z_img [B, n_cam, N, 256], z_pos [B, n_cam, N, 3] — flattens both to
        [B, n_cam*N, ...] identically (so the shapes can't drift apart),
        then applies Linear -> RMSNorm and nothing more.

        Returns (z_flat, pos_flat) if z_pos is given, else z_flat.
        """
        dev = self.residual_param.device
        z_img = z_img.to(dev)
        # accept raw 4-D [B, n_cam, N, C] (encode_visual_tokens) or
        # pre-flattened 3-D [B, M, C] (e.g. from the replay buffer)
        if z_img.dim() == 4:
            z_img = z_img.flatten(1, 2)
            z_pos = z_pos.flatten(1, 2) if z_pos is not None else None
        z_flat = self.scene_adapter(z_img)
        if z_pos is None:
            return z_flat
        pos_flat = z_pos.to(dev)
        return z_flat, pos_flat


class HoloBrainEncoder(nn.Module):
    """HoloBrain-0 robot state encoder + spatial enhancer (GD/BIP3D variant)."""

    def __init__(self, ckpt_dir: str, device="cuda", dtype=torch.float32):
        super().__init__()
        if os.path.isdir(os.path.join(ckpt_dir, "pretrain")):
            ckpt_dir = os.path.join(ckpt_dir, "pretrain")

        with open(os.path.join(ckpt_dir, "model.config.json")) as f:
            cfg = json.load(f)
        if "BIP3D" not in cfg["class_type"]:
            raise ValueError(
                "This wrapper targets the GD/BIP3D variant "
                f"(HoloBrain_v0.0_GD); got {cfg['class_type']}"
            )
        self.cfg = cfg

        # --- robot state encoder ---
        self.robot_state_encoder = build(cfg["decoder"]["robot_encoder"])

        # --- spatial enhancer ---
        self.spatial_enhancer = build(cfg["spatial_enhancer"])

        # --- vision path (features consumed by the spatial enhancer) ---
        self.backbone = build(cfg["backbone"])          # 2D Swin on RGB
        self.neck = build(cfg["neck"])                  # -> 256-d multi-scale maps
        self.backbone_3d = build(cfg["backbone_3d"])    # Swin on depth
        self.neck_3d = build(cfg["neck_3d"])            # -> 128-d 3D features

        # --- load pretrained weights ---
        from safetensors.torch import load_file

        state = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        _load_state_dict_prefix(
            self.robot_state_encoder, state, "decoder.robot_encoder.", "robot_state_encoder"
        )
        _load_state_dict_prefix(
            self.spatial_enhancer, state, "spatial_enhancer.", "spatial_enhancer"
        )
        _load_state_dict_prefix(self.backbone, state, "backbone.", "backbone")
        _load_state_dict_prefix(self.neck, state, "neck.", "neck")
        _load_state_dict_prefix(self.backbone_3d, state, "backbone_3d.", "backbone_3d")
        _load_state_dict_prefix(self.neck_3d, state, "neck_3d.", "neck_3d")

        self.to(device=device, dtype=dtype).eval()
        self._device = device
        self._dtype = dtype

    # ------------------------------------------------------------------
    # robot state encoding
    # ------------------------------------------------------------------
    @staticmethod
    def make_kinematics(urdf_path: str, **kwargs):
        """Create the dual-arm kinematics helper (FK + joint graph distances).

        Defaults match the RoboTwin aloha-agilex dual-arm setup
        (fl_link*/fr_link*). Returns a DualArmKinematics instance.
        """
        from robo_orchard_lab.dataset.robotwin.transforms import DualArmKinematics

        return DualArmKinematics(urdf=urdf_path, **kwargs)

    @torch.no_grad()
    def encode_robot_state(self, robot_state, joint_relative_pos, joint_mask=None):
        """Encode per-link robot states.

        Args:
            robot_state: [B, T, n_link, 8] — per link
                [theta, x, y, z, qw, qx, qy, qz] (theta is the joint angle,
                only the gripper entries carry a meaningful value).
            joint_relative_pos: [n_link, n_link] pairwise joint-index
                distance matrix (kinematics.joint_relative_pos).
            joint_mask: optional [B, T, n_link] mask for apply_joint_mask.

        Returns:
            [B, n_link, T, embed_dims] link embeddings.
        """
        robot_state = robot_state.to(self._device, self._dtype)
        joint_relative_pos = joint_relative_pos.to(self._device)
        if joint_mask is not None:
            joint_mask = joint_mask.to(self._device)
        return self.robot_state_encoder(robot_state, joint_relative_pos, joint_mask)

    @torch.no_grad()
    def encode_joint_angles(self, joint_state, kinematics, joint_mask=None,
                            embodiedment_mat=None):
        """Encode raw joint angles [B, T, n] (or [B, n]) via FK + state encoder.

        Layout: [left_arm, left_gripper, right_arm, right_gripper].
        embodiedment_mat: optional [4, 4] base->ego transform applied to the
        FK link poses (HoloBrain trains the state encoder on ego-frame poses).
        Returns [B, n_link, T, embed_dims].
        """
        if joint_state.dim() == 2:
            joint_state = joint_state[:, None]
        if embodiedment_mat is not None:
            embodiedment_mat = torch.as_tensor(
                embodiedment_mat, dtype=torch.float32
            )
        robot_state = kinematics.joint_state_to_robot_state(
            joint_state, embodiedment_mat
        )
        # HoloBrain parity (processor joint_mask + AddScaleShift): the robot
        # state encoder sees arm thetas masked to -1 (theta is redundant
        # with the FK pose channels) and the gripper scalar normalized
        # (g - 0.5) / 0.5 -> [-1, 1] (both processor JSONs use [0.5, 0.5]
        # on the gripper entries).
        n_arm = len(kinematics.arm_link_keys[0])
        n_per_arm = n_arm + 1
        n_link = robot_state.shape[-2]
        assert n_link == 2 * n_per_arm, (
            f"expected [arm x {n_arm} + gripper] x 2 links, got {n_link}"
        )
        idx = torch.arange(n_link, device=robot_state.device)
        is_arm = (idx % n_per_arm) < n_arm
        robot_state = robot_state.clone()
        robot_state[..., ~is_arm, 0] = (robot_state[..., ~is_arm, 0] - 0.5) / 0.5
        if joint_mask is None:
            joint_mask = is_arm.unsqueeze(0).expand(robot_state.shape[0], -1)
        return self.encode_robot_state(
            robot_state, kinematics.joint_relative_pos, joint_mask
        )

    # ------------------------------------------------------------------
    # spatial encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_vision_features(self, imgs, depths):
        """Run the 2D/3D vision backbones.

        Args:
            imgs:   [B, n_cam, 3, H, W] RGB (model-normalized) images.
            depths: [B, n_cam, 1, H, W] metric depth maps.

        Returns:
            feature_maps: list of [B, n_cam, 256, h_i, w_i] (3 scales).
            feature_3d:   list of [B, n_cam, 128, h_i, w_i] (3 scales).
        """
        bs, n_cam = imgs.shape[:2]
        imgs = imgs.flatten(0, 1).to(self._device, self._dtype)
        depths = depths.flatten(0, 1).to(self._device, self._dtype)

        feature_maps = self.backbone(imgs)
        if self.neck is not None:
            feature_maps = self.neck(feature_maps)
        feature_maps = [x.unflatten(0, (bs, n_cam)) for x in feature_maps]

        feature_3d = self.backbone_3d(depths)
        if self.neck_3d is not None:
            feature_3d = self.neck_3d(feature_3d)
        feature_3d = [x.unflatten(0, (bs, n_cam)) for x in feature_3d]
        return feature_maps, feature_3d

    @torch.no_grad()
    def encode_spatial(self, feature_maps, image_wh, projection_mat, feature_3d=None):
        """Fuse vision features with camera-geometry priors.

        Args:
            feature_maps: list of [B, n_cam, 256, h_i, w_i] multi-scale maps.
            image_wh:     [n_cam, 2] image (width, height).
            projection_mat: [B, n_cam, 4, 4] camera projection
                (intrinsic @ extrinsic, mapping world -> image).
            feature_3d:   list of [B, n_cam, 128, h_i, w_i] depth features
                (required unless the enhancer was configured with
                with_feature_3d=False).

        Returns:
            fused:      list of [B, n_cam, 256, h_i, w_i] geometry-aware maps.
            depth_prob: predicted depth distribution per feature location.
        """
        inputs = {
            "image_wh": torch.as_tensor(image_wh).to(self._device),
            "projection_mat": torch.as_tensor(projection_mat).to(
                self._device, self._dtype
            ),
        }
        fused, depth_prob, _ = self.spatial_enhancer(
            feature_maps=feature_maps,
            inputs=inputs,
            feature_3d=feature_3d,
        )
        return fused, depth_prob

    @torch.no_grad()
    def encode_visual(self, imgs, depths, image_wh, projection_mat):
        """End-to-end: raw RGB + depth + camera params -> spatial features."""
        feature_maps, feature_3d = self.extract_vision_features(imgs, depths)
        return self.encode_spatial(feature_maps, image_wh, projection_mat, feature_3d)

    @torch.no_grad()
    def encode_visual_tokens(
        self,
        imgs,
        depths,
        image_wh,
        projection_mat,
        out_extrinsic=None,
        feature_level=(1, 2),
    ):
        """Observation -> (vision tokens, token 3D positions), efficiently.

        Only the scales in `feature_level` are passed through the spatial
        enhancer (default [1, 2], matching the HoloBrain GD decoder) — the
        unused stride-8 map (~75% of all tokens) and stride-64 map are
        dropped before enhancement, not after.

        Args:
            imgs:   [B, n_cam, 3, H, W] RGB (model-normalized) images.
            depths: [B, n_cam, 1, H, W] metric depth maps.
            image_wh:     [n_cam, 2] image (width, height).
            projection_mat: [B, n_cam, 4, 4] camera projection
                (intrinsic @ extrinsic, mapping world -> image).
            out_extrinsic: optional world->camera [4, 4] (or [3, 4])
                transform; pass the third-person (head) camera's extrinsic
                to get positions in that camera frame.
            feature_level: scale indices to use (default (1, 2)).

        Returns:
            tokens:   [B, n_cam, N, 256] vision representation, scales
                concatenated finest-selected first, row-major per scale.
            pos_mean: [B, n_cam, N, 3] depth-weighted 3D position per token.
            pos_max:  [B, n_cam, N, 3] argmax-depth 3D position per token.
        """
        feature_level = list(feature_level)
        feature_maps, feature_3d = self.extract_vision_features(imgs, depths)

        # drop unused strides BEFORE the enhancer (token ops are per-scale)
        feature_maps = [feature_maps[i] for i in feature_level]
        if feature_3d is not None:
            feature_3d = [feature_3d[i] for i in feature_level]

        fused, depth_prob = self.encode_spatial(
            feature_maps, image_wh, projection_mat, feature_3d
        )

        # flatten scales to tokens, row-major within each scale
        tokens = torch.cat(
            [f.flatten(3).transpose(2, 3) for f in fused], dim=2
        )  # [B, n_cam, N, C]

        pos_mean, pos_max = self.spatial_token_positions(
            depth_prob, image_wh, projection_mat, fused,
            out_extrinsic=out_extrinsic,
        )
        return tokens, pos_mean, pos_max

    @torch.no_grad()
    def spatial_token_positions(
        self,
        depth_prob,
        image_wh,
        projection_mat,
        fused,
        out_extrinsic=None,
        feature_level=None,
    ):
        """3D position of each spatial token.

        Each token has `num_depth` candidate points along its camera ray
        (the depth anchors, min_depth..max_depth), back-projected through
        the camera projection matrix. Two reductions are returned:

        Args:
            out_extrinsic: optional world->camera [4, 4] (or [3, 4])
                transform. Pass the third-person (head) camera's extrinsic
                to get all positions in that camera's frame instead of the
                world frame.
            feature_level: optional int or list of ints. If given, only
                return positions of tokens from those scales of `fused`
                (e.g. [1, 2] to match the HoloBrain GD decoder's choice).

        Returns:
            pos_mean: [B, n_cam, N, 3] depth-probability-weighted mean position.
            pos_max:  [B, n_cam, N, 3] position at the argmax depth bin.

        Token order matches the flattened feature maps: scale by scale
        (finest first), row-major within each scale — the same order as
        depth_prob's token dimension.
        """
        spatial_shapes = torch.tensor(
            [[f.shape[-2], f.shape[-1]] for f in fused],
            device=depth_prob.device,
        )
        image_wh = torch.as_tensor(image_wh).to(depth_prob.device)
        projection_mat = torch.as_tensor(projection_mat).to(depth_prob)
        pts = self.spatial_enhancer.get_pts(
            spatial_shapes, image_wh, projection_mat,
            depth_prob.device, depth_prob.dtype,
        )  # [B, n_cam, N, num_depth, 3]
        pos_mean = (depth_prob.unsqueeze(-1) * pts).sum(dim=-2)
        idx = depth_prob.argmax(dim=-1, keepdim=True)  # [B, n_cam, N, 1]
        pos_max = pts.gather(
            3, idx.unsqueeze(-1).expand(*idx.shape, 3)
        ).squeeze(3)
        if feature_level is not None:
            if isinstance(feature_level, int):
                feature_level = [feature_level]
            sizes = [f.shape[-2] * f.shape[-1] for f in fused]
            offsets = [0]
            for s in sizes:
                offsets.append(offsets[-1] + s)
            sel = torch.cat(
                [
                    torch.arange(offsets[i], offsets[i + 1], device=pts.device)
                    for i in feature_level
                ]
            )
            pos_mean = pos_mean.index_select(2, sel)
            pos_max = pos_max.index_select(2, sel)
        if out_extrinsic is not None:
            T = torch.as_tensor(out_extrinsic).to(depth_prob)
            if T.shape == (3, 4):
                T = torch.cat(
                    [T, torch.tensor([0.0, 0.0, 0.0, 1.0]).to(T)[None]], dim=0
                )
            pos_mean = pos_mean @ T[:3, :3].T + T[:3, 3]
            pos_max = pos_max @ T[:3, :3].T + T[:3, 3]
        return pos_mean, pos_max
