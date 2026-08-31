"""Frozen X-VLA representation encoder for the CEHJ baseline.

Replaces HoloBrainEncoder: per control tick, 3 camera views (224x224,
ImageNet-normalized) + a fixed task instruction go through X-VLA's
Florence-2 VLM (encoder-only), giving:

  vlm_features [1, 100, 1024]   view-0 image tokens fused with text
  aux_visual   [1, 100, 1024]   views 1-2 image tokens

We concatenate them -> [1, 200, 1024], run a LEARNED adapter down to the
project's 256-d token width, and hand over tokens + zero positions (X-VLA
carries no metric 3D — geometry-free baseline).

Robot state: X-VLA's own proprio layout (EE6D per arm: xyz + rot6d + grip),
20-dim, encoded by a small learned MLP to per-arm state tokens.

Nothing here touches the action head or the transformer trunk of X-VLA —
only the frozen VLM encoder is used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

XVLA_DIM = 1024
PROPRIO_DIM = 20  # ee6d x 2 arms


class XVLAEncoder(nn.Module):
    """Frozen X-VLA VLM + learned adapter to the project token width."""

    def __init__(self, ckpt_dir: str, device: str = "cuda",
                 out_dim: int = 256, instruction: str = ""):
        super().__init__()
        import sys
        import types
        from pathlib import Path

        ckpt_dir = str(Path(ckpt_dir).resolve())
        # the checkpoint's modeling_xvla.py uses RELATIVE imports
        # (from .modeling_florence2 import ...) — import it as a package
        # anchored at the ckpt dir, not as a top-level sys.path module
        pkg_name = "xvla_ckpt"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [ckpt_dir]
            sys.modules[pkg_name] = pkg
        from xvla_ckpt import modeling_florence2 as _mf2
        from xvla_ckpt.modeling_xvla import XVLA
        from xvla_ckpt.processing_xvla import XVLAProcessor

        # transformers >= 4.52 compatibility shims for the checkpoint's
        # vendored Florence-2 code (X-VLA pins transformers<=4.51.3; the
        # RoboTwin env has 4.57):
        # - _supports_sdpa/_supports_flash_attn_2 are PROPERTIES reading
        #   self.language_model, which does not exist yet when
        #   PreTrainedModel.__init__ queries them — shadow with plain False
        #   class attrs (eager attention is fine for a frozen encoder)
        # - XVLA.__init__ deletes lm.lm_head (encoder-only); 4.57's
        #   tie_weights then crashes in get_output_embeddings — make it
        #   lm_head-safe (None is handled by tie_embeddings_and_encoder_decoder)
        _mf2.Florence2PreTrainedModel._supports_sdpa = False
        _mf2.Florence2PreTrainedModel._supports_flash_attn_2 = False
        _mf2.Florence2LanguageForConditionalGeneration.get_output_embeddings = (
            lambda self: getattr(self, "lm_head", None)
        )

        self.processor = XVLAProcessor.from_pretrained(ckpt_dir)
        self.model = XVLA.from_pretrained(
            ckpt_dir, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._device = device
        self.model.to(device)

        # learned adapter: X-VLA 1024-d tokens -> project 256-d
        self.adapter = nn.Sequential(
            nn.Linear(XVLA_DIM, out_dim),
            nn.RMSNorm(out_dim),
        ).to(device)
        # proprio encoder: raw EE6D x2 -> per-arm tokens
        self.proprio_enc = nn.Sequential(
            nn.Linear(PROPRIO_DIM // 2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        ).to(device)
        self.instruction = instruction
        self._input_ids = None

    def set_instruction(self, text: str) -> None:
        self.instruction = text
        self._input_ids = None

    @torch.no_grad()
    def _encode_text(self):
        if self._input_ids is None:
            out = self.processor.encode_language(self.instruction or "")
            self._input_ids = out["input_ids"].to(self._device)
        return self._input_ids

    @torch.no_grad()
    def encode_scene(self, images_uint8: list[np.ndarray]):
        """3 camera RGB uint8 frames -> (tokens [1, 200, out_dim], pos zeros).

        images in X-VLA view order: [head, left_wrist, right_wrist] — view 0
        is the text-fused one (the head camera).
        """
        from PIL import Image

        proc = self.processor.encode_image(
            [Image.fromarray(img) for img in images_uint8]
        )
        out = self.model.forward_vlm(
            self._encode_text(),
            proc["image_input"].to(self._device),
            proc["image_mask"].to(self._device),
        )
        feats = torch.cat(
            [out["vlm_features"], out["aux_visual_inputs"]], dim=1
        )                                     # [1, 200, 1024]
        tokens = self.adapter(feats)          # [1, 200, out_dim]
        pos = torch.zeros(1, tokens.shape[1], 3, device=self._device)
        return tokens, pos

    def encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """proprio [B, 20] (ee6d x2 arms) -> [B, 2, out_dim] per-arm tokens."""
        B = proprio.shape[0]
        per_arm = proprio.view(B, 2, PROPRIO_DIM // 2).to(self._device)
        return self.proprio_enc(per_arm)      # [B, 2, out_dim]
