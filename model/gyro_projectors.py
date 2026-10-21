"""Riemannian GyroKernel projectors for aligning expert representations."""

import math
from typing import Optional

import torch
import torch.nn as nn

from geoopt import ManifoldParameter, ManifoldTensor
from geoopt.manifolds.stereographic.math import project as stereographic_project


class _MoEEncoderReferenceMixin:
    """Keep the projector-to-expert link out of PyTorch module registration."""

    def __setattr__(self, name, value):
        if name == "moe_encoder":
            self.bind_moe_encoder(value)
            return
        super().__setattr__(name, value)

    @property
    def moe_encoder(self):
        return self._moe_encoder_ref

    def bind_moe_encoder(self, moe_encoder) -> None:
        experts = getattr(moe_encoder, "experts", None)
        if experts is None:
            raise TypeError("moe_encoder must expose an experts collection")
        if len(experts) != self.num_experts:
            raise ValueError(
                f"Expected {self.num_experts} experts, received {len(experts)}"
            )

        # Bypass nn.Module.__setattr__: the encoder is owned by the enclosing
        # model and is only a live curvature reference for this projector.
        object.__setattr__(self, "_moe_encoder_ref", moe_encoder)


class RiemannianGyroKernelProjector(_MoEEncoderReferenceMixin, nn.Module):
    """Map each expert's manifold features independently to Euclidean space.

    Input and output use ``[nodes, features, experts]``. Expert curvatures are
    read live during each forward pass.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        moe_encoder,
        hidden_dim: Optional[int] = None,
        use_residual: bool = True,  # kept for API compatibility, unused
        dropout: float = 0.0,       # kept for API compatibility, unused
        norm_type: str = "layernorm",
        eps: float = 1e-8,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.eps = eps
        self.num_experts = len(moe_encoder.experts)

        self.bind_moe_encoder(moe_encoder)

        # Per-expert learnable gyro-kernel parameters (each [hidden_dim, hidden_dim])
        self.rm_features = nn.ParameterList()
        self.Ws = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.lambdas = nn.ParameterList()

        for _ in range(self.num_experts):
            feat = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
            nn.init.uniform_(feat, -1e-2, 1e-2)

            pre_w = torch.randn(self.hidden_dim, self.hidden_dim)
            pre_w = pre_w / (torch.norm(pre_w, dim=-1, keepdim=True) + eps)
            w = nn.Parameter(pre_w)

            b = nn.Parameter(2 * math.pi * torch.rand(self.hidden_dim, self.hidden_dim))
            lam = nn.Parameter(torch.rand(self.hidden_dim, self.hidden_dim))

            self.rm_features.append(feat)
            self.Ws.append(w)
            self.biases.append(b)
            self.lambdas.append(lam)

        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.norm = nn.Identity()

    def _curvature_distance(
        self, x: torch.Tensor, w: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """Return the gyro-distance matrix for two feature matrices."""
        if torch.abs(k).item() < 1e-12:
            return x @ w.t()

        x_sq = (x * x).sum(dim=-1, keepdim=True)          # [D, 1]
        div = ((x[:, None, :] - w[None, :, :]) ** 2).sum(dim=-1)  # [D, D]
        numerator = 1.0 + k * x_sq
        return torch.log(torch.abs(numerator / (div + self.eps)) + self.eps)

    def general_fourier_mapping(self):
        """Return one gyro-kernel matrix per live expert curvature."""
        kernel_matrices = []
        for i in range(self.num_experts):
            curvature = self.moe_encoder.experts[i].manifold.k
            feature_anchor = self.rm_features[i]
            weight = self.Ws[i]
            bias = self.biases[i]
            frequency = self.lambdas[i]

            distance = self._curvature_distance(
                feature_anchor, weight, curvature
            )
            dimension = feature_anchor.shape[-1]
            kernel = torch.exp((dimension - 1) * distance / 2.0) * torch.cos(
                frequency * distance + bias
            )
            kernel_matrices.append(kernel)
        return kernel_matrices

    def forward(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Project expert outputs while preserving the expert dimension."""

        kernel_matrices = self.general_fourier_mapping()

        projected_outputs = []
        for i in range(self.num_experts):
            expert_output = expert_outputs[:, :, i]
            projected = self.norm(expert_output @ kernel_matrices[i])
            projected_outputs.append(projected.unsqueeze(-1))

        return torch.cat(projected_outputs, dim=-1)


class RiemannianGyroKernelProjectorStrict(_MoEEncoderReferenceMixin, nn.Module):
    """Project experts with geometry-constrained gyro-kernel anchors.

    Anchors are bound to their associated expert manifold and reprojected on
    every forward pass. Input and output follow the per-expert convention:
    [N, in_dim, num_experts] -> [N, hidden_dim, num_experts].
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        moe_encoder,
        hidden_dim: Optional[int] = None,
        use_residual: bool = True,   # kept for API compatibility, unused
        dropout: float = 0.0,        # kept for API compatibility, unused
        norm_type: str = "layernorm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        self.eps = eps
        self.num_experts = len(moe_encoder.experts)

        self.bind_moe_encoder(moe_encoder)

        # Keep one manifold-constrained anchor matrix per expert.
        self.rm_features = nn.ParameterList()
        for i in range(self.num_experts):
            manifold = moe_encoder.experts[i].manifold
            tensor = ManifoldTensor(
                torch.empty(self.hidden_dim, self.hidden_dim),
                manifold=manifold,
            )
            feat = ManifoldParameter(tensor)
            self._init_manifold_feature(feat, manifold)
            self.rm_features.append(feat)

        self.Ws = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(self.num_experts):
            pre_w = torch.randn(self.hidden_dim, self.hidden_dim)
            pre_w = pre_w / (torch.norm(pre_w, dim=-1, keepdim=True) + eps)
            self.Ws.append(nn.Parameter(pre_w))
            # Bias is shared across the second kernel dimension.
            self.biases.append(nn.Parameter(2 * math.pi * torch.rand(self.hidden_dim)))

        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.norm = nn.Identity()

    def bind_moe_encoder(self, moe_encoder) -> None:
        super().bind_moe_encoder(moe_encoder)

        # Rebind copied anchors before RiemannianAdam sees them. This keeps the
        # projector anchors and their corresponding expert on the same manifold.
        if hasattr(self, "rm_features"):
            for expert, feature in zip(moe_encoder.experts, self.rm_features):
                if isinstance(feature, ManifoldParameter):
                    feature.manifold = expert.manifold

    @staticmethod
    def _init_manifold_feature(param, manifold, scale: float = 1e-2):
        """Initialize an anchor safely within the current manifold radius."""
        with torch.no_grad():
            param.data.uniform_(-scale, scale)

            k = manifold.k
            k_val = float(k.detach().abs().item()) if torch.is_tensor(k) else abs(float(k))
            if k_val < 1e-8:
                return

            w_norm = param.data.norm(p=2, dim=-1, keepdim=True) + 1e-6
            # Initializers run before the enclosing model is moved to its target device.
            radius = manifold.radius.to(param.data.device)
            scale_factor = radius * 0.9 * torch.rand(1, device=param.data.device)
            param.data = param.data / w_norm * scale_factor

    def _curvature_distance(self, x: torch.Tensor, w: torch.Tensor, k) -> torch.Tensor:
        """Return gyro distance, with an inner-product limit near zero curvature."""
        k_val = float(k.detach().abs().item()) if torch.is_tensor(k) else abs(float(k))
        if k_val < 1e-12:
            return x @ w.t()

        x_sq = (x * x).sum(dim=-1, keepdim=True)                  # [D, 1]
        div = ((x[:, None, :] - w[None, :, :]) ** 2).sum(dim=-1)  # [D, D]
        return torch.log((1.0 + k * x_sq) / (div + self.eps) + self.eps)

    def general_fourier_mapping(self):
        """Return one geometry-constrained kernel matrix per expert."""
        kernel_matrices = []
        for i in range(self.num_experts):
            manifold = self.moe_encoder.experts[i].manifold
            curvature = manifold.k
            feature_anchor = self.rm_features[i]

            # Keep the anchor on the manifold defined by the current curvature.
            projected_anchor = stereographic_project(
                feature_anchor, k=curvature
            )
            weight = self.Ws[i]
            bias = self.biases[i]

            distance = self._curvature_distance(
                projected_anchor, weight, curvature
            )
            dimension = projected_anchor.shape[-1]
            kernel = torch.exp((dimension - 1) * distance / 2.0) * torch.cos(
                distance + bias
            )
            kernel_matrices.append(kernel)
        return kernel_matrices

    def forward(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Project expert outputs while preserving the expert dimension."""
        kernel_matrices = self.general_fourier_mapping()

        projected_outputs = []
        for i in range(self.num_experts):
            expert_output = expert_outputs[:, :, i]
            projected = self.norm(expert_output @ kernel_matrices[i])
            projected_outputs.append(projected.unsqueeze(-1))

        return torch.cat(projected_outputs, dim=-1)
