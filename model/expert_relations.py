"""Expert-distribution and relation-matrix utilities for aggregation."""

import torch


def flatten_model_params(model: torch.nn.Module) -> torch.Tensor:
    """Flatten all parameters of one expert into a single vector."""
    vec = []
    for p in model.parameters():
        vec.append(p.detach().reshape(-1))
    return torch.cat(vec, dim=0)


def compute_expert_weight_distribution(
    routing_weights_list,
    temperature: float = 0.7,
    eps: float = 1e-12,
):
    """Return raw and smoothed routing distributions over experts.

    Each input tensor has shape ``[batch, experts]`` and both outputs have
    shape ``[experts]``.
    """
    assert len(routing_weights_list) > 0, "routing_weights_list cannot be empty."

    usage = None
    total_samples = 0

    for alpha in routing_weights_list:
        if alpha.dim() != 2:
            raise ValueError(f"Expected alpha shape [B, M], got {alpha.shape}")
        batch_sum = alpha.sum(dim=0)  # [M]
        usage = batch_sum if usage is None else usage + batch_sum
        total_samples += alpha.size(0)

    usage = usage / max(total_samples, 1)
    p_raw = usage / (usage.sum() + eps)

    p_smooth = torch.pow(torch.clamp(p_raw, min=eps), temperature)
    p_smooth = p_smooth / (p_smooth.sum() + eps)

    return p_raw.detach(), p_smooth.detach()


def compute_expert_relation_matrix(
    experts,
    curvatures,
    alpha: float = 0.5,
    eps: float = 1e-12,
    normalize: bool = True,
):
    """Return pairwise parameter and curvature distances between experts.

    The symmetric result uses ``alpha`` to blend the two distances and is
    normalized to ``[0, 1]`` when requested.
    """
    M = len(experts)
    assert M > 0
    assert len(curvatures) == M

    param_vecs = []
    for expert in experts:
        v = flatten_model_params(expert)
        v = v / (torch.norm(v, p=2) + eps)
        param_vecs.append(v)

    device = param_vecs[0].device
    curv_tensor = torch.tensor(
        [k.item() if hasattr(k, 'item') else float(k) for k in curvatures],
        dtype=torch.float32, device=device
    )

    C = torch.zeros(M, M, device=device)
    for i in range(M):
        for j in range(M):
            d_param = torch.sum((param_vecs[i] - param_vecs[j]) ** 2)
            d_curv = (curv_tensor[i] - curv_tensor[j]) ** 2
            C[i, j] = alpha * d_param + (1.0 - alpha) * d_curv

    C = 0.5 * (C + C.t())
    C.fill_diagonal_(0.0)

    if normalize:
        maxv = C.max()
        if maxv > eps:
            C = C / maxv

    return C.detach()


def low_rank_factor(C: torch.Tensor, rank: int):
    """Return a rank-limited factor and reconstruction for ``C ~= L L^T``."""
    U, S, _ = torch.linalg.svd(C, full_matrices=False)
    r = min(rank, S.numel())

    U_r = U[:, :r]
    S_r = S[:r]
    L = U_r * torch.sqrt(torch.clamp(S_r, min=0.0)).unsqueeze(0)

    C_hat = L @ L.t()
    C_hat = 0.5 * (C_hat + C_hat.t())
    C_hat.fill_diagonal_(0.0)

    return L.detach(), C_hat.detach()


def reconstruct_relation_matrix_from_factor(L) -> torch.Tensor:
    """Reconstruct ``C_hat = L L^T`` from a tensor or NumPy factor."""
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, dtype=torch.float32)
    C_hat = L @ L.t()
    C_hat = 0.5 * (C_hat + C_hat.t())
    C_hat.fill_diagonal_(0.0)
    return C_hat
