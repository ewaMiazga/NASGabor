import torch


def get_basis_parameterized(cosθ, cosϕ, cosτ):
    clamp_min = -0.999999
    clamp_max = 0.999999
    cosθ = torch.clamp(cosθ, clamp_min, clamp_max)
    cosϕ = torch.clamp(cosϕ, clamp_min, clamp_max)
    cosτ = torch.clamp(cosτ, clamp_min, clamp_max)

    sinθ = torch.sqrt(1.0 - cosθ * cosθ)
    sinϕ = torch.sqrt(1.0 - cosϕ * cosϕ)
    sinτ = torch.sqrt(1.0 - cosτ * cosτ)

    x = torch.stack(
        [
            cosθ * cosϕ * cosτ - sinθ * sinτ,
            sinθ * cosϕ * cosτ + cosθ * sinτ,
            -sinϕ * cosτ,
        ],
        dim=2,
    )

    z = torch.stack([cosθ * sinϕ, sinθ * sinϕ, cosϕ], dim=2)
    return [x, z]
