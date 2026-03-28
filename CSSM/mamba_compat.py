# mamba_compat.py
# A pure-PyTorch compatibility implementation aligned with mamba-ssm==1.1.4
# - Preserves parameter names/shapes for strict state_dict loading.
# - Avoids all custom CUDA/Triton extensions (P100-safe).
#
# Intended usage:
#   from mamba_compat import Mamba
#   self.mamba = Mamba(..., use_fast_path=False)  # use_fast_path kept for API compat

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    # Inverse of softplus; stable variant used in official code.
    # y + log(-expm1(-y))
    return y + torch.log(-torch.expm1(-y))


def selective_scan_ref(
    x: torch.Tensor,                      # (B, d_inner, L)
    dt: torch.Tensor,                     # (B, d_inner, L)  pre-bias
    A: torch.Tensor,                      # (d_inner, d_state)
    B: torch.Tensor,                      # (B, d_state, L)
    C: torch.Tensor,                      # (B, d_state, L)
    D: torch.Tensor,                      # (d_inner,)
    z: Optional[torch.Tensor] = None,     # (B, d_inner, L)
    delta_bias: Optional[torch.Tensor] = None,  # (d_inner,)
    delta_softplus: bool = True,
    return_last_state: bool = False,
) -> torch.Tensor:
    """
    Reference selective scan (pure PyTorch, sequential in L).
    Matches the functional intent of mamba_ssm slow path.

    Returns:
      y: (B, d_inner, L) or (y, last_state) if return_last_state
    """
    Bsz, d_inner, L = x.shape
    d_state = A.shape[1]
    dtype = x.dtype
    device = x.device

    # Do scan in fp32 for numerical stability, like official code keeps A_log/D in fp32.
    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    D_f = D.float()
    B_f = B.float()
    C_f = C.float()
    z_f = z.float() if z is not None else None

    if delta_bias is not None:
        bias = delta_bias.float().view(1, d_inner)  # (1, d_inner)
    else:
        bias = None

    state = torch.zeros(Bsz, d_inner, d_state, device=device, dtype=torch.float32)
    ys = []

    for t in range(L):
        dt_t = dt_f[:, :, t]  # (B, d_inner)
        if bias is not None:
            dt_t = dt_t + bias
        if delta_softplus:
            dt_t = F.softplus(dt_t)

        # Discretize A and B
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt_t, A_f))               # (B, d_inner, d_state)
        dB = torch.einsum("bd,bn->bdn", dt_t, B_f[:, :, t])                 # (B, d_inner, d_state)

        state = state * dA + x_f[:, :, t].unsqueeze(-1) * dB

        y_t = torch.einsum("bdn,bn->bd", state, C_f[:, :, t])               # (B, d_inner)
        y_t = y_t + D_f * x_f[:, :, t]                                      # skip

        if z_f is not None:
            # Official step() uses y = y * act(z) where act is SiLU
            y_t = y_t * F.silu(z_f[:, :, t])

        ys.append(y_t)

    y = torch.stack(ys, dim=-1).to(dtype=dtype)  # (B, d_inner, L)

    if return_last_state:
        return y, state.to(dtype=dtype)
    return y


class Mamba(nn.Module):
    """
    Compatibility Mamba aligned with mamba-ssm==1.1.4 parameterization.

    Key points:
      - Parameter names/shapes match official mamba_simple.Mamba:
        in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj
      - forward uses pure PyTorch reference scan (no CUDA extensions).
      - use_fast_path exists for API compatibility but is ignored (always slow path).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,  # kept for compatibility; ignored here
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # --- Modules / parameters (names must match official) ---
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # --- Init dt_proj (match official logic) ---
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"dt_init={dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = _inverse_softplus(dt)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # for parity with official

        # --- S4D real initialization (A_log fp32, shape: (d_inner, d_state)) ---
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).view(1, -1)
        A = A.repeat(self.d_inner, 1).contiguous()
        A_log = torch.log(A)  # fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # --- D skip (fp32) ---
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device, dtype=torch.float32))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        hidden_states: (B, L, D)
        Returns: (B, L, D)
        Note: inference_params/step cache not supported in this compat implementation.
        """
        if inference_params is not None:
            raise NotImplementedError("mamba_compat.Mamba does not support inference_params/step cache.")

        B, L, D = hidden_states.shape

        # (B, L, 2*d_inner) -> (B, 2*d_inner, L)
        xz = self.in_proj(hidden_states).transpose(1, 2)

        # A: (d_inner, d_state)
        A = -torch.exp(self.A_log.float())

        # Slow path: split, depthwise causal conv, then scan
        x, z = xz.chunk(2, dim=1)  # each: (B, d_inner, L)

        # Causal depthwise conv: Conv1d already has padding=d_conv-1, crop to L
        x = self.act(self.conv1d(x)[..., :L])

        # x_proj: operate on (B*L, d_inner)
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(B * L, self.d_inner))  # (B*L, dt_rank+2*d_state)
        dt, B_in, C_in = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt: (B*L, dt_rank) -> (B*L, d_inner) -> (B, d_inner, L)
        dt = F.linear(dt, self.dt_proj.weight)  # no bias here (bias handled inside scan)
        dt = dt.view(B, L, self.d_inner).transpose(1, 2).contiguous()

        # B,C: (B*L, d_state) -> (B, d_state, L)
        B_in = B_in.view(B, L, self.d_state).transpose(1, 2).contiguous()
        C_in = C_in.view(B, L, self.d_state).transpose(1, 2).contiguous()

        y = selective_scan_ref(
            x=x,
            dt=dt,
            A=A,
            B=B_in,
            C=C_in,
            D=self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        # (B, d_inner, L) -> (B, L, d_inner) -> out
        y = y.transpose(1, 2).contiguous()
        out = self.out_proj(y)
        return out
