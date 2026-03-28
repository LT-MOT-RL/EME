import math
import torch
import torch.nn as nn

# Import the core Mamba module directly to avoid pulling transformer deps.
from CSSM.mamba_compat import Mamba



def safe_numerics(x, clip=1e4):
    """Clamp/clean NaN/inf to keep downstream ops stable."""
    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(x, -clip, clip)


def sinusoidal_pos_enc(length, dim, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe  # [length, dim]


class ConvTemporalGraphical(nn.Module):
    """Graph convolution over spatial adjacency + temporal conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        print("x : ",x.shape)
        x = self.conv(x)
        print("x : ",x.shape)
        print("A : ",A.shape)
        input("##")
        x = torch.einsum("nctv,tvw->nctw", (x, A))
        return x.contiguous(), A







class ConvTemporalGraphicalAggressive(nn.Module):
    """Graph conv with input-conditioned temporal coupling matrix A (more aggressive)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,      # T
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
        hidden=32,        # A-generator hidden dim
        temperature=1.0,  # softmax temperature (lower => sharper)
        use_identity_bias=True,
        identity_bias=1.0 # strength of identity bias if enabled
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.use_identity_bias = use_identity_bias
        self.identity_bias = identity_bias

        # temporal conv (same as your original)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

        # A-generator: produce 4 logits per time step (2x2)
        # We pool over V=2 to get a per-time-step descriptor, then MLP -> 4 logits.
        self.a_mlp = nn.Sequential(
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
        )

        # identity bias buffer
        self.register_buffer("I", torch.eye(2).unsqueeze(0))  # [1, 2, 2]

    def forward(self, x, Temp):
        """
        x: [N, C_in, T, V],  V must be 2
        Returns:
            x_out: [N, C_out, T, 2]
            A_eff: [N, T, 2, 2] (input-conditioned)
        """
        assert x.size(-1) == 2, "This aggressive version assumes V=2."
        assert x.size(2) == self.kernel_size, "T mismatch with kernel_size."

        # 1) temporal conv
        x = self.conv(x)  # [N, C_out, T, 2]
        N, C, T, V = x.shape


        # 2) build input-conditioned A
        # Pool over V to get per-time-step feature: [N, C, T]
        x_pool = x.mean(dim=-1)              # [N, C, T]
        x_pool = x_pool.permute(0, 2, 1)     # [N, T, C]

        # MLP per time step -> 4 logits -> [N, T, 2, 2]
        A_logits = self.a_mlp(x_pool).view(N, T, 2, 2)

        # Optional identity bias (keeps training sane but still more aggressive than fixed-A)
        if self.use_identity_bias:
            A_logits = A_logits + self.identity_bias * self.I  # broadcast to [N,T,2,2]

        # Row-softmax with temperature for stability/control
        A_eff = torch.softmax(A_logits / self.temperature, dim=-1)  # [N, T, 2, 2]

        # 3) mixing: now A is per-sample, so einsum signature changes
        # x: [N,C,T,V], A_eff: [N,T,V,W] -> out: [N,C,T,W]
        x = torch.einsum("nctv,ntvw->nctw", x, A_eff)

        return x.contiguous(), A_eff










class st_mamba(nn.Module):
    """ST block: graph conv + Mamba (time) with residual."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_mdn=False,
        stride=1,
        dropout=0.0,
        d_state=16,
        d_conv=4,
        expand=2,
        apply_gcn=True,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        self.use_mdn = use_mdn
        self.apply_gcn = apply_gcn

        self.gcn = ConvTemporalGraphicalAggressive(in_channels, out_channels, kernel_size[1])
        self.mamba = Mamba(
            d_model=out_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

        if stride != 1:
            # temporal stride via pooling on residual path
            self.time_pool = nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))
        else:
            self.time_pool = None

        if in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        if self.apply_gcn:
            x, A = self.gcn(x, A)

        n, c, t, v = x.shape
        print("x before permute: ",x.shape)
        x = x.permute(0, 3, 2, 1).contiguous().view(n * v, t, c)  # (B, T, C)

        print("x before mamba: ",x.shape)
        x = self.mamba(x)
        print("x after mamba: ",x.shape)

        x = safe_numerics(x)
        x = x.view(n, v, t, c).permute(0, 3, 2, 1).contiguous()
        print("x after permute: ",x.shape)

        x = self.dropout(x)

        if self.time_pool is not None:
            x = self.time_pool(x)
            res = self.time_pool(res)
        print("x after time_pool: ",x.shape)
        x = x + res
        if not self.use_mdn:
            x = self.prelu(x)
        print("x after res: ",x.shape)
        input("##")
        # x before permute:  torch.Size([1, 5, 6, 2])
        # x before mamba:  torch.Size([2, 6, 5])
        # x after mamba:  torch.Size([2, 6, 5])
        # x after permute:  torch.Size([1, 5, 6, 2])
        # x after time_pool:  torch.Size([1, 5, 6, 2])
        # x after res:  torch.Size([1, 5, 6, 2])




        return x, A


class MambaTemporal(nn.Module):
    """Temporal Mamba + linear time projection."""

    def __init__(self, in_steps, out_steps, channels, dropout=0.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.time_proj = nn.Linear(in_steps, out_steps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, T, C, V)

        print("### MambaTemporal ####")
        n, t, c, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(n * v, t, c)  # (N*V, T, C)
        x = self.mamba(x)
        x = safe_numerics(x)
        x = x.transpose(1, 2)  # (N*V, C, T)

        print("x before time_proj: ",x.shape)
        x = self.time_proj(x)  # linear over time dim
        print("x after time_proj: ",x.shape)

        # x before time_proj:  torch.Size([2, 5, 6])
        # x after time_proj:  torch.Size([2, 5, 4])

        x = self.dropout(x)
        x = x.transpose(1, 2)  # (N*V, T_out, C)
        x = x.view(n, v, -1, c).permute(0, 2, 3, 1).contiguous()  # (N, T_out, C, V)
        x = safe_numerics(x)
        return x


class social_stgcnn(nn.Module):
    """Social-STGCNN variant using Mamba-SSM for spatial-temporal modeling."""

    def __init__(
        self,
        n_stgcnn=1,
        n_txpcnn=1,
        input_feat=2,
        output_feat=5,
        seq_len=8,
        pred_seq_len=12,
        kernel_size=3,
        dropout=0.0,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(
            st_mamba(
                input_feat,
                output_feat,
                (kernel_size, seq_len),
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                apply_gcn=True,
            )
        )
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(
                st_mamba(
                    output_feat,
                    output_feat,
                    (kernel_size, seq_len),
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    apply_gcn=False,  # only first block applies spatial graph conv
                )
            )

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(
            MambaTemporal(
                seq_len,
                pred_seq_len,
                output_feat,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        )
        for _ in range(1, self.n_txpcnn):
            self.tpcnns.append(
                MambaTemporal(
                    pred_seq_len,
                    pred_seq_len,
                    output_feat,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
        # Lightweight task head: 3x3 conv over (pred_seq_len, V) with channel mixing.
        self.tpcnn_output = nn.Sequential(
            nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=pred_seq_len),
            nn.PReLU(),
        )
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(self.n_txpcnn)])

    def forward(self, v, a):
        # v: (N, C, T, V), a: (K, V, V)
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])  # (N, T, C, V)
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        print("v before output conv: ",v.shape)
        v = self.tpcnn_output(v)
        print("v after output conv: ",v.shape)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])  # (N, C, T, V)
        print("v shape: ",v.shape)
        v = safe_numerics(v)

        # v before output conv:  torch.Size([1, 4, 5, 2])
        # v after output conv:  torch.Size([1, 4, 5, 2])
        # v shape:  torch.Size([1, 5, 4, 2])

        return v, a
