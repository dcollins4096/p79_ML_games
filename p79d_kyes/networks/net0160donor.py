import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn

# -------------------------
# Helpers
# -------------------------
def init_e2conv_weights(module):
    """Kaiming init for R2Conv weights (call after model instantiation)."""
    for m in module.modules():
        if isinstance(m, enn.R2Conv):
            # R2Conv stores weights in .weights (tensor)
            try:
                nn.init.kaiming_normal_(m.weights, nonlinearity='relu')
            except Exception:
                # fallback: try to access param directly if different API
                for p in m.parameters():
                    if p.ndim >= 2:
                        nn.init.kaiming_normal_(p, nonlinearity='relu')

# -------------------------
# Residual block (no SE by default)
# -------------------------
class ResidualBlockE2(nn.Module):
    def __init__( self, r2_act, in_type, out_fields, kernel_size=3, use_se=False, reduction=16, dropout_p=0.0):
        """
        out_fields: number of FieldType entries using regular_repr (this *times* group order
                    gives the number of scalar components).
        in_type: enn.FieldType for input
        """
        super().__init__()
        self.r2_act = r2_act
        self.in_type = in_type
        # use regular_repr fields for capacity
        self.out_type = enn.FieldType(r2_act, out_fields * [r2_act.regular_repr])

        # conv stack (equivariant)
        self.block = enn.SequentialModule(
            enn.R2Conv(in_type, self.out_type, kernel_size=kernel_size, padding=1, bias=False),
            enn.PointwiseBatchNorm(self.out_type),
            enn.NormNonLinearity(self.out_type),
            enn.R2Conv(self.out_type, self.out_type, kernel_size=kernel_size, padding=1, bias=False),
            enn.PointwiseBatchNorm(self.out_type),
        )

        # skip projection if needed
        if in_type != self.out_type:
            self.skip = enn.R2Conv(in_type, self.out_type, kernel_size=1, bias=False)
        else:
            self.skip = None

        # optional SE-like gating (equivariant-friendly)
        self.use_se = use_se
        if use_se:
            # gating operates on per-field magnitudes -> outputs scalar gate per field
            # store number of fields (not scalar components)
            self.num_fields = len(self.out_type)  # number of Field instances
            # small MLP on pooled magnitudes (per-field)
            self.se_fc1 = nn.Linear(self.num_fields, max(4, self.num_fields // reduction))
            self.se_fc2 = nn.Linear(max(4, self.num_fields // reduction), self.num_fields)

        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: enn.GeometricTensor):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(identity)

        out = out + identity
        out = enn.GeometricTensor(out.tensor, out.type)  # ensure proper type

        # equivariant nonlinearity already applied inside block
        # optional SE gating (equivariant): use field magnitudes
        if self.use_se:
            t = out.tensor  # [B, C, H, W] where C = sum(field sizes)
            # compute per-field magnitudes: iterate fields and sum squares across components of each field
            mags = []
            idx = 0
            for field in out.type:
                comp = field.size  # number of scalar components for this field
                # take components idx:idx+comp
                f = t[:, idx:idx+comp, :, :]
                idx += comp
                # L2 norm over component axis -> shape [B, H, W]
                f_mag = torch.sqrt((f**2).sum(dim=1, keepdim=False) + 1e-9)
                # global avg pool -> [B]
                f_pool = f_mag.mean(dim=(1,2))
                mags.append(f_pool)
            # stack -> [B, num_fields]
            mags = torch.stack(mags, dim=1)
            g = torch.relu(self.se_fc1(mags))
            g = torch.sigmoid(self.se_fc2(g))  # [B, num_fields]
            # expand gates to component channels
            idx = 0
            gated = []
            for i, field in enumerate(out.type):
                comp = field.size
                gate = g[:, i].view(-1, 1, 1, 1).expand(-1, comp, out.tensor.shape[2], out.tensor.shape[3])
                gated.append(out.tensor[:, idx:idx+comp, :, :] * gate)
                idx += comp
            t_gated = torch.cat(gated, dim=1)
            out = enn.GeometricTensor(t_gated, out.type)

        out.tensor = self.dropout(out.tensor)
        return out

# -------------------------
# Cleaned up UNet (spin-2 outputs)
# -------------------------
class MainNetE2_Clean(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,            # real/imag parts of spin-2 (E+iB)
        base_channels=16,          # like your original base_channels — we'll scale internally
        channel_mult=None,         # if None, defaults to N (makes scalar capacity ~ base_channels * N)
        use_fc_bottleneck=True,
        fc_hidden=512,
        fc_spatial=4,
        use_se=False,
        dropout_1=0.0,
        dropout_2=0.0,
        dropout_3=0.0,
        N=4
    ):
        """
        Clean equivariant U-Net:
        - channel_mult: multiplier for out_fields per block. Default = N (rotation order)
        - base_channels is the "logical" base; actual field count = base_channels * channel_mult
        """
        super().__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        self.N = N
        if channel_mult is None:
            channel_mult = max(1, N)
        self.channel_mult = channel_mult

        # input type: scalar channels
        in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # compute field counts (number of regular_repr fields)
        def f(x):
            return max(1, int(x * channel_mult))

        # Encoder: out_fields are counts of regular_repr fields
        self.enc1 = ResidualBlockE2(self.r2_act, in_type, out_fields=f(base_channels), dropout_p=dropout_1, use_se=use_se)
        self.pool1 = enn.PointwiseAvgPoolAntialiased(self.enc1.out_type, sigma=0.66, stride=2)

        self.enc2 = ResidualBlockE2(self.r2_act, self.enc1.out_type, out_fields=f(base_channels*2), dropout_p=dropout_1, use_se=use_se)
        self.pool2 = enn.PointwiseAvgPoolAntialiased(self.enc2.out_type, sigma=0.66, stride=2)

        self.enc3 = ResidualBlockE2(self.r2_act, self.enc2.out_type, out_fields=f(base_channels*4), dropout_p=dropout_1, use_se=use_se)
        self.pool3 = enn.PointwiseAvgPoolAntialiased(self.enc3.out_type, sigma=0.66, stride=2)

        self.enc4 = ResidualBlockE2(self.r2_act, self.enc3.out_type, out_fields=f(base_channels*8), dropout_p=dropout_1, use_se=use_se)
        self.pool4 = enn.PointwiseAvgPoolAntialiased(self.enc4.out_type, sigma=0.66, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlockE2(self.r2_act, self.enc4.out_type, out_fields=f(base_channels*8), dropout_p=dropout_2, use_se=use_se)

        # FC bottleneck over field magnitudes; operate on scalar component dimension
        self.use_fc_bottleneck = use_fc_bottleneck
        if use_fc_bottleneck:
            Cb = self.bottleneck.out_type.size
            self.fc1 = nn.Linear(Cb * fc_spatial * fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, Cb * fc_spatial * fc_spatial)
            self.fc_spatial = fc_spatial

        # Decoder
        # We create dec blocks with in_type matching the bottleneck out_type; their in/out field counts will be set accordingly
        self.up = None
        # try to use e2cnn upsampling if available; fallback to nn.Upsample
        if hasattr(enn, "R2Upsampling"):
            self.up = enn.R2Upsampling  # class, will be instantiated per-call
            use_e2_upsample = True
        else:
            use_e2_upsample = False

        # Decoder blocks: note we will handle skip merging via transform_to
        self.dec4 = ResidualBlockE2(self.r2_act, self.bottleneck.out_type, out_fields=f(base_channels*8), dropout_p=dropout_3, use_se=use_se)
        self.dec3 = ResidualBlockE2(self.r2_act, self.dec4.out_type, out_fields=f(base_channels*4), dropout_p=dropout_3, use_se=use_se)
        self.dec2 = ResidualBlockE2(self.r2_act, self.dec3.out_type, out_fields=f(base_channels*2), dropout_p=dropout_3, use_se=use_se)
        self.dec1 = ResidualBlockE2(self.r2_act, self.dec2.out_type, out_fields=f(base_channels), dropout_p=dropout_3, use_se=use_se)

        # Multi-scale spin-2 output heads (map field types -> spin-2)
        spin2_repr = self.r2_act.irrep(2)
        out_type_main = enn.FieldType(self.r2_act, out_channels * [spin2_repr])
        self.out_main = enn.R2Conv(self.dec1.out_type, out_type_main, kernel_size=3, padding=1)

        self.out_d2 = enn.R2Conv(self.dec2.out_type, out_type_main, kernel_size=3, padding=1)
        self.out_d3 = enn.R2Conv(self.dec3.out_type, out_type_main, kernel_size=3, padding=1)
        self.out_d4 = enn.R2Conv(self.dec4.out_type, out_type_main, kernel_size=3, padding=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B,1,H,W)

        x = enn.GeometricTensor(x, self.enc1.in_type)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # optional FC bottleneck (operate on scalar component tensor)
        if self.use_fc_bottleneck:
            B, C, H, W = b.tensor.shape
            z = F.adaptive_avg_pool2d(b.tensor, (self.fc_spatial, self.fc_spatial)).view(B, -1)
            z = F.relu(self.fc1(z))
            z = F.dropout(z, p=0.0, training=self.training)
            z = F.relu(self.fc2(z))
            z = F.dropout(z, p=0.0, training=self.training)
            b_tensor = b.tensor + z.view(B, C, self.fc_spatial, self.fc_spatial).repeat(1, 1, H // self.fc_spatial, W // self.fc_spatial)[:,:,:H,:W]
            b = enn.GeometricTensor(b_tensor, b.type)

        # Decoder: upsample using equivariant upsample if available
        # We'll implement a simple equivariant upsample fallback
        def upsample_geometric(gtensor):
            if hasattr(enn, "R2Upsampling"):
                # instantiate per-call
                up = enn.R2Upsampling(gtensor.type, scale_factor=2, mode='bilinear')
                return up(gtensor)
            else:
                # fallback: plain interpolation on tensor + rewrap (still works but not strictly equivariant)
                t = F.interpolate(gtensor.tensor, scale_factor=2, mode='bilinear', align_corners=False)
                return enn.GeometricTensor(t, gtensor.type)

        d4 = self.dec4(upsample_geometric(b))
        # skip: transform encoder field type to decoder's type before add
        e4_t = e4.transform_to(d4.type) if e4.type != d4.type else e4
        d4 = enn.GeometricTensor(d4.tensor + e4_t.tensor, d4.type)

        d3 = self.dec3(upsample_geometric(d4))
        e3_t = e3.transform_to(d3.type) if e3.type != d3.type else e3
        d3 = enn.GeometricTensor(d3.tensor + e3_t.tensor, d3.type)

        d2 = self.dec2(upsample_geometric(d3))
        e2_t = e2.transform_to(d2.type) if e2.type != d2.type else e2
        d2 = enn.GeometricTensor(d2.tensor + e2_t.tensor, d2.type)

        d1 = self.dec1(upsample_geometric(d2))
        e1_t = e1.transform_to(d1.type) if e1.type != d1.type else e1
        d1 = enn.GeometricTensor(d1.tensor + e1_t.tensor, d1.type)

        # outputs (unwrap to plain tensors)
        out_main = self.out_main(d1).tensor
        out_d2 = self.out_d2(d2).tensor
        out_d3 = self.out_d3(d3).tensor
        out_d4 = self.out_d4(d4).tensor

        # Preserve same return signature as your original (main, d2, d3, d4)
        return out_main, out_d2, out_d3, out_d4
