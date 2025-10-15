import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class main_net(nn.Module):
    def __init__(self, 
                 in_channels=1, out_channels=3, base_channels=32,
                 use_fc_bottleneck=True, fc_hidden=512, fc_spatial=4, epochs=200,
                 pool_type='max', dropout_1=0, dropout_2=0, dropout_3=0):
        super().__init__()
        self.use_fc_bottleneck = use_fc_bottleneck
        self.fc_spatial = fc_spatial
        self.dropout_2 = dropout_2

        # Encoder
        self.enc1 = ResidualBlockSE(in_channels, base_channels, pool_type=pool_type, dropout_p=dropout_1)
        self.enc2 = ResidualBlockSE(base_channels, base_channels*2, pool_type=pool_type, dropout_p=dropout_1)
        self.enc3 = ResidualBlockSE(base_channels*2, base_channels*4, pool_type=pool_type, dropout_p=dropout_1)
        self.enc4 = ResidualBlockSE(base_channels*4, base_channels*8, pool_type=pool_type, dropout_p=dropout_1)
        self.pool = nn.MaxPool2d(2)

        # Multi-level Attention blocks after each encoder stage
        self.att1 = AttentionBlock(base_channels, base_channels*2, base_channels)
        self.att2 = AttentionBlock(base_channels*2, base_channels*4, base_channels*2)
        self.att3 = AttentionBlock(base_channels*4, base_channels*8, base_channels*4)

        # Optional FC bottleneck
        if use_fc_bottleneck:
            self.fc1 = nn.Linear(base_channels*8*fc_spatial*fc_spatial, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, base_channels*8*fc_spatial*fc_spatial)

        # Decoder with upsampling + attention-refined skip connections
        self.up4 = nn.ConvTranspose2d(base_channels*8, base_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.dec4 = ResidualBlockSE(base_channels*8 + base_channels*4, base_channels*4, pool_type=pool_type, dropout_p=dropout_3)
        self.dec3 = ResidualBlockSE(base_channels*4 + base_channels*2, base_channels*2, pool_type=pool_type, dropout_p=dropout_3)
        self.dec2 = ResidualBlockSE(base_channels*2 + base_channels, base_channels, pool_type=pool_type, dropout_p=dropout_3)
        self.dec1 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # Multi-scale output heads (unchanged)
        self.out_d4 = nn.Conv2d(base_channels*4, out_channels, 3, padding=1)
        self.out_d3 = nn.Conv2d(base_channels*2, out_channels, 3, padding=1)
        self.out_d2 = nn.Conv2d(base_channels,   out_channels, 3, padding=1)

        self.register_buffer("train_curve", torch.zeros(epochs))
        self.register_buffer("val_curve", torch.zeros(epochs))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Encoder + Attention
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Attention refinement on skip paths
        a1 = self.att1(e1, e2)
        a2 = self.att2(e2, e3)
        a3 = self.att3(e3, e4)

        # Optional FC bottleneck
        if self.use_fc_bottleneck:
            B, C, H, W = e4.shape
            z = F.adaptive_avg_pool2d(e4, (self.fc_spatial, self.fc_spatial)).view(B, -1)
            z = F.relu(self.fc1(z))
            z = F.dropout(z, p=self.dropout_2, training=self.training)
            z = F.relu(self.fc2(z))
            z = F.dropout(z, p=self.dropout_2, training=self.training)
            e4 = F.interpolate(z.view(B, C, self.fc_spatial, self.fc_spatial),
                               size=(H, W), mode='bilinear', align_corners=False)

        # Decoder with attention-refined skip connections
        d4 = self.up4(e4)
        d4 = torch.cat([d4, a3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, a2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, a1], dim=1)
        d2 = self.dec2(d2)

        out_main = self.dec1(d2)
        out_d4 = self.out_d4(d4)
        out_d3 = self.out_d3(d3)
        out_d2 = self.out_d2(d2)

        return out_main, out_d2, out_d3, out_d4

