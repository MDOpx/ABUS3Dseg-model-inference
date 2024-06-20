# Python build-ins
from typing import Tuple, Union, List
# PyTorch
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
# nnUNet modules
from nnunet.network_architecture.static_UNet import Static_UNet, Conv3dBlock, NIConv3dBlock
from nnunet.network_architecture.initialization import InitWeights_He
import torchvision
from nnunet.network_architecture.self_attention import LinearAttention, ManualAttention, ManualAttention1D, ManualAttention3D
import pdb
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch.nn.modules.activation import LeakyReLU
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelGate_wSA_v3(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, reduce_size = 5):
        super().__init__()
        self.gate_channels = gate_channels
        self.reduce_size = reduce_size
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, int(math.pow(reduce_size,3))), 
            nn.Sigmoid(),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.sigmoid = nn.Sigmoid()

        self.attn = ManualAttention(1, heads=1, dim_head=1, attn_drop=0., proj_drop=0.)

        self.conv3d = nn.Conv3d(gate_channels, int(math.pow(reduce_size,3)), kernel_size=1, bias=False)
        self.instnorm0 = InstanceNorm3d(int(math.pow(reduce_size,3)), affine=True)
        self.act0 = LeakyReLU(inplace=True)

    def forward(self, x):
        x_re = self.conv3d(x)
        x_re = self.act0(self.instnorm0(x_re))
        x_avg_pool = self.mlp(self.avgpool(x)) 
        q_reshape = x_avg_pool.reshape(-1,1,self.reduce_size,self.reduce_size,self.reduce_size) 
        x_max_pool = self.mlp(self.maxpool(x)) 
        k_reshape = x_max_pool.reshape(-1,1,self.reduce_size,self.reduce_size,self.reduce_size)
        attention = self.sigmoid(x_avg_pool + x_max_pool) 
        v_reshape = attention.reshape(-1,1,self.reduce_size,self.reduce_size,self.reduce_size) 

        # Do Cross SA
        sa, q_k_attn = self.attn(q_reshape, k_reshape, v_reshape)
        sa = sa.reshape(-1,int(math.pow(self.reduce_size,3))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x_re)
        return x_re * sa
    
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, 7, padding=3),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_avg_pool = torch.mean(x, dim=1).unsqueeze(1)
        x_max_pool = torch.max(x, dim=1)[0].unsqueeze(1)
        attention = torch.cat((x_avg_pool, x_max_pool), dim=1)
        attention = self.conv(attention)
        return x*attention

class CBAM_wSA_small_v3(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, channel_attention=True, spatial_attention=True, reduce_size=5):
        super(CBAM_wSA_small_v3, self).__init__()
        self.channel_attention, self.spatial_attention = channel_attention, spatial_attention
        if channel_attention:
            self.channel_gate = ChannelGate_wSA_v3(gate_channels, reduction_ratio, reduce_size)
        if spatial_attention:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        if self.channel_attention:
            x = self.channel_gate(x)
        if self.spatial_attention:
            x = self.spatial_gate(x)
        return x
    
class AGUpConvTwinCBAM3D_wSA_small_v3(nn.Module):
    def __init__(self, g_channels, x_channels, kernel_size, stride, bias, reduce_size = 5, **kwargs):
        super().__init__()
        self.g_conv = nn.Conv3d(int(math.pow(reduce_size,3)), int(math.pow(reduce_size,3)), kernel_size=1)
        self.g_upconv = nn.ConvTranspose3d(int(math.pow(reduce_size,3)), int(math.pow(reduce_size,3)), kernel_size=kernel_size, stride=stride, bias=bias)
        self.x_conv = nn.Conv3d(int(math.pow(reduce_size,3)), int(math.pow(reduce_size,3)), kernel_size=1)
        self.ψ_conv = nn.Conv3d(int(math.pow(reduce_size,3)), 1, kernel_size=1)
        self.g_cbamblock = CBAM_wSA_small_v3(g_channels, channel_attention=True if 'C' in kwargs['cbammode'] else False, spatial_attention=True if 'S' in kwargs['cbammode'] else False, reduce_size = reduce_size)
        self.x_cbamblock = CBAM_wSA_small_v3(x_channels, channel_attention=True if 'C' in kwargs['cbammode'] else False, spatial_attention=True if 'S' in kwargs['cbammode'] else False, reduce_size = reduce_size)

    def forward(self, g, x, **kwargs):
        g_cbam = self.g_cbamblock(g) # CBAM under
        x_cbam = self.x_cbamblock(x)
        
        g_out = self.g_upconv(self.g_conv(g_cbam))
        x_out = self.x_conv(x_cbam)
        ψ = self.ψ_conv(F.relu(g_out + x_out))
        α = torch.sigmoid(ψ)
        return x * α    

class MultitaskAGIBA_NI(Static_UNet):
    def __init__(
        self,
        num_classes, 
        in_channels,
        encoder_module: nn.Module=Conv3dBlock,
        decoder_module: nn.Module=Conv3dBlock,
        pool_op_kernel_sizes = None,
        conv_kernel_sizes = None,
        patch_size = None,
        train_fp_in_seg=False,
        cascade_gap=False,
        attention_module=None,
        cbammode='CS',
        apply_skips='0,1,2,3,4',
        start_iba_ep=0,
        depth = [0, 0, 0, 0, 0, 0] ,       #무조건 0이라도 들어가야함, 모두 0이면 no NA
        num_heads = [0, 0, 0, 0, 0, 0],   #일단 3 통일
        reduce_size = [5, 5, 5, 5, 5, 5],
        projection = [[0], [0], [0], [0], [0], [0]], 
        in_feature = 32,
        is_grad = False
    ):
        super().__init__(num_classes=num_classes, in_channels=in_channels, encoder_module=encoder_module, decoder_module=decoder_module)
        # self.bottleneck = None # bottleneck layers are in encoder part
        self.decoder = None # Multitask uses not decoder but seg_decoder
        self.upsamplers = None # upsamplers are in seg_decoder
        self.deepsupervision = None # deepsupervision are in seg_decoder
        self.cascade_gaps = {}
        self.is_grad = is_grad
        self.num_levels = len(depth)

        OUT_CHs = [in_feature]
        layer_idx = 0

        # Calculate s_th
        min_patch_size = min(patch_size) 
        layer_min_axis = []
        for pool in pool_op_kernel_sizes:
            min_patch_size = min_patch_size / min(pool)
            layer_min_axis.append(min_patch_size) 
        min_axis = min(layer_min_axis)
        s_th = [math.ceil(l/min_axis) for l in layer_min_axis] 

        target_reduce_size = []
        min_patch_size = min(patch_size)
        for s, min_axis in zip(s_th, layer_min_axis):
            target_reduce_size.append(math.ceil(min_axis/s))

        self.encoder = nn.ModuleList([encoder_module(in_channels, in_feature, padding=1, depth=depth[layer_idx], num_heads=num_heads[layer_idx], reduce_size=target_reduce_size[0], projection=projection[layer_idx])])
        for ix, (pool, conv) in enumerate(zip(pool_op_kernel_sizes[:-1], conv_kernel_sizes[:-1])):
            out_feature = in_feature*2
            if out_feature > 256: out_feature = 320
            kernel = tuple(conv)
            stride = tuple(pool)
            self.encoder.append(encoder_module(in_feature, out_feature, kernel_size= kernel, padding=1, stride0=stride, stride1=1, depth=depth[ix+1], num_heads=num_heads[ix+1], reduce_size=target_reduce_size[ix+1], projection=projection[ix+1]))
            in_feature = in_feature * 2
            OUT_CHs.append(out_feature)

        kernel = tuple(conv_kernel_sizes[-1])
        stride = tuple(pool_op_kernel_sizes[-1])
        # Bottle-neck
        self.bottleneck = encoder_module(out_feature, out_feature, kernel_size=kernel, padding=1, stride0=stride, stride1=1, depth=depth[-1], num_heads=num_heads[-1], reduce_size=reduce_size[-1], projection=projection[-1]) 

        # Classification
        if cascade_gap:
            self.cascade_gaps = {i: nn.AdaptiveAvgPool3d(1) for i in cascade_gap}
            self.cls_head = CLS_Head(
                in_channels=OUT_CHs[-1] + np.array([OUT_CHs[i] for i in cascade_gap]).sum(),
                out_channels=num_classes if not num_classes == -1 else 1,
                pooling='avg',
                dropout=0.5,
            )
        else:
            self.cls_head = CLS_Head(
                in_channels=OUT_CHs[-1],
                out_channels=num_classes if not num_classes == -1 else 1,
                pooling='avg',
                dropout=0.5,
            )
        
        # Segmentation
        self.seg_decoder = SEG_Decoder_with_AG(decoder_module, OUT_CHs, pool_op_kernel_sizes, conv_kernel_sizes, out_channels=num_classes if not num_classes == -1 else 2, attention_module=attention_module, cbammode=cbammode, apply_skips=apply_skips, start_iba_ep=start_iba_ep, do_ds= not is_grad, deep_supervision=not is_grad, target_reduce_size=target_reduce_size)

        # Reconstruction
        self.rec_decoder = None # Res_UNet_rec_decoder(self.encoder)

        self.apply(InitWeights_He)

        # IBA
        self.sigmoid = nn.Sigmoid()
        from nnunet.network_architecture.iba.pytorch import _SpatialGaussianKernel
        self.sigma = 1.0
        smooth_kernel_size = int(round(2 * self.sigma)) * 2 + 1  # Cover 2.5 stds in both directions
        self.smooth = _SpatialGaussianKernel(smooth_kernel_size, self.sigma, 32).to('cuda')

    def forward(self, x):
        
        # Encoder phase
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
                
        last_feature = self.bottleneck(x) 
        skips.append(last_feature)
        
        # Classification
        cls_output = self.cls_head(last_feature)

        # Segmentation
        seg_decoder_output = self.seg_decoder(last_feature, skips)

        if self.is_grad:
            return seg_decoder_output
        else:
            return  cls_output, seg_decoder_output

class SEG_Decoder_with_AG(nn.Module):
    def __init__(
        self,
        decoder_module: nn.Module=Conv3dBlock,
        channels=[],
        pool_op_kernel_sizes=None,
        conv_kernel_sizes = None,
        out_channels=2,
        attention_module=None,
        cbammode='CS',
        apply_skips='0,1,2,3,4',
        start_iba_ep=0,
        do_ds=True, deep_supervision = True,
        target_reduce_size=None
    ):
        super().__init__()
        self.do_ds = do_ds
        self.final_nonlin = lambda x: x
        self.deep_supervision =deep_supervision
        out_channels = out_channels

        kernel = tuple(conv_kernel_sizes[-1])
        stride = tuple(pool_op_kernel_sizes[-1])
        self.decoder = nn.ModuleList([decoder_module(channels[-1]*2, channels[-1], kernel, padding=1)])
        self.upsamplers = nn.ModuleList([nn.ConvTranspose3d(channels[-1], channels[-1], kernel_size=stride, stride=stride, bias=False)])
        self.attention = nn.ModuleList([eval(attention_module)(channels[-1], channels[-1], kernel_size=stride, stride=stride, bias=False, reduce_size=target_reduce_size[-1], cbammode=cbammode)])
        self.deepsupervision = nn.ModuleList()

        for c in channels[::-1][1:]: # 256 128 64 32
            self.decoder.append(decoder_module(c*2, c, padding=1))
        for ix, pool in enumerate(pool_op_kernel_sizes[::-1][1:]):
            dec = tuple(pool)
            self.upsamplers.append(nn.ConvTranspose3d(channels[-(ix+1)], channels[-(ix+2)], kernel_size=dec, stride=dec, bias=False))
            self.attention.append(eval(attention_module)(channels[-(ix+1)], channels[-(ix+2)], kernel_size=dec, stride=dec, bias=False, reduce_size=target_reduce_size[-(ix+2)], cbammode=cbammode))
        for c in channels[::-1][:]: # 320 256 128 64 32
            self.deepsupervision.append(nn.Conv3d(c, out_channels, kernel_size=1, stride=1, bias=False))
        
        self.upscale_logits_ops = [lambda x: x for _ in range(4)]
        self.apply_skips = [int(i) for i in apply_skips.split(',')]
        self.start_iba_ep = start_iba_ep

    def forward(self, x, skips):
        seg_outputs = []

        for ix, (dec, up, ds, att) in enumerate(zip(self.decoder, self.upsamplers, self.deepsupervision, self.attention)):
            x_up = up(x)
            if len(skips)-(ix+2) in self.apply_skips:
                x = att(x, skips[-(ix+2)])
                x = torch.cat((x_up, x), dim=1)
            else:
                x = torch.cat((x_up, skips[-(ix+2)]), dim=1)
            x = dec(x)
            seg_outputs.append(self.final_nonlin(ds(x)))

        if self.deep_supervision and self.do_ds:
            return tuple(
                [seg_outputs[-1]] +
                [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
            )
        else:
            return seg_outputs[-1]
        
class CLS_Head(nn.Sequential):

    def __init__(self, in_channels, out_channels, pooling="avg", dropout=0.2):
        super().__init__()

        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        self.pool = nn.AdaptiveAvgPool3d(1) if pooling == 'avg' else nn.AdaptiveMaxPool3d(1)
        self.flatten = Flatten()
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        
        self.linear  = nn.Linear(in_channels, out_channels, bias=True)
        #super().__init__(self.pool, self.flatten, self.dropout, self.linear)
    
    def forward(self, input, cascade_gaps=None):
        #input = [batch, last_conv_output_channels, last_conv_output_width, last_conv_output_height] ex: [1, 320, 5, 5, 6]
        avg_pool = self.pool(input) # ex: avg_pool = [1, 320, 1, 1, 1]
        if cascade_gaps:
            flat = self.flatten(torch.concat([torch.concat(cascade_gaps, dim=1), avg_pool], dim=1))
        else:
            flat = self.flatten(avg_pool) # ex: flat = [1, 320] ← [1, 320*1*1*1]
        drop_out = self.dropout(flat) 
        linear = self.linear(drop_out) # ex: linear = [1, 1]
        return linear
