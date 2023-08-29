
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from model.modulesoneconv10146181localgparapoolpe import *

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)
        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        affinity = (-a+b) / math.sqrt(CK)   # B, THW, HW
        affinity = F.softmax(affinity, dim=1)
        return affinity


    def readout1(self, affinity, mv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W)
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        return mem

class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object
        self.conv_fusion1v = nn.Conv2d(6656, 512, kernel_size=1)
        self.conv_fusion = nn.Conv2d(512, 512, kernel_size=1)

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO() 
        else:
            self.value_encoder = ValueEncoder() 

        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.memory = MemoryReader()
        self.decoder = Decoder()
        self.mask = nn.Conv2d(1536, 512, kernel_size=3, padding=1)
        self.mask2 = nn.Conv2d(1536, 512, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        f16_thin = self.key_comp(f16)
        k16 = f16_thin

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, query,  kf16, mask, other_mask=None):
        if self.single_object:
            f16 = self.value_encoder(frame, query, kf16, mask)
        else:
            f16 = self.value_encoder(frame, query, kf16, mask, other_mask)
        return f16.unsqueeze(2) # B*512*T*H*W



    def encode_valuepe(self, frame, query,  kf16, mask, other_mask=None):
        if self.single_object:
            f16 = self.value_encoder(frame, query, kf16, mask)

            f161 = f16.transpose(1, -1)
            pe = positional_encoding(f161)
            pe = pe.transpose(1, -1)
            pe = self.conv_fusion1v(pe)
            f16_thin = f16 + pe

            f16_thin = self.conv_fusion(f16_thin)
            f16 = f16_thin.unsqueeze(2)

        else:
            f16 = self.value_encoder(frame, query, kf16, mask, other_mask)
            f161 = f16.transpose(1, -1)
            pe = positional_encoding(f161)
            pe = pe.transpose(1, -1)
            pe = self.conv_fusion1v(pe)
            f16_thin = f16 + pe

            f16_thin = self.conv_fusion(f16_thin)
            f16 = f16_thin.unsqueeze(2)

        return f16 # B*512*T*H*W

    def segment1(self, qk16, qv16, qf8, qf4, mk16, mv16, firstk, firstv,adjk,adfv, selector=None):
        affinitylocal = self.memory.get_affinity(mk16, qk16)
        affinityglobal = self.memory.get_affinity(firstk, qk16)
        affinityadj = self.memory.get_affinity(adjk, qk16)

        if self.single_object:

            locals = self.memory.readout1(affinitylocal, mv16)
            globals= self.memory.readout1(affinityglobal, firstv)
            adjs= self.memory.readout1(affinityadj, adfv)

            fusion_masks = torch.cat((locals, globals, adjs), dim=1)
            B2s, CK2s, H2s, W2s = fusion_masks.shape

            fusion_maskpools = self.avg_pool(fusion_masks)
            fusion_maskpools = fusion_maskpools.repeat(1, 1,H2s,W2s)


            fusion_maskadds = fusion_masks + fusion_maskpools


            masks = self.mask(fusion_maskadds)  # 1/32, 512
            masks = torch.sigmoid(masks)  # broadcasting
            local1masks = locals*masks + locals
            global1masks = globals*masks + globals
            adj1masks = adjs*masks + adjs
            decodes = local1masks + global1masks + adj1masks

            decodes = torch.cat([decodes, qv16], dim=1)

            logits = self.decoder(decodes, qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            local1 = self.memory.readout1(affinitylocal, mv16[:, 0])
            local2 = self.memory.readout1(affinitylocal, mv16[:,1])
            global1 = self.memory.readout1(affinityglobal, firstv[:, 0])
            global2 = self.memory.readout1(affinityglobal, firstv[:, 1])
            adj1 = self.memory.readout1(affinityadj, adfv[:, 0])
            adj2 = self.memory.readout1(affinityadj, adfv[:, 1])
            fusion_mask = torch.cat((local1, global1, adj1), dim=1)
            B2, CK2, H2, W2 = fusion_mask.shape

            fusion_maskpool = self.avg_pool(fusion_mask)
            fusion_maskpool = fusion_maskpool.repeat(1, 1,H2,W2)
            fusion_maskadd = fusion_mask + fusion_maskpool

            mask = self.mask(fusion_maskadd)  # 1/32, 512
            mask = torch.sigmoid(mask)  # broadcasting
            local1mask = local1*mask + local1
            global1mask = global1*mask + global1
            adj1mask = adj1*mask + adj1
            decode1 = local1mask + global1mask + adj1mask
            decode1 = torch.cat([decode1, qv16], dim=1)

            fusion_mask2 = torch.cat((local2, global2, adj2), dim=1)
            B2, CK2, H2, W2 = fusion_mask2.shape
            fusion_mask2pool = self.avg_pool(fusion_mask2)
            fusion_mask2pool = fusion_mask2pool.repeat(1, 1,H2,W2)

            fusion_mask2add = fusion_mask2 + fusion_mask2pool

            mask2 = self.mask2(fusion_mask2add)  # 1/32, 512
            mask2 = torch.sigmoid(mask2)  # broadcasting

            local2mask = local2*mask2 + local2
            global2mask = global2*mask2 + global2
            adj2mask = adj2*mask2 + adj2
            decode2 = local2mask + global2mask + adj2mask

            decode2 = torch.cat([decode2, qv16], dim=1)

            logits = torch.cat([
                self.decoder(decode1, qf8, qf4),
                self.decoder(decode2, qf8, qf4),
            ], 1)
            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob



    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'encode_valuepe':
            return self.encode_valuepe(*args, **kwargs)
        elif mode == 'segment1':
            return self.segment1(*args, **kwargs)
        else:
            raise NotImplementedError


