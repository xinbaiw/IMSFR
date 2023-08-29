import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modulesoneconv10146181localgparapoolpe import *
from model.networkoneconv10146181localgparapoolpe import Decoder

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
        return torch.cat(encoding, dim=-1).cuda()
class STCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder()
        self.conv_fusion1v = nn.Conv2d(6656, 512, kernel_size=1)
        self.conv_fusion = nn.Conv2d(512, 512, kernel_size=1)
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decoder = Decoder()
        self.mask = nn.Conv2d(1536, 512, kernel_size=3, padding=1)
        self.mask2 = nn.Conv2d(1536, 512, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def encode_value(self, frame,curframe, kf16, masks):
        k, _, h, w = masks.shape
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        curframe = curframe.view(1, 3, h, w).repeat(k, 1, 1, 1)
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame,curframe, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)
    def encode_valuepe(self, frame, curframe, kf16, masks):
        k, _, h, w = masks.shape
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        curframe = curframe.view(1, 3, h, w).repeat(k, 1, 1, 1)
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame,curframe, kf16.repeat(k,1,1,1), masks, others)

        f161 = f16.transpose(1, -1)
        pe = positional_encoding(f161)
        pe = pe.transpose(1, -1)
        pe = self.conv_fusion1v(pe)
        f16_thin = f16 + pe
        f16_thin = self.conv_fusion(f16_thin)
        f16 = f16_thin.unsqueeze(2)

        return f16


    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        f16_thin = self.key_comp(f16)
        k16 = f16_thin
        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16):
        k = mem_bank.num_objects

        local1,global1,adj1 = mem_bank.match_memory(qk16)

        fusion_mask = torch.cat((local1, global1, adj1), dim=1)
        B2, CK2, H2, W2 = fusion_mask.shape
        fusion_maskpool = self.avg_pool(fusion_mask)
        fusion_maskpool = fusion_maskpool.repeat(1, 1, H2, W2)
        fusion_maskadd = fusion_mask + fusion_maskpool

        mask = self.mask(fusion_maskadd)  # 1/32, 512
        mask = torch.sigmoid(mask)  # broadcasting
        local1mask = local1 * mask + local1
        global1mask = global1 * mask + global1
        adj1mask = adj1 * mask + adj1
        decode1 = local1mask + global1mask + adj1mask

        qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([decode1, qv16], 1)
        decode = self.decoder(qv16, qf8, qf4)
        return torch.sigmoid(decode)
